"""Regression checks for batched FK and the Cartesian training loss."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import pytorch_kinematics as pk
import torch

from lbks_planner.train import IKFeedForward, Train


class TestTrainShapes(unittest.TestCase):
    def setUp(self):
        urdf_path = (
            Path(__file__).resolve().parents[2]
            / 'lbks_description/urdf/ur/ur.urdf'
        )
        chain = pk.build_serial_chain_from_urdf(
            urdf_path.read_text(), end_link_name='tool0')
        self.dof = len(chain.get_joint_parameter_names())
        # Exercise tensor operations on CPU without starting a ROS node.
        self.trainer = object.__new__(Train)
        self.trainer._Train__chain = chain
        self.trainer._Train__dof = self.dof

    def test_single_and_batch_positions_match(self):
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                joints = torch.rand(batch_size, self.dof)
                positions = self.trainer.find_fk(joints)
                self.assertEqual(positions.shape, (batch_size, 3))
                expected = torch.stack([
                    self.trainer.find_fk(q) for q in joints])
                torch.testing.assert_close(positions, expected)

    def test_invalid_joint_shapes(self):
        for shape in ((), (self.dof + 1,), (2, self.dof + 1), (1, 2, self.dof)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, 'Expected joint shape'):
                    self.trainer.find_fk(torch.zeros(shape))

    def test_position_gradients(self):
        joints = torch.rand(4, self.dof, requires_grad=True)
        self.trainer.find_fk(joints).square().sum().backward()
        self.assertEqual(joints.grad.shape, joints.shape)
        self.assertTrue(torch.isfinite(joints.grad).all())
        self.assertGreater(joints.grad.abs().sum().item(), 0)

    def test_training_updates_model(self):
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                torch.manual_seed(42)
                model = IKFeedForward(self.dof)
                self.trainer._Train__model = model
                self.trainer._Train__batch_size = batch_size
                self.trainer._Train__epoch = 2
                before = [p.detach().clone() for p in model.parameters()]
                with patch.object(Train, 'get_logger', return_value=Mock()):
                    with patch('lbks_planner.train.torch.save') as save:
                        self.trainer.train_ik()
                        save.assert_called_once()
                self.assertTrue(any(
                    not torch.equal(old, new)
                    for old, new in zip(before, model.parameters())))
                self.assertTrue(all(
                    torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
