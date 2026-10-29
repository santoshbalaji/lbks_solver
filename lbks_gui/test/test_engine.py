"""CPU regression tests for model variants, IK, and portable robot assets."""

import math
import json
from pathlib import Path
import tempfile
import unittest

import torch

from lbks_gui.engine import PositionMLP, PreviewEngine
from lbks_gui.robot import load_robot, load_variants
from lbks_gui.network_trace import MAX_VALUES, predict_with_trace


ROBOT = Path(__file__).resolve().parents[2] / 'lbks_description/urdf/ur/ur.urdf'


class TestPreview(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.urdf, self.root = load_robot(ROBOT)
        self.engine = PreviewEngine(self.urdf)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_portable_meshes(self):
        import xml.etree.ElementTree as ET
        self.assertEqual(self.root, 'world')
        for mesh in ET.fromstring(self.urdf).iter('mesh'):
            url = mesh.get('filename')
            self.assertTrue(url.startswith('package://lbks_description/'))
            self.assertTrue((ROBOT.parents[2] / url.split('lbks_description/', 1)[1]).is_file())

    def test_numerical_reaches_target(self):
        result = self.engine.solve([0.4, 0., 0.4], [0.] * 6)
        self.assertEqual(len(result['joints']), 6)
        self.assertLess(result['error'], 0.002)
        self.assertEqual(result['network_trace']['mode'], 'numerical')
        self.assertEqual(result['network_trace']['stages'], [])

    def test_unreached_target_reports_residual(self):
        result = self.engine.solve([4., 4., 4.], [0.] * 6)
        self.assertGreater(result['error'], 1.)
        q = torch.tensor(result['joints'])
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(((q >= self.engine.lower) & (q <= self.engine.upper)).all())

    def test_state_dict_variants_match_direct_inference(self):
        target = torch.tensor([[0.4, 0., 0.4]])
        for hidden in ([128, 128], [64, 32, 16]):
            model = PositionMLP(6, hidden).eval()
            checkpoint = self.path / f'model-{len(hidden)}.pth'
            torch.save(model.state_dict(), checkpoint)
            result = self.engine.solve(target[0], [0.] * 6,
                                       {'path': str(checkpoint), 'hidden_sizes': hidden})
            torch.testing.assert_close(torch.tensor(result['joints']), model(target)[0])

    def test_scripted_variant(self):
        model = PositionMLP(6, [32]).eval()
        path = self.path / 'scripted.pt'
        torch.jit.trace(model, torch.zeros(1, 3)).save(str(path))
        result = self.engine.solve([.4, 0., .4], [0.] * 6,
                                   {'path': str(path), 'format': 'torchscript'})
        self.assertEqual(len(result['joints']), 6)
        self.assertEqual(result['network_trace']['mode'], 'endpoints')
        self.assertEqual([s['kind'] for s in result['network_trace']['stages']],
                         ['input', 'output', 'limits', 'fk'])

    def test_trace_records_actual_layers_and_contributions(self):
        model = PositionMLP(6, [12, 8]).eval()
        target = torch.tensor([.4, -.2, .3])
        prediction, trace = predict_with_trace(model, target, True, self.engine.names)
        torch.testing.assert_close(prediction, model(target.unsqueeze(0)))
        value = target.unsqueeze(0)
        for layer, stage in zip(model.net, trace['stages'][1:-1]):
            value = layer(value)
            torch.testing.assert_close(torch.tensor(stage['values']), value[0])
            for index, detail in enumerate(stage.get('neurons', [])):
                reconstructed = detail['bias'] + detail['other_sum'] + sum(
                    term['product'] for term in detail['terms'])
                self.assertAlmostEqual(reconstructed, value[0, index].item(), places=5)
        torch.testing.assert_close(torch.tensor(trace['stages'][-1]['values']), prediction[0])
        self.assertTrue(all(not layer._forward_hooks for layer in model.net))
        json.dumps(trace, allow_nan=False)

    def test_trace_is_bounded_and_hooks_removed_after_failure(self):
        model = PositionMLP(6, [MAX_VALUES + 16]).eval()
        _, trace = predict_with_trace(model, torch.zeros(3), True, self.engine.names)
        stage = trace['stages'][1]
        self.assertEqual(stage['count'], MAX_VALUES + 16)
        self.assertEqual(len(stage['values']), MAX_VALUES)
        self.assertEqual(len(stage['neurons']), MAX_VALUES)
        with torch.no_grad():
            model.net[0].weight.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'Non-finite activation'):
            predict_with_trace(model, torch.zeros(3), True, self.engine.names)
        self.assertTrue(all(not layer._forward_hooks for layer in model.net))

    def test_trace_separates_raw_predictions_from_joint_limits(self):
        model = PositionMLP(6, [8], output_scale=20.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[-2].bias.fill_(2.)
        path = self.path / 'clipped.pth'
        torch.save(model.state_dict(), path)
        result = self.engine.solve([.4, 0., .4], [0.] * 6,
                                   {'path': str(path), 'hidden_sizes': [8], 'output_scale': 20.})
        stages = result['network_trace']['stages']
        self.assertTrue(result['clipped'])
        self.assertGreater(stages[-3]['values'][0], self.engine.upper[0].item())
        self.assertEqual(stages[-2]['values'], result['joints'])
        self.assertEqual(stages[-1]['values'], result['actual'])

    def test_invalid_target_and_architecture(self):
        for target in ([1., 2.], [math.nan, 0., 0.]):
            with self.assertRaises(ValueError):
                self.engine.solve(target, [0.] * 6)
        path = self.path / 'wrong.pth'
        torch.save(PositionMLP(7).state_dict(), path)
        with self.assertRaises(RuntimeError):
            self.engine.solve([.4, 0., .4], [0.] * 6, {'path': str(path)})

    def test_variant_relative_paths(self):
        config = self.path / 'models.yaml'
        config.write_text('models:\n  baseline:\n    path: baseline.pth\n')
        variants = load_variants(config)
        self.assertEqual(variants['baseline']['path'], str(self.path / 'baseline.pth'))


if __name__ == '__main__':
    unittest.main()
