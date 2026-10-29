from torch import nn
import pytorch_kinematics as pk

import rclpy
from rclpy.node import Node

import math

import torch

from lbks_planner.utils import check_cuda_state


class IKFeedForward(nn.Module):
    def __init__(self, dof : int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dof),
            nn.Tanh())


    def forward(self, xy):
        return math.pi * self.net(xy)  # joint limits [-pi, pi]



class Train(Node):
    def __init__(
        self,
        urdf_path : str,
        dof : int,
        batch_size : int,
        epoch: int):

        super().__init__('train_node')
        with open(urdf_path) as f:
            urdf = f.read()

        # create kinematic chain
        chain = pk.build_serial_chain_from_urdf(
            urdf,
            end_link_name="tool0")

        self.__dof = dof
        self.__batch_size = batch_size
        self.__epoch = epoch
        self.__device = "cuda"

        # move to GPU if available
        self.__chain = chain.to(device=self.__device, dtype=torch.float32)

        self.__model = IKFeedForward(dof).to(self.__device)

        self.get_logger().info("trainer initialised")


    def find_fk(self, q: torch.Tensor):
        # can accept batch or single input (multiple joint positions / single joint position)
        q = torch.as_tensor(
            q,
            dtype=self.__chain.dtype,
            device=self.__chain.device)
        matrices = self.__chain.forward_kinematics(q).get_matrix()
        positions = matrices[:, :3, 3]  # XYZ is the last column of each matrix.
        return positions


    def train_ik(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=1e-3)

        # set the model in training mode
        self.__model.train()
        for epoch in range(self.__epoch):
            # generate random joint configurations
            train_outputs = (
                torch.rand(
                    self.__batch_size,
                    self.__dof,
                    device=self.__chain.device,
                    dtype=self.__chain.dtype) * 2 - 1) * math.pi

            # generate target XYZ using FK
            with torch.no_grad():
                train_inputs = self.find_fk(train_outputs)

            # predict batch joint angles (batch training input positions & batch joint positions)
            q_pred = self.__model(train_inputs)

            # compute FK of predicted joints
            xyz_pred = self.find_fk(q_pred)

            # cartesian position loss
            loss = torch.mean(
                torch.sum(
                    (xyz_pred - train_inputs) ** 2,
                    dim=1))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                self.get_logger().info(
                    f"Epoch {epoch} | Loss: {loss.item():.6f}")

        # save trained model
        torch.save(
            self.__model.state_dict(),
            "ik_model.pth")

        self.get_logger().info("training completed")


    def predict_ik(self, pose : torch.Tensor):
        # set the model in evaluation mode
        self.__model.eval()

        pose = torch.tensor(
            pose,
            dtype=torch.float32,
            device=self.__device).unsqueeze(0)

        with torch.no_grad():
            q_pred = self.__model(pose)
            pose_actual = self.find_fk(q_pred)

        error = torch.linalg.norm(
            pose_actual - pose,
            dim=1)

        self.get_logger().info("prediction error: " + str(error))


def main(args=None):
    rclpy.init(args=args)
    train = None
    try:
        check_cuda_state()
        train = Train(
            urdf_path="/home/santosh/purdue_ws/src/milestone_2/lbks_description/urdf/ur/ur.urdf",
            dof=6,
            batch_size=2000,
            epoch=10000)
        train.train_ik()
        train.predict_ik([0.3, 0.7, 0.4])
    finally:
        if train is not None:
            train.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

