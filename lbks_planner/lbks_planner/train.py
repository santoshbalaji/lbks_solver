from torch import nn
import pytorch_kinematics as pk

import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as R

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

        self.__batch_size = batch_size
        self.__epoch = epoch
        self.__dof = dof

        # move to GPU if available
        self.__chain = chain.to(device="cuda", dtype=torch.float32)

        self.__model = IKFeedForward(dof).to("cuda")

        self.get_logger().info("trainer initialised")


    def find_fk(self, q : torch.Tensor):
        jp = torch.as_tensor(
            q,
            dtype=self.__chain.dtype,
            device=self.__chain.device).unsqueeze(0)

        # compute FK
        transform = self.__chain.forward_kinematics(jp)

        # 4x4 transformation matrix
        T = transform.get_matrix()[0]

        position = T[:3, 3]
        quaternion = R.from_matrix(T[:3, :3].detach().cpu().numpy()).as_quat()
        return position, quaternion


    def train_ik(self):
        optimizer = torch.optim.Adam(self.__model.parameters(), lr=1e-3)
        
        self.__model.train()
        for epoch in range(self.__epoch):

            # generate random joint configurations
            train_outputs = (
                torch.rand(
                    self.__batch_size,
                    self.__dof,
                    device="cuda") * 2 - 1) * math.pi

            # generate target XYZ using FK
            with torch.no_grad():
                train_inputs = self.find_fk(train_outputs)

            # predict joint angles
            q_pred = self.__model(train_inputs)

            # compute FK of predicted joints
            xyz_pred = self.find_fk(q_pred)

            # Cartesian position loss
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



def main(args=None):
    rclpy.init(args=args)
    train = None
    try:
        check_cuda_state()
        train = Train(
            urdf_path="/home/santosh/purdue_ws/src/milestone_2/lbks_description/urdf/ur/ur.urdf",
            dof=6,
            batch_size=2000,
            epoch=3000)
        train.train_ik()
    finally:
        if train is not None:
            train.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

