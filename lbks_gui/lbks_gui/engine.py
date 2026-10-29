"""CPU inference and position-only numerical IK; no ROS or GUI dependency."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytorch_kinematics as pk
import torch
from torch import nn

from lbks_gui.network_trace import activation_stage, predict_with_trace


class PositionMLP(nn.Module):
    """Matches the planner's net.* state_dict and configurable MLP variants."""

    def __init__(self, dof, hidden_sizes=(128, 128), activation='relu',
                 output_scale=math.pi):
        super().__init__()
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        if activation not in activations:
            raise ValueError(f'Unsupported activation: {activation}')
        layers = []
        previous = 3
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, int(width)), activations[activation]()])
            previous = int(width)
        layers.extend([nn.Linear(previous, dof), nn.Tanh()])
        self.net = nn.Sequential(*layers)
        self.output_scale = float(output_scale)

    def forward(self, xyz):
        return self.output_scale * self.net(xyz)


class PreviewEngine:
    def __init__(self, urdf, tip='tool0'):
        self.chain = pk.build_serial_chain_from_urdf(urdf, tip).to(
            device='cpu', dtype=torch.float32)
        self.names = self.chain.get_joint_parameter_names()
        joints = {j.get('name'): j for j in ET.fromstring(urdf).findall('joint')}
        bounds = []
        for name in self.names:
            joint = joints[name]
            limit = joint.find('limit')
            if joint.get('type') == 'continuous':
                bounds.append((-math.pi, math.pi))
            else:
                bounds.append((float(limit.get('lower')), float(limit.get('upper'))))
        self.lower, self.upper = torch.tensor(bounds).T
        self.models = {}

    def fk(self, q):
        q = torch.as_tensor(q, dtype=torch.float32).reshape(1, len(self.names))
        return self.chain.forward_kinematics(q).get_matrix()[0, :3, 3]

    def model(self, spec):
        path = Path(spec['path']).expanduser().resolve()
        key = (str(path), path.stat().st_mtime_ns, repr(spec))
        if key not in self.models:
            if spec.get('format', 'state_dict') == 'torchscript':
                model = torch.jit.load(str(path), map_location='cpu')
            elif spec.get('format', 'state_dict') == 'state_dict':
                model = PositionMLP(
                    len(self.names), spec.get('hidden_sizes', [128, 128]),
                    spec.get('activation', 'relu'), spec.get('output_scale', math.pi))
                weights = torch.load(path, map_location='cpu', weights_only=True)
                if 'state_dict' in weights:
                    weights = weights['state_dict']
                model.load_state_dict(weights, strict=True)
            else:
                raise ValueError('Model format must be state_dict or torchscript')
            self.models[key] = model.eval()
        return self.models[key]

    def solve(self, target, seed, spec=None):
        target = torch.as_tensor(target, dtype=torch.float32)
        if target.shape != (3,) or not torch.isfinite(target).all():
            raise ValueError('Target must contain three finite XYZ coordinates')
        if spec is not None:
            model = self.model(spec)
            result, trace = predict_with_trace(
                model, target, isinstance(model, PositionMLP), self.names)
            q = result[0]
            if not torch.isfinite(q).all():
                raise ValueError('Model returned non-finite joint values')
        else:
            trace = {'mode': 'numerical', 'stages': [],
                     'note': 'Numerical IK has no neural network. Load a trained checkpoint to inspect predictions.'}
            # Damped least squares, with deterministic restarts near singularities.
            best_error = float('inf')
            q = torch.as_tensor(seed, dtype=torch.float32).clone()
            generator = torch.Generator().manual_seed(7)
            for attempt in range(3):
                candidate = q.clone() if attempt == 0 else (
                    torch.rand(len(self.names), generator=generator) * 2 - 1) * math.pi
                candidate = candidate.clamp(self.lower, self.upper)
                for _ in range(100):
                    position = self.fk(candidate)
                    error = target - position
                    distance = error.norm().item()
                    if distance < best_error:
                        best_error, q = distance, candidate.clone()
                    if distance < 0.002:
                        break
                    jac = torch.autograd.functional.jacobian(self.fk, candidate)
                    delta = jac.T @ torch.linalg.solve(
                        jac @ jac.T + 0.0025 * torch.eye(3), error)
                    candidate = (candidate + delta.clamp(-0.25, 0.25)).clamp(
                        self.lower, self.upper).detach()
                if best_error < 0.002:
                    break
        clipped = bool(((q < self.lower) | (q > self.upper)).any())
        q = q.clamp(self.lower, self.upper).detach()
        actual = self.fk(q).detach()
        if spec is not None:
            trace['stages'].append(activation_stage(
                'URDF joint limits · rad', 'limits', q, self.names))
            trace['stages'].append(activation_stage(
                'Forward kinematics · m', 'fk', actual, ['X', 'Y', 'Z']))
            trace['clipped'] = clipped
        return {'joints': q.tolist(), 'actual': actual.tolist(),
                'error': (target - actual).norm().item(), 'clipped': clipped,
                'network_trace': trace}
