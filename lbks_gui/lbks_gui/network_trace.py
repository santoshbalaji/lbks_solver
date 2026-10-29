"""Capture real forward-pass values for the network inspector.

Only a bounded number of values and weighted contributions are sent to the GUI.
The model still computes its full output, with its normal forward method.
"""

import torch
from torch import nn


MAX_VALUES = 256
TOP_INPUTS = 4


def activation_stage(name, kind, values, labels=None):
    flat = values.detach().cpu().reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError(f'Non-finite activation in {name}')
    return {
        'name': name, 'kind': kind, 'count': flat.numel(),
        'values': flat[:MAX_VALUES].tolist(),
        'labels': labels or [],
        'min': flat.min().item(), 'max': flat.max().item(),
        'mean': flat.mean().item(), 'zeros': int((flat == 0).sum().item()),
    }


def linear_details(layer, inputs):
    """Explain y[j] = sum(w[j, i] * x[i]) + bias[j] for displayed neurons."""
    values = inputs.detach().reshape(-1)
    weights = layer.weight.detach()[:MAX_VALUES]
    products = weights * values
    top_indices = products.abs().topk(min(TOP_INPUTS, values.numel()), dim=1).indices
    details = []
    for index, top in enumerate(top_indices):
        terms = [{'input': i, 'value': values[i].item(),
                  'weight': weights[index, i].item(), 'product': products[index, i].item()}
                 for i in top.tolist()]
        details.append({
            'bias': layer.bias[index].item() if layer.bias is not None else 0.0,
            'terms': terms,
            'other_sum': (products[index].sum() - products[index, top].sum()).item(),
        })
    return details


def predict_with_trace(model, target, inspect_layers, joint_names):
    """Run once and remove all temporary hooks, including on failure."""
    stages = [activation_stage('Cartesian input · m', 'input', target, ['X', 'Y', 'Z'])]
    handles = []

    def record_layer(name, kind):
        def hook(layer, inputs, output):
            stage = activation_stage(name, kind, output)
            if isinstance(layer, nn.Linear):
                stage['neurons'] = linear_details(layer, inputs[0])
            stages.append(stage)
        return hook

    if inspect_layers:
        for index, layer in enumerate(model.net):
            kind = type(layer).__name__
            name = (f'Linear {layer.in_features} → {layer.out_features}'
                    if isinstance(layer, nn.Linear) else kind)
            handles.append(layer.register_forward_hook(record_layer(f'{index + 1}. {name}', kind)))
    try:
        with torch.inference_mode():
            prediction = model(target.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(prediction, torch.Tensor) or prediction.shape != (1, len(joint_names)):
        raise ValueError(f'Model must return shape (1, {len(joint_names)})')
    name = (f'Scale × {model.output_scale:.5g} · rad' if inspect_layers
            else 'Model output · rad')
    stages.append(activation_stage(name, 'output', prediction, joint_names))
    return prediction, {
        'mode': 'layers' if inspect_layers else 'endpoints',
        'parameters': sum(p.numel() for p in model.parameters()),
        'stages': stages,
        'note': ('Captured activations from this prediction. Replay is slowed down for inspection.'
                 if inspect_layers else
                 'TorchScript: internal activations unavailable. Only real inputs and outputs are shown.'),
    }
