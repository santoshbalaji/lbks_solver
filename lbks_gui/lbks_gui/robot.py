"""Shared robot and model configuration, independent of Torch and Qt."""

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def load_robot(path):
    root = ET.parse(path).getroot()
    # The exported training URDF contains machine-specific mesh paths.
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename', '')
        if '/lbks_description/meshes/' in filename:
            mesh.set('filename', 'package://lbks_description/meshes/' +
                     filename.split('/lbks_description/meshes/', 1)[1])
    children = {j.find('child').get('link') for j in root.findall('joint')}
    roots = [link.get('name') for link in root.findall('link')
             if link.get('name') not in children]
    if len(roots) != 1:
        raise ValueError('URDF must have exactly one root link')
    return ET.tostring(root, encoding='unicode'), roots[0]


def load_variants(path):
    config_path = Path(path).expanduser().resolve()
    entries = (yaml.safe_load(config_path.read_text()) or {}).get('models', {})
    for name, spec in entries.items():
        if name in ('numerical', '__custom__'):
            raise ValueError(f'Reserved model name: {name}')
        if spec.get('path'):
            checkpoint = Path(spec['path']).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = config_path.parent / checkpoint
            spec['path'] = str(checkpoint)
    return entries
