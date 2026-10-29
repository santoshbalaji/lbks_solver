from pathlib import Path
from setuptools import find_packages, setup

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/lbks_gui']),
    ('share/lbks_gui', ['package.xml', 'README.md']),
    ('share/lbks_gui/launch', ['launch/visualize.launch.py']),
    ('share/lbks_gui/config', ['config/models.yaml']),
]
for directory in sorted(Path('web').rglob('*')):
    if directory.is_dir():
        files = [str(path) for path in directory.iterdir() if path.is_file()]
        if files:
            data_files.append(('share/lbks_gui/' + str(directory), files))
data_files.append(('share/lbks_gui/web', [str(path) for path in Path('web').iterdir() if path.is_file()]))

setup(
    name='lbks_gui',
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=False,
    maintainer='santosh',
    maintainer_email='selvarajs@a-star.edu.sg',
    description='Local web interface for robot and network visualization.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['web_gui = lbks_gui.server:main']},
)
