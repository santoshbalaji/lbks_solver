# LBKS GUI

A single RViz window with a custom **IK Studio** panel, URDF rendering, and a
background Python inference process. No commands are sent to robot hardware.
The preview uses the same URDF and `tool0` endpoint as the current trainer.

## Build and run

```bash
cd ~/purdue_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lbks_description lbks_gui
source install/setup.bash
ros2 launch lbks_gui visualize.launch.py \
  python_executable:=/home/santosh/purdue_ws/venv/bin/python
```

The backend Python needs `torch`, `pytorch-kinematics`, and `PyYAML`; the existing
training virtualenv can be used. The default interpreter is the activated
virtualenv's Python, or the launch interpreter if none is activated. RViz and
the compiled Qt panel use the ROS/system libraries, so Qt need not be installed
in the training virtualenv. Build dependencies are `rviz_common`, `pluginlib`,
`rclcpp`, `std_msgs`, `ament_cmake_python`, and Qt5 Widgets. ROS runtime also
requires `robot_state_publisher`, `rviz2`, and `interactive_markers`.

## Interaction

- Edit XYZ in metres and click **Solve target**, or enable live solving.
- Drag the XYZ gizmo with RViz's **Interact** tool; releasing it solves the target.
- **Publish Point** selects a target on rendered geometry.
- Select numerical IK before a checkpoint is available; it uses bounded,
  damped least squares with deterministic restarts.
- Load a `.pth` state dictionary from the current trainer with **Load trained
  checkpoint**. The default network is `3 → 128 → 128 → dof`, ReLU, final Tanh
  scaled by pi. Checkpoints load on CPU even when trained on CUDA.
- Joint sliders preview configurations directly. **Zero joints** resets them.
- Blue marks the target, green the achieved tip position, and amber shows the
  position error. Error in mm and solver latency are shown in the panel.

The solver is **position-only**. It does not constrain orientation or perform
collision checking. Approximate solutions are displayed with their residual;
they are not declared unreachable or collision-free. Predictions outside URDF
joint limits are clamped and reported. Neural inference is shown directly,
without numerical refinement, so variants can be compared honestly.

## Model variants

Copy `config/models.yaml` to a writable location and add named models, then pass
`models_config:=/absolute/path/models.yaml`. Paths in YAML are relative to the
YAML file. Restart the launch after changing the model list.

```yaml
models:
  baseline:
    path: /path/to/ik_model.pth
    format: state_dict
    hidden_sizes: [128, 128]
    activation: relu
    output_scale: 3.141592653589793
  wider:
    path: /path/to/wider.pth
    hidden_sizes: [256, 256, 128]
  custom_architecture:
    path: /path/to/scripted_model.pt
    format: torchscript
```

State dictionaries must use `net.*` keys (as in `IKFeedForward`); a dictionary
containing `state_dict` is also supported. MLP activations: relu, tanh, gelu.
For other architectures, provide a trusted TorchScript module. Its interface
must be float32 XYZ `(batch, 3)` in metres in the URDF root frame to joint angles
`(batch, dof)` in radians, in the serial chain's joint order. Any normalization
must be included in the exported model. Models must match the selected robot
and tip. Dimension errors are reported in the panel without replacing the pose.

## Network inspector

The **Network inspector** dock opens below the robot view. Load a checkpoint and
solve a target to see the real forward pass: XYZ input → linear layers → activation
functions → output scaling → URDF limits → forward kinematics. New predictions
update the view automatically. Numerical IK shows an explanation instead of a
neural network.

- Each colored cell is an actual activation. Blue is negative, amber positive,
  and gray zero. Brightness is normalized per stage; compare exact values, not
  colors, across different layers. Hover for values and click for details.
- Click a linear-layer neuron to see its four strongest `weight × input`
  contributions, the sum of remaining contributions, and its bias. These sum to
  the recorded activation (up to floating-point rounding).
- **Replay forward pass**, the arrow buttons, and the stage slider step through
  a captured prediction. Replay slows down the explanation; it is not the actual
  computation timing. New targets cancel replay and replace the old trace.
- The final stages distinguish raw joint predictions from joint-limit clamping
  and the achieved Cartesian pose. These postprocessing stages are not learned.
- MLP variants expose all layers; custom TorchScript modules expose inputs and
  outputs only. No synthetic hidden activations are shown.

Up to 256 activations per stage are displayed (larger layers are explicitly
marked); min/max/mean/zero counts cover the full layer. Inference always uses all
neurons. The displayed solve latency includes activation inspection overhead.
Temporary hooks are removed after each prediction, and traces are published once
per update on the transient-local `/lbks_gui/network_trace` topic, rather than
repeated with the 20 Hz robot state. Errors and manual joint edits clear the view.
For an existing RViz layout, add **lbks_gui/NetworkPanel** using Panels → Add New
Panel, or launch with the supplied `studio.rviz` configuration.

## Collision extensions

Enable **Collision geometry** in RViz Displays to inspect the URDF collision
meshes. **Show example obstacle** adds a translucent box, purely for preview.
Future collision checking code can publish `visualization_msgs/MarkerArray` to
`/lbks_gui/collision_markers` for contacts, highlighted links, distances, or
obstacles. Use the URDF root frame or a frame in `/lbks_gui/tf`; clear obsolete
markers with DELETE or DELETEALL. The display is already configured.

Topics are isolated under `/lbks_gui`: `joint_states`, `robot_description`, `tf`,
`tf_static`, `markers`, `state`, and `request`. The panel exchanges JSON inside
`std_msgs/String`; requests support `solve` (target/model/path), `joints`, `home`,
and `obstacle`. The latest request wins; completed stale solves are discarded.

Other launch options: `urdf_path` (expanded URDF, not Xacro), `tip_link`, and
`rviz_config`. Machine-specific lbks_description mesh URLs are normalized to
`package://lbks_description/meshes/...` without editing the original URDF.

## Tests

From this repository, with the training environment available:

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD/lbks_gui:$PYTHONPATH" ~/purdue_ws/venv/bin/python \
  -m unittest discover -s lbks_gui/test -v
```
