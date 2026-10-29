# LBKS web GUI

A local browser app for exploring robot poses and learned inverse kinematics.
Three.js renders the actual URDF meshes. A small Python HTTP server runs model
predictions. There is no RViz, Qt, ROS messaging, frontend build, or web framework.

## Run

```bash
cd ~/purdue_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select lbks_gui
source install/setup.bash
ros2 launch lbks_gui visualize.launch.py \
  python_executable:=/home/santosh/purdue_ws/venv/bin/python
```

Open **http://127.0.0.1:8080**. The launch opens your default browser; pass
`open_browser:=false` to disable this or `port:=8081` to change the port.
Stop the server with Ctrl+C in its terminal.

You can also run directly from this repository without sourcing ROS:

```bash
PYTHONPATH="$PWD/lbks_gui" ~/purdue_ws/venv/bin/python -m lbks_gui.server
```

Python needs `torch`, `pytorch-kinematics`, and `PyYAML`, as in the training
virtualenv. Browser libraries are bundled under `web/vendor`; no npm install,
CDN connection, or internet access is required at runtime. Use a current browser
with WebGL2 enabled.

## Controls

- Enter XYZ in metres and click **Predict joint angles**. Live prediction updates
  after a short typing pause. Drag the blue XYZ handles to solve on release.
- Orbit by dragging the scene and zoom with the scroll wheel. Reset view restores
  the initial camera.
- Select numerical IK to preview before loading a model. Load a `.pth` / `.pt`
  state dictionary from the file picker, or enter its absolute path on this
  machine. Uploaded files live in a temporary directory until the server stops.
- Joint sliders preview the arm directly; Reset sets joint angles to zero.
- The achieved XYZ, position error, and solve time appear over the robot view.
- Toggle URDF collision meshes or an example obstacle. Collision detection is
  not implemented. Position-only IK leaves tool orientation unconstrained.

The browser discards stale responses when targets or joints change. Failed
predictions preserve the displayed pose. Joint predictions are clamped to URDF
limits and reported; approximate poses retain their measured position error.

## Model variants

Copy `config/models.yaml`, then launch with `models_config:=/path/models.yaml`:

```yaml
models:
  baseline:
    path: /path/to/ik_model.pth
    hidden_sizes: [128, 128]
    activation: relu
    output_scale: 3.141592653589793
  wider:
    path: /path/to/wider.pth
    hidden_sizes: [256, 256, 128]
  custom:
    path: /path/to/scripted_model.pt
    format: torchscript
```

Relative paths resolve against the YAML file. Reload the server after editing
it. The file picker and path field use the current trainer's default architecture:
`3 → 128 → 128 → dof`, ReLU, final Tanh scaled by pi. Other widths and activations
(relu, tanh, gelu) belong in YAML. State dictionaries use `net.*` keys; a wrapping
`state_dict` key is also supported. Only trusted TorchScript files should be used.
Models accept float32 `(batch, 3)` XYZ in metres and return `(batch, dof)` radians
in the serial chain's joint order. Include normalization in exported models.

## Network inspector

The lower panel shows actual recorded activations for each MLP layer, output
scaling, joint limits, and achieved XYZ. Click a cell for its value; linear neurons
also show their four strongest weighted inputs, remaining contribution sum, and
bias. Blue is negative, amber positive, and gray zero. Brightness is normalized
per layer. Layer statistics cover all neurons; at most 256 values are displayed.

Replay and the stage slider step through one recorded prediction, not real-time
execution. New inputs cancel replay. Solve time includes inspection overhead.
TorchScript exposes inputs and outputs only; numerical IK has no network view.

## Code map

- `lbks_gui/server.py`: static files, checkpoint uploads, and a small JSON API.
- `lbks_gui/engine.py`: model loading, numerical IK, and FK.
- `lbks_gui/network_trace.py`: actual activation capture and neuron contributions.
- `lbks_gui/robot.py`: URDF and model configuration.
- `web/app.js`: controls and prediction requests.
- `web/viewer.js`: URDF rendering, target dragging, and collision geometry.
- `web/network.js`: activation cards and replay.
- `web/index.html`, `web/style.css`: layout and appearance.

The server binds only to `127.0.0.1`, accepts same-origin requests, and serves
assets only from its web directory and the configured robot description folder.
It does not publish robot commands or model data to ROS/DDS. Uploaded checkpoints
are limited to 64 MiB; use a local path for larger files.

API: `GET /api/config`, `GET /robot.urdf`, `POST /api/predict` with
`{target, seed, model, path}`, and `POST /api/joints` with `{joints}`.
A future collision checker can feed root-frame boxes to
`RobotView.setObstacles([{position: [x,y,z], size: [sx,sy,sz], color}])`.
The current example uses this same rendering method.

## Tests

```bash
PYTHONPATH="$PWD/lbks_gui" ~/purdue_ws/venv/bin/python \
  -m unittest discover -s lbks_gui/test -v
```

ROS launch is only a convenience. Direct server execution, API tests, and the
browser application do not require a running ROS graph.

If Node.js is available, the optional frontend regression test checks controls,
stale-response handling, neuron inspection, and replay using a small test DOM:

```bash
node --experimental-vm-modules lbks_gui/test/test_frontend.mjs
```

This test does not verify WebGL rendering or replace a visual browser check.
