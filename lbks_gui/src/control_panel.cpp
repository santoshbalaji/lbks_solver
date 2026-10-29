#include "lbks_gui/control_panel.hpp"

#include <cmath>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>

namespace lbks_gui
{
ControlPanel::ControlPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  auto outer = new QVBoxLayout(this);
  auto scroll = new QScrollArea;
  scroll->setWidgetResizable(true);
  auto content = new QWidget;
  auto layout = new QVBoxLayout(content);
  scroll->setWidget(content);
  outer->addWidget(scroll);
  auto title = new QLabel("<h2>LBKS · IK Studio</h2>");
  layout->addWidget(title);
  frame_ = new QLabel("Waiting for robot…");
  frame_->setWordWrap(true);
  layout->addWidget(frame_);

  auto target = new QGroupBox("Cartesian target · metres");
  auto target_layout = new QFormLayout(target);
  for (int i = 0; i < 3; ++i) {
    xyz_[i] = new QDoubleSpinBox;
    xyz_[i]->setRange(-5.0, 5.0);
    xyz_[i]->setDecimals(3);
    xyz_[i]->setSingleStep(0.01);
    xyz_[i]->setValue(i == 1 ? 0.0 : 0.4);
    target_layout->addRow(QString("XYZ").mid(i, 1), xyz_[i]);
  }
  live_ = new QCheckBox("Live solve while editing");
  live_->setChecked(true);
  target_layout->addRow(live_);
  layout->addWidget(target);
  debounce_ = new QTimer(this);
  debounce_->setSingleShot(true);
  debounce_->setInterval(200);
  connect(debounce_, &QTimer::timeout, this, &ControlPanel::solve);
  for (auto spin : xyz_) {
    connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
      [this](double) {if (live_->isChecked()) {debounce_->start();}});
  }
  connect(live_, &QCheckBox::toggled, this, [this](bool enabled) {
    if (!enabled) {debounce_->stop();}
  });

  auto model_group = new QGroupBox("IK model");
  auto model_layout = new QVBoxLayout(model_group);
  models_ = new QComboBox;
  models_->addItem("Numerical IK · preview", "numerical");
  model_layout->addWidget(models_);
  auto browse = new QPushButton("Load trained checkpoint…");
  model_layout->addWidget(browse);
  checkpoint_ = new QLabel("Default MLP: 3 → 128 → 128 → joints");
  checkpoint_->setWordWrap(true);
  model_layout->addWidget(checkpoint_);
  layout->addWidget(model_group);
  connect(browse, &QPushButton::clicked, this, [this]() {
    const auto path = QFileDialog::getOpenFileName(
      this, "Load planner state_dict", QString(), "PyTorch checkpoints (*.pth *.pt)");
    if (path.isEmpty()) {return;}
    custom_path_ = path;
    const QSignalBlocker blocker(models_);
    if (models_->findData("__custom__") < 0) {
      models_->addItem("Loaded checkpoint · default MLP", "__custom__");
    }
    models_->setCurrentIndex(models_->findData("__custom__"));
    checkpoint_->setText(path);
    solve();
  });
  connect(models_, qOverload<int>(&QComboBox::currentIndexChanged), this,
    [this](int) {solve();});
  auto buttons = new QHBoxLayout;
  auto solve_button = new QPushButton("Solve target");
  auto home = new QPushButton("Zero joints");
  buttons->addWidget(solve_button);
  buttons->addWidget(home);
  layout->addLayout(buttons);
  connect(solve_button, &QPushButton::clicked, this, &ControlPanel::solve);
  connect(home, &QPushButton::clicked, this, [this]() {
    debounce_->stop(); send({{"op", "home"}});
  });
  status_ = new QLabel("Connecting to IK backend…");
  status_->setWordWrap(true);
  layout->addWidget(status_);
  metrics_ = new QLabel("Achieved XYZ: —\nPosition error: —");
  metrics_->setWordWrap(true);
  layout->addWidget(metrics_);
  auto joints = new QGroupBox("Joint preview · degrees");
  joints_layout_ = new QVBoxLayout(joints);
  layout->addWidget(joints);
  auto scene = new QGroupBox("Collision visualization");
  auto scene_layout = new QVBoxLayout(scene);
  auto obstacle = new QCheckBox("Show example obstacle");
  scene_layout->addWidget(obstacle);
  auto note = new QLabel(
    "Collision checking: not connected.\n"
    "Toggle Collision geometry in Displays to inspect URDF collision meshes. "
    "Future results can publish to /lbks_gui/collision_markers.");
  note->setWordWrap(true);
  scene_layout->addWidget(note);
  connect(obstacle, &QCheckBox::toggled, this, [this](bool checked) {
    send({{"op", "obstacle"}, {"enabled", checked}});
  });
  layout->addWidget(scene);
  auto help = new QLabel(
    "Drag the XYZ gizmo with RViz Interact. Orbit with the mouse; scroll to zoom. "
    "Blue = target · green = achieved · amber = position error. "
    "Position-only IK; tool orientation is unconstrained.");
  help->setWordWrap(true);
  layout->addWidget(help);
  layout->addStretch();
  setMinimumWidth(330);
}

void ControlPanel::onInitialize()
{
  node_ = std::make_shared<rclcpp::Node>(
    "lbks_gui_controls", rclcpp::NodeOptions().use_global_arguments(false));
  publisher_ = node_->create_publisher<std_msgs::msg::String>("/lbks_gui/request", 10);
  subscription_ = node_->create_subscription<std_msgs::msg::String>(
    "/lbks_gui/state", 10, [this](std_msgs::msg::String::ConstSharedPtr msg) {
      receive(*msg);
    });
  executor_.add_node(node_);
  auto timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, [this]() {
    executor_.spin_some();
    if (connected_ && std::chrono::steady_clock::now() - last_state_ > std::chrono::seconds(3)) {
      connected_ = false;
      status_->setText("Backend disconnected — check the launch terminal.");
      status_->setStyleSheet("color: #df724a;");
    }
  });
  timer->start(30);
}

void ControlPanel::send(const QJsonObject & request)
{
  if (!publisher_ || !connected_) {
    status_->setText("Waiting for IK backend — check the launch terminal.");
    return;
  }
  std_msgs::msg::String message;
  message.data = QJsonDocument(request).toJson(QJsonDocument::Compact).toStdString();
  publisher_->publish(message);
}

void ControlPanel::solve()
{
  debounce_->stop();
  QJsonArray target;
  for (auto spin : xyz_) {target.append(spin->value());}
  send({{"op", "solve"}, {"target", target},
    {"model", models_->currentData().toString()}, {"path", custom_path_}});
}

void ControlPanel::createJoints(const QJsonObject & state)
{
  const auto names = state["names"].toArray();
  const auto lower = state["lower"].toArray();
  const auto upper = state["upper"].toArray();
  for (int i = 0; i < names.size(); ++i) {
    auto name = new QLabel(names[i].toString());
    joints_layout_->addWidget(name);
    auto row = new QHBoxLayout;
    auto slider = new QSlider(Qt::Horizontal);
    slider->setRange(static_cast<int>(lower[i].toDouble() * 1000),
      static_cast<int>(upper[i].toDouble() * 1000));
    slider->setSingleStep(17);
    auto value = new QLabel;
    value->setMinimumWidth(65);
    row->addWidget(slider);
    row->addWidget(value);
    joints_layout_->addLayout(row);
    sliders_.push_back(slider);
    joint_values_.push_back(value);
    connect(slider, &QSlider::valueChanged, this, [this, i](int raw) {
      joint_values_[i]->setText(QString::number(raw * 0.180 / M_PI, 'f', 1) + "°");
      debounce_->stop();
      QJsonArray joints;
      for (auto item : sliders_) {joints.append(item->value() / 1000.0);}
      send({{"op", "joints"}, {"joints", joints}});
    });
  }
}

void ControlPanel::receive(const std_msgs::msg::String & message)
{
  const auto state = QJsonDocument::fromJson(QByteArray::fromStdString(message.data)).object();
  if (state.isEmpty()) {return;}
  last_state_ = std::chrono::steady_clock::now();
  connected_ = true;
  if (!initialized_) {
    createJoints(state);
    const QSignalBlocker blocker(models_);
    for (auto model : state["models"].toArray()) {
      models_->addItem(model.toString(), model.toString());
    }
    frame_->setText("Frame: " + state["frame"].toString() + "  ·  Tip: " + state["tip"].toString());
    initialized_ = true;
  }
  const auto target = state["target"].toArray();
  if (target != last_target_) {
    for (int i = 0; i < 3; ++i) {
      if (!xyz_[i]->hasFocus() && !debounce_->isActive()) {
        const QSignalBlocker blocker(xyz_[i]);
        xyz_[i]->setValue(target[i].toDouble());
      }
    }
    last_target_ = target;
  }
  const auto q = state["joints"].toArray();
  for (size_t i = 0; i < sliders_.size(); ++i) {
    if (!sliders_[i]->isSliderDown()) {
      const QSignalBlocker blocker(sliders_[i]);
      sliders_[i]->setValue(static_cast<int>(q[static_cast<int>(i)].toDouble() * 1000));
    }
    joint_values_[i]->setText(QString::number(
      sliders_[i]->value() * 0.180 / M_PI, 'f', 1) + "°");
  }
  const auto actual = state["actual"].toArray();
  metrics_->setText(QString("Achieved: %1, %2, %3 m\nError: %4 mm  ·  Solve: %5 ms")
    .arg(actual[0].toDouble(), 0, 'f', 3).arg(actual[1].toDouble(), 0, 'f', 3)
    .arg(actual[2].toDouble(), 0, 'f', 3).arg(state["error"].toDouble() * 1000, 0, 'f', 1)
    .arg(state["elapsed_ms"].toDouble(), 0, 'f', 1));
  status_->setText(state["status"].toString());
  status_->setStyleSheet(state["status"].toString().startsWith("Error") ?
    "color: #df724a;" : "");
}
}  // namespace lbks_gui

PLUGINLIB_EXPORT_CLASS(lbks_gui::ControlPanel, rviz_common::Panel)
