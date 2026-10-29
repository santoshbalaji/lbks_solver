#include "lbks_gui/network_panel.hpp"

#include <algorithm>
#include <cmath>
#include <QDockWidget>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsTextItem>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSplitter>
#include <QTextBrowser>
#include <QTimer>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>

namespace lbks_gui
{
namespace
{
QString number(double value) {return QString::number(value, 'g', 6);}

QString operation(const QString & kind)
{
  if (kind == "Linear") {return "Weighted sum: y[j] = Σ w[j,i] × x[i] + bias[j].";}
  if (kind == "ReLU") {return "ReLU: max(0, x). Negative inputs become zero.";}
  if (kind == "Tanh") {return "Tanh maps each input to the range [−1, +1].";}
  if (kind == "GELU") {return "GELU: x × Φ(x), where Φ is the standard normal CDF.";}
  if (kind == "input") {return "Target XYZ in metres, in the robot root frame.";}
  if (kind == "output") {return "Predicted joint angles in radians, before URDF limits.";}
  if (kind == "limits") {return "Postprocessing: clamp each joint to its URDF limits.";}
  if (kind == "fk") {return "Robot kinematics converts the applied joints to achieved XYZ. This is not a learned layer.";}
  return kind;
}
}  // namespace

NetworkPanel::NetworkPanel(QWidget * parent) : rviz_common::Panel(parent)
{
  auto layout = new QVBoxLayout(this);
  summary_ = new QLabel("Network inspector · load a checkpoint and solve a target.");
  summary_->setWordWrap(true);
  layout->addWidget(summary_);
  auto controls = new QHBoxLayout;
  replay_ = new QPushButton("Replay forward pass");
  replay_->setEnabled(false);
  auto previous = new QPushButton("◀");
  auto next = new QPushButton("▶");
  previous->setMaximumWidth(35);
  next->setMaximumWidth(35);
  stage_slider_ = new QSlider(Qt::Horizontal);
  stage_slider_->setRange(0, 0);
  stage_label_ = new QLabel;
  controls->addWidget(replay_);
  controls->addWidget(previous);
  controls->addWidget(stage_slider_, 1);
  controls->addWidget(next);
  controls->addWidget(stage_label_);
  layout->addLayout(controls);

  auto split = new QSplitter(Qt::Horizontal);
  scene_ = new QGraphicsScene(this);
  view_ = new QGraphicsView(scene_);
  view_->setBackgroundBrush(QColor("#141c28"));
  view_->setRenderHint(QPainter::Antialiasing);
  view_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
  view_->setMinimumHeight(210);
  explanation_ = new QTextBrowser;
  explanation_->setMinimumWidth(240);
  explanation_->setMaximumWidth(420);
  split->addWidget(view_);
  split->addWidget(explanation_);
  split->setStretchFactor(0, 3);
  split->setStretchFactor(1, 1);
  layout->addWidget(split, 1);
  auto legend = new QLabel(
    "Blue: negative · amber: positive · gray: zero. Brightness is normalized within each stage. "
    "Click a value to inspect it. Replay shows recorded values, not execution time.");
  legend->setWordWrap(true);
  layout->addWidget(legend);
  animation_ = new QTimer(this);
  animation_->setInterval(650);
  connect(animation_, &QTimer::timeout, this, [this]() {
    if (stage_slider_->value() == stage_slider_->maximum()) {
      animation_->stop(); replay_->setText("Replay forward pass");
    } else {
      stage_slider_->setValue(stage_slider_->value() + 1);
    }
  });
  connect(replay_, &QPushButton::clicked, this, [this]() {
    if (animation_->isActive()) {
      animation_->stop(); replay_->setText("Replay forward pass");
    } else {
      stage_slider_->setValue(0); showStage(0);
      replay_->setText("Pause replay"); animation_->start();
    }
  });
  connect(stage_slider_, &QSlider::valueChanged, this, &NetworkPanel::showStage);
  connect(stage_slider_, &QSlider::sliderPressed, this, [this]() {
    animation_->stop(); replay_->setText("Replay forward pass");
  });
  auto step = [this](int delta) {
      animation_->stop(); replay_->setText("Replay forward pass");
      stage_slider_->setValue(stage_slider_->value() + delta);
    };
  connect(previous, &QPushButton::clicked, this, [step]() {step(-1);});
  connect(next, &QPushButton::clicked, this, [step]() {step(1);});
  connect(scene_, &QGraphicsScene::selectionChanged, this, &NetworkPanel::explainSelection);
}

void NetworkPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  bool bottom = false;
  config.mapGetBool("Dock at bottom", &bottom);
  if (bottom) {
    QTimer::singleShot(0, this, [this]() {
      auto dock = qobject_cast<QDockWidget *>(parentWidget());
      auto main = qobject_cast<QMainWindow *>(window());
      if (dock && main) {
        main->addDockWidget(Qt::BottomDockWidgetArea, dock);
        main->resizeDocks({dock}, {370}, Qt::Vertical);
      }
    });
  }
}

void NetworkPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  // User-saved layouts should retain their own dock arrangement.
  config.mapSetValue("Dock at bottom", false);
}

void NetworkPanel::onInitialize()
{
  node_ = std::make_shared<rclcpp::Node>(
    "lbks_network_inspector", rclcpp::NodeOptions().use_global_arguments(false));
  subscription_ = node_->create_subscription<std_msgs::msg::String>(
    "/lbks_gui/network_trace", rclcpp::QoS(1).transient_local(),
    [this](std_msgs::msg::String::ConstSharedPtr message) {receive(*message);});
  executor_.add_node(node_);
  auto timer = new QTimer(this);
  connect(timer, &QTimer::timeout, this, [this]() {executor_.spin_some();});
  timer->start(40);
}

void NetworkPanel::receive(const std_msgs::msg::String & message)
{
  const auto trace = QJsonDocument::fromJson(QByteArray::fromStdString(message.data)).object();
  animation_->stop();
  replay_->setText("Replay forward pass");
  stages_ = trace["stages"].toArray();
  replay_->setEnabled(!stages_.isEmpty());
  QString model = trace["model"].toString();
  if (!trace["checkpoint"].toString().isEmpty()) {model = trace["checkpoint"].toString();}
  summary_->setText(stages_.isEmpty() ? trace["note"].toString() :
    QString("%1  ·  Prediction #%2  ·  %3 parameters  ·  %4 ms including inspection\n%5")
    .arg(model).arg(trace["prediction_id"].toInt()).arg(trace["parameters"].toInt())
    .arg(trace["elapsed_ms"].toDouble(), 0, 'f', 1).arg(trace["note"].toString()));
  drawNetwork();
  const QSignalBlocker blocker(stage_slider_);
  stage_slider_->setRange(0, std::max(0, stages_.size() - 1));
  stage_slider_->setValue(0);
  showStage(0);
  if (stages_.isEmpty()) {explanation_->setPlainText(trace["note"].toString());}
}

void NetworkPanel::drawNetwork()
{
  const QSignalBlocker blocker(scene_);
  scene_->clear();
  cards_.clear();
  for (int column = 0; column < stages_.size(); ++column) {
    const auto stage = stages_[column].toObject();
    const auto values = stage["values"].toArray();
    const auto labels = stage["labels"].toArray();
    const bool compact = values.size() > 8;
    const double x = column * 225.0;
    const double height = std::max(230.0, 110.0 + (compact ?
      std::ceil(values.size() / 8.0) * 13 : values.size() * 23));
    auto card = scene_->addRect(x, 0, 205, height, QPen(QColor("#40516a")), QColor("#1c2839"));
    cards_.push_back(card);
    auto title = scene_->addText(stage["name"].toString());
    title->setTextWidth(195); title->setDefaultTextColor(QColor("#eff6ff"));
    title->setPos(x + 5, 4);
    auto count = scene_->addText(QString("%1 values · %2 zeros")
      .arg(stage["count"].toInt()).arg(stage["zeros"].toInt()));
    count->setDefaultTextColor(QColor("#a6b8d0")); count->setPos(x + 5, 44);
    const double maximum = std::max(std::abs(stage["min"].toDouble()), std::abs(stage["max"].toDouble()));
    for (int index = 0; index < values.size(); ++index) {
      const double value = values[index].toDouble();
      QColor color("#3c4656");
      if (value != 0 && maximum > 0) {
        color = QColor::fromHsvF(value < 0 ? 0.57 : 0.10, 0.72,
          0.35 + 0.65 * std::abs(value) / maximum);
      }
      const double cell_x = x + 12 + (compact ? (index % 8) * 23 : 0);
      const double cell_y = 78 + (compact ? (index / 8) * 13 : index * 23);
      auto cell = scene_->addRect(cell_x, cell_y, compact ? 19 : 176,
        compact ? 10 : 20, QPen(Qt::NoPen), color);
      cell->setFlag(QGraphicsItem::ItemIsSelectable);
      cell->setData(0, column); cell->setData(1, index);
      const QString label = index < labels.size() ? labels[index].toString() : QString("Neuron %1").arg(index);
      cell->setToolTip(label + " = " + number(value));
      if (!compact) {
        const auto text = scene_->addText(QString("%1: %2").arg(
          labels.size() == 3 ? label : QString::number(index), number(value)));
        text->setDefaultTextColor(Qt::white); text->setPos(cell_x + 2, cell_y - 2);
        text->setAcceptedMouseButtons(Qt::NoButton);
      }
    }
    if (values.size() < stage["count"].toInt()) {
      auto truncated = scene_->addText(QString("First %1 values shown").arg(values.size()));
      truncated->setDefaultTextColor(QColor("#e6b15a")); truncated->setPos(x + 5, height - 25);
    }
    if (column > 0) {
      auto arrow = scene_->addText("→"); arrow->setDefaultTextColor(QColor("#8dbbff"));
      arrow->setPos(x - 22, 95);
    }
  }
  scene_->setSceneRect(scene_->itemsBoundingRect().adjusted(-5, -5, 5, 5));
}

void NetworkPanel::showStage(int index)
{
  if (stages_.isEmpty()) {stage_label_->clear(); return;}
  for (size_t i = 0; i < cards_.size(); ++i) {
    cards_[i]->setPen(QPen(QColor(static_cast<int>(i) == index ? "#7dd3fc" : "#40516a"),
      static_cast<int>(i) == index ? 3 : 1));
  }
  view_->ensureVisible(cards_[index], 20, 10);
  const auto stage = stages_[index].toObject();
  stage_label_->setText(QString("%1 / %2").arg(index + 1).arg(stages_.size()));
  explanation_->setPlainText(QString("%1\n\n%2\n\nMin: %3\nMax: %4\nMean: %5\nZeros: %6 / %7\n\nClick a value for its calculation.")
    .arg(stage["name"].toString(), operation(stage["kind"].toString()),
    number(stage["min"].toDouble()), number(stage["max"].toDouble()), number(stage["mean"].toDouble()))
    .arg(stage["zeros"].toInt()).arg(stage["count"].toInt()));
}

void NetworkPanel::explainSelection()
{
  const auto selected = scene_->selectedItems();
  if (selected.isEmpty()) {return;}
  const int column = selected[0]->data(0).toInt();
  const int index = selected[0]->data(1).toInt();
  animation_->stop(); replay_->setText("Replay forward pass");
  stage_slider_->setValue(column);
  const auto stage = stages_[column].toObject();
  const auto labels = stage["labels"].toArray();
  const auto label = index < labels.size() ? labels[index].toString() : QString("Neuron %1").arg(index);
  QString detail = stage["name"].toString() + "\n" + label + " = " +
    number(stage["values"].toArray()[index].toDouble()) + "\n\n" + operation(stage["kind"].toString());
  const auto neurons = stage["neurons"].toArray();
  if (index < neurons.size()) {
    const auto neuron = neurons[index].toObject();
    detail += "\n\nStrongest contributions (by |weight × input|):\n";
    for (const auto term_value : neuron["terms"].toArray()) {
      const auto term = term_value.toObject();
      detail += QString("x[%1]: %2 × %3 = %4\n").arg(term["input"].toInt())
        .arg(number(term["value"].toDouble()), number(term["weight"].toDouble()), number(term["product"].toDouble()));
    }
    detail += "Other contributions: " + number(neuron["other_sum"].toDouble()) +
      "\nBias: " + number(neuron["bias"].toDouble()) + "\nTheir sum gives this neuron's value.";
  } else if (column > 0) {
    const auto inputs = stages_[column - 1].toObject()["values"].toArray();
    if (index < inputs.size() && stage["kind"].toString() != "fk") {
      detail += "\n\nPrevious value: " + number(inputs[index].toDouble());
    }
  }
  explanation_->setPlainText(detail);
}
}  // namespace lbks_gui

PLUGINLIB_EXPORT_CLASS(lbks_gui::NetworkPanel, rviz_common::Panel)
