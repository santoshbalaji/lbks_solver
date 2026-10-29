#pragma once

#include <array>
#include <chrono>
#include <vector>
#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSlider;
class QTimer;
class QVBoxLayout;

namespace lbks_gui
{
class ControlPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit ControlPanel(QWidget * parent = nullptr);
  void onInitialize() override;

private:
  void send(const QJsonObject & request);
  void solve();
  void receive(const std_msgs::msg::String & message);
  void createJoints(const QJsonObject & state);
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::array<QDoubleSpinBox *, 3> xyz_;
  std::vector<QSlider *> sliders_;
  std::vector<QLabel *> joint_values_;
  QComboBox * models_;
  QCheckBox * live_;
  QLabel * status_;
  QLabel * metrics_;
  QLabel * frame_;
  QLabel * checkpoint_;
  QVBoxLayout * joints_layout_;
  QTimer * debounce_;
  QString custom_path_;
  QJsonArray last_target_;
  bool initialized_ = false;
  bool connected_ = false;
  std::chrono::steady_clock::time_point last_state_;
};
}  // namespace lbks_gui
