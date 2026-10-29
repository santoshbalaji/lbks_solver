#pragma once

#include <vector>
#include <QJsonArray>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class QGraphicsScene;
class QGraphicsView;
class QGraphicsRectItem;
class QLabel;
class QPushButton;
class QSlider;
class QTextBrowser;
class QTimer;

namespace lbks_gui
{
class NetworkPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit NetworkPanel(QWidget * parent = nullptr);
  void onInitialize() override;
  void load(const rviz_common::Config & config) override;
  void save(rviz_common::Config config) const override;

private:
  void receive(const std_msgs::msg::String & message);
  void drawNetwork();
  void showStage(int index);
  void explainSelection();
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  QGraphicsScene * scene_;
  QGraphicsView * view_;
  QLabel * summary_;
  QLabel * stage_label_;
  QTextBrowser * explanation_;
  QSlider * stage_slider_;
  QPushButton * replay_;
  QTimer * animation_;
  QJsonArray stages_;
  std::vector<QGraphicsRectItem *> cards_;
};
}  // namespace lbks_gui
