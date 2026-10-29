cmake_minimum_required(VERSION 3.8)
project(lbks_gui)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
add_library(lbks_control_panel SHARED
  src/control_panel.cpp include/lbks_gui/control_panel.hpp
  src/network_panel.cpp include/lbks_gui/network_panel.hpp)
target_compile_features(lbks_control_panel PUBLIC cxx_std_17)
target_include_directories(lbks_control_panel PRIVATE include)
ament_target_dependencies(lbks_control_panel rclcpp rviz_common pluginlib std_msgs)
target_link_libraries(lbks_control_panel Qt5::Widgets)
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_python_install_package(${PROJECT_NAME})
install(TARGETS lbks_control_panel LIBRARY DESTINATION lib)
install(PROGRAMS scripts/ik_preview DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
ament_package()
