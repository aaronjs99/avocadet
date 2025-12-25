cmake_minimum_required(VERSION 3.8)
project(avocadet)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/Color.msg"
  "msg/AvocadoDetection.msg"
  "msg/AvocadoDetectionArray.msg"
  DEPENDENCIES std_msgs
)

# Install Python modules
ament_python_install_package(avocadet_ros)
ament_python_install_package(src/avocadet PACKAGE_DIR src/avocadet)

# Install Python executables
install(PROGRAMS
  avocadet_ros/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
