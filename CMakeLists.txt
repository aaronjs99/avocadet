cmake_minimum_required(VERSION 3.8)
project(avocadet)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Dependencies
# ============================================================================

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ============================================================================
# Message Generation
# ============================================================================

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/Color.msg"
  "msg/AvocadoDetection.msg"
  "msg/AvocadoDetectionArray.msg"
  DEPENDENCIES std_msgs
)

# ============================================================================
# Python Packages
# ============================================================================

# Install the ROS2 node package
ament_python_install_package(avocadet_ros)

# Install the core detection library
install(DIRECTORY
  src/avocadet_lib
  DESTINATION lib/python3.8/site-packages
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

# ============================================================================
# Executables
# ============================================================================

install(PROGRAMS
  avocadet_ros/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================================
# Launch, Config, and Other Resources
# ============================================================================

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install documentation
install(FILES
  README.md
  LICENSE
  DESTINATION share/${PROJECT_NAME}
)

# ============================================================================
# Testing
# ============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Python tests
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_detector tests/)
endif()

# ============================================================================
# Package Configuration
# ============================================================================

ament_export_dependencies(
  rclpy
  std_msgs
  sensor_msgs
  cv_bridge
)

ament_package()
