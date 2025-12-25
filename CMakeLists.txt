cmake_minimum_required(VERSION 3.0.2)
project(avocadet)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_generation
)

## ============================================================================
## Message Generation
## ============================================================================

add_message_files(
  FILES
  BoundingBox.msg
  Color.msg
  AvocadoDetection.msg
  AvocadoDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## ============================================================================
## Catkin Package Configuration
## ============================================================================

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    image_transport
    message_runtime
)

## ============================================================================
## Build
## ============================================================================

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## ============================================================================
## Install
## ============================================================================

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/detector_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python modules
catkin_python_setup()

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

## ============================================================================
## Testing
## ============================================================================

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  # Add tests here
endif()
