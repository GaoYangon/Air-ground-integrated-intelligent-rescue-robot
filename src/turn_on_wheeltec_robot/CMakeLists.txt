cmake_minimum_required(VERSION 3.5)
project(turn_on_wheeltec_robot)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(serial REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(turtlesim REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_srvs REQUIRED)
find_package(wheeltec_robot_msg REQUIRED)
find_package(serial_driver REQUIRED)  # 替换为 serial_driver
find_package(rosidl_default_generators REQUIRED)

target_link_libraries(wheeltec_robot_node
  ${serial_LIBRARIES}  # 添加此行
  ${rclcpp_LIBRARIES}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Position.msg"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(
  include
)

set(wheeltec_robot_node_SRCS
  src/wheeltec_robot.cpp
  src/Quaternion_Solution.cpp
)

add_executable(wheeltec_robot_node src/wheeltec_robot.cpp src/Quaternion_Solution.cpp)
ament_target_dependencies(wheeltec_robot_node 
  tf2_ros 
  tf2 
  tf2_geometry_msgs 
  rclcpp 
  std_msgs 
  nav_msgs 
  std_srvs 
  sensor_msgs 
  serial_driver  # 替换为 serial_driver
  wheeltec_robot_msg 
  turtlesim
)

install(TARGETS
  wheeltec_robot_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
