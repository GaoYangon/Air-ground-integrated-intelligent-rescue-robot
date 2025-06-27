cmake_minimum_required(VERSION 3.8)
project(my_project_controller)

# 默认使用C++14标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖项
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# 添加深度距离检测节点
add_executable(depth_distance_node src/depth_distance_node.cpp)
ament_target_dependencies(depth_distance_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)

# 原有的控制器节点配置
add_executable(controller_node src/controller_node.cpp)
ament_target_dependencies(controller_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
)
target_link_libraries(controller_node ${OpenCV_LIBS})

# 安装规则
install(TARGETS
  controller_node
  depth_distance_node  # 新增节点的安装配置
  DESTINATION lib/${PROJECT_NAME}
)

# 测试配置（可选）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
