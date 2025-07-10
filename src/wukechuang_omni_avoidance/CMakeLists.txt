cmake_minimum_required(VERSION 3.8)
project(wukechuang_omni_avoidance)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(OpenCV REQUIRED)

# PCL库配置
find_package(PCL REQUIRED COMPONENTS common filters segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 头文件目录
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 生成可执行文件
add_executable(omni_avoidance_node
  src/omni_avoidance_node.cpp
)

# 链接ROS依赖
ament_target_dependencies(omni_avoidance_node
  rclcpp
  sensor_msgs
  geometry_msgs
  mavros_msgs
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  pcl_conversions
)

# 链接第三方库
target_link_libraries(omni_avoidance_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# 安装可执行文件
install(TARGETS
  omni_avoidance_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装配置文件和launch文件
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

