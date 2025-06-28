cmake_minimum_required(VERSION 3.8)
project(mlx90640_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 添加头文件路径
include_directories(
  include
  ${PROJECT_NAME}
  ${OpenCV_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(mlx90640_node 
  src/mlx90640_node.cpp  # 新建的 ROS2 节点文件
  src/MLX90640_API.cpp
  src/MLX90640_LINUX_I2C_Driver.cpp
)

# 链接库
ament_target_dependencies(mlx90640_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  OpenCV
)

# 链接 SDL2 和 I2C 库
find_library(SDL2_LIBRARY SDL2)
find_library(SDL2_TTF_LIBRARY SDL2_ttf)
find_library(I2C_LIBRARY i2c)

target_link_libraries(mlx90640_node
  ${SDL2_LIBRARY}
  ${SDL2_TTF_LIBRARY}
  ${I2C_LIBRARY}
  pthread
)

# 安装可执行文件
install(TARGETS
  mlx90640_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装资源文件
install(DIRECTORY resources/
  DESTINATION share/${PROJECT_NAME}/resources
)

ament_package()

