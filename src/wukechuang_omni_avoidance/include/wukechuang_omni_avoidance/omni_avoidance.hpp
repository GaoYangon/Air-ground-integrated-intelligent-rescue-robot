#ifndef WUKECHUANG_OMNI_AVOIDANCE_HPP_
#define WUKECHUANG_OMNI_AVOIDANCE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "mavros_msgs/srv/command_bool.hpp"  // 服务类型
#include "mavros_msgs/srv/set_mode.hpp"      // 服务类型
#include "mavros_msgs/msg/state.hpp"         // 消息类型
#include <tf2_ros/transform_listener.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>

namespace wukechuang_omni_avoidance {  // 修正：命名空间小写

// 运动模式
enum class MotionMode {
  GROUND,    // 地面模式
  FLIGHT,    // 飞行模式
  STOP       // 停止模式
};

// 障碍物信息
struct Obstacle {
  float distance;  // 距离（米）
  float angle;     // 角度（弧度）
  float height;    // 高度（米）
  bool is_thermal; // 是否由热成像检测
};

class OmniAvoidanceNode : public rclcpp::Node {
public:
  OmniAvoidanceNode();

private:
  // 回调函数
  void depth_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void thermal_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void color_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void mavros_state_callback(const mavros_msgs::msg::State::SharedPtr msg);

  // 控制循环
  void control_loop();

  // 障碍物检测
  std::vector<Obstacle> detect_obstacles();
  std::vector<Obstacle> detect_ground_obstacles(const sensor_msgs::msg::PointCloud2& cloud_msg);
  std::vector<Obstacle> detect_thermal_obstacles(const cv::Mat& thermal_img);

  // 决策与指令生成
  MotionMode decide_mode(const std::vector<Obstacle>& obstacles);
  bool has_ground_passage(const std::vector<Obstacle>& obstacles);
  geometry_msgs::msg::Twist generate_ground_command(const std::vector<Obstacle>& obstacles);
  geometry_msgs::msg::Twist generate_flight_command();

  // 辅助函数
  bool is_low_light(const cv::Mat& color_img);

  // 订阅者
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr thermal_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr mavros_state_sub_;

  // 发布者与服务客户端
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arm_client_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;

  // 定时器
  rclcpp::TimerBase::SharedPtr control_timer_;

  // 数据缓存
  sensor_msgs::msg::PointCloud2 depth_cloud_;  // 修正：使用ROS 2原生类型
  sensor_msgs::msg::LaserScan lidar_scan_;
  cv::Mat thermal_img_;
  cv::Mat color_img_;
  mavros_msgs::msg::State current_state_;
  std::mutex data_mutex_;

  // 状态变量
  MotionMode current_mode_ = MotionMode::GROUND;
  enum FlightState {
    IDLE, TAKEOFF, CROSS, LAND
  } flight_state_ = FlightState::IDLE;
  rclcpp::Time flight_start_time_;
};

}  // namespace wukechuang_omni_avoidance

#endif  // WUKECHUANG_OMNI_AVOIDANCE_HPP_

