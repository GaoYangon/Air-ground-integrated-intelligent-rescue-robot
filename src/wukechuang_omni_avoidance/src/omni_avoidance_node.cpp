#include "wukechuang_omni_avoidance/omni_avoidance.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/imgproc.hpp>

namespace wukechuang_omni_avoidance {

OmniAvoidanceNode::OmniAvoidanceNode() : Node("omni_avoidance_node") {
  // 声明参数
  this->declare_parameter<float>("obstacle.ground_max_height", 0.3f);
  this->declare_parameter<float>("obstacle.safety_distance", 0.5f);
  this->declare_parameter<float>("robot.width", 0.4f);
  this->declare_parameter<float>("flight.takeoff_height", 0.8f);
  this->declare_parameter<float>("flight.cross_distance", 1.5f);
  this->declare_parameter<float>("velocity.ground_max_linear", 0.3f);
  this->declare_parameter<float>("velocity.ground_max_angular", 0.5f);
  this->declare_parameter<float>("velocity.flight_max_linear", 0.3f);
  this->declare_parameter<int>("thermal.light_threshold", 30);

  // 订阅者、发布者初始化
  depth_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/camera/depth/color/points", 10,
    std::bind(&OmniAvoidanceNode::depth_callback, this, std::placeholders::_1));

  lidar_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 10,
    std::bind(&OmniAvoidanceNode::lidar_callback, this, std::placeholders::_1));

  thermal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/thermal_image", 10,
    std::bind(&OmniAvoidanceNode::thermal_callback, this, std::placeholders::_1));

  color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw", 10,
    std::bind(&OmniAvoidanceNode::color_callback, this, std::placeholders::_1));

  mavros_state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
    "/mavros/state", 10,
    std::bind(&OmniAvoidanceNode::mavros_state_callback, this, std::placeholders::_1));

  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
    "/mavros/setpoint_velocity/cmd_vel_unstamped", 10);

  arm_client_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
  set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");

  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&OmniAvoidanceNode::control_loop, this));

  RCLCPP_INFO(this->get_logger(), "omni_avoidance_node启动成功");
}

// 回调函数
void OmniAvoidanceNode::depth_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  depth_cloud_ = *msg;
}

void OmniAvoidanceNode::lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  lidar_scan_ = *msg;
}

void OmniAvoidanceNode::thermal_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  thermal_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
}

void OmniAvoidanceNode::color_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  color_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
}

void OmniAvoidanceNode::mavros_state_callback(const mavros_msgs::msg::State::SharedPtr msg) {
  current_state_ = *msg;
}

// 主控制循环
void OmniAvoidanceNode::control_loop() {
  auto obstacles = detect_obstacles();
  current_mode_ = decide_mode(obstacles);

  geometry_msgs::msg::Twist cmd_vel;
  if (current_mode_ == MotionMode::GROUND) {
    cmd_vel = generate_ground_command(obstacles);
    flight_state_ = FlightState::IDLE;
  } else if (current_mode_ == MotionMode::FLIGHT) {
    cmd_vel = generate_flight_command();
  } else {
    cmd_vel.linear.x = 0.0;
    cmd_vel.angular.z = 0.0;
  }

  cmd_vel_pub_->publish(cmd_vel);
}

// 障碍物检测
std::vector<Obstacle> OmniAvoidanceNode::detect_obstacles() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  std::vector<Obstacle> obstacles;

  auto ground_obstacles = detect_ground_obstacles(depth_cloud_);
  obstacles.insert(obstacles.end(), ground_obstacles.begin(), ground_obstacles.end());

  if (!color_img_.empty() && is_low_light(color_img_) && !thermal_img_.empty()) {
    auto thermal_obstacles = detect_thermal_obstacles(thermal_img_);
    obstacles.insert(obstacles.end(), thermal_obstacles.begin(), thermal_obstacles.end());
  }

  return obstacles;
}

// 地面障碍物检测
std::vector<Obstacle> OmniAvoidanceNode::detect_ground_obstacles(const sensor_msgs::msg::PointCloud2& cloud_msg) {
  std::vector<Obstacle> obstacles;
  
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(cloud_msg, cloud);
  
  if (cloud.empty()) return obstacles;

  // 地面分割
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.05);
  seg.setInputCloud(cloud.makeShared());
  seg.segment(*inliers, *coefficients);

  // 提取非地面点
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud.makeShared());
  extract.setIndices(inliers);
  extract.setNegative(true);
  pcl::PointCloud<pcl::PointXYZ> obstacle_cloud;
  extract.filter(obstacle_cloud);

  // 生成障碍物信息
  float ground_max_height = 0.0;
  float safety_distance = 0.0;
  this->get_parameter("obstacle.ground_max_height", ground_max_height);
  this->get_parameter("obstacle.safety_distance", safety_distance);

  for (const auto& p : obstacle_cloud.points) {
    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z) || p.z < 0) continue;

    float distance = sqrt(p.x*p.x + p.y*p.y);
    float angle = atan2(p.y, p.x);

    if (distance < safety_distance && p.z > ground_max_height) {
      obstacles.push_back({distance, angle, p.z, false});
    }
  }

  return obstacles;
}

// 热成像障碍物检测
std::vector<Obstacle> OmniAvoidanceNode::detect_thermal_obstacles(const cv::Mat& thermal_img) {
  std::vector<Obstacle> obstacles;
  if (thermal_img.empty()) return obstacles;

  cv::Mat hsv_img;
  cv::cvtColor(thermal_img, hsv_img, cv::COLOR_BGR2HSV);
  cv::Mat mask;
  cv::inRange(hsv_img, cv::Scalar(0, 100, 100), cv::Scalar(30, 255, 255), mask);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  for (const auto& contour : contours) {
    if (cv::contourArea(contour) < 50) continue;

    cv::Moments m = cv::moments(contour);
    float cx = m.m10 / m.m00;

    float angle = (cx - 16.0f) / 16.0f * 0.523f;

    obstacles.push_back({1.0f, angle, 0.5f, true});
  }

  return obstacles;
}

// 低光判断
bool OmniAvoidanceNode::is_low_light(const cv::Mat& color_img) {
  cv::Mat gray;
  cv::cvtColor(color_img, gray, cv::COLOR_BGR2GRAY);
  double mean_brightness = cv::mean(gray)[0];
  
  int light_threshold = 0;
  this->get_parameter("thermal.light_threshold", light_threshold);
  return mean_brightness < light_threshold;
}

// 模式决策
MotionMode OmniAvoidanceNode::decide_mode(const std::vector<Obstacle>& obstacles) {
  bool has_high_obstacle = false;
  float ground_max_height = 0.0;
  this->get_parameter("obstacle.ground_max_height", ground_max_height);
  
  for (const auto& obs : obstacles) {
    if (obs.height > ground_max_height) {
      has_high_obstacle = true;
      break;
    }
  }

  bool has_passage = has_ground_passage(obstacles);

  if (has_high_obstacle && !has_passage) {
    RCLCPP_INFO(this->get_logger(), "切换至飞行模式");
    return MotionMode::FLIGHT;
  } else {
    return MotionMode::GROUND;
  }
}

// 地面通道判断
bool OmniAvoidanceNode::has_ground_passage(const std::vector<Obstacle>& obstacles) {
  float robot_width = 0.0;
  this->get_parameter("robot.width", robot_width);
  float min_passage_width = robot_width + 0.2f;

  std::vector<Obstacle> sorted_obs = obstacles;
  std::sort(sorted_obs.begin(), sorted_obs.end(),
    [](const Obstacle& a, const Obstacle& b) { return a.angle < b.angle; });

  if (sorted_obs.size() <= 1) return true;

  for (size_t i = 0; i < sorted_obs.size() - 1; i++) {
    float gap_angle = sorted_obs[i+1].angle - sorted_obs[i].angle;
    float avg_distance = (sorted_obs[i].distance + sorted_obs[i+1].distance) / 2.0f;
    float gap_width = avg_distance * gap_angle;

    if (gap_width > min_passage_width) {
      return true;
    }
  }

  return false;
}

// 地面模式指令
geometry_msgs::msg::Twist OmniAvoidanceNode::generate_ground_command(const std::vector<Obstacle>& obstacles) {
  geometry_msgs::msg::Twist cmd;
  float max_linear = 0.0;
  float max_angular = 0.0;
  float safety_distance = 0.0;
  
  this->get_parameter("velocity.ground_max_linear", max_linear);
  this->get_parameter("velocity.ground_max_angular", max_angular);
  this->get_parameter("obstacle.safety_distance", safety_distance);

  if (obstacles.empty()) {
    cmd.linear.x = max_linear;
    return cmd;
  }

  float min_distance = std::numeric_limits<float>::max();
  float nearest_angle = 0.0f;
  for (const auto& obs : obstacles) {
    if (obs.distance < min_distance) {
      min_distance = obs.distance;
      nearest_angle = obs.angle;
    }
  }

  if (min_distance < safety_distance) {
    cmd.linear.x = max_linear * (min_distance / safety_distance);
    cmd.angular.z = -nearest_angle * max_angular;
  } else {
    cmd.linear.x = max_linear;
  }

  return cmd;
}

// 飞行模式指令
geometry_msgs::msg::Twist OmniAvoidanceNode::generate_flight_command() {
  geometry_msgs::msg::Twist cmd;
  float max_linear = 0.0;
  float cross_distance = 0.0;
  
  this->get_parameter("velocity.flight_max_linear", max_linear);
  this->get_parameter("flight.cross_distance", cross_distance);

  switch (flight_state_) {
    case FlightState::IDLE:
      flight_state_ = FlightState::TAKEOFF;
      flight_start_time_ = this->now();
      RCLCPP_INFO(this->get_logger(), "开始起飞");

      if (current_state_.mode != "OFFBOARD") {
        auto set_mode_req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        set_mode_req->custom_mode = "OFFBOARD";
        set_mode_client_->async_send_request(set_mode_req);
      }

      if (!current_state_.armed) {
        auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        arm_req->value = true;
        arm_client_->async_send_request(arm_req);
      }

      cmd.linear.z = 0.2f;
      break;

    case FlightState::TAKEOFF: {
      rclcpp::Duration elapsed = this->now() - flight_start_time_;
      if (elapsed.seconds() < 5.0) {
        cmd.linear.z = 0.2f;
      } else {
        flight_state_ = FlightState::CROSS;
        flight_start_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "开始跨越");
      }
      break;
    }

    case FlightState::CROSS: {
      rclcpp::Duration elapsed = this->now() - flight_start_time_;
      float cross_time = cross_distance / max_linear;
      if (elapsed.seconds() < cross_time) {
        cmd.linear.x = max_linear;
        cmd.linear.z = 0.05f;
      } else {
        flight_state_ = FlightState::LAND;
        flight_start_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "开始降落");
      }
      break;
    }

    case FlightState::LAND: {
      rclcpp::Duration elapsed = this->now() - flight_start_time_;
      if (elapsed.seconds() < 5.0) {
        cmd.linear.z = -0.1f;
      } else {
        if (current_state_.armed) {
          auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
          arm_req->value = false;
          arm_client_->async_send_request(arm_req);
        }
        
        flight_state_ = FlightState::IDLE;
        current_mode_ = MotionMode::GROUND;
        RCLCPP_INFO(this->get_logger(), "降落完成，切换至地面模式");
      }
      break;
    }
  }

  return cmd;
}

}  // namespace wukechuang_omni_avoidance

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<wukechuang_omni_avoidance::OmniAvoidanceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


