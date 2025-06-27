#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std::chrono_literals;

class DepthDistanceNode : public rclcpp::Node {
public:
  DepthDistanceNode() : Node("depth_distance_node") {
    // 订阅深度图像话题（根据实际输出调整话题名）
    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", 10,
      std::bind(&DepthDistanceNode::depth_callback, this, std::placeholders::_1));
  }

private:
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // 将ROS图像转换为OpenCV格式（深度图像为16UC1类型）
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    cv::Mat depth_image = cv_ptr->image;

    // 示例：提取图像中心像素的深度值（单位：毫米）
    int height = depth_image.rows;
    int width = depth_image.cols;
    int center_y = height / 2;
    int center_x = width / 2;
    uint16_t depth_value = depth_image.at<uint16_t>(center_y, center_x);

    // 转换为米（mm -> m）
    float distance_meters = depth_value / 1000.0f;

    // 输出距离（仅显示有效深度值，跳过0值）
    if (depth_value != 0) {
      RCLCPP_INFO(this->get_logger(), "Center distance: %.2f meters", distance_meters);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthDistanceNode>());
  rclcpp::shutdown();
  return 0;
}
