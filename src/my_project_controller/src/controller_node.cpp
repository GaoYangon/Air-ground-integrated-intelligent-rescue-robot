#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class MyProjectController : public rclcpp::Node
{
public:
  MyProjectController() : Node("my_project_controller")
  {
    // 订阅深度图像话题
    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", 10, 
      std::bind(&MyProjectController::depthCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "Controller node initialized. Listening for depth images...");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      // 将ROS图像消息转换为OpenCV格式
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      
      // 获取深度图像
      cv::Mat depth_image = cv_ptr->image;
      
      // 在这里添加你的深度图像处理逻辑
      // 例如：计算深度平均值、检测特定距离的物体等
      
      // 示例：打印图像中心像素的深度值（单位：毫米）
      cv::Point center(depth_image.cols / 2, depth_image.rows / 2);
      uint16_t depth_value = depth_image.at<uint16_t>(center);
      RCLCPP_INFO(this->get_logger(), "Depth at center: %d mm", depth_value);
      
      // 显示深度图像（可选，需要GUI支持）
      // cv::imshow("Depth Image", depth_image / 1000.0);  // 缩放以便于显示
      // cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MyProjectController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
