#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "mlx90640_ros2/MLX90640_API.h"
#include "mlx90640_ros2/MLX90640_I2C_Driver.h"

class MLX90640Node : public rclcpp::Node
{
public:
  MLX90640Node() : Node("mlx90640_node")
  {
    // 初始化参数
    declare_parameter("i2c_bus", 4);
    declare_parameter("frame_rate", 8);
    declare_parameter("resolution", 32);  // 32x24
    
    // 获取参数
    i2c_bus_ = get_parameter("i2c_bus").as_int();
    frame_rate_ = get_parameter("frame_rate").as_int();
    resolution_ = get_parameter("resolution").as_int();
    
    // 初始化 MLX90640
    InitializeMLX90640(i2c_bus_);
    
    // 创建图像发布者
    image_pub_ = image_transport::create_publisher(this, "thermal_image");
    
    // 创建定时器
    timer_ = create_wall_timer(
      std::chrono::milliseconds(1000 / frame_rate_),
      std::bind(&MLX90640Node::publishThermalImage, this));
    
    RCLCPP_INFO(get_logger(), "MLX90640 node started on I2C bus %d", i2c_bus_);
  }
  
  ~MLX90640Node()
  {
    // 清理资源
    MLX90640_Shutdown(i2c_bus_);
  }
  
private:
  int i2c_bus_;
  int frame_rate_;
  int resolution_;
  float mlx90640To[768];  // 32x24=768
  paramsMLX90640 params_;
  image_transport::Publisher image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  void InitializeMLX90640(int bus)
  {
    // 初始化 I2C 通信
    MLX90640_I2CInit(bus);
    
    // 读取并校准参数
    uint16_t eeMLX90640[832];
    MLX90640_DumpEE(bus, eeMLX90640);
    MLX90640_ExtractParameters(eeMLX90640, &params_);
    
    // 设置刷新速率
    MLX90640_SetRefreshRate(bus, 0x03);  // 8Hz
  }
  
  void publishThermalImage()
  {
    // 读取帧数据
    uint16_t frame[834];
    MLX90640_GetFrameData(i2c_bus_, frame);
    
    // 计算温度
    float Ta = MLX90640_GetTa(frame, &params_);
    MLX90640_CalculateTo(frame, &params_, 1.0f, Ta, mlx90640To);
    
    // 创建 OpenCV 图像
    cv::Mat thermal_image(24, 32, CV_32FC1, mlx90640To);
    
    // 归一化并应用伪彩色
    cv::Mat normalized, colored;
    cv::normalize(thermal_image, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    
    // 转换为 ROS 消息并发布
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", colored).toImageMsg();
    msg->header.stamp = now();
    msg->header.frame_id = "thermal_camera";
    image_pub_.publish(msg);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MLX90640Node>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

