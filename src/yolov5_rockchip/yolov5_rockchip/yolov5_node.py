#!/home/elf/miniconda3/envs/rknn-env/bin/python3  # 替换为你的Python路径
# -*- coding: utf-8 -*-
import os
import sys

# 强制添加rknn_toolkit_lite2到Python路径
rknn_path = '/home/elf/miniconda3/envs/rknn-env/lib/python3.10/site-packages'  # 替换为你的site-packages路径
if rknn_path not in sys.path:
    sys.path.append(rknn_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# 全局参数（仅声明，不在此处赋值）
OBJ_THRESH = None
NMS_THRESH = None
IMG_SIZE = 640
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop", "mouse", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

class YOLOv5Node(Node):
    def __init__(self):
        super().__init__('yolov5_rockchip_node')

        # 声明可配置参数
        self.declare_parameter('model_path', '/home/elf/Astra_ws/src/yolov5_rockchip/models/yolov5s_relu_rk3588.rknn')
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/yolov5/detections')
        self.declare_parameter('obj_thresh', 0.25)  # 默认值直接写在这里
        self.declare_parameter('nms_thresh', 0.45)  # 默认值直接写在这里

        # 正确使用global声明（在赋值前，且不引用全局默认值）
        global OBJ_THRESH, NMS_THRESH
        OBJ_THRESH = self.get_parameter('obj_thresh').value
        NMS_THRESH = self.get_parameter('nms_thresh').value

        # 初始化RKNN模型
        self.rknn = self.load_rknn_model()
        if not self.rknn:
            self.get_logger().fatal('模型加载失败，节点退出')
            exit(1)

        # 创建图像转换桥接
        self.bridge = CvBridge()

        # 创建订阅者和发布者
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('input_topic').value,
            self.image_callback,
            10)
        self.subscription  # 防止未使用变量警告

        self.publisher = self.create_publisher(
            Image,
            self.get_parameter('output_topic').value,
            10)

        self.get_logger().info(f'YOLOv5 Rockchip节点已启动，目标检测阈值: {OBJ_THRESH}')

    def load_rknn_model(self):
        """加载RKNN模型并初始化运行时"""
        try:
            # 从rknn_toolkit_lite2导入RKNNLite
            from rknn_toolkit_lite2.rknnlite.api import RKNNLite
            rknn = RKNNLite()
            model_path = self.get_parameter('model_path').value

            # 加载模型
            if rknn.load_rknn(model_path) != 0:
                self.get_logger().error(f'加载模型失败: {model_path}')
                return None

            # 初始化模型运行时
            if rknn.init_runtime() != 0:
                self.get_logger().error('初始化模型运行时失败')
                return None

            self.get_logger().info('RKNN Lite 初始化成功')
            return rknn
        except ImportError as e:
            self.get_logger().error(f'模块导入失败: {e}，请检查rknn_toolkit_lite2安装')
            return None

    def preprocess(self, image):
        """图像预处理（与转换时完全一致）"""
        # BGR->RGB转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 调整大小到640x640
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # 添加batch维度 (NHWC格式)
        return np.expand_dims(image, axis=0)

    def xywh2xyxy(self, x):
        """中心坐标转边界坐标"""
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def process(self, input, mask, anchors):
        """处理单个输出层"""
        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        # 提取置信度和类别概率
        box_confidence = np.expand_dims(input[..., 4], axis=-1)
        box_class_probs = input[..., 5:]

        # 计算边界框中心
        box_xy = input[..., :2] * 2 - 0.5

        # 构建网格坐标
        col = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w)
        row = np.tile(np.arange(grid_h).reshape(-1, 1), grid_w)
        grid = np.stack([col, row], axis=-1).reshape(grid_h, grid_w, 1, 2)

        # 调整边界框位置
        box_xy += grid
        box_xy *= int(IMG_SIZE / grid_h)

        # 计算边界框尺寸
        box_wh = pow(input[..., 2:4] * 2, 2) * anchors

        return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs

    def filter_boxes(self, boxes, confidences, class_probs):
        """过滤低置信度边界框"""
        boxes = boxes.reshape(-1, 4)
        confidences = confidences.reshape(-1)
        class_probs = class_probs.reshape(-1, class_probs.shape[-1])

        # 第一次过滤：目标置信度
        keep = confidences >= OBJ_THRESH
        boxes = boxes[keep]
        confidences = confidences[keep]
        class_probs = class_probs[keep]

        # 第二次过滤：类别置信度
        class_max = np.max(class_probs, axis=-1)
        classes = np.argmax(class_probs, axis=-1)
        keep = class_max >= OBJ_THRESH

        return boxes[keep], classes[keep], (class_max * confidences)[keep]

    def nms_boxes(self, boxes, scores):
        """非极大值抑制"""
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留低重叠框
            inds = np.where(iou <= NMS_THRESH)[0]
            order = order[inds + 1]

        return np.array(keep)

    def yolov5_post_process(self, inputs):
        """YOLOv5后处理主函数"""
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        all_boxes, all_classes, all_scores = [], [], []

        # 处理三个输出层
        for input, mask in zip(inputs, masks):
            boxes, confs, probs = self.process(input, mask, anchors)
            boxes, classes, scores = self.filter_boxes(boxes, confs, probs)
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)

        # 合并所有检测结果
        boxes = np.concatenate(all_boxes, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # 无检测结果处理
        if len(boxes) == 0:
            return None, None, None

        # 转换坐标格式
        boxes = self.xywh2xyxy(boxes)

        # 按类别进行NMS
        final_boxes, final_classes, final_scores = [], [], []
        for cls in set(classes):
            idx = classes == cls
            cls_boxes = boxes[idx]
            cls_scores = scores[idx]

            keep = self.nms_boxes(cls_boxes, cls_scores)
            final_boxes.append(cls_boxes[keep])
            final_classes.append(classes[idx][keep])
            final_scores.append(cls_scores[keep])

        return (np.concatenate(final_boxes),
                np.concatenate(final_classes),
                np.concatenate(final_scores))

    def prepare_outputs(self, outputs):
        """对齐虚拟机输出处理逻辑"""
        out0 = outputs[0].reshape([3, -1] + list(outputs[0].shape[-2:]))
        out1 = outputs[1].reshape([3, -1] + list(outputs[1].shape[-2:]))
        out2 = outputs[2].reshape([3, -1] + list(outputs[2].shape[-2:]))

        return [
            np.transpose(out0, (2, 3, 0, 1)),
            np.transpose(out1, (2, 3, 0, 1)),
            np.transpose(out2, (2, 3, 0, 1))
        ]

    def draw_results(self, image, boxes, classes, scores):
        """绘制检测结果"""
        orig_h, orig_w = image.shape[:2]
        if boxes is not None:
            # 坐标映射回原图
            scale_x, scale_y = orig_w/IMG_SIZE, orig_h/IMG_SIZE
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            # 绘制结果
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f"{CLASSES[cls]} {score:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def image_callback(self, msg):
        """图像回调函数，处理每一帧相机数据"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换错误: {e}')
            return

        # 预处理图像
        input_data = self.preprocess(cv_image)

        # 模型推理
        try:
            outputs = self.rknn.inference(inputs=[input_data], data_format=["nhwc"])
        except Exception as e:
            self.get_logger().error(f'模型推理错误: {e}')
            return

        # 处理输出
        try:
            processed_outs = self.prepare_outputs(outputs)
            boxes, classes, scores = self.yolov5_post_process(processed_outs)
        except Exception as e:
            self.get_logger().error(f'后处理错误: {e}')
            return

        # 绘制检测结果
        result_image = self.draw_results(cv_image.copy(), boxes, classes, scores)

        # 发布结果图像
        try:
            result_msg = self.bridge.cv2_to_imgmsg(result_image, 'bgr8')
            self.publisher.publish(result_msg)
        except Exception as e:
            self.get_logger().error(f'结果发布错误: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断')
    finally:
        # 释放资源
        if hasattr(node, 'rknn') and node.rknn:
            node.rknn.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

