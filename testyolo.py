
import cv2
import torch
import numpy as np
import yaml
from mss import mss  # 屏幕捕获库
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes


class ScreenDetector:
    def __init__(self, data_yaml_path, model_weights_path):
        """
        初始化屏幕检测器
        :param data_yaml_path: 数据配置文件路径（包含类别信息）
        :param model_weights_path: 模型权重文件路径
        """
        # 加载数据配置（关键参数）
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)
        self.class_names = data_cfg['names']
        self.colors = np.random.randint(0, 255, (len(self.class_names), 3))

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载训练好的YOLOv5模型
        self.model = attempt_load(model_weights_path, device=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = 640  # 与训练时设置的输入尺寸一致

        # 模型预热
        self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device))

    def process_frame(self, frame):
        """完整的帧处理流程"""
        # 预处理
        img = self.preprocess(frame)

        # 推理
        with torch.no_grad():
            pred = self.model(img)[0]

        # 后处理
        return self.postprocess(pred, frame.shape)

    def preprocess(self, frame):
        """图像预处理（保持宽高比）"""
        # 自动调整大小+填充
        img = letterbox(frame, self.img_size, stride=self.stride, auto=True)[0]

        # 转换通道顺序和归一化
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        return img.unsqueeze(0)  # 添加batch维度

    def postprocess(self, prediction, orig_shape):
        """结果后处理"""
        # 应用NMS
        pred = non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.45)

        # 转换坐标到原始尺寸
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.img_size, det[:, :4], orig_shape).round()
                detections.extend(det.cpu().numpy())
        return detections


def screen_capture(monitor=0):
    """屏幕捕获生成器"""
    with mss() as sct:
        target_monitor = sct.monitors[monitor]  # 默认选择第二个显示器
        while True:
            yield np.array(sct.grab(target_monitor))[:, :, :3]  # 去除alpha通道


def visualize(frame, detections, class_names, colors):
    """可视化标注"""
    for *xyxy, conf, cls in detections:
        # 解析参数
        x1, y1, x2, y2 = map(int, xyxy)
        cls = int(cls)

        # 绘制边界框
        color = tuple(map(int, colors[cls]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 构建标签
        label = f"{class_names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # 绘制标签背景
        cv2.rectangle(frame,
                      (x1, y1 - th - 4),
                      (x1 + tw, y1),
                      color, -1)

        # 添加文本
        cv2.putText(frame, label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255),
                    1, cv2.LINE_AA)
    return frame


# 配置文件使用说明 ---------------------------------------------------
"""
需要指定两个关键路径：
1. data.yaml: 包含类别信息的配置文件（必须）
   示例内容：
   names:
     - cat
     - dog

2. 模型YAML: 网络结构配置文件（自动嵌入在.pt文件中，无需指定）
"""

if __name__ == "__main__":
    # 初始化检测器（按实际路径配置）
    detector = ScreenDetector(
        data_yaml_path=r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\data\narutot.yaml",  # 包含类别信息的YAML
        model_weights_path=r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"  # 训练好的模型权重
    )

    # 初始化屏幕捕获（默认捕获第二个显示器）
    screen_gen = screen_capture(monitor=0)

    # 实时检测循环
    while True:
        # 捕获屏幕帧
        screen_frame = next(screen_gen)

        # 执行检测
        detections = detector.process_frame(screen_frame)

        # 可视化结果
        result_frame = visualize(screen_frame.copy(), detections,
                                 detector.class_names, detector.colors)

        # 显示结果（按Q退出）
        cv2.imshow("Screen Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
