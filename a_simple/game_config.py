import os
import json
import numpy as np

# ============= 1. 动态路径配置 =============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PPO_MODEL_PATH = os.path.join(CURRENT_DIR, "ppo_realtime.zip")
TRAINING_METADATA_PATH = os.path.join(CURRENT_DIR, "training_metadata.json")
TENSORBOARD_LOG_DIR = os.path.join(CURRENT_DIR, "logs")

# ============= 2. 外部依赖路径 =============
YOLO_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0"
MODEL_PATH = r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\best.pt"

# CNN 模型相关路径
CNN_SAVE_DIR = r"G:\god\pycharm\PythonProject\test\cnn\models"
CNN_WEIGHT_PATH = os.path.join(CNN_SAVE_DIR, "best.pth")
CLASS_MAP_PATH = os.path.join(CNN_SAVE_DIR, "class_to_idx.json")

# ============= 3. 游戏区域 =============
GAME_REGION = {"left": 0, "top": 46, "width": 945, "height": 535}
MONITOR_1P = {"left": 98, "top": 91, "width": 176, "height": 5}
MONITOR_2P = {"left": 672, "top": 91, "width": 176, "height": 5}

# ============= 4. 视觉参数 =============
CONF_THRES, IOU_THRES = 0.8, 0.8
TARGET_CLASSES = {'p', 'b', 'r', 's', 'wm', 'rm'}
IMG_SIZE = 70

# ============= 5. 动作映射 =============
MOVE_MAP = {
    0: set(), 1: {'w'}, 2: {'s'}, 3: {'a'}, 4: {'d'},
    5: {'w', 'a'}, 6: {'w', 'd'}, 7: {'a', 's'}, 8: {'s', 'd'},
}

SKILL_MAP = {
    0: None, 1: 'j', 2: 'k', 3: 'i', 4: 'q', 5: 'e', 6: 'space'
}

KEYS = ['w', 's', 'a', 'd', 'j', 'k', 'i', 'q', 'e', 'space']

# ============= 6. CNN 区域定义 =============
cnn_regions = [
    {"left": 30, "top": 58, "width": 70, "height": 70, "name": "1p"},
    {"left": 845, "top": 58, "width": 70, "height": 70, "name": "2p"},
    {"left": 824, "top": 151, "width": 56, "height": 56, "name": "summon"},
    {"left": 824, "top": 238, "width": 56, "height": 56, "name": "scroll"},
    {"left": 624, "top": 506, "width": 30, "height": 30, "name": "stand"},
    {"left": 726, "top": 505, "width": 30, "height": 30, "name": "skill1"},
    {"left": 738, "top": 400, "width": 30, "height": 30, "name": "skill2"},
    {"left": 428, "top": 83, "width": 88, "height": 66, "name": "time"},
]

# ============= 7. 加载 Class Map & 关键 ID =============
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
else:
    class_to_idx = {"none": 92}

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# 获取特殊的 ID
NONE_ID = class_to_idx.get("none", 92)

# [已确认] 死亡判定 ID ("x": 128)
DEATH_ID = 128