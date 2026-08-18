"""
update_yaml_names.py
一次性脚本，把 classes.txt 的类别名写入 narutot.yaml 的 names 字段
"""

from pathlib import Path
import yaml

# ============== 路径按需修改 ==============
classes_path = Path(r"G:\god\yolov5\yolov5-7.0\naruto\labels\classes.txt")
yaml_path    = Path(r"G:\god\yolov5\yolov5-7.0\yolov5-7.0\data\narutot.yaml")
# =========================================

# 1. 读 classes.txt
with open(classes_path, encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

# 2. 读原 yaml
with open(yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

# 3. 构造新的 names 字典
data["names"] = {idx: name for idx, name in enumerate(class_names)}

# 4. 写回 yaml（覆盖原文件；如需备份，先复制一份）
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)

print("✅ 已更新 names 字段，共写入", len(class_names), "个类别。")