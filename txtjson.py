import os
import json
import cv2
from glob import glob
import base64

def convert_yolo_to_labelme():
    # ============== 在这里硬编码路径 ==============
    image_dir = r"C:\Users\Ginmart1\Desktop\txtjson\images"  # 图片目录
    txt_dir = r"C:\Users\Ginmart1\Desktop\txtjson\labels"  # YOLO标注目录
    classes_path = r"C:\Users\Ginmart1\Desktop\txtjson\classes.txt"  # 类别文件路径
    output_dir = r"C:\Users\Ginmart1\Desktop\txtjson\json"  # 输出目录
    # ============================================

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取类别列表
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    # 遍历所有图片文件
    for img_path in glob(os.path.join(image_dir, "*.*")):
        if not img_path.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # 生成对应文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(txt_dir, f"{base_name}.txt")

        # 跳过没有标注的文件
        if not os.path.exists(txt_path):
            print(f"警告: 未找到标注文件 {txt_path}，已跳过")
            continue

        # 读取图片尺寸
        try:
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
        except Exception as e:
            print(f"错误: 无法读取图片 {img_path} ({str(e)})，已跳过")
            continue

        # 解析YOLO标注
        shapes = []
        with open(txt_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError("无效的行格式")

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    # 转换为矩形坐标
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2

                    # 验证类别ID
                    if class_id < 0 or class_id >= len(classes):
                        raise ValueError(f"无效的类别ID {class_id}")

                    # 构建形状数据
                    shapes.append({
                        "label": classes[class_id],
                        "points": [
                            [round(x_min, 2), round(y_min, 2)],
                            [round(x_max, 2), round(y_max, 2)]
                        ],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })
                except Exception as e:
                    print(f"错误: 文件 {txt_path} 第{line_num}行解析失败 ({str(e)})")
        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        # 构建JSON结构
        labelme_json = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": image_data,
            "imageHeight": height,
            "imageWidth": width
        }

        # 保存JSON文件
        output_path = os.path.join(output_dir, f"{base_name}.json")
        with open(output_path, "w") as f:
            json.dump(labelme_json, f, indent=2)
        print(f"已生成: {output_path}")


if __name__ == "__main__":
    convert_yolo_to_labelme()  # 直接运行无需参数
