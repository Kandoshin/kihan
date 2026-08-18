from torchvision import datasets
import json
dataset = datasets.ImageFolder("G:/god/pycharm/PythonProject/test/cnn/dataset/train")
print("num samples:", len(dataset))
print("classes:", dataset.classes)  # 列表，顺序即 label id
with open("class_to_idx.json", "w") as f:
    json.dump(dataset.class_to_idx, f, indent=2)
