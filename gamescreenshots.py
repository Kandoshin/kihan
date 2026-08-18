
import os
import tkinter as tk
from tkinter import ttk
import mss
import mss.tools

# 截图区域
ROI = {"left": 0, "top": 46, "width": 945, "height": 535}
SAVE_DIR = r"G:\god\yolov5\yolov5-7.0\naruto\images"

# 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# 计算下一个文件名
def next_index():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.png')]
    nums = [int(os.path.splitext(f)[0]) for f in files if f[:-4].isdigit()]
    return max(nums, default=0) + 1

# 截图函数
def take_shot():
    idx = next_index()
    filename = os.path.join(SAVE_DIR, f"{idx}.png")
    with mss.mss() as sct:
        img = sct.grab(ROI)
        mss.tools.to_png(img.rgb, img.size, output=filename)
    print(f"已保存：{filename}")

# 简单 GUI
root = tk.Tk()
root.title("ROI 截图工具")
root.geometry("200x100+300+300")   # 宽x高+x+y
btn = ttk.Button(root, text="截图", command=take_shot)
btn.pack(expand=True)
root.mainloop()