import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab
import time
import os

# 设置截图区域
region = (428, 83, 428 + 88, 83 + 66)  # (left, top, right, bottom)

# 设置保存路径
save_path = r'G:\\god\\pycharm\\PythonProject\\test\\time\\images\\'

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

def take_screenshot(index):
    filename = f"{save_path}\{index}.png"
    img = ImageGrab.grab(region)
    img.save(filename)
    return filename

def screenshot_loop():
    for i in range(60, -1, -1):  # 截图60次，文件名从60递减到0
        filename = take_screenshot(i)
        time.sleep(1)  # 每隔一秒截图一次

# 创建主窗口
root = tk.Tk()
root.title("Screenshot Tool")

# 创建按钮
button = tk.Button(root, text="Start Screenshot", command=lambda: screenshot_loop())
button.pack(pady=20)

# 运行主循环
root.mainloop()