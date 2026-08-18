# soft_blackout_fixed.py
import tkinter as tk
import ctypes
from ctypes import wintypes

# ---------- 让窗口鼠标穿透 ----------
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080

def set_click_through(hwnd):
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

# ---------- 主窗口 ----------
def soft_blackout():
    root = tk.Tk()
    root.configure(bg='black')               # 正常黑色背景
    root.attributes('-fullscreen', True)     # 全屏
    root.attributes('-topmost', True)        # 置顶
    # root.config(cursor='none')               # 隐藏鼠标
    root.overrideredirect(True)              # 去掉标题栏
    root.bind('<Escape>', lambda e: root.destroy())

    root.update_idletasks()                  # 必须先更新，hwnd 才有效
    set_click_through(root.winfo_id())       # 设穿透

    root.mainloop()

if __name__ == '__main__':
    soft_blackout()