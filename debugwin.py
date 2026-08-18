import ctypes, sys
hwnd = ctypes.windll.user32.FindWindowW(0, '雷电模拟器')
if not hwnd: sys.exit('未找到雷电模拟器')
ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, -2, 755, 466, 0x0044)
print('已固定到 0,-2,755×466')