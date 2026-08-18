import ctypes,json
h=ctypes.windll.user32.FindWindowW(0,'雷电模拟器')
if not h:exit('未找到窗口')
r=ctypes.wintypes.RECT()
ctypes.windll.user32.GetWindowRect(h,ctypes.byref(r))
print(json.dumps({"left":r.left,"top":r.top,"width":r.right-r.left,"height":r.bottom-r.top}))