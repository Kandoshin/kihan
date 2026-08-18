import pyautogui, time

def job():
    for _ in range(3):
        pyautogui.click(864,490, duration=0.5)
        time.sleep(2)
    time.sleep(20)
    pyautogui.click(466,514, duration=0.5)
    time.sleep(2)
    pyautogui.click(746,488, duration=0.5)

while True:
    job()
    time.sleep(300)          # 5 分钟