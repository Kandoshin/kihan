@echo off
:: 设置控制台编码为 UTF-8，防止中文乱码 (可选)
chcp 65001

echo ==========================================
echo      Starting Imitation Learning Training
echo ==========================================

:: 1. 激活 Anaconda 环境
:: 注意：这里假设你的 Anaconda 安装在默认位置或已添加环境变量。
:: 如果 call activate 失败，请尝试使用绝对路径，例如:
:: call "G:\god\anaconda\Scripts\activate.bat" yolov5gpu
call activate yolov5gpu

:: 2. 切换到脚本所在驱动器 (G盘)
G:

:: 3. 进入项目目录
cd "G:\god\pycharm\PythonProject\test\b_imitation"

:: 4. 运行训练脚本
echo Running bc_train.py...
python bc_train.py

:: 5. 训练结束后暂停，方便查看日志
echo.
echo ==========================================
echo      Training Finished. Press any key to exit.
echo ==========================================
pause