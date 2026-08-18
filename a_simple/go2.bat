@echo off
:: 1. 调整终端窗口位置 (755,-6) 和大小 (780x432)
powershell -Command "$c='[DllImport(\"user32.dll\")]public static extern bool MoveWindow(IntPtr hwnd,int x,int y,int w,int h,bool r);';$type=Add-Type -MemberDefinition $c -Name WinAPI -PassThru;$proc=Get-Process -Name 'WindowsTerminal' | Select-Object -First 1;$type::MoveWindow($proc.MainWindowHandle,755,-6,780,432,$true)"

:: 2. 激活环境并运行 Python
call activate yolov5gpu
G:
cd G:\god\pycharm\PythonProject\test\a_simple
python run_inference.py

pause