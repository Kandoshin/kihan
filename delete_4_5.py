from pathlib import Path

# 1. 改成你的 labels 目录
LABEL_DIR = Path(r'G:\god\yolov5\yolov5-7.0\naruto\labels')

# 2. 遍历所有 txt
for txt_file in LABEL_DIR.glob('*.txt'):
    lines = txt_file.read_text().splitlines()
    # 3. 过滤掉 4 或 5 开头的整行
    new_lines = [ln for ln in lines if not ln.strip().startswith(('4 ', '5 '))]
    # 4. 写回（空文件就留空）
    txt_file.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''))

print('✅ 已删除所有 4、5 开头行')