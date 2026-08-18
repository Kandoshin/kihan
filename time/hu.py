import os
import numpy as np

# 目录
HU_DIR = r'G:\god\pycharm\PythonProject\test\time\hutxt'
OUT_PATH = os.path.join(HU_DIR, 'allhu.txt')

lines = []
for idx in range(61):
    txt = os.path.join(HU_DIR, f'{idx}.txt')
    if not os.path.exists(txt):
        print(f'{txt} 不存在，跳过')
        continue
    hu = np.loadtxt(txt)        # 7 维
    hu_str = ','.join(f'{x:.8f}' for x in hu)
    lines.append(f'{idx}:({hu_str})\n')

# 写文件
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('✅ 已生成 allhu.txt')