"""
Transform the original SST-2 dataset to GLUE style
"""


import os

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 处理train和dev文件
for f in ['train.tsv', 'dev.tsv']:
    if os.path.exists(f):
        # 读取文件内容
        with open(f, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        
        # 写入转换后的内容
        with open(f, 'w', encoding='utf-8') as fout:
            fout.write('sentence\tlabel\n')
            for line in lines[1:]:  # 跳过表头
                fout.write(line.strip() + '\n')
        

# 处理test文件
if os.path.exists('test.tsv'):
    # 读取文件内容
    with open('test.tsv', 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    # 写入转换后的内容
    with open('test.tsv', 'w', encoding='utf-8') as fout:
        fout.write('sentence\tlabel\n')
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label, sentence = parts[0], parts[1]
                    fout.write(f"{sentence}\t{label}\n")
    

print("\n所有文件转换并覆盖完成！")