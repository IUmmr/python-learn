"""
第五周 任务1：整理前4周代码并分类归档
将各周作业代码按主题分类复制到 week5/archive/ 下
"""
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # program/
ARCHIVE = os.path.join(BASE, 'archive')

# 归档映射: 目标分类 -> [(源路径, 归档后文件名), ...]
ARCHIVE_MAP = {
    '01_Python编程基础': [
        (os.path.join(ROOT, 'week1', 'average.py'), 'average.py'),
        (os.path.join(ROOT, 'week1', 'prime number.py'), 'prime_number.py'),
        (os.path.join(ROOT, 'week1', 'files statistics.py'), 'file_statistics.py'),
        (os.path.join(ROOT, 'week1', 'students.py'), 'students.py'),
        (os.path.join(ROOT, 'week1', 'bubble Sort.py'), 'bubble_sort.py'),
        (os.path.join(ROOT, 'week1', 'multiplication table.py'), 'multiplication_table.py'),
        (os.path.join(ROOT, 'week1', 'tabulated function.py'), 'tabulated_function.py'),
    ],
    '02_数据分析与可视化': [
        (os.path.join(ROOT, 'week2', 'Numpy.py'), 'numpy_practice.py'),
        (os.path.join(ROOT, 'week2', 'pandas-practice.py'), 'pandas_practice.py'),
        (os.path.join(ROOT, 'week2', 'pandas-practice2.py'), 'pandas_practice2.py'),
        (os.path.join(ROOT, 'week2', 'matplotlib-practice.py'), 'matplotlib_practice.py'),
    ],
    '03_医学影像与医疗大数据': [
        (os.path.join(ROOT, 'week3', 'medical_image_preprocess.py'), 'medical_image_preprocess.py'),
        (os.path.join(ROOT, 'week3', 'medical_insurance_analysis.py'), 'medical_insurance_analysis.py'),
    ],
    '04_深度学习': [
        (os.path.join(ROOT, 'week4', 'cnn_cifar10.py'), 'cnn_cifar10.py'),
    ],
}


def main():
    total = 0
    lines = ['# 代码分类归档清单\n', '> 依据第五周任务1生成，按主题整理前4周作业代码。\n']
    for category, files in ARCHIVE_MAP.items():
        cat_dir = os.path.join(ARCHIVE, category)
        os.makedirs(cat_dir, exist_ok=True)
        lines.append(f'\n## {category}\n')
        lines.append('| 归档文件名 | 原始位置 |')
        lines.append('|------------|----------|')
        for src, dst_name in files:
            if not os.path.exists(src):
                print(f'[警告] 源文件不存在: {src}')
                continue
            dst = os.path.join(cat_dir, dst_name)
            shutil.copy2(src, dst)
            rel_src = os.path.relpath(src, ROOT).replace('\\', '/')
            lines.append(f'| `{dst_name}` | `{rel_src}` |')
            total += 1
            print(f'[OK] {os.path.basename(src)} -> {category}/{dst_name}')

    lines.append(f'\n---\n\n共归档 **{total}** 个文件。')
    with open(os.path.join(ARCHIVE, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n归档完成，共 {total} 个文件，清单见 archive/README.md')


if __name__ == '__main__':
    main()
