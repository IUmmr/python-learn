"""
第五周 任务2：绘制知识思维导图
生成五周 Python 数据分析学习知识体系思维导图 PNG
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 思维导图数据
# ============================================================
ROOT = 'Python数据分析\n五周学习体系'

# 一级分支: (标题, 颜色, [子节点...])
BRANCHES = [
    ('第1周 Python基础', '#E74C3C', [
        '基础语法\n(数据类型/容器)',
        '控制结构\n(分支/循环)',
        '函数与面向对象',
        '文件与异常处理',
    ]),
    ('第2周 数据分析', '#E67E22', [
        'NumPy 数值计算',
        'Pandas 表格处理\n(清洗/分组聚合)',
        'Matplotlib 可视化',
    ]),
    ('第3周 医学影像+医保', '#27AE60', [
        'SimpleITK 影像预处理\n(生成/归一化/统计)',
        '医保大数据清洗\n(缺失填充/异常剔除)',
    ]),
    ('第4周 深度学习', '#2980B9', [
        'PyTorch 框架\n(张量/自动求导)',
        'CNN 卷积神经网络\n(Conv/BN/池化)',
        'CIFAR-10 分类\n(77.98%)',
    ]),
    ('通用能力', '#8E44AD', [
        '环境与工具\n(conda/pip/VS Code)',
        '工程习惯\n(注释/编码/Git)',
        '学习方法论\n(预处理优先/复盘)',
    ]),
]

# ============================================================
# 布局绘制（按分支占用空间动态分配纵向位置）
# ============================================================
fig, ax = plt.subplots(figsize=(17, 10), dpi=150)
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# 中心主题
root_x, root_y = 1.2, 5.0
ax.add_patch(FancyBboxPatch(
    (root_x - 1.3, root_y - 0.65), 2.6, 1.3,
    boxstyle='round,pad=0.12,rounding_size=0.25',
    facecolor='#2C3E50', edgecolor='none'))
ax.text(root_x, root_y, ROOT, ha='center', va='center',
        fontsize=15, fontweight='bold', color='white')

# 节点尺寸
BRANCH_H = 0.7
BRANCH_W = 3.0
CHILD_H = 0.58
CHILD_W = 2.4
CHILD_VSPACE = 0.85
COLUMN_X = [8.0, 12.0]

# 计算每个分支的高度（两列布局）
branch_heights = []
for _, _, children in BRANCHES:
    half = (len(children) + 1) // 2
    height = max(BRANCH_H, half * CHILD_VSPACE + 0.3)
    branch_heights.append(height)

total_height = sum(branch_heights)
available_h = 9.2  # 顶部到底部可用空间
margin = (available_h - total_height) / (len(BRANCHES) + 1)

# 计算每个一级分支的 y 中心
current_y = margin
branch_centers = []
for h in branch_heights:
    branch_centers.append(current_y + h / 2)
    current_y += h + margin

# 一级节点 x 坐标
b_x = 4.0

for i, (title, color, children) in enumerate(BRANCHES):
    y = branch_centers[i]

    # ---- 中心 -> 一级 ----
    ax.plot([root_x + 1.3, b_x - BRANCH_W / 2], [root_y, y],
            color=color, lw=1.8, alpha=0.7, zorder=1)

    # ---- 一级节点 ----
    ax.add_patch(FancyBboxPatch(
        (b_x - BRANCH_W / 2, y - BRANCH_H / 2), BRANCH_W, BRANCH_H,
        boxstyle='round,pad=0.08,rounding_size=0.15',
        facecolor=color, edgecolor='none', zorder=3))
    ax.text(b_x, y, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # ---- 二级节点（两列） ----
    half = (len(children) + 1) // 2
    col1 = children[:half]
    col2 = children[half:]

    def column_positions(n, center_y):
        """计算一列 n 个节点的 y 位置"""
        total = n * CHILD_VSPACE
        start = center_y + total / 2 - CHILD_VSPACE / 2
        return [start - k * CHILD_VSPACE for k in range(n)]

    pos1 = column_positions(len(col1), y)
    pos2 = column_positions(len(col2), y)

    for k, (text, cy) in enumerate(zip(col1, pos1)):
        cx = COLUMN_X[0]
        ax.plot([b_x + BRANCH_W / 2, cx - CHILD_W / 2], [y, cy],
                color=color, lw=1.2, alpha=0.5, zorder=1)
        ax.add_patch(FancyBboxPatch(
            (cx - CHILD_W / 2, cy - CHILD_H / 2), CHILD_W, CHILD_H,
            boxstyle='round,pad=0.06,rounding_size=0.12',
            facecolor='white', edgecolor=color, lw=1.5, zorder=3))
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=8.5, color='#2C3E50', zorder=4)

    for k, (text, cy) in enumerate(zip(col2, pos2)):
        cx = COLUMN_X[1]
        ax.plot([COLUMN_X[0] + CHILD_W / 2, cx - CHILD_W / 2], [pos1[k], cy],
                color=color, lw=1.2, alpha=0.5, zorder=1)
        ax.add_patch(FancyBboxPatch(
            (cx - CHILD_W / 2, cy - CHILD_H / 2), CHILD_W, CHILD_H,
            boxstyle='round,pad=0.06,rounding_size=0.12',
            facecolor='white', edgecolor=color, lw=1.5, zorder=3))
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=8.5, color='#2C3E50', zorder=4)

ax.set_title('Python 数据分析五周学习 · 知识思维导图',
             fontsize=20, fontweight='bold', color='#2C3E50', pad=15)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_map.png')
fig.savefig(out, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'思维导图已生成: {out}')
plt.close(fig)
