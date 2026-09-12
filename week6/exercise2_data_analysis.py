"""
============================================================
第6周 习题2：数据分析综合题
============================================================
题目要求：
  自动生成一份包含「姓名、年龄、收入」共 60 行的数据集；
  1. 检测缺失值，用均值填充数值字段；
  2. 剔除收入大于 200000 的异常样本；
  3. 按年龄段分组：青年 (18-35)、中年 (36-59)、老年 (≥60)；
  4. 统计每组平均收入；
  5. 绘制各组平均收入柱状图。

核心知识点：
  1. pandas DataFrame 构建与随机数据生成
  2. isnull / fillna 缺失值处理
  3. 条件筛选剔除异常值
  4. cut / groupby 分组聚合
  5. matplotlib 柱状图可视化
============================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_dataset(n=60, seed=42):
    """生成 60 行模拟数据集，并人为制造缺失值"""
    np.random.seed(seed)
    surnames = ['张', '李', '王', '刘', '陈', '杨', '赵', '黄', '周', '吴']
    names = [np.random.choice(surnames) + np.random.choice(['伟', '芳', '娜', '敏', '静',
                                                            '磊', '军', '洋', '勇', '艳'])
             for _ in range(n)]
    # 年龄 18~75，收入 3000~300000（含少量高收入异常）
    ages = np.random.randint(18, 76, size=n)
    incomes = np.random.randint(3000, 300000, size=n).astype(float)

    df = pd.DataFrame({'姓名': names, '年龄': ages, '收入': incomes})

    # 人为制造缺失值：随机选 5 个"收入"置为 NaN
    miss_idx = np.random.choice(n, size=5, replace=False)
    df.loc[miss_idx, '收入'] = np.nan

    print(f"  已生成 {n} 行数据，其中 {len(miss_idx)} 个收入缺失")
    return df


def main():
    print("=" * 60)
    print("第6周 习题2：数据分析综合题")
    print("=" * 60)

    # ---------- 生成数据集 ----------
    print("\n[步骤0] 生成模拟数据集（含缺失值）")
    df = generate_dataset()
    print(df.head(10).to_string(index=False))

    # ---------- 步骤1：缺失值检测与均值填充 ----------
    print("\n[步骤1] 检测缺失值并用均值填充数值字段")
    print(f"  缺失值统计:\n{df.isnull().sum().to_string()}")
    print(f"  填充前收入均值: {df['收入'].mean():.2f}")

    income_mean = df['收入'].mean()
    df['收入'] = df['收入'].fillna(income_mean)   # 用均值填充数值字段
    print(f"  填充后缺失值数量: {df['收入'].isnull().sum()}")
    print(f"  填充后收入均值: {df['收入'].mean():.2f}")

    # ---------- 步骤2：剔除收入 > 200000 的异常样本 ----------
    print("\n[步骤2] 剔除收入大于 200000 的异常样本")
    before = len(df)
    df = df[df['收入'] <= 200000]
    after = len(df)
    print(f"  剔除前: {before} 行, 剔除后: {after} 行, 共剔除 {before - after} 行")

    # ---------- 步骤3：按年龄段分组 ----------
    print("\n[步骤3] 按年龄段分组：青年(18-35)、中年(36-59)、老年(≥60)")
    bins = [17, 35, 59, 200]
    labels = ['青年(18-35)', '中年(36-59)', '老年(60+)']
    df['年龄段'] = pd.cut(df['年龄'], bins=bins, labels=labels, right=True)
    print(f"  分组统计:\n{df['年龄段'].value_counts().sort_index().to_string()}")

    # ---------- 步骤4：统计每组平均收入 ----------
    print("\n[步骤4] 统计每组平均收入")
    group_stats = df.groupby('年龄段', observed=True)['收入'].agg(['mean', 'count'])
    group_stats['mean'] = group_stats['mean'].round(2)
    print(group_stats.to_string())

    # ---------- 步骤5：绘制各组平均收入柱状图 ----------
    print("\n[步骤5] 绘制各组平均收入柱状图")
    means = group_stats['mean']

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#5DADE2', '#F5B041', '#58D68D']
    bars = ax.bar(means.index.astype(str), means.values, color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_title('各年龄段平均收入对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('年龄段')
    ax.set_ylabel('平均收入（元）')
    ax.grid(axis='y', alpha=0.3)

    fig_path = os.path.join(OUT_DIR, 'exercise2_avg_income.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  柱状图已保存: {fig_path}")

    # 保存清洗后数据
    csv_path = os.path.join(OUT_DIR, 'exercise2_cleaned.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  清洗后数据已保存: {csv_path}")
    print("\n✅ 习题2完成")


if __name__ == '__main__':
    main()
