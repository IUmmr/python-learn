"""
============================================================
第6周 习题4：医保大数据综合复习题
============================================================
题目要求：
  生成 250 条医保模拟数据：年龄、就诊费用、就诊次数；
  1. 填充费用字段缺失值；
  2. 剔除费用 > 80000 的异常记录；
  3. 筛选 65 岁以上老年患者；
  4. 统计老年患者平均就诊费用与平均就诊次数；
  5. 画出老年患者报销费用直方图。

核心知识点：
  1. pandas 模拟数据生成（含缺失值）
  2. fillna 缺失值填充
  3. 条件筛选异常值剔除
  4. 老年患者筛选与分组统计
  5. matplotlib 直方图
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


def generate_insurance_data(n=250, seed=2026):
    """生成 250 条医保模拟数据：年龄、就诊费用、就诊次数、报销比例"""
    np.random.seed(seed)
    ages = np.random.randint(20, 95, size=n)
    visits = np.random.randint(1, 12, size=n)
    # 就诊费用与年龄正相关；人为抬高少量样本费用制造 >80000 的异常记录
    fees = (np.random.rand(n) * 4000 + ages * 300 + np.random.randn(n) * 5000).clip(100, 60000)
    # 随机挑 20 条样本，费用放大 2.0~2.8 倍制造高费用异常
    outlier_idx = np.random.choice(n, size=20, replace=False)
    fees[outlier_idx] = fees[outlier_idx] * np.random.uniform(2.0, 2.8, size=20)
    fees = np.clip(fees, 100, 150000).round(2)
    # 报销比例 0.5~0.95
    reimbursement_ratio = np.random.uniform(0.5, 0.95, size=n).round(3)
    # 报销费用 = 就诊费用 * 报销比例
    reimbursement = (fees * reimbursement_ratio).round(2)

    df = pd.DataFrame({
        '年龄': ages,
        '就诊费用': fees,
        '就诊次数': visits,
        '报销比例': reimbursement_ratio,
        '报销费用': reimbursement,
    })

    # 人为制造缺失值：随机选 12 个"就诊费用"置为 NaN
    miss_idx = np.random.choice(n, size=12, replace=False)
    df.loc[miss_idx, '就诊费用'] = np.nan
    # 缺失费用对应的报销费用也置为 NaN，保持一致
    df.loc[miss_idx, '报销费用'] = np.nan

    print(f"  已生成 {n} 条医保记录，其中 {len(miss_idx)} 条费用缺失")
    return df


def main():
    print("=" * 60)
    print("第6周 习题4：医保大数据综合复习题")
    print("=" * 60)

    # ---------- 生成数据 ----------
    print("\n[步骤0] 生成 250 条医保模拟数据")
    df = generate_insurance_data()
    print(f"  数据形状: {df.shape}")
    print(df.head(8).to_string(index=False))

    # ---------- 步骤1：填充费用字段缺失值 ----------
    print("\n[步骤1] 填充费用字段缺失值（用均值）")
    print(f"  填充前缺失统计:\n{df.isnull().sum().to_string()}")

    fee_mean = df['就诊费用'].mean()
    df['就诊费用'] = df['就诊费用'].fillna(fee_mean)
    # 同步补齐报销费用
    df['报销费用'] = df['报销比例'] * df['就诊费用']
    df['报销费用'] = df['报销费用'].round(2)

    print(f"  费用均值填充: {fee_mean:.2f}")
    print(f"  填充后缺失值数量: {df.isnull().sum().sum()}")

    # ---------- 步骤2：剔除费用 > 80000 的异常记录 ----------
    print("\n[步骤2] 剔除费用 > 80000 的异常记录")
    before = len(df)
    df = df[df['就诊费用'] <= 80000]
    after = len(df)
    print(f"  剔除前: {before} 条, 剔除后: {after} 条, 共剔除 {before - after} 条")

    # ---------- 步骤3：筛选 65 岁以上老年患者 ----------
    print("\n[步骤3] 筛选 65 岁以上老年患者")
    elderly = df[df['年龄'] >= 65]
    print(f"  老年患者人数: {len(elderly)}")

    # ---------- 步骤4：统计老年患者平均就诊费用与平均就诊次数 ----------
    print("\n[步骤4] 统计老年患者平均就诊费用与平均就诊次数")
    avg_fee = elderly['就诊费用'].mean()
    avg_visits = elderly['就诊次数'].mean()
    avg_reimburse = elderly['报销费用'].mean()
    print(f"  平均就诊费用: {avg_fee:.2f} 元")
    print(f"  平均就诊次数: {avg_visits:.2f} 次")
    print(f"  平均报销费用: {avg_reimburse:.2f} 元")

    # ---------- 步骤5：绘制老年患者报销费用直方图 ----------
    print("\n[步骤5] 绘制老年患者报销费用直方图")
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(elderly['报销费用'], bins=20, color='#58D68D',
                               edgecolor='white', alpha=0.85)

    ax.axvline(avg_reimburse, color='#E74C3C', linestyle='--', linewidth=2,
               label=f'平均报销 {avg_reimburse:.0f} 元')
    ax.set_title(f'65岁以上老年患者报销费用分布（n={len(elderly)}）',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('报销费用（元）')
    ax.set_ylabel('人数')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig_path = os.path.join(OUT_DIR, 'exercise4_elderly_hist.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  直方图已保存: {fig_path}")

    # 保存清洗后数据
    csv_path = os.path.join(OUT_DIR, 'exercise4_cleaned.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  清洗后数据已保存: {csv_path}")
    print("\n✅ 习题4完成")


if __name__ == '__main__':
    main()
