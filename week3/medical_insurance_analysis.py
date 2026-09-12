"""
============================================================
第3周 作业2：医保大数据清洗分析
============================================================
任务说明：
  1. 自动构造生成医保模拟数据
  2. 人为制造缺失值、异常值
  3. 完成缺失值填充、异常金额剔除
  4. 数据统计与结果可视化

核心知识点：
  1. Pandas 模拟数据生成 (随机抽样)
  2. 缺失值处理策略 (均值/中位数/众数填充)
  3. 异常值检测 (IQR / 3σ 原则)
  4. 数据分组统计与可视化

环境要求：
  pip install pandas numpy matplotlib
============================================================
"""

import sys
import io
# Windows GBK 编码环境下强制 UTF-8 输出，避免 emoji 等字符报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 中文显示设置
# ============================================================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第3周 作业2：医保大数据清洗分析")
print("=" * 60)

# ============================================================
# 第一步：自动构造医保模拟数据
# ============================================================
print("\n>>> 第一步：生成医保模拟数据")

np.random.seed(42)

n_records = 500  # 生成500条记录

# 参保人信息
genders = np.random.choice(['男', '女'], size=n_records, p=[0.52, 0.48])
ages = np.random.randint(18, 85, size=n_records)

# 医保类型
insurance_types = np.random.choice(
    ['城镇职工医保', '城乡居民医保', '新农合', '商业补充医保'],
    size=n_records,
    p=[0.40, 0.30, 0.20, 0.10]
)

# 就诊类型
visit_types = np.random.choice(
    ['门诊', '住院', '急诊', '体检'],
    size=n_records,
    p=[0.45, 0.25, 0.15, 0.15]
)

# 总费用 (元) - 不同就诊类型不同分布
total_cost = np.zeros(n_records)
for i, vt in enumerate(visit_types):
    if vt == '门诊':
        total_cost[i] = np.random.lognormal(mean=5.5, sigma=0.8)
    elif vt == '住院':
        total_cost[i] = np.random.lognormal(mean=9.0, sigma=0.6)
    elif vt == '急诊':
        total_cost[i] = np.random.lognormal(mean=6.5, sigma=1.0)
    else:  # 体检
        total_cost[i] = np.random.lognormal(mean=6.0, sigma=0.5)

# 报销比例 (根据医保类型)
reimburse_rate = np.zeros(n_records)
for i, it in enumerate(insurance_types):
    if it == '城镇职工医保':
        reimburse_rate[i] = np.random.uniform(0.70, 0.90)
    elif it == '城乡居民医保':
        reimburse_rate[i] = np.random.uniform(0.50, 0.70)
    elif it == '新农合':
        reimburse_rate[i] = np.random.uniform(0.40, 0.60)
    else:
        reimburse_rate[i] = np.random.uniform(0.60, 0.85)

# 报销金额
reimburse_amount = total_cost * reimburse_rate
# 自付金额
self_pay = total_cost - reimburse_amount

# 日期
dates = pd.date_range(start='2025-01-01', end='2025-12-31', periods=n_records)
dates = np.random.choice(dates, size=n_records, replace=False)
dates = np.sort(dates)

# 科室
departments = np.random.choice(
    ['内科', '外科', '儿科', '妇产科', '骨科', '眼科', '心内科', '神经科'],
    size=n_records
)

# 构建 DataFrame
df = pd.DataFrame({
    '记录ID': range(1, n_records + 1),
    '日期': dates,
    '性别': genders,
    '年龄': ages,
    '医保类型': insurance_types,
    '就诊类型': visit_types,
    '科室': departments,
    '总费用': np.round(total_cost, 2),
    '报销比例': np.round(reimburse_rate, 4),
    '报销金额': np.round(reimburse_amount, 2),
    '自付金额': np.round(self_pay, 2)
})

print(f"  生成记录数: {len(df)}")
print(f"  数据字段: {list(df.columns)}")
print(f"\n  数据预览 (前10行):")
print(df.head(10).to_string(index=False))


# ============================================================
# 第二步：人为制造缺失值和异常值
# ============================================================
print("\n>>> 第二步：人为制造缺失值和异常值")

df_dirty = df.copy()

# --- 制造缺失值 ---
# 随机缺失：总费用 (5%)
missing_cost_idx = np.random.choice(df_dirty.index, size=int(n_records * 0.05), replace=False)
df_dirty.loc[missing_cost_idx, '总费用'] = np.nan

# 随机缺失：年龄 (3%)
missing_age_idx = np.random.choice(df_dirty.index, size=int(n_records * 0.03), replace=False)
df_dirty.loc[missing_age_idx, '年龄'] = np.nan

# 随机缺失：科室 (8%)
missing_dept_idx = np.random.choice(df_dirty.index, size=int(n_records * 0.08), replace=False)
df_dirty.loc[missing_dept_idx, '科室'] = np.nan

# --- 制造异常值 ---
# 极端高费用 (前1%的费用 × 100倍)
outlier_high_idx = np.random.choice(df_dirty.index, size=10, replace=False)
df_dirty.loc[outlier_high_idx, '总费用'] = df_dirty.loc[outlier_high_idx, '总费用'].fillna(0) * 100

# 负费用 (异常)
outlier_neg_idx = np.random.choice(df_dirty.index, size=5, replace=False)
df_dirty.loc[outlier_neg_idx, '总费用'] = df_dirty.loc[outlier_neg_idx, '总费用'].fillna(0) * -1

# 年龄异常 (超过120岁)
outlier_age_idx = np.random.choice(df_dirty.index, size=8, replace=False)
df_dirty.loc[outlier_age_idx, '年龄'] = np.random.randint(130, 200, size=8)

# 报销比例异常 (>1.0)
outlier_rate_idx = np.random.choice(df_dirty.index, size=6, replace=False)
df_dirty.loc[outlier_rate_idx, '报销比例'] = np.random.uniform(1.5, 3.0, size=6)
df_dirty.loc[outlier_rate_idx, '报销金额'] = np.nan  # 相应报销金额也改为缺失

print(f"\n  缺失值统计:")
print(f"    总费用缺失: {df_dirty['总费用'].isnull().sum()} 条")
print(f"    年龄缺失:   {df_dirty['年龄'].isnull().sum()} 条")
print(f"    科室缺失:   {df_dirty['科室'].isnull().sum()} 条")
print(f"    报销金额缺失: {df_dirty['报销金额'].isnull().sum()} 条")
print(f"\n  异常值统计:")
print(f"    负费用: {(df_dirty['总费用'] < 0).sum()} 条")
print(f"    异常高费用(>5万): {(df_dirty['总费用'] > 50000).sum()} 条")
print(f"    年龄异常(>120): {(df_dirty['年龄'] > 120).sum()} 条")
print(f"    报销比例异常(>1): {(df_dirty['报销比例'] > 1).sum()} 条")

# 保存脏数据
dirty_path = os.path.join(os.path.dirname(__file__) or '.', 'medical_insurance_dirty.csv')
df_dirty.to_csv(dirty_path, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 脏数据已保存: medical_insurance_dirty.csv")


# ============================================================
# 第三步：数据清洗 - 缺失值填充
# ============================================================
print("\n>>> 第三步：数据清洗")

df_clean = df_dirty.copy()
cleaning_log = []

# --- 3.1 异常值剔除 / 修正 ---
# 负费用 → 标记为NaN，后续填充
neg_count = (df_clean['总费用'] < 0).sum()
df_clean.loc[df_clean['总费用'] < 0, '总费用'] = np.nan
cleaning_log.append(f"剔除负费用: {neg_count} 条")

# 年龄异常(>120) → 标记为NaN
age_outlier_count = (df_clean['年龄'] > 120).sum()
df_clean.loc[df_clean['年龄'] > 120, '年龄'] = np.nan
cleaning_log.append(f"剔除异常年龄(>120): {age_outlier_count} 条")

# 极高费用 → 使用 IQR 方法检测并标记
cost_valid = df_clean['总费用'].dropna()
Q1_cost = cost_valid.quantile(0.25)
Q3_cost = cost_valid.quantile(0.75)
IQR_cost = Q3_cost - Q1_cost
cost_lower = Q1_cost - 1.5 * IQR_cost
cost_upper = Q3_cost + 1.5 * IQR_cost

high_cost_outlier = (df_clean['总费用'] > cost_upper)
high_cost_count = high_cost_outlier.sum()
df_clean.loc[high_cost_outlier, '总费用'] = np.nan
cleaning_log.append(f"剔除IQR异常高费用(>{cost_upper:.0f}元): {high_cost_count} 条")

# 报销比例异常(>1)
rate_outlier_count = (df_clean['报销比例'] > 1).sum()
df_clean.loc[df_clean['报销比例'] > 1, '报销比例'] = np.nan
cleaning_log.append(f"剔除异常报销比例(>1): {rate_outlier_count} 条")

# --- 3.2 缺失值填充 ---
# 总费用 → 按就诊类型分组，用组内中位数填充
df_clean['总费用'] = df_clean.groupby('就诊类型')['总费用'].transform(
    lambda x: x.fillna(x.median())
)
fill_cost_count = df_dirty['总费用'].isnull().sum() + neg_count + high_cost_count
cleaning_log.append(f"填充总费用缺失: {fill_cost_count} 条 (按就诊类型中位数)")

# 年龄 → 用全局中位数填充
age_median = df_clean['年龄'].median()
age_missing = df_clean['年龄'].isnull().sum()
df_clean['年龄'] = df_clean['年龄'].fillna(age_median)
cleaning_log.append(f"填充年龄缺失: {age_missing} 条 (中位数={age_median:.0f})")

# 科室 → 用众数填充
dept_mode = df_clean['科室'].mode()[0]
dept_missing = df_clean['科室'].isnull().sum()
df_clean['科室'] = df_clean['科室'].fillna(dept_mode)
cleaning_log.append(f"填充科室缺失: {dept_missing} 条 (众数={dept_mode})")

# 报销比例 → 按医保类型分组中位数填充
df_clean['报销比例'] = df_clean.groupby('医保类型')['报销比例'].transform(
    lambda x: x.fillna(x.median())
)
cleaning_log.append(f"填充报销比例缺失: {rate_outlier_count} 条 (按医保类型中位数)")

# 重新计算报销金额和自付金额
df_clean['报销金额'] = np.round(df_clean['总费用'] * df_clean['报销比例'], 2)
df_clean['自付金额'] = np.round(df_clean['总费用'] - df_clean['报销金额'], 2)

print("\n  清洗日志:")
for log in cleaning_log:
    print(f"    · {log}")

print(f"\n  清洗后缺失值检查:")
print(f"    总费用缺失: {df_clean['总费用'].isnull().sum()}")
print(f"    年龄缺失:   {df_clean['年龄'].isnull().sum()}")
print(f"    科室缺失:   {df_clean['科室'].isnull().sum()}")
print(f"    报销金额缺失: {df_clean['报销金额'].isnull().sum()}")

# 保存清洗后数据
clean_path = os.path.join(os.path.dirname(__file__) or '.', 'medical_insurance_clean.csv')
df_clean.to_csv(clean_path, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 清洗后数据已保存: medical_insurance_clean.csv")


# ============================================================
# 第四步：数据统计分析
# ============================================================
print("\n>>> 第四步：数据统计分析")

print("\n  --- 4.1 描述性统计 (数值字段) ---")
desc_stats = df_clean[['年龄', '总费用', '报销比例', '报销金额', '自付金额']].describe()
print(desc_stats.to_string())

print("\n  --- 4.2 按就诊类型统计 ---")
visit_stats = df_clean.groupby('就诊类型').agg(
    记录数=('记录ID', 'count'),
    平均费用=('总费用', 'mean'),
    中位费用=('总费用', 'median'),
    总报销金额=('报销金额', 'sum'),
    平均自付比例=('报销比例', lambda x: 1 - x.mean())
).round(2)
print(visit_stats.to_string())

print("\n  --- 4.3 按医保类型统计 ---")
insurance_stats = df_clean.groupby('医保类型').agg(
    记录数=('记录ID', 'count'),
    平均年龄=('年龄', 'mean'),
    平均费用=('总费用', 'mean'),
    平均报销比例=('报销比例', 'mean'),
    总报销金额=('报销金额', 'sum')
).round(2)
print(insurance_stats.to_string())

print("\n  --- 4.4 按科室统计 (Top 5 费用最高) ---")
dept_cost = df_clean.groupby('科室')['总费用'].mean().sort_values(ascending=False)
print(dept_cost.head(5).to_string())


# ============================================================
# 第五步：结果可视化
# ============================================================
print("\n>>> 第五步：结果可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('医保大数据清洗分析 - 可视化报告', fontsize=16, fontweight='bold')

# ---------- 子图1: 就诊类型分布 (饼图) ----------
visit_counts = df_clean['就诊类型'].value_counts()
colors1 = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']
wedges1, texts1, autotexts1 = axes[0, 0].pie(
    visit_counts.values, labels=visit_counts.index,
    autopct='%1.1f%%', colors=colors1, startangle=90
)
axes[0, 0].set_title('就诊类型分布', fontsize=12, fontweight='bold')

# ---------- 子图2: 各就诊类型平均费用对比 (柱状图) ----------
visit_cost_mean = df_clean.groupby('就诊类型')['总费用'].mean().sort_values()
bars2 = axes[0, 1].bar(visit_cost_mean.index, visit_cost_mean.values,
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 1].set_title('各就诊类型平均费用', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('平均费用 (元)')
axes[0, 1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars2, visit_cost_mean.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{val:.0f}', ha='center', fontsize=9)

# ---------- 子图3: 年龄分布直方图 ----------
axes[0, 2].hist(df_clean['年龄'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 2].axvline(df_clean['年龄'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"均值={df_clean['年龄'].mean():.1f}")
axes[0, 2].axvline(df_clean['年龄'].median(), color='orange', linestyle='--',
                   linewidth=2, label=f"中位数={df_clean['年龄'].median():.0f}")
axes[0, 2].set_title('年龄分布', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('年龄')
axes[0, 2].set_ylabel('频数')
axes[0, 2].legend(fontsize=8)

# ---------- 子图4: 医保类型报销比例对比 (箱线图) ----------
insurance_order = df_clean.groupby('医保类型')['报销比例'].median().sort_values().index
box_data = [df_clean[df_clean['医保类型'] == it]['报销比例'].dropna().values
            for it in insurance_order]
bp = axes[1, 0].boxplot(box_data, labels=insurance_order, patch_artist=True)
colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_title('各医保类型报销比例分布', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('报销比例')
axes[1, 0].tick_params(axis='x', rotation=15)

# ---------- 子图5: 费用 vs 年龄 散点图 ----------
scatter = axes[1, 1].scatter(df_clean['年龄'], df_clean['总费用'],
                              c=df_clean['报销比例'], cmap='RdYlGn',
                              alpha=0.6, edgecolors='grey', linewidth=0.3)
axes[1, 1].set_title('总费用 vs 年龄 (颜色=报销比例)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('年龄')
axes[1, 1].set_ylabel('总费用 (元)')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('报销比例', fontsize=9)

# 费用与年龄的相关性
cost_age_corr = df_clean['总费用'].corr(df_clean['年龄'])
axes[1, 1].text(0.05, 0.95, f'相关系数 r={cost_age_corr:.3f}',
                transform=axes[1, 1].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---------- 子图6: 科室平均费用 (横向柱状图) ----------
dept_cost_sorted = df_clean.groupby('科室')['总费用'].mean().sort_values()
colors6 = get_cmap('viridis')(np.linspace(0.2, 0.9, len(dept_cost_sorted)))
bars6 = axes[1, 2].barh(dept_cost_sorted.index, dept_cost_sorted.values, color=colors6)
axes[1, 2].set_title('各科室平均费用', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('平均费用 (元)')
for bar, val in zip(bars6, dept_cost_sorted.values):
    axes[1, 2].text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8)

plt.tight_layout()

# 保存图表
fig.savefig(os.path.join(os.path.dirname(__file__) or '.', 'medical_insurance_report.png'),
            dpi=150, bbox_inches='tight')
print("  ✅ 可视化报告已保存: medical_insurance_report.png")

# ---------- 额外的月度趋势图 ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('医保费用月度趋势', fontsize=14, fontweight='bold')

df_clean['月份'] = pd.to_datetime(df_clean['日期']).dt.month
monthly_cost = df_clean.groupby('月份').agg(
    总费用=('总费用', 'sum'),
    报销金额=('报销金额', 'sum'),
    记录数=('记录ID', 'count')
)

axes2[0].fill_between(monthly_cost.index, monthly_cost['总费用'],
                       alpha=0.3, color='steelblue')
axes2[0].plot(monthly_cost.index, monthly_cost['总费用'], 'o-',
              color='steelblue', linewidth=2, markersize=6, label='总费用')
axes2[0].plot(monthly_cost.index, monthly_cost['报销金额'], 's--',
              color='#FF6B6B', linewidth=2, markersize=6, label='报销金额')
axes2[0].set_xlabel('月份')
axes2[0].set_ylabel('金额 (元)')
axes2[0].set_title('月度费用与报销趋势')
axes2[0].legend()
axes2[0].set_xticks(range(1, 13))

axes2[1].bar(monthly_cost.index, monthly_cost['记录数'],
             color='#96CEB4', edgecolor='white')
axes2[1].set_xlabel('月份')
axes2[1].set_ylabel('记录数')
axes2[1].set_title('月度就诊记录数')
axes2[1].set_xticks(range(1, 13))

plt.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__) or '.', 'medical_insurance_monthly.png'),
             dpi=150, bbox_inches='tight')
print("  ✅ 月度趋势图已保存: medical_insurance_monthly.png")

plt.show()

# ============================================================
# 第六步：清洗前后对比汇总
# ============================================================
print("\n" + "=" * 60)
print("清洗前后数据对比汇总")
print("=" * 60)

print(f"\n{'指标':<25}{'清洗前':>12}{'清洗后':>12}")
print("-" * 50)
print(f"{'总记录数':<25}{len(df_dirty):>12}{len(df_clean):>12}")
print(f"{'总费用缺失数':<25}{df_dirty['总费用'].isnull().sum():>12}{df_clean['总费用'].isnull().sum():>12}")
print(f"{'负费用数':<25}{(df_dirty['总费用'] < 0).sum():>12}{(df_clean['总费用'] < 0).sum():>12}")
print(f"{'异常高费用数(>5万)':<25}{(df_dirty['总费用'] > 50000).sum():>12}{(df_clean['总费用'] > 50000).sum():>12}")
print(f"{'年龄异常数(>120)':<25}{(df_dirty['年龄'] > 120).sum():>12}{(df_clean['年龄'] > 120).sum():>12}")
print(f"{'报销比例异常(>1)':<25}{(df_dirty['报销比例'] > 1).sum():>12}{(df_clean['报销比例'] > 1).sum():>12}")
print(f"{'总费用均值':<25}{df_dirty['总费用'].mean():>12.2f}{df_clean['总费用'].mean():>12.2f}")
print(f"{'年龄均值':<25}{df_dirty['年龄'].mean():>12.2f}{df_clean['年龄'].mean():>12.2f}")

print("\n✅ 医保大数据清洗分析作业完成！")
