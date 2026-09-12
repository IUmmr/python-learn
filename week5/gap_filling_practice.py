"""
第五周 任务3：补齐个人知识短板
针对前4周暴露的薄弱环节，编写针对性练习脚本。
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import json
from typing import Optional, Union, List, Dict, Tuple

print("=" * 60)
print("第五周 任务3：知识短板补强练习")
print("=" * 60)

# ============================================================
# 练习1：列表/字典推导式 + 类型注解
# ============================================================
print("\n[练习1] 推导式与类型注解")

numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 列表推导式：偶数平方
squares_even: List[int] = [x ** 2 for x in numbers if x % 2 == 0]
print(f"  偶数平方列表: {squares_even}")

# 字典推导式：数字->奇偶标签
label_map: Dict[int, str] = {x: '奇数' if x % 2 else '偶数' for x in numbers}
print(f"  数字标签字典: {label_map}")

# 函数类型注解
def calc_bmi(weight: float, height: float) -> Optional[float]:
    """计算 BMI，身高输入非法返回 None"""
    if height <= 0:
        return None
    return round(weight / (height ** 2), 2)

print(f"  BMI(70kg, 1.75m): {calc_bmi(70, 1.75)}")
print("  ✅ 练习1完成")


# ============================================================
# 练习2：编码安全处理
# ============================================================
print("\n[练习2] 文件读写与编码处理")

sample_text: str = "Python数据分析 - 包含中文与 emoji ✅"
test_file: str = "test_utf8.txt"

# 安全写入（指定编码）
with open(test_file, 'w', encoding='utf-8') as f:
    f.write(sample_text)

# 安全读取
try:
    with open(test_file, 'r', encoding='utf-8') as f:
        content: str = f.read()
    print(f"  读取内容: {content}")
except UnicodeDecodeError as e:
    print(f"  编码错误: {e}")

# JSON 读写
json_data: Dict[str, Union[str, int, List[float]]] = {
    "name": "汪超",
    "week": 5,
    "scores": [85.5, 90.0, 78.0]
}
json_file: str = "test_data.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

with open(json_file, 'r', encoding='utf-8') as f:
    loaded: Dict = json.load(f)
print(f"  JSON 读取: {loaded}")
print("  ✅ 练习2完成")


# ============================================================
# 练习3：Pandas 高级数据清洗
# ============================================================
print("\n[练习3] Pandas 高级数据清洗")

np.random.seed(42)
# 构造带时间索引的模拟销售数据
dates = pd.date_range('2025-01-01', periods=20, freq='D')
sales = pd.DataFrame({
    'date': dates,
    'amount': np.random.randint(100, 500, 20).astype(float)
})
# 人为制造缺失和异常
sales.loc[[2, 5, 12], 'amount'] = np.nan
sales.loc[15, 'amount'] = 9999  # 异常高值
sales.loc[18, 'amount'] = -50   # 异常低值

print(f"  原始数据:\n{sales.head(10).to_string(index=False)}")

# 3.1 异常值标记为缺失（3-sigma 原则）
mean_val = sales['amount'].mean()
std_val = sales['amount'].std()
lower, upper = mean_val - 3 * std_val, mean_val + 3 * std_val
sales['amount'] = sales['amount'].where(sales['amount'].between(lower, upper))

# 3.2 时间序列线性插值
sales['amount_filled'] = sales['amount'].interpolate(method='linear')
# 首尾缺失用前后填充
sales['amount_filled'] = sales['amount_filled'].ffill().bfill()

# 3.3 重复值处理
df_dup = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3], 'B': ['x', 'y', 'y', 'z', 'z', 'z']})
df_unique = df_dup.drop_duplicates()

print(f"\n  清洗后数据:\n{sales[['date', 'amount', 'amount_filled']].to_string(index=False)}")
print(f"\n  重复值处理前: {len(df_dup)} 行, 处理后: {len(df_unique)} 行")
print("  ✅ 练习3完成")


# ============================================================
# 练习4：混淆矩阵与分类评估指标
# ============================================================
print("\n[练习4] 模型评估指标")

# 模拟真实标签与预测标签（二分类：正例=1，负例=0）
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  手动计算: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print(f"  sklearn  report:\n{classification_report(y_true, y_pred, target_names=['负例', '正例'])}")
print("  ✅ 练习4完成")


# ============================================================
# 练习5：自定义 PyTorch Dataset 模板
# ============================================================
print("\n[练习5] 自定义 PyTorch Dataset")

import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    """表格数据自定义 Dataset 模板
    输入: features (n_samples, n_features), labels (n_samples,)
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# 构造模拟样本：2 特征，3 类别
X = np.random.randn(100, 2)
y = np.random.randint(0, 3, size=100)

dataset = TabularDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_x, batch_y in loader:
    print(f"  一个 batch: features shape={batch_x.shape}, labels shape={batch_y.shape}")
    break

print("  ✅ 练习5完成")


# ============================================================
# 练习6：Git 提交规范示例
# ============================================================
print("\n[练习6] Git 提交规范示例")

COMMIT_CONVENTION = """
<type>(<scope>): <subject>

<body>

常见 type:
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 代码格式（不影响功能）
- refactor: 重构
- test: 测试相关
- chore: 构建/工具相关

示例:
feat(week4): 添加 CIFAR-10 CNN 训练脚本
fix(week3): 修复 SimpleITK GaussianSource 参数格式
"""
print(COMMIT_CONVENTION)
print("  ✅ 练习6完成")


print("\n" + "=" * 60)
print("所有短板补强练习完成！")
print("=" * 60)
