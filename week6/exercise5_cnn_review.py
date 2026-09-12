"""
============================================================
第6周 习题5：深度学习 CNN 巩固任务
============================================================
复习任务（不新增网络结构，重在代码重构优化）：
  1. 将第四周 CIFAR-10 CNN 代码重新抄写一遍，不加复制粘贴；
  2. 在原有代码基础上新增：
     (1) 训练过程每 200 轮打印一次损失；
     (2) 增加验证思路，划分一小部分训练集作为验证；
     (3) 优化代码注释，把每一层网络的作用写上。

与第4周版本的差异（重构优化点）：
  - 新增：训练迭代每 200 步打印一次实时损失
  - 新增：从 50000 张训练集中划分 5000 张作为验证集（random_split）
  - 新增：训练结束给出训练/验证/测试三组指标的完整对比
  - 新增：每一层 CNN 结构的作用注释
  - 复用第4周已下载的 CIFAR-10 数据集缓存，避免重复下载

核心知识点：
  1. torchvision.transforms 数据预处理与增强
  2. torch.utils.data.random_split 验证集划分
  3. nn.Conv2d / BatchNorm2d / ReLU / MaxPool2d / Dropout / Linear
  4. 训练循环中按 step 打印日志
  5. 训练集 / 验证集 / 测试集三路评估
============================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 全局配置
# ============================================================
BATCH_SIZE = 64
EPOCHS = 5                 # CPU 环境控制训练时长
LEARNING_RATE = 0.001
NUM_CLASSES = 10
PRINT_EVERY = 200          # 每 200 步打印一次损失
VAL_SIZE = 5000            # 从训练集中划出 5000 张作为验证集
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 复用已下载的 CIFAR-10 数据缓存（位于项目根目录 data/），避免重复下载
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("第6周 习题5：深度学习 CNN 巩固任务（代码重构优化版）")
print("=" * 60)
print(f"  设备: {DEVICE}")
print(f"  Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, 每 {PRINT_EVERY} 步打印损失")
print(f"  验证集: 训练集中划分 {VAL_SIZE} 张")

# ============================================================
# 1. 数据预处理（与第4周一致：增强 + 归一化）
# ============================================================
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),                       # 随机裁剪：扩充空间多样性，防过拟合
    transforms.RandomHorizontalFlip(),                          # 随机水平翻转：扩充样本方向多样性
    transforms.ColorJitter(brightness=0.2, contrast=0.2),       # 颜色抖动：提升对光照变化的鲁棒性
    transforms.ToTensor(),                                      # PIL -> Tensor，像素缩放到 [0,1]
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),         # 按通道标准化：均值为0、方差为1
                         std=(0.2023, 0.1994, 0.2010)),         # 加速收敛、稳定梯度
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

# ============================================================
# 2. 数据集加载 + 训练集/验证集划分
# ============================================================
print("\n>>> 加载数据集（复用第4周缓存，不重复下载）")
full_train_set = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_transform)
test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=test_transform)

# ---- 新增：验证集划分 ----
# 训练集 50000 张 -> 训练 45000 张 + 验证 5000 张（固定随机种子保证可复现）
train_set, val_set = random_split(
    full_train_set,
    [len(full_train_set) - VAL_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  训练集: {len(train_set)} 张 | 验证集: {len(val_set)} 张 | 测试集: {len(test_set)} 张")

# ============================================================
# 3. CNN 模型（结构同第4周，逐层注释作用）
# ============================================================
class SimpleCNN(nn.Module):
    """
    简易 CNN：3 组「卷积-归一化-激活-池化」特征提取 + 2 层全连接分类。
    输入: (B, 3, 32, 32) 的 CIFAR-10 彩色图像
    输出: (B, 10) 的类别得分（logits）
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # ---- 特征提取部分 ----
        # conv1: 3通道 -> 32通道，3x3卷积 + padding=1 保持尺寸，提取低级特征（边缘/颜色/纹理）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # bn1: 对 32 个特征图做批归一化，归一化激活值，加速收敛、减轻过拟合
        self.bn1 = nn.BatchNorm2d(32)

        # conv2: 32 -> 64 通道，组合低级特征为中级特征（局部形状/角点）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # conv3: 64 -> 128 通道，提取更抽象的高级语义特征（物体部件）
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # pool: 2x2 最大池化，下采样减半尺寸，保留显著特征、扩大感受野、减少计算量
        #       经过 3 次池化: 32 -> 16 -> 8 -> 4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- 分类部分 ----
        # fc1: 将 4x4x128=2048 维展平特征映射到 256 维，做特征组合与非线性变换
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        # dropout: 训练时随机丢弃 30% 神经元，防止全连接层过拟合
        self.dropout = nn.Dropout(0.3)
        # fc2: 256 -> 10 维，输出 10 个类别的得分
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一卷积块: (B,3,32,32) -> conv1 -> (B,32,32,32) -> bn+relu -> pool -> (B,32,16,16)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二卷积块: -> (B,64,16,16) -> pool -> (B,64,8,8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 第三卷积块: -> (B,128,8,8) -> pool -> (B,128,4,4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平: (B,128,4,4) -> (B, 2048)
        x = x.view(x.size(0), -1)

        # 全连接: 2048 -> 256 (relu+dropout) -> 10
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n>>> CNN 模型构建完成，总参数量: {total_params:,}")

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)


# ============================================================
# 4. 训练 / 验证 / 测试函数
# ============================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    """执行一个 epoch。
    - 传入 optimizer 时处于训练模式，返回平均损失与准确率；
    - 不传 optimizer 时处于评估模式（验证/测试）。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    step = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # ---- 新增(1)：每 200 步打印一次训练损失 ----
            step += 1
            if step % PRINT_EVERY == 0:
                print(f"    [步 {step:>4d}] 训练损失: {loss.item():.4f}")
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, 100.0 * correct / total


# ============================================================
# 5. 训练主循环（训练集训练 + 验证集评估）
# ============================================================
print("\n>>> 开始训练（每 200 步打印损失）")
print("-" * 74)
print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<10}")
print("-" * 74)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc, best_state = 0.0, None
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, DEVICE, optimizer)
    # ---- 新增(2)：每个 epoch 结束用验证集评估 ----
    val_loss, val_acc = run_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<12.2f}{val_loss:<12.4f}{val_acc:<10.2f}")

elapsed = time.time() - start_time
print("-" * 74)
print(f"训练完成! 耗时 {elapsed:.1f}s, 最佳验证准确率: {best_val_acc:.2f}%")

# 加载最佳模型并在测试集上做最终评估
model.load_state_dict(best_state)
test_loss, test_acc = run_epoch(model, test_loader, criterion, DEVICE)
print(f"\n>>> 最终测试集评估: 损失 {test_loss:.4f}, 准确率 {test_acc:.2f}%")
print(f"  指标对比: 训练 {history['train_acc'][-1]:.2f}% | "
      f"验证 {history['val_acc'][-1]:.2f}% | 测试 {test_acc:.2f}%")

# 保存模型
torch.save(best_state, os.path.join(OUT_DIR, 'exercise5_cnn.pth'))
print(f"  模型已保存: exercise5_cnn.pth")

# ============================================================
# 6. 训练曲线可视化（新增训练/验证对比）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'习题5：CNN 训练曲线（测试准确率 {test_acc:.2f}%）', fontsize=14, fontweight='bold')

axes[0].plot(history['train_loss'], 'o-', color='steelblue', label='训练损失')
axes[0].plot(history['val_loss'], 's--', color='#FF6B6B', label='验证损失')
axes[0].set_title('损失曲线'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['train_acc'], 'o-', color='#2E8B57', label='训练准确率')
axes[1].plot(history['val_acc'], 's--', color='#FFA500', label='验证准确率')
axes[1].set_title('准确率曲线'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('准确率 (%)')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'exercise5_training_curves.png'), dpi=150, bbox_inches='tight')
print(f"  训练曲线已保存: exercise5_training_curves.png")
print("\n✅ 习题5完成")
