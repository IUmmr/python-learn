"""
============================================================
第4周 作业：基于PyTorch搭建简易CNN网络 + CIFAR10图像分类
============================================================
任务说明：
  1. 数据预处理（归一化 + 数据增强）
  2. 加载数据集（torchvision CIFAR10 + DataLoader）
  3. 搭建简易 CNN 模型（Conv + Pool + FC）
  4. 完整训练与评估（损失曲线 + 准确率曲线 + 预测可视化）

核心知识点：
  1. torchvision.transforms 图像预处理管道
  2. torch.utils.data.DataLoader 数据加载
  3. nn.Conv2d / nn.BatchNorm2d / nn.ReLU / nn.MaxPool2d 搭建 CNN
  4. 训练循环、优化器、损失函数、学习率调度

环境要求：
  pip install torch torchvision matplotlib numpy
============================================================
"""

import sys
import io
# Windows GBK 编码环境下强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 全局配置
# ============================================================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("第4周 作业：PyTorch 简易 CNN + CIFAR10 图像分类")
print("=" * 60)
print(f"  设备: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, 学习率: {LEARNING_RATE}")


# ============================================================
# 第一步：数据预处理
# ============================================================
print("\n>>> 第一步：数据预处理 (Data Preprocessing)")

# CIFAR-10 类别标签
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# ---------- 1.1 训练集预处理：数据增强 + 归一化 ----------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     # 随机裁剪（数据增强）
    transforms.RandomHorizontalFlip(),         # 随机水平翻转（数据增强）
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动（数据增强）
    transforms.ToTensor(),                     # PIL -> Tensor [0, 1]
    transforms.Normalize(                      # 标准化 (x - mean) / std
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# ---------- 1.2 测试集预处理：仅归一化（不做增强） ----------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

print(f"  训练集变换: {train_transform}")
print(f"  测试集变换: {test_transform}")
print("  ✅ 预处理管道构建完成")


# ============================================================
# 第二步：加载数据集
# ============================================================
print("\n>>> 第二步：加载数据集 (Load Dataset)")

# 首次运行会自动下载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)

print(f"  训练集样本数: {len(train_dataset)}")
print(f"  测试集样本数: {len(test_dataset)}")
print(f"  训练批次数量: {len(train_loader)}")
print(f"  单批样本数:   {BATCH_SIZE}")
print(f"  图像尺寸:     {train_dataset[0][0].shape}")
print(f"  类别数量:     {len(train_dataset.classes)}")
print("  ✅ 数据集加载完成")


# ============================================================
# 第三步：搭建简易 CNN 模型
# ============================================================
print("\n>>> 第三步：搭建简易 CNN 模型")


class SimpleCNN(nn.Module):
    """简易卷积神经网络
    结构: Conv1 -> BN -> ReLU -> Pool | Conv2 -> BN -> ReLU -> Pool
          | Conv3 -> BN -> ReLU -> Pool | Flatten -> FC1 -> Dropout -> FC2
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # ---------- 卷积特征提取部分 ----------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 每次减半: 32->16->8->4

        # ---------- 全连接分类部分 ----------
        # 三次池化后特征图尺寸: 32/2/2/2 = 4, 即 4x4x128
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一卷积块: 32x32x3 -> 32x32x32 -> 16x16x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二卷积块: 16x16x32 -> 16x16x64 -> 8x8x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 第三卷积块: 8x8x64 -> 8x8x128 -> 4x4x128
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平: 4x4x128 -> 2048
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

print(f"\n  模型结构:")
print(model)

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  总参数量: {total_params:,}")
print(f"  可训练参数量: {trainable_params:,}")

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# 学习率调度：每 3 个 epoch 衰减为原来的 0.7
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

print("  ✅ 模型搭建完成")


# ============================================================
# 第四步：训练与评估函数
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均损失和准确率"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """评估模型，返回平均损失和准确率"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ============================================================
# 第五步：训练模型
# ============================================================
print("\n>>> 第四步：开始训练")
print(f"  总训练轮数: {EPOCHS}")
print("-" * 78)
print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<12}{'Test Loss':<13}{'Test Acc':<12}")
print("-" * 78)

history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

start_time = time.time()
best_acc = 0.0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_state = copy.deepcopy(model.state_dict())

    print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<12.2f}"
          f"{test_loss:<13.4f}{test_acc:<12.2f}")

elapsed = time.time() - start_time
print("-" * 78)
print(f"  训练完成! 耗时: {elapsed:.1f}s, 最佳测试准确率: {best_acc:.2f}%")

# 加载最佳模型
model.load_state_dict(best_model_state)
torch.save(best_model_state, os.path.join(os.path.dirname(__file__) or '.', 'cifar10_cnn.pth'))
print(f"  ✅ 最佳模型已保存: cifar10_cnn.pth")


# ============================================================
# 第六步：结果可视化
# ============================================================
print("\n>>> 第五步：结果可视化")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'CIFAR-10 CNN 训练曲线 (最佳测试准确率 {best_acc:.2f}%)',
             fontsize=14, fontweight='bold')

# 损失曲线
axes[0].plot(range(1, EPOCHS + 1), history['train_loss'], 'o-',
             color='steelblue', linewidth=2, label='训练损失')
axes[0].plot(range(1, EPOCHS + 1), history['test_loss'], 's--',
             color='#FF6B6B', linewidth=2, label='测试损失')
axes[0].set_title('损失曲线', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 准确率曲线
axes[1].plot(range(1, EPOCHS + 1), history['train_acc'], 'o-',
             color='#2E8B57', linewidth=2, label='训练准确率')
axes[1].plot(range(1, EPOCHS + 1), history['test_acc'], 's--',
             color='#FFA500', linewidth=2, label='测试准确率')
axes[1].set_title('准确率曲线', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('准确率 (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__) or '.', 'cifar10_training_curves.png'),
            dpi=150, bbox_inches='tight')
print("  ✅ 训练曲线已保存: cifar10_training_curves.png")


# ---------- 6.2 预测结果可视化 ----------
print("\n>>> 第六步：预测结果可视化")

# 取一批测试数据做预测
model.eval()
test_iter = iter(test_loader)
sample_images, sample_labels = next(test_iter)
sample_images, sample_labels = sample_images[:16].to(DEVICE), sample_labels[:16]

with torch.no_grad():
    outputs = model(sample_images)
    _, preds = outputs.max(1)

# 反归一化用于显示
def denormalize(tensor):
    """将归一化后的张量还原为可显示的 [0,1] 图像"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

fig2, axes2 = plt.subplots(4, 4, figsize=(12, 12))
fig2.suptitle('CIFAR-10 预测结果（绿=正确, 红=错误）', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes2.flat):
    img = denormalize(sample_images[idx]).cpu()
    # CHW -> HWC
    img = img.permute(1, 2, 0).numpy()
    ax.imshow(img)
    true_label = CLASS_NAMES[sample_labels[idx]]
    pred_label = CLASS_NAMES[preds[idx]]
    is_correct = (preds[idx] == sample_labels[idx]).item()
    color = '#2E8B57' if is_correct else '#DC143C'
    ax.set_title(f'真实: {true_label}\n预测: {pred_label}', fontsize=9, color=color)
    ax.axis('off')

plt.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__) or '.', 'cifar10_predictions.png'),
             dpi=150, bbox_inches='tight')
print("  ✅ 预测结果图已保存: cifar10_predictions.png")


# ---------- 6.3 每类准确率分析 ----------
print("\n>>> 第七步：各类别准确率分析")

class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct_mask = predicted.eq(labels)
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if correct_mask[i].item():
                class_correct[label] += 1

print(f"\n  {'类别':<10}{'准确率':<12}{'预测正确/总数':<15}")
print("  " + "-" * 40)
for i, name in enumerate(CLASS_NAMES):
    acc = 100.0 * class_correct[i] / class_total[i]
    print(f"  {name:<10}{acc:<12.2f}{class_correct[i]}/{class_total[i]}")

overall_acc = 100.0 * sum(class_correct) / sum(class_total)
print(f"\n  总体测试准确率: {overall_acc:.2f}%")

plt.show()

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 60)
print("作业完成情况汇总")
print("=" * 60)
print(f"\n  ① 数据预处理:  归一化 + 数据增强 (随机裁剪/翻转/颜色抖动)")
print(f"  ② 数据集加载:  CIFAR-10 训练集 {len(train_dataset)} 张 + 测试集 {len(test_dataset)} 张")
print(f"  ③ CNN 模型:    3 层卷积 + BatchNorm + 3 层池化 + 2 层全连接")
print(f"  ④ 训练结果:    最佳测试准确率 {best_acc:.2f}%")
print(f"  ⑤ 产出文件:")
print(f"     - cifar10_cnn.pth               (最佳模型权重)")
print(f"     - cifar10_training_curves.png   (损失/准确率曲线)")
print(f"     - cifar10_predictions.png       (预测结果可视化)")
print("\n✅ 第4周作业完成！")
