# Python数据分析第四周复盘报告

## 一、本周学习内容

### 1. PyTorch 深度学习框架入门
- [1] 张量（Tensor）基础：创建、运算、设备迁移（`.to(device)`）
- [2] `nn.Module` 模型定义：`__init__` 搭建层结构、`forward` 定义前向传播
- [3] 自动求导机制：`loss.backward()` 反向传播、`optimizer.step()` 参数更新
- [4] 模型训练三要素：损失函数（CrossEntropyLoss）、优化器（Adam）、学习率调度（StepLR）

### 2. 卷积神经网络（CNN）核心概念
- [1] 卷积层 `nn.Conv2d`：通过卷积核提取图像局部特征，参数为 in_channels / out_channels / kernel_size / padding
- [2] 批归一化 `nn.BatchNorm2d`：加速收敛、缓解梯度消失/爆炸
- [3] 激活函数 `ReLU`：引入非线性，缓解梯度消失
- [4] 池化层 `nn.MaxPool2d`：下采样降维，保留主要特征，增强平移不变性
- [5] 全连接层 `nn.Linear` + Dropout：特征展平后分类，Dropout 防止过拟合

### 3. 数据预处理与数据加载
- [1] `torchvision.transforms` 预处理管道：
  - `ToTensor()`：PIL 图像 → 张量，自动归一化到 [0,1]
  - `Normalize()`：标准化 (x - mean) / std，按 CIFAR-10 统计量设置
- [2] 数据增强：`RandomCrop`（随机裁剪）、`RandomHorizontalFlip`（随机翻转）、`ColorJitter`（颜色抖动）
- [3] `DataLoader` 批量加载：batch_size、shuffle、num_workers、pin_memory
- [4] 训练集与测试集差异：训练集做增强提升泛化，测试集仅归一化保证评估公平

### 4. 模型训练与评估流程
- [1] 训练/评估模式切换：`model.train()` vs `model.eval()`，`torch.no_grad()` 关闭梯度
- [2] 模型保存与加载：`torch.save(state_dict)` / `load_state_dict`
- [3] 训练曲线分析：损失下降趋势、准确率收敛情况
- [4] 混淆矩阵思想：各类别准确率分析发现模型薄弱类别

---

## 二、本周作业完成情况

| 作业 | 任务 | 完成状态 | 核心技术栈 |
|------|------|----------|-----------|
| 作业 1 | 搭建简易 CNN + CIFAR-10 图像分类 | ✅ | PyTorch、torchvision、Matplotlib |

### 作业任务拆解

| 任务 | 实现要点 |
|------|----------|
| 数据预处理 | ToTensor + Normalize + 数据增强（随机裁剪/翻转/颜色抖动） |
| 加载数据集 | `datasets.CIFAR10` 自动下载 + `DataLoader`（batch_size=64） |
| 搭建 CNN 模型 | 3 层 Conv2d + BatchNorm + ReLU + MaxPool + 2 层全连接（62 万参数） |
| 训练评估 | Adam 优化器 + CrossEntropyLoss + StepLR 调度，10 个 epoch |

### 训练结果

| Epoch | Train Loss | Train Acc | Test Acc |
|-------|-----------|-----------|----------|
| 1 | 1.5149 | 44.54% | 58.68% |
| 5 | 0.8866 | 68.87% | 75.17% |
| 10 | 0.7043 | 75.49% | **77.98%** |

- 最佳测试准确率：**77.98%**（CPU 训练约 12 分钟）
- 模型总参数量：620,810

### 各类别准确率分析

| 类别 | 准确率 | 类别 | 准确率 |
|------|--------|------|--------|
| 卡车 | 93.1% | 鸟 | 64.6% |
| 汽车 | 88.9% | 鹿 | 67.7% |
| 青蛙 | 88.4% | 狗 | 73.2% |
| 马 | 85.5% | 猫 | 58.5%（最弱） |
| 船 | 83.5% | 飞机 | 76.4% |

> 分析：猫、鸟、鹿这类类内差异大、纹理相似的类别最难区分；卡车、汽车等轮廓鲜明的类别准确率高。这与 CNN 依赖局部纹理特征的性质一致。

### 产出文件（program/week4/）

| 文件 | 说明 |
|------|------|
| `cnn_cifar10.py` | 完整训练脚本 |
| `cifar10_cnn.pth` | 最佳模型权重 |
| `cifar10_training_curves.png` | 损失曲线 + 准确率曲线 |
| `cifar10_predictions.png` | 预测结果可视化（绿=正确，红=错误） |
| `data/` | CIFAR-10 数据集缓存（下次运行无需重新下载） |

---

## 三、遇到的报错及解决方案

### 报错 1：`ModuleNotFoundError: No module named 'torch'`

- **场景**：第一次运行脚本时找不到 torch
- **解决**：使用 CPU 版安装（无需 CUDA，体积小、兼容性好）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> 注意：默认 `pip install torch` 会安装 CUDA 版（约 2.5GB），CPU 版只有约 120MB。

### 报错 2：CIFAR-10 数据集下载慢 / 连接中断

- **场景**：首次运行需从国外服务器下载 170MB 数据集，多次连接中断
- **现象**：pip 下载中断后自动断点续传，最终耗时约 3 分 20 秒
- **解决**：耐心等待自动重试；如多次失败，可手动下载后放到 `./data` 目录

### 报错 3：`pin_memory` 警告

- **场景**：`UserWarning: 'pin_memory' argument is set as true but no accelerator is found`
- **原因**：CPU 环境下 `pin_memory=True` 无效，该参数仅对 GPU 加速有意义
- **解决**：CPU 环境下设置为 `pin_memory=False`；代码保留该参数以兼容 GPU 环境

### 报错 4：CPU 训练耗时较长

- **场景**：10 个 epoch 共耗时约 709 秒（约 12 分钟）
- **分析**：62 万参数的模型在 CPU 上单 batch 前向+反向约 0.6 秒
- **优化思路**：
  - 减少 epoch 数（如 5 个即可到 75%）
  - 使用 `num_workers` 加速数据加载
  - 有 GPU 时用 `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 自动切换

---

## 四、本周收获总结

1. 打通了 **"数据预处理 → 数据加载 → 模型搭建 → 训练 → 评估 → 可视化"** 的深度学习完整流程，与前三周的数据分析流程一脉相承
2. 理解了 **CNN 各层的作用**：卷积提特征、BN 加速收敛、池化降维、Dropout 防过拟合、全连接分类
3. 掌握了 **数据增强的实践价值**：通过随机裁剪/翻转/颜色抖动扩充数据多样性，有效缓解过拟合、提升泛化能力
4. 学会了 **模型评估的维度**：不仅看总体准确率，还通过各类别准确率定位模型薄弱环节（猫类最难）
5. 深化了对 **归一化/标准化的理解**：第三周在 SimpleITK 上做的影像归一化，与本周 CIFAR-10 的 `Normalize()` 本质相同——统一数据分布，加速收敛、提升稳定性
6. 建立了 **超参数调优意识**：学习率、batch_size、dropout、epoch 数对训练效果的影响

---

## 五、下周学习目标

- 深入学习 PyTorch 进阶：自定义 Dataset、迁移学习（ResNet 预训练模型）
- 尝试提高分类准确率：更深的网络结构（VGG/ResNet）、学习率预热、更多数据增强
- 学习模型评估进阶指标：精确率、召回率、F1-score、混淆矩阵可视化
- 结合第三周医学影像主题，尝试用 CNN 做医学影像分类（如肺结节、皮肤病变数据集）
- 继续文献阅读，跟进深度学习在医学影像领域的最新进展

---

**提交日期**：2026年8月16日
**提交人**：[汪超]
