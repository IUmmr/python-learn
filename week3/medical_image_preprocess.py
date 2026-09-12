"""
============================================================
第3周 作业1：医学影像预处理
============================================================
任务说明：
  使用 SimpleITK 生成测试影像（高斯球体 + 噪声），
  完成影像归一化预处理，输出影像参数和结果。

核心知识点：
  1. SimpleITK 生成合成医学影像
  2. 影像强度归一化（Min-Max / Z-Score）
  3. 影像统计参数提取（均值、标准差、最大/最小值等）
  4. 结果可视化对比

环境要求：
  pip install SimpleITK matplotlib numpy
============================================================
"""

import sys
import io
# Windows GBK 编码环境下强制 UTF-8 输出，避免 emoji 等字符报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ============================================================
# 0. 中文显示设置
# ============================================================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("第3周 作业1：医学影像预处理")
print("=" * 60)

# ============================================================
# 第一步：使用 SimpleITK 生成测试影像
# ============================================================
print("\n>>> 第一步：生成合成测试影像")

# 参数设置
image_size = [128, 128, 64]      # 3D 影像尺寸 (x, y, z)
sphere_center = [64, 64, 32]      # 球体中心位置
sphere_radius = 25                # 球体半径
sphere_intensity = 200            # 球体信号强度
noise_std = 30                    # 高斯噪声标准差

print(f"  影像尺寸: {image_size}")
print(f"  球体中心: {sphere_center}, 半径: {sphere_radius}, 强度: {sphere_intensity}")
print(f"  噪声标准差: {noise_std}")

# 生成高斯球体 (模仿肿瘤/病灶信号)
# GaussianSource 签名: (pixelType, size, sigma, mean, scale)
# sigma: 各方向的 sigma 值 (list of float)
# mean:  球体中心位置 (list of float)
# scale: 峰值强度
gaussian_sphere = sitk.GaussianSource(
    sitk.sitkFloat32,
    image_size,
    [float(sphere_radius)] * 3,              # sigma
    [float(sphere_center[0]), float(sphere_center[1]), float(sphere_center[2])],  # mean
    float(sphere_intensity)                  # scale
)

# 添加背景（灰度渐变）
background = sitk.Image(image_size, sitk.sitkFloat32)
background += 50  # 背景基线强度

# 合成影像 = 背景 + 球体
test_image = background + gaussian_sphere

# 添加高斯噪声
noise_image = sitk.AdditiveGaussianNoise(test_image, noise_std)

# 转换为 sitkFloat32 便于处理
noise_image = sitk.Cast(noise_image, sitk.sitkFloat32)

print("  ✅ 测试影像生成完成")


# ============================================================
# 第二步：检查影像参数（处理前）
# ============================================================
print("\n>>> 第二步：原始影像参数")

# 获取numpy数组便于计算统计信息
original_array = sitk.GetArrayFromImage(noise_image)

print(f"  影像维度: {noise_image.GetDimension()}")
print(f"  影像尺寸: {noise_image.GetSize()}")
print(f"  像素间距: {noise_image.GetSpacing()}")
print(f"  原点坐标: {noise_image.GetOrigin()}")
print(f"  方向矩阵: \n{np.array(noise_image.GetDirection()).reshape(3, 3)}")
print(f"  像素类型: {noise_image.GetPixelIDTypeAsString()}")

# 统计参数（使用SimpleITK统计滤波器）
stats_filter = sitk.StatisticsImageFilter()
stats_filter.Execute(noise_image)

print(f"\n  --- 强度统计 ---")
print(f"  最小值: {stats_filter.GetMinimum():.4f}")
print(f"  最大值: {stats_filter.GetMaximum():.4f}")
print(f"  均值:   {stats_filter.GetMean():.4f}")
print(f"  标准差: {stats_filter.GetSigma():.4f}")
print(f"  方差:   {stats_filter.GetVariance():.4f}")


# ============================================================
# 第三步：影像归一化预处理
# ============================================================
print("\n>>> 第三步：影像归一化预处理")

# ---------- 方法1: Min-Max 归一化 [0, 1] ----------
print("\n  --- 方法1: Min-Max 归一化 (映射到 [0, 1]) ---")
minmax_filter = sitk.MinimumMaximumImageFilter()
minmax_filter.Execute(noise_image)
img_min = minmax_filter.GetMinimum()
img_max = minmax_filter.GetMaximum()

# Min-Max 公式: (x - min) / (max - min)
img_norm_minmax = sitk.ShiftScale(noise_image, shift=-img_min, scale=1.0 / (img_max - img_min))

stats_minmax = sitk.StatisticsImageFilter()
stats_minmax.Execute(img_norm_minmax)
print(f"  归一化后 - 最小值: {stats_minmax.GetMinimum():.4f}, 最大值: {stats_minmax.GetMaximum():.4f}")
print(f"  归一化后 - 均值: {stats_minmax.GetMean():.4f}, 标准差: {stats_minmax.GetSigma():.4f}")

# ---------- 方法2: Z-Score 标准化 ----------
print("\n  --- 方法2: Z-Score 标准化 (均值为0, 标准差为1) ---")
stats_full = sitk.StatisticsImageFilter()
stats_full.Execute(noise_image)
global_mean = stats_full.GetMean()
global_std = stats_full.GetSigma()

# Z-Score 公式: (x - mean) / std
img_norm_zscore = sitk.ShiftScale(noise_image, shift=-global_mean, scale=1.0 / global_std)

stats_zscore = sitk.StatisticsImageFilter()
stats_zscore.Execute(img_norm_zscore)
print(f"  标准化后 - 最小值: {stats_zscore.GetMinimum():.4f}, 最大值: {stats_zscore.GetMaximum():.4f}")
print(f"  标准化后 - 均值: {stats_zscore.GetMean():.4f}, 标准差: {stats_zscore.GetSigma():.4f}")

# ---------- 方法3: 直方图匹配（强度归一化到[0,255]） ----------
print("\n  --- 方法3: 强度归一到 [0, 255] (8-bit 显示) ---")
img_norm_255 = sitk.RescaleIntensity(noise_image, 0, 255)
stats_255 = sitk.StatisticsImageFilter()
stats_255.Execute(img_norm_255)
print(f"  归一化后 - 最小值: {stats_255.GetMinimum():.4f}, 最大值: {stats_255.GetMaximum():.4f}")
print(f"  归一化后 - 均值: {stats_255.GetMean():.4f}, 标准差: {stats_255.GetSigma():.4f}")

print("\n  ✅ 三种归一化方法完成")


# ============================================================
# 第四步：结果可视化
# ============================================================
print("\n>>> 第四步：结果可视化")

# 取中间切片用于2D展示
slice_idx_z = image_size[2] // 2   # z方向中间切片
slice_idx_y = image_size[1] // 2   # y方向中间切片

# 提取各影像的 numpy 数组
arr_original = sitk.GetArrayFromImage(noise_image)
arr_minmax = sitk.GetArrayFromImage(img_norm_minmax)
arr_zscore = sitk.GetArrayFromImage(img_norm_zscore)
arr_255 = sitk.GetArrayFromImage(img_norm_255)

# ---------- 子图1: 对比四种影像的横断面 ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('医学影像归一化预处理 - 横断面对比 (轴向中间切片)', fontsize=14, fontweight='bold')

titles = ['原始影像', 'Min-Max [0,1]', 'Z-Score 标准化', 'Rescale [0,255]']
images = [arr_original[slice_idx_z], arr_minmax[slice_idx_z],
          arr_zscore[slice_idx_z], arr_255[slice_idx_z]]

for ax, img, title in zip(axes.flat, images, titles):
    im = ax.imshow(img, cmap='gray', aspect='equal')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()

# ---------- 子图2: 多方位对比 (原始 vs Min-Max) ----------
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
fig2.suptitle('多方位影像对比：原始影像 vs Min-Max归一化', fontsize=14, fontweight='bold')

slice_positions = [
    ('横断面 (Axial)', arr_original[slice_idx_z], arr_minmax[slice_idx_z]),
    ('冠状面 (Coronal)', arr_original[:, slice_idx_y, :], arr_minmax[:, slice_idx_y, :]),
    ('矢状面 (Sagittal)', arr_original[:, :, slice_idx_y], arr_minmax[:, :, slice_idx_y]),
]

for col, (label, orig, norm) in enumerate(slice_positions):
    # 原始
    im1 = axes2[0, col].imshow(orig, cmap='gray', aspect='equal')
    axes2[0, col].set_title(f'原始 {label}', fontsize=10)
    axes2[0, col].axis('off')
    plt.colorbar(im1, ax=axes2[0, col], fraction=0.046)
    # 归一化
    im2 = axes2[1, col].imshow(norm, cmap='gray', aspect='equal')
    axes2[1, col].set_title(f'归一化 {label}', fontsize=10)
    axes2[1, col].axis('off')
    plt.colorbar(im2, ax=axes2[1, col], fraction=0.046)

plt.tight_layout()

# ---------- 子图3: 强度直方图对比 ----------
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))
fig3.suptitle('强度分布直方图对比', fontsize=14, fontweight='bold')

hist_data = [arr_original.flatten(), arr_minmax.flatten(),
             arr_zscore.flatten(), arr_255.flatten()]
hist_titles = ['原始影像', 'Min-Max [0,1]', 'Z-Score', '[0,255]']

for ax, data, title in zip(axes3, hist_data, hist_titles):
    ax.hist(data, bins=100, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('像素强度')
    ax.set_ylabel('频数')
    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1.5, label=f'均值={np.mean(data):.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()

# 保存图像
os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)
fig.savefig(os.path.join(os.path.dirname(__file__) or '.', 'medical_image_comparison.png'), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(os.path.dirname(__file__) or '.', 'medical_image_multiplanar.png'), dpi=150, bbox_inches='tight')
fig3.savefig(os.path.join(os.path.dirname(__file__) or '.', 'medical_image_histogram.png'), dpi=150, bbox_inches='tight')

print("  ✅ 可视化图表已保存")
print(f"     - medical_image_comparison.png")
print(f"     - medical_image_multiplanar.png")
print(f"     - medical_image_histogram.png")

plt.show()

# ============================================================
# 第五步：汇总输出
# ============================================================
print("\n" + "=" * 60)
print("预处理结果汇总")
print("=" * 60)

results = [
    ("原始影像", original_array),
    ("Min-Max [0,1]", arr_minmax),
    ("Z-Score 标准化", arr_zscore),
    ("Rescale [0,255]", arr_255),
]

print(f"\n{'方法':<18}{'最小值':>10}{'最大值':>10}{'均值':>10}{'标准差':>10}")
print("-" * 58)
for name, arr in results:
    print(f"{name:<18}{arr.min():>10.4f}{arr.max():>10.4f}"
          f"{arr.mean():>10.4f}{arr.std():>10.4f}")

print("\n✅ 医学影像预处理作业完成！")
