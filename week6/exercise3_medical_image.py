"""
============================================================
第6周 习题3：医学影像巩固题
============================================================
题目要求：
  生成二维测试影像；完成：
  1. 获取影像 numpy 数组；
  2. 像素归一化到 [0, 1]；
  3. 将大于 0.5 的像素置为 1，其余置 0（简单二值分割）；
  4. 打印原始数组形状、二值图像素总和。

核心知识点：
  1. SimpleITK 影像生成（GaussianSource）
  2. GetArrayFromImage 影像转 numpy 数组
  3. 像素归一化（Min-Max 到 [0,1]）
  4. numpy 布尔索引二值分割
  5. 可视化对比
============================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("第6周 习题3：医学影像巩固题")
    print("=" * 60)

    # ---------- 生成二维测试影像 ----------
    print("\n[步骤0] 生成二维测试影像（高斯球体 + 噪声）")
    image_size = [128, 128]
    sphere_radius = 30.0
    sphere_center = [64.0, 64.0]
    sphere_intensity = 255.0

    # 创建高斯"球体"作为伪病灶（二维高斯斑）
    gaussian = sitk.GaussianSource(sitk.sitkFloat32, image_size,
                                   [sphere_radius] * 2,
                                   [sphere_center[0], sphere_center[1]],
                                   sphere_intensity)
    # 叠加高斯噪声，模拟真实影像噪声
    noise = sitk.AdditiveGaussianNoise(gaussian, mean=0.0, standardDeviation=15.0)
    # 将噪声范围外的强度限制在 [0, 255]，保证原始影像为 8bit 灰度风格
    image = sitk.Clamp(noise, lowerBound=0.0, upperBound=255.0)

    print(f"  影像尺寸: {image.GetSize()}, 像素类型: {image.GetPixelIDTypeAsString()}")

    # ---------- 步骤1：获取影像 numpy 数组 ----------
    print("\n[步骤1] 获取影像 numpy 数组")
    arr = sitk.GetArrayFromImage(image)          # 返回 2D numpy 数组
    print(f"  numpy 数组形状: {arr.shape}")
    print(f"  数据类型: {arr.dtype}")
    print(f"  像素范围: [{arr.min():.2f}, {arr.max():.2f}]")

    # ---------- 步骤2：像素归一化到 [0, 1] ----------
    print("\n[步骤2] 像素归一化到 [0, 1]")
    arr_min, arr_max = arr.min(), arr.max()
    arr_norm = (arr - arr_min) / (arr_max - arr_min)   # Min-Max 归一化到 [0,1]
    print(f"  归一化后范围: [{arr_norm.min():.4f}, {arr_norm.max():.4f}]")
    print(f"  归一化后均值: {arr_norm.mean():.4f}")

    # ---------- 步骤3：简单二值分割 ----------
    print("\n[步骤3] 二值分割（>0.5 置 1，其余置 0）")
    arr_binary = (arr_norm > 0.5).astype(np.uint8)   # 布尔索引转 0/1
    print(f"  二值数组取值: {np.unique(arr_binary)}")
    print(f"  二值化后均值: {arr_binary.mean():.4f}")

    # ---------- 步骤4：打印形状与二值像素总和 ----------
    print("\n[步骤4] 输出结果")
    print(f"  原始数组形状: {arr.shape}")
    print(f"  二值图像素总和（前景像素数）: {arr_binary.sum()}")
    print(f"  前景占比: {arr_binary.sum() / arr_binary.size * 100:.2f}%")

    # ---------- 可视化对比 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('习题3：影像归一化与二值分割', fontsize=14, fontweight='bold')

    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title(f'原始影像\n[{arr.min():.0f}, {arr.max():.0f}]')
    axes[0].axis('off')

    axes[1].imshow(arr_norm, cmap='gray')
    axes[1].set_title('归一化到 [0,1]')
    axes[1].axis('off')

    axes[2].imshow(arr_binary, cmap='gray')
    axes[2].set_title(f'二值分割\n前景像素={arr_binary.sum()}')
    axes[2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'exercise3_binary_seg.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  可视化已保存: {fig_path}")
    print("\n✅ 习题3完成")


if __name__ == '__main__':
    main()
