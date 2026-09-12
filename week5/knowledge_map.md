# Python数据分析 · 五周知识思维导图

> 使用 Mermaid mindmap 语法绘制，可在 VS Code（Markdown Preview Mermaid Support）、Typora 或 GitHub 中直接预览渲染。

```mermaid
mindmap
  root((Python数据分析<br/>五周学习))
    第一周 Python基础
      基础语法
        数据类型 int float str bool
        列表 字典 元组
      控制结构
        if-elif-else 分支
        for / while 循环
      函数与面向对象
        函数定义与参数
        类 class 与 __init__
      文件与异常
        with open 读写
        try-except 捕获
    第二周 数据分析三件套
      NumPy 数值计算
        数组 ndarray
        向量化运算
        广播机制
      Pandas 表格处理
        DataFrame 结构
        缺失值处理
        分组聚合 groupby
        排序与筛选
      Matplotlib 可视化
        折线图 柱状图
        饼图 散点图
        多子图布局
    第三周 医学影像+医保
      SimpleITK 影像预处理
        合成测试影像生成
        统计参数提取
        Min-Max / Z-Score 归一化
        多方位可视化
      医保大数据清洗
        模拟数据自动构造
        缺失值填充策略
        IQR 异常值剔除
        多维度统计与可视化
    第四周 深度学习
      PyTorch 框架
        张量 Tensor
        自动求导 autograd
        DataLoader 数据加载
        transforms 预处理
      CNN 卷积神经网络
        Conv2d 卷积层
        BatchNorm 批归一化
        ReLU 激活函数
        MaxPool 池化层
        Dropout 防过拟合
      CIFAR-10 图像分类
        数据增强
        训练与评估
        最佳准确率 77.98%
    通用能力
      环境与工具
        conda / venv 虚拟环境
        pip 包管理
        VS Code 开发
      工程习惯
        代码注释与规范
        编码兼容 UTF-8
        Git 版本管理与归档
      方法论
        数据预处理优先
        问题驱动学习
        复盘迭代改进
```

---

## 核心主线

```
数据(原始) → 预处理(清洗/归一化) → 分析(统计/建模) → 可视化(洞察) → 总结(复盘)
```

| 周次 | 数据形态 | 核心工具 | 核心能力 |
|------|----------|----------|----------|
| 第1周 | 基础数据 | Python 原生 | 语法、算法、工程习惯 |
| 第2周 | 表格数据 | NumPy/Pandas/Matplotlib | 清洗、统计、可视化 |
| 第3周 | 影像+医保 | SimpleITK + Pandas | 医学影像预处理、数据清洗 |
| 第4周 | 图像数据 | PyTorch/torchvision | 深度学习、CNN 建模 |
| 第5周 | 全部沉淀 | Git + 归档 | 知识体系化、复盘输出 |
