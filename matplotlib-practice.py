import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']        # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示为方块的问题

# 1. 折线图（时间序列、趋势）
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)', linewidth=2, color='blue')
plt.xlabel('X轴')     
plt.ylabel('Y轴')      
plt.title('图示例')        
plt.legend()              # 图例
plt.grid(True, alpha=0.3) # 网格
plt.show()

# 2. 柱状图
categories = ['A', 'B', 'C', 'D']
values = [15, 30, 45, 20]
plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('数值')
plt.title('柱状图示例')
plt.show()

# 3. 热力图
df = pd.DataFrame(np.random.randn(10, 5), columns=['A','B','C','D','E'])
corr = df.corr()  # 计算相关系数
plt.imshow(corr, cmap='coolwarm', aspect='auto')
plt.colorbar(label='相关系数')
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('相关性热力图')
plt.show()

# 4. 子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x))
axes[0, 1].plot(x, np.cos(x))
axes[1, 0].bar(categories, values)
axes[1, 1].hist(np.random.randn(1000), bins=30)
plt.show()