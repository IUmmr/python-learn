import numpy as np

# 1. 创建数组
arr1 = np.array([1, 2, 3, 4, 5])      
arr2 = np.zeros((3, 4))                    # 全0矩阵
arr3 = np.random.randn(100, 5)             # 随机数

# 2. 切片
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(arr[0, :])     
print(arr[:, 1])     
print(arr[0:2, 1:3]) 

# 3. 运算
arr = np.array([1, 2, 3, 4, 5])
print(arr + 10)      
print(arr * 2)       
print(f'平均值是:{arr.mean()}')    # 平均值
print(f'总和是:{arr.sum()}')     # 总和

# 4. 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))  # 矩阵乘法
