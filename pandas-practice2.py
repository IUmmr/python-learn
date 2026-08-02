import pandas as pd
import numpy as np

# 创建原始数据（模拟真实数据，包含各种脏数据）
print("\n【1】原始数据（包含缺失值、异常值、重复值）")

df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '孙七', '钱八', '张三'],  # 张三重复
    '年龄': [23, 25, 999, 22, 19, 200, 23],     # 999 和 200 是异常值
    '专业': ['计算机', '计算机', '数学', '数学', '计算机', '数学', '计算机'],
    '成绩': [85, np.nan, 78, 88, 92, 75, 85]    # 李四缺考，成绩缺失；张三重复
})

print(df)
print(f"\n原始数据形状：{df.shape} 行，{df.shape[1]} 列")

# 第二步：缺失值处理

print("\n" + "=" * 60)
print("【2】缺失值处理")
print("\n缺失值统计：")
print(df.isnull().sum())
# 用平均分填充缺失值
avg_score = df['成绩'].mean()
df['成绩'] = df['成绩'].fillna(avg_score)
print(f"\n用平均分 {avg_score:.1f} 填充成绩缺失值")
print(df)

# 第三步：异常值处理

print("\n" + "=" * 60)
print("【3】异常值处理（年龄 > 150 视为异常）")

print("\n过滤前数据条数：", len(df))
df = df[df['年龄'] < 150]   # 过滤年龄>150的异常值
print("过滤后数据条数：", len(df))
print(df)

# 第四步：数据去重

print("\n" + "=" * 60)
print("【4】数据去重（按姓名去重）")

print("\n去重前数据条数：", len(df))
df = df.drop_duplicates(subset=['姓名'], keep='first')
print("去重后数据条数：", len(df))
print(df)

# 第五步：数据合并

print("\n" + "=" * 60)
print("【5】数据合并（添加班级信息）")

# 创建班级信息表
df_class = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '孙七'],
    '班级': ['1班', '2班', '1班', '3班', '2班']
})
print("班级信息表：")
print(df_class)

# 合并
df = pd.merge(df, df_class, on='姓名', how='left')
print("\n合并后的数据：")
print(df)

# 第六步：分组统计

print("\n" + "=" * 60)
print("【6】分组统计（按专业计算平均成绩）")

grouped = df.groupby('专业')['成绩'].agg(['mean', 'max', 'min', 'count'])
print("各专业成绩统计：")
print(grouped)

# 按班级分组统计
print("\n各班级成绩统计：")
print(df.groupby('班级')['成绩'].mean())

# 第七步：保存清洗后的数据

print("【7】保存清洗后的数据")

df.to_csv('学生成绩_清洗后.csv', index=False, encoding='utf-8-sig')
print("已保存到：学生成绩_清洗后.csv")

print("\n最终清洗后的数据：")
print(df)
print("【数据清洗完成！】")
print(f"最终数据：{len(df)} 条记录，{len(df.columns)} 列")
print("=" * 60)