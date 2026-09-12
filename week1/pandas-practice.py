import pandas as pd
import os

# 第一步：如果 data.csv 不存在，先创建一个

if not os.path.exists('data.csv'):
    print("\n⚠️ data.csv 不存在，正在创建示例数据...")
    df_demo = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '孙七'],
        '年龄': [23, 25, 22, 24, 21],
        '成绩': [85, 92, 78, 90, 88]
    })
    df_demo.to_csv('data.csv', index=False, encoding='utf-8-sig')
    print("✅ data.csv 已创建\n")


# 第二步：从 CSV 读取数据

print("【1】从 data.csv 读取数据")
df = pd.read_csv('data.csv', encoding='utf-8')
print("读取成功！数据如下：")
print(df)
print(f"\n数据形状：{df.shape[0]} 行，{df.shape[1]} 列")

# 第三步：所有操作都基于这个 df（从 CSV 读出来的数据）

print("【2】基础筛选（成绩大于80的学生）")
print(df[df['成绩'] > 80])


print("【3】只取姓名和成绩两列")
print(df[['姓名', '成绩']])


print("【4】复合筛选（成绩>80 且 年龄>22）")
print(df[(df['成绩'] > 80) & (df['年龄'] > 22)])


print("【5】排序：按成绩从高到低")
print(df.sort_values('成绩', ascending=False))


print("【6】排序：按年龄从小到大")
print(df.sort_values('年龄'))


print("【7】新增列：根据成绩划分等级")
df['等级'] = df['成绩'].apply(lambda x: '优秀' if x >= 90 else ('良好' if x >= 80 else '及格'))
print(df)


print("【8】分组统计：各等级人数")
print(df.groupby('等级')['姓名'].count())



print("【9】保存处理后的数据到新文件")
df.to_csv('data_处理后.csv', index=False, encoding='utf-8-sig')
print("✅ 已保存到 data_处理后.csv")


print("【全部完成！】所有操作都基于 data.csv 读取的数据")
print("=" * 60)
