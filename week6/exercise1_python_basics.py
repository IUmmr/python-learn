"""
============================================================
第6周 习题1：Python 基础综合题
============================================================
题目要求：
  1. 定义函数接收一个数字列表；
  2. 过滤掉列表中小于 0 的数据；
  3. 返回过滤后列表的最大值、最小值、平均值；
  4. 使用冒泡排序对过滤后列表升序输出。

核心知识点：
  1. 函数定义与参数传递
  2. 列表推导式 / filter 过滤
  3. max / min / sum 内置函数
  4. 冒泡排序算法实现
============================================================
"""

import sys
import io
# Windows GBK 编码环境下强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def filter_and_analyze(numbers):
    """
    过滤负数并返回最大值、最小值、平均值，同时给出升序排序结果。

    参数:
        numbers: 数字列表

    返回:
        dict: 包含过滤后列表、最大值、最小值、平均值、升序排序结果
    """
    # 步骤2：过滤掉小于 0 的数据
    filtered = [x for x in numbers if x >= 0]

    if not filtered:
        # 空列表时避免 max() 报错
        return {
            'filtered': filtered,
            'max': None,
            'min': None,
            'avg': None,
            'sorted': filtered
        }

    # 步骤3：计算最大值、最小值、平均值
    max_value = max(filtered)
    min_value = min(filtered)
    avg_value = sum(filtered) / len(filtered)

    # 步骤4：冒泡排序升序（逐趟比较相邻元素，大的往后冒）
    sorted_list = filtered.copy()
    n = len(sorted_list)
    for i in range(n - 1):                 # 共需 n-1 趟
        swapped = False
        for j in range(n - 1 - i):         # 每趟比较 n-1-i 对
            if sorted_list[j] > sorted_list[j + 1]:
                sorted_list[j], sorted_list[j + 1] = sorted_list[j + 1], sorted_list[j]
                swapped = True
        if not swapped:                    # 本趟无交换说明已有序，提前结束
            break

    return {
        'filtered': filtered,
        'max': max_value,
        'min': min_value,
        'avg': avg_value,
        'sorted': sorted_list
    }


def main():
    # 测试数据：包含正数、负数、小数、0
    test_list = [3.5, -2, 7, 0, -9.5, 12, 4, -1, 8.2, 5, -3.3, 2]

    print("=" * 56)
    print("第6周 习题1：Python 基础综合题")
    print("=" * 56)
    print(f"原始列表: {test_list}")

    result = filter_and_analyze(test_list)

    print(f"\n[步骤2] 过滤负数后的列表: {result['filtered']}")
    print(f"\n[步骤3] 统计结果:")
    print(f"  最大值: {result['max']}")
    print(f"  最小值: {result['min']}")
    print(f"  平均值: {result['avg']:.4f}")

    print(f"\n[步骤4] 冒泡排序升序结果: {result['sorted']}")
    print(f"  排序验证: 与内置 sorted() 一致 -> {result['sorted'] == sorted(result['filtered'])}")

    # 边界测试：全为负数的列表
    print("\n--- 边界测试：全负数列表 ---")
    all_negative = [-5, -3, -8]
    edge_result = filter_and_analyze(all_negative)
    print(f"原始: {all_negative} -> 过滤后: {edge_result['filtered']}, "
          f"max={edge_result['max']}, min={edge_result['min']}, avg={edge_result['avg']}")


if __name__ == '__main__':
    main()
