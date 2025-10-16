"""
项目: 滑动窗口特征提取

描述:
本项目旨在实现一个多尺寸滑动窗口的特征转换.
给定一个时间序列数据 input_array 和一组窗口大小 window_array.
对于 window_array 中的每个窗口大小 w, 在 input_array 上滑动窗口.
从每个窗口中提取5个特征: 均值(mean), 标准差(std), 最小值(min), 最大值(max), 和斜率(slope).

特征计算说明:
- 均值: 窗口内元素的平均值.
- 标准差: 样本标准差, ddof为0.
- 最小值: 窗口内的最小值.
- 最大值: 窗口内的最大值.
- 斜率: 使用线性最小二乘法回归计算. 时间索引 x = [0, 1, 2, ..., w-1].
  斜率 beta 计算公式为:
  beta = [n * sum(xy) - sum(x) * sum(y)] / [n * sum(x^2) - (sum(x))^2]

数值修约规则:
- 如果一个数的整数部分不为0, 则小数点后最多保留3位.
- 如果整数部分为0, 则最多保留4位有效数字.
- 采用四舍五入的方式.

输入:
- input_array: 一个一维数组, 表示时间序列数据.
- window_array: 一个一维数组, 包含多个窗口的大小.

输出:
- 一个二维数组, 形状为 (n, m), 其中:
  - n = len(input_array) - max(window_array) + 1
  - m = len(window_array) * 5
- 每一行对应一个滑动窗口的起始位置, 包含了所有窗口大小计算出的特征的拼接结果.
- 每个窗口大小对应的特征顺序为 [mean, std, min, max, slope].
"""

import numpy as np
import math


def round_number(value):
    """
    根据题目要求的数值修约规则进行四舍五入
    
    规则:
    - 如果整数部分不为0, 则小数点后最多保留3位
    - 如果整数部分为0, 则最多保留4位有效数字
    
    参数:
        value: 要修约的数值
    
    返回:
        修约后的数值
    """
    if np.isnan(value) or np.isinf(value):
        return value
    
    abs_value = abs(value)
    
    # 整数部分不为0: 保留小数点后3位
    if abs_value >= 1:
        return round(value, 3)
    
    # 整数部分为0: 保留4位有效数字
    if abs_value == 0:
        return 0.0
    
    # 计算需要保留的小数位数以达到4位有效数字
    # 例如: 0.001234 -> log10(0.001234) ≈ -2.91, floor(-2.91)=-3
    # 需要保留到 -(-3) + 4 - 1 = 6 位小数
    decimal_places = -int(math.floor(math.log10(abs_value))) + 3
    return round(value, decimal_places)


def calculate_slope(window_data):
    """
    使用线性最小二乘法计算窗口数据的斜率
    
    斜率公式: beta = [n * sum(xy) - sum(x) * sum(y)] / [n * sum(x^2) - (sum(x))^2]
    其中 x = [0, 1, 2, ..., n-1]
    
    参数:
        window_data: 窗口内的数据数组
    
    返回:
        斜率值
    """
    n = len(window_data)
    x = np.arange(n)  # [0, 1, 2, ..., n-1]
    y = window_data
    
    # 计算各项求和
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x * x)
    
    # 计算斜率
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x * sum_x
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return slope


def extract_features(window_data):
    """
    从窗口数据中提取5个特征
    
    参数:
        window_data: 窗口内的数据数组
    
    返回:
        包含5个特征的列表: [mean, std, min, max, slope]
    """
    # 均值
    mean = np.mean(window_data)
    
    # 标准差 (ddof=0, 即总体标准差)
    std = np.std(window_data, ddof=0)
    
    # 最小值
    min_val = np.min(window_data)
    
    # 最大值
    max_val = np.max(window_data)
    
    # 斜率
    slope = calculate_slope(window_data)
    
    # 应用数值修约规则
    features = [
        round_number(mean),
        round_number(std),
        round_number(min_val),
        round_number(max_val),
        round_number(slope)
    ]
    
    return features


def sliding_window_features(input_array, window_array):
    """
    对时间序列数据应用多尺寸滑动窗口并提取特征
    
    参数:
        input_array: 一维数组, 时间序列数据
        window_array: 一维数组, 包含多个窗口大小
    
    返回:
        二维数组, 形状为 (n, m)
        n = len(input_array) - max(window_array) + 1
        m = len(window_array) * 5
    """
    input_array = np.array(input_array)
    window_array = np.array(window_array)
    
    # 计算输出数组的维度
    max_window = int(np.max(window_array))
    n_rows = len(input_array) - max_window + 1
    n_cols = len(window_array) * 5
    
    # 初始化输出数组
    result = np.zeros((n_rows, n_cols))
    
    # 对每个滑动窗口位置
    for i in range(n_rows):
        col_idx = 0
        
        # 对每个窗口大小
        for window_size in window_array:
            window_size = int(window_size)
            
            # 提取当前窗口的数据
            # 关键理解: 窗口的结束位置应该与max_window对齐
            # 即所有窗口都结束在相同的相对位置
            start_idx = i + (max_window - window_size)
            end_idx = start_idx + window_size
            window_data = input_array[start_idx:end_idx]
            
            # 提取特征
            features = extract_features(window_data)
            
            # 将特征填入结果数组
            result[i, col_idx:col_idx+5] = features
            col_idx += 5
    
    return result


def format_output(value):
    """
    格式化输出数字，去除不必要的小数点和零
    
    参数:
        value: 要格式化的数值
    
    返回:
        格式化后的字符串
    """
    if abs(value) >= 1:
        # 整数部分不为0，保留3位小数后去除末尾0
        s = f"{value:.3f}".rstrip('0').rstrip('.')
        return s if s else "0"
    elif value == 0:
        return "0"
    else:
        # 整数部分为0，已经通过round_number处理过
        return str(value)


def main():
    """
    主函数: 从标准输入读取数据并输出结果
    输入格式: [a, b, c, ...] (第一行输入数组，带方括号和逗号)
             [w1, w2, ...] (第二行窗口大小数组，带方括号和逗号)
    输出格式: [v1,v2,v3,...] (每行一个数组，带方括号和逗号，无空格)
    """
    import sys
    import re
    
    # 读取所有输入
    sall = sys.stdin.read()
    
    # 提取两个数组的内容
    blocks = re.findall(r'\[([^\]]*)\]', sall)
    
    if len(blocks) < 2:
        print("[]")
        sys.exit(0)
    
    # 提取数字（支持负数和浮点数）
    input_array = list(map(float, re.findall(r'-?\d+\.?\d*', blocks[0])))
    window_array = list(map(int, re.findall(r'-?\d+', blocks[1])))
    
    if not input_array or not window_array:
        print("[]")
        sys.exit(0)
    
    # 计算特征
    result = sliding_window_features(input_array, window_array)
    
    # 输出结果（格式化为 [v1,v2,v3,...]，无空格）
    for row in result:
        # 格式化每个值
        formatted_values = [format_output(val) for val in row]
        # 输出为 [v1,v2,v3,...] 格式
        print("[" + ",".join(formatted_values) + "]")


if __name__ == "__main__":
    main()
