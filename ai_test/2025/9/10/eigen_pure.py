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
import sys, math

def round_number(value):
    """
    根据题目要求的数值规则进行四舍五入
    规则:
    --整数则不带小数点
    --有小数则最多保留3位(四舍五入)
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
    elif abs_value > 0:
        digits = -int(math.floor(math.log10(abs_value))) + 3
        return round(value, digits)
    
    return value




def main():
    """
    主函数: 读取输入, 计算特征, 输出结果
    """
    return

if __name__ == "__main__":
    main()