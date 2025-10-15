'''
某工业制造企业在其生产线上部署了多台传感器以监控关键设备（如电机、泵、压缩机等）的运行状态。
这些传感器周期性地采集设备的多维度运行数据（如温度、振动、压力、电流、转速等），每隔固定时间窗口会生成一组时序特征数据。

为了实现设备早期故障预警，需要对每一组采集到的时序数据进行异常检测和评分。
工程师通过人工标记历史数据集，训练出一套多层自注意力（Self-Attention）+多层全连接层（FC）结构的神经网络模型。
现在，为了模型的快速部署与测试，需要根据题目中给定的网络权重参数，编写代码完成前向推理，输出每一组传感器数据的最终异常分数。

网络结构：传感器数据 -> Self-Attention -> FC -> Self-Attention -> FC -> 异常分数

输入说明：
- 第1行：L,D（L为时序长度，D为特征维度）
- 第2行：输入序列数据（L*D个逗号分隔的浮点数）
- 第3-7行：第一层Self-Attention和FC的权重（Wq1, Wk1, Wv1, Wmlp1, bmlp1）
- 第8-12行：第二层Self-Attention和FC的权重（Wq2, Wk2, Wv2, Wmlp2, bmlp2）

Self-Attention计算公式：
Q = X @ Wq, K = X @ Wk, V = X @ Wv
S = (Q @ K^T) / sqrt(D)
A = Softmax(S)
Output = A @ V
'''

import numpy as np
import math


def softmax_rows(M: np.ndarray) -> np.ndarray:
    """对矩阵的每一行进行softmax归一化"""
    mx = np.max(M, axis=1, keepdims=True)
    E = np.exp(M - mx)
    S = np.sum(E, axis=1, keepdims=True)
    return E / S


def attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, D: int) -> np.ndarray:
    """
    Self-Attention机制
    
    参数:
        X: (L, D) 输入矩阵
        Wq, Wk, Wv: (D, D) Query, Key, Value权重矩阵
        D: 特征维度
    
    返回:
        (L, D) 输出矩阵
    """
    # 计算Query, Key, Value
    Q = X @ Wq  # (L, D) @ (D, D) = (L, D)
    K = X @ Wk  # (L, D) @ (D, D) = (L, D)
    V = X @ Wv  # (L, D) @ (D, D) = (L, D)
    
    # 计算注意力分数并缩放
    S = (Q @ K.T) / math.sqrt(D)  # (L, D) @ (D, L) = (L, L)
    
    # Softmax归一化得到注意力权重
    A = softmax_rows(S)  # (L, L)
    
    # 加权求和得到输出
    return A @ V  # (L, L) @ (L, D) = (L, D)


def read_line_as_array(expected_size: int, shape=None) -> np.ndarray:
    """
    读取一行并转换为指定形状的numpy数组
    
    参数:
        expected_size: 期望的元素数量
        shape: 目标形状（可选）
    
    返回:
        numpy数组
    """
    line = input().strip()
    values = np.array(list(map(float, line.split(','))), dtype=np.float64)
    
    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {len(values)}")
    
    return values.reshape(shape) if shape else values


def analyze_sensor_data(L: int, D: int,
                       seq: np.ndarray,
                       Wq1: np.ndarray, Wk1: np.ndarray, Wv1: np.ndarray,
                       Wmlp1: np.ndarray, bmlp1: np.ndarray,
                       Wq2: np.ndarray, Wk2: np.ndarray, Wv2: np.ndarray,
                       Wmlp2: np.ndarray, bmlp2: np.ndarray) -> np.ndarray:
    """
    两层Self-Attention + FC网络进行异常检测
    
    网络结构:
        Input (L, D)
        -> Self-Attention1 -> FC1
        -> Self-Attention2 -> FC2
        -> Output (L, D)
    
    参数:
        L: 序列长度
        D: 特征维度
        seq: 输入序列 (L, D)
        Wq1, Wk1, Wv1: 第一层Self-Attention权重
        Wmlp1, bmlp1: 第一层全连接层权重和偏置
        Wq2, Wk2, Wv2: 第二层Self-Attention权重
        Wmlp2, bmlp2: 第二层全连接层权重和偏置
    
    返回:
        (L, D) 异常分数矩阵
    """
    # 第一层：Self-Attention + FC
    Y1 = attention(seq, Wq1, Wk1, Wv1, D)  # (L, D)
    Z1 = Y1 @ Wmlp1 + bmlp1  # (L, D) @ (D, D) + (D,) = (L, D)

    # 第二层：Self-Attention + FC
    Y2 = attention(Z1, Wq2, Wk2, Wv2, D)  # (L, D)
    Z2 = Y2 @ Wmlp2 + bmlp2  # (L, D) @ (D, D) + (D,) = (L, D)

    return Z2


def main():
    """主函数：读取输入，执行推理，输出结果"""
    # 第1行：读取维度信息
    L, D = map(int, input().split(','))
    
    # 第2行：读取输入序列
    seq = read_line_as_array(L * D, (L, D))
    
    # 第3-7行：读取第一层权重
    Wq1 = read_line_as_array(D * D, (D, D))
    Wk1 = read_line_as_array(D * D, (D, D))
    Wv1 = read_line_as_array(D * D, (D, D))
    Wmlp1 = read_line_as_array(D * D, (D, D))
    bmlp1 = read_line_as_array(D, (D,))
    
    # 第8-12行：读取第二层权重
    Wq2 = read_line_as_array(D * D, (D, D))
    Wk2 = read_line_as_array(D * D, (D, D))
    Wv2 = read_line_as_array(D * D, (D, D))
    Wmlp2 = read_line_as_array(D * D, (D, D))
    bmlp2 = read_line_as_array(D, (D,))
    
    # 执行前向推理
    result = analyze_sensor_data(
        L, D, seq,
        Wq1, Wk1, Wv1, Wmlp1, bmlp1,
        Wq2, Wk2, Wv2, Wmlp2, bmlp2
    )
    
    # 输出结果（保留两位小数）
    flat = result.flatten()
    print(','.join(f"{x:.2f}" for x in flat))


if __name__ == '__main__':
    main()
