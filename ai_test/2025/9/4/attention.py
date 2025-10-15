'''
某工业制造企业在其生产线上部署了多台传感器以监控关键设备（如电机、泵、压缩机等）的运行状态。这些传感器周期性地采集设备的多维度运行数据（如温度、振动、压力、电流、转速等），每隔固定时间窗口会生成一组时序特征数据。

为了实现设备早期故障预警，需要对每一组采集到的时序数据进行异常检测和评分。工程师通过人工标记历史数据集，训练出一套多层自注意力（Self-Attention）+多层全连接层（FC）结构的神经网络模型。现在，为了模型的快速部署与测试，需要根据题目中给定的网络权重参数，编写代码完成前向推理，输出每一组传感器数据的最终异常分数。结构如下图所示：

[结构图：传感器数据 -> Self-Attention -> FC -> Self-Attention -> FC -> 异常分数]

具体说明如下：

每一组采集数据为一个二维矩阵，尺寸为 L， L 采样时序长度，D 为每次采样包含的特征数（如：10个时间点，每点5个特征）。

网络结构为两层 Self-Attention，每层后接全连接层 FC，最终输出异常分数。为简化起见，网络中无非线性激活函数。

Self-Attention 采用 Dot-product Attention，计算公式如下：
'''

import sys
import numpy as np
import math

def softmax_rows(M: np.ndarray) -> np.ndarray:
    """对矩阵的每一行进行softmax"""
    mx = np.max(M, axis=1, keepdims=True)
    E = np.exp(M - mx)
    S = np.sum(E, axis=1, keepdims=True)
    return E / S   
    
def attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, D: int) -> np.ndarray:
    """
    Self-Attention机制
    X: (L, D) 输入矩阵
    Wq, Wk, Wv: (D, D) 权重矩阵
    返回: (L, D) 输出矩阵
    """
    # 矩阵乘法，不是元素乘法
    Q = X @ Wq  # (L, D) @ (D, D) = (L, D)
    K = X @ Wk  # (L, D) @ (D, D) = (L, D)
    V = X @ Wv  # (L, D) @ (D, D) = (L, D)
    
    # 计算注意力分数
    S = (Q @ K.T) / math.sqrt(D)  # (L, D) @ (D, L) = (L, L)
    
    # Softmax归一化
    A = softmax_rows(S)  # (L, L)
    
    # 加权求和
    return A @ V  # (L, L) @ (L, D) = (L, D)

class Solution:
    def analyze_data(self, L: int, D: int,
                     seq: np.ndarray,
                     Wq1: np.ndarray, Wk1: np.ndarray, Wv1: np.ndarray,
                     Wmlp1: np.ndarray, bmlp1: np.ndarray,
                     Wq2: np.ndarray, Wk2: np.ndarray, Wv2: np.ndarray,
                     Wmlp2: np.ndarray, bmlp2: np.ndarray) -> np.ndarray:
        """
        两层Self-Attention + FC网络
        """
        X = seq  # (L, D)

        # 第一层：Self-Attention + FC
        Y1 = attention(X, Wq1, Wk1, Wv1, D)  # (L, D)
        Z1 = Y1 @ Wmlp1 + bmlp1  # (L, D) @ (D, D) + (D,) = (L, D)

        # 第二层：Self-Attention + FC (注意这里Wq2拼写错误)
        Y2 = attention(Z1, Wq2, Wk2, Wv2, D)  # (L, D)
        Z2 = Y2 @ Wmlp2 + bmlp2  # (L, D) @ (D, D) + (D,) = (L, D)

        return Z2

if __name__ == '__main__':
    # 读取12行输入
    lines = [sys.stdin.readline().strip() for _ in range(12)]
    L, D = map(int, lines[0].split(','))

    def parse_line(idx: int, count: int):
        """解析一行数据并转换为numpy数组"""
        values = list(map(float, lines[idx].split(',')))
        assert len(values) == count, f"Line {idx} expected {count} values, got {len(values)}"
        return np.array(values, dtype=np.float64), idx + 1
    
    # 逐行解析数据
    idx = 1
    seq_flat, idx = parse_line(idx, L * D)
    seq = seq_flat.reshape(L, D)

    Wq1_flat, idx = parse_line(idx, D * D)
    Wk1_flat, idx = parse_line(idx, D * D)
    Wv1_flat, idx = parse_line(idx, D * D)
    Wmlp1_flat, idx = parse_line(idx, D * D)
    bmlp1_flat, idx = parse_line(idx, D)

    Wq2_flat, idx = parse_line(idx, D * D)
    Wk2_flat, idx = parse_line(idx, D * D)
    Wv2_flat, idx = parse_line(idx, D * D)
    Wmlp2_flat, idx = parse_line(idx, D * D)
    bmlp2_flat, idx = parse_line(idx, D)

    # 重塑为矩阵
    Wq1 = Wq1_flat.reshape(D, D)
    Wk1 = Wk1_flat.reshape(D, D)
    Wv1 = Wv1_flat.reshape(D, D)
    Wmlp1 = Wmlp1_flat.reshape(D, D)
    bmlp1 = bmlp1_flat  # (D,) 向量，不需要reshape

    Wq2 = Wq2_flat.reshape(D, D)
    Wk2 = Wk2_flat.reshape(D, D)
    Wv2 = Wv2_flat.reshape(D, D)
    Wmlp2 = Wmlp2_flat.reshape(D, D)
    bmlp2 = bmlp2_flat  # (D,) 向量，不需要reshape

    # 执行前向推理
    solver = Solution()
    result = solver.analyze_data(L, D, seq,
                                 Wq1, Wk1, Wv1, Wmlp1, bmlp1,
                                 Wq2, Wk2, Wv2, Wmlp2, bmlp2)
    
    # 输出结果
    flat = result.flatten()
    print(','.join(f"{x:.2f}" for x in flat))
    
