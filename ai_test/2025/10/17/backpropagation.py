""""
题目:实现反向传播
"""
# If you need to import additional packages or classes, please import here.
import sys
import numpy as np

def print_formatted_matrix(matrix):
    """Prints a matrix with 4 decimal places."""
    for row in matrix:
        print(" ".join(f"{x:.4f}" for x in row))

def print_formatted_vector(vector):
    """Prints a vector (as a single row) with 4 decimal places."""
    print(" ".join(f"{x:.4f}" for x in vector))
    
def stable_softmax(z):
    """Numerically stable softmax for a batch of vectors."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(z_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def backprop(dZ, K, A, M, Z):
    """
    实现反向传播算法
    
    参数:
        dZ: 输出层的梯度 (已计算好的 dZ[K])
        K: 层数
        A: 各层的激活值列表 [A[0], A[1], ..., A[K]]
        M: 权重矩阵列表 [None, M[1], M[2], ..., M[K]]
        Z: 各层的线性输出列表 [None, Z[1], Z[2], ..., Z[K]]
    
    返回:
        grad_M: 权重矩阵的梯度
        grad_b: 偏置向量的梯度
    """
    # 初始化梯度存储
    grad_M = [None] * (K + 1)
    grad_b = [None] * (K + 1)
    
    # 当前层的梯度（从输出层开始）
    dZ_current = dZ
    
    # 从最后一层反向传播到第一层
    for i in range(K, 0, -1):
        # 计算权重梯度: dL/dM[i] = A[i-1]^T @ dZ[i]
        grad_M[i] = A[i-1].T @ dZ_current
        
        # 计算偏置梯度: dL/db[i] = sum(dZ[i], axis=0)
        # 保持形状为 (1, dim) 以匹配 b[i] 的形状
        grad_b[i] = np.sum(dZ_current, axis=0, keepdims=True)
        
        # 如果不是第一层，继续反向传播
        if i > 1:
            # 计算传递到前一层的梯度: dZ[i-1] = dZ[i] @ M[i]^T
            dA_prev = dZ_current @ M[i].T
            
            # 应用 ReLU 的导数
            # ReLU 导数: d(ReLU(z))/dz = 1 if z > 0, else 0
            dZ_current = dA_prev * (Z[i-1] > 0)
    
    return grad_M, grad_b

def solve():
    try:
        K_str = sys.stdin.readline()
        if not K_str.strip(): return
        K = int(K_str)
        
        h_str = sys.stdin.readline()
        h = list(map(int, h_str.split()))
        dims = h + [10]

        M = [None] * (K + 1)
        b = [None] * (K + 1)

        for i in range(1, K + 1):
            rows, cols = dims[i-1], dims[i]
            M[i] = np.array([list(map(float, sys.stdin.readline().split())) for _ in range(rows)])

        for i in range(1, K + 1):
            b[i] = np.array([list(map(float, sys.stdin.readline().split()))])

        N = int(sys.stdin.readline())
        X = np.array([list(map(float, sys.stdin.readline().split())) for _ in range(N)])
        Y_labels = [int(sys.stdin.readline()) for _ in range(N)]


        Y_true_onehot = np.zeros((N, 10))
        Y_true_onehot[np.arange(N), Y_labels] = 1

        A = [None] * (K + 1)
        Z = [None] * (K + 1)
        A[0] = X

        for i in range(1, K):
            Z[i] = A[i-1] @ M[i] + b[i]
            A[i] = np.maximum(0, Z[i])

        Z[K] = A[K-1] @ M[K] + b[K]
        A[K] = stable_softmax(Z[K])
        output_probabilities = A[K]

        dZK = (output_probabilities - Y_true_onehot) / N # Gradient for the last layer
        
        grad_M, grad_b = backprop(dZK, K, A, M, Z)
        
        for i in range(1, K + 1):
            print_formatted_matrix(grad_M[i])
            print_formatted_vector(grad_b[i])

    except (IOError, ValueError, IndexError):
        # Gracefully exit on any parsing or reading error
        return

if __name__ == "__main__":
    solve()


