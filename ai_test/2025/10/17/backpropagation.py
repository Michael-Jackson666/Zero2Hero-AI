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
    # 如果是2D数组 (1, n)，展平为1D数组
    if hasattr(vector, 'ndim') and vector.ndim == 2:
        vector = vector.flatten()
    print(" ".join(f"{x:.4f}" for x in vector))
    
def stable_softmax(z):
    """Numerically stable softmax for a batch of vectors."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(z_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def backprop(dZ, K, A, M, Z):
    """
    实现反向传播算法，计算神经网络中所有权重和偏置的梯度
    
    算法思路:
    1. 从输出层(第K层)开始，逐层向前计算梯度
    2. 对每一层i，计算两个梯度:
       - grad_M[i]: 权重矩阵 M[i] 的梯度
       - grad_b[i]: 偏置向量 b[i] 的梯度
    3. 使用链式法则将梯度传播到前一层
    4. 对隐藏层，需要乘以 ReLU 激活函数的导数
    
    参数:
        dZ: 输出层的梯度 (shape: (N, output_dim))，已经通过 Softmax+CrossEntropy 计算得到
        K: 网络层数（不包括输入层）
        A: 各层的激活值列表 [A[0], A[1], ..., A[K]]
           - A[0] 是输入 X (shape: N × input_dim)
           - A[i] 是第i层的激活输出 (shape: N × dims[i])，i=1..K
        M: 权重矩阵列表 [None, M[1], M[2], ..., M[K]]
           - M[i] 是第i层的权重矩阵 (shape: dims[i-1] × dims[i])
        Z: 各层的线性输出列表 [None, Z[1], Z[2], ..., Z[K]]
           - Z[i] = A[i-1] @ M[i] + b[i] (shape: N × dims[i])
    
    返回:
        grad_M: 权重矩阵的梯度列表 [None, grad_M[1], ..., grad_M[K]]
        grad_b: 偏置向量的梯度列表 [None, grad_b[1], ..., grad_b[K]]
    
    数学推导:
        对于第i层: Z[i] = A[i-1] @ M[i] + b[i], A[i] = activation(Z[i])
        
        权重梯度: ∂L/∂M[i] = A[i-1]^T @ (∂L/∂Z[i])
        偏置梯度: ∂L/∂b[i] = sum(∂L/∂Z[i], axis=0)
        
        传播到前一层: ∂L/∂A[i-1] = (∂L/∂Z[i]) @ M[i]^T
        应用激活函数导数: ∂L/∂Z[i-1] = ∂L/∂A[i-1] ⊙ activation'(Z[i-1])
    """
    # 初始化梯度存储列表，索引0位置留空（不使用）
    grad_M = [None] * (K + 1)
    grad_b = [None] * (K + 1)
    
    # 当前层的梯度 dZ_current = ∂L/∂Z[i]，从输出层的 dZ[K] 开始
    dZ_current = dZ
    
    # 反向遍历所有层: 从第K层到第1层
    for i in range(K, 0, -1):
        # === 步骤1: 计算权重梯度 ===
        # grad_M[i] = ∂L/∂M[i] = A[i-1]^T @ dZ_current
        # 矩阵维度: (dims[i-1], N) @ (N, dims[i]) = (dims[i-1], dims[i])
        grad_M[i] = A[i-1].T @ dZ_current
        
        # === 步骤2: 计算偏置梯度 ===
        # grad_b[i] = ∂L/∂b[i] = sum(dZ_current, axis=0)
        # 对批次维度求和，保持形状为 (1, dims[i]) 以匹配 b[i] 的形状
        grad_b[i] = np.sum(dZ_current, axis=0, keepdims=True)
        
        # === 步骤3: 将梯度传播到前一层 ===
        # 只有当不是第一层时才需要继续反向传播
        if i > 1:
            # 3.1 计算对前一层激活值的梯度: ∂L/∂A[i-1] = dZ_current @ M[i]^T
            # 矩阵维度: (N, dims[i]) @ (dims[i], dims[i-1]) = (N, dims[i-1])
            dA_prev = dZ_current @ M[i].T
            
            # 3.2 应用 ReLU 激活函数的导数
            # ReLU'(z) = 1 if z > 0, else 0
            # ∂L/∂Z[i-1] = ∂L/∂A[i-1] ⊙ ReLU'(Z[i-1])
            # 元素级乘法: 将梯度传递到未激活的线性输出上
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


