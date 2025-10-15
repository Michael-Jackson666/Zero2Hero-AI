"""
测试 attention.py 的Self-Attention实现
"""

import numpy as np
import math
from attention import softmax_rows, attention, Solution


def test_softmax():
    """测试softmax函数"""
    print("=" * 60)
    print("测试 softmax_rows 函数")
    print("=" * 60)
    
    M = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0]
    ])
    
    result = softmax_rows(M)
    print(f"输入:\n{M}")
    print(f"\n输出:\n{result}")
    
    # 验证每行和为1
    row_sums = np.sum(result, axis=1)
    print(f"\n每行和: {row_sums}")
    assert np.allclose(row_sums, 1.0), "Softmax每行和应该为1"
    
    # 验证正确性
    expected_row1 = np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))
    assert np.allclose(result[0], expected_row1), "第一行softmax计算错误"
    
    print("✓ softmax_rows 测试通过\n")


def test_attention_basic():
    """测试基本的attention计算"""
    print("=" * 60)
    print("测试 attention 函数")
    print("=" * 60)
    
    L, D = 3, 2  # 3个时间步，每步2个特征
    
    # 简单的输入
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    
    # 单位矩阵作为权重
    Wq = np.eye(D)
    Wk = np.eye(D)
    Wv = np.eye(D)
    
    result = attention(X, Wq, Wk, Wv, D)
    
    print(f"输入 X:\n{X}")
    print(f"\n输出:\n{result}")
    print(f"\n输出形状: {result.shape}")
    
    assert result.shape == (L, D), f"输出形状应该是({L}, {D})"
    print("✓ attention 基本测试通过\n")


def test_solution_simple():
    """测试Solution类的简单案例"""
    print("=" * 60)
    print("测试 Solution.analyze_data")
    print("=" * 60)
    
    L, D = 2, 2
    
    # 简单输入
    seq = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    # 使用单位矩阵和零偏置
    I = np.eye(D)
    zero_bias = np.zeros(D)
    
    solver = Solution()
    result = solver.analyze_data(
        L, D, seq,
        I, I, I, I, zero_bias,  # 第一层
        I, I, I, I, zero_bias   # 第二层
    )
    
    print(f"输入:\n{seq}")
    print(f"\n输出:\n{result}")
    print(f"输出形状: {result.shape}")
    
    assert result.shape == (L, D), f"输出形状应该是({L}, {D})"
    print("✓ Solution 简单测试通过\n")


def test_attention_dimensions():
    """测试不同维度下的attention"""
    print("=" * 60)
    print("测试不同维度的attention")
    print("=" * 60)
    
    test_cases = [
        (2, 2),  # 小规模
        (5, 3),  # 中等规模
        (10, 5), # 较大规模
    ]
    
    for L, D in test_cases:
        X = np.random.randn(L, D)
        Wq = np.random.randn(D, D)
        Wk = np.random.randn(D, D)
        Wv = np.random.randn(D, D)
        
        result = attention(X, Wq, Wk, Wv, D)
        
        assert result.shape == (L, D), f"L={L}, D={D}: 形状错误"
        assert not np.isnan(result).any(), f"L={L}, D={D}: 包含NaN"
        assert not np.isinf(result).any(), f"L={L}, D={D}: 包含Inf"
        
        print(f"✓ L={L}, D={D}: 通过")
    
    print("\n✓ 所有维度测试通过\n")


def test_manual_example():
    """手动计算一个简单例子验证正确性"""
    print("=" * 60)
    print("手动验证例子")
    print("=" * 60)
    
    L, D = 2, 2
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    # Q, K, V都等于X (使用单位矩阵)
    Wq = Wk = Wv = np.eye(D)
    
    # 手动计算
    Q = X @ Wq  # = X
    K = X @ Wk  # = X
    V = X @ Wv  # = X
    
    print(f"Q:\n{Q}")
    print(f"K:\n{K}")
    print(f"V:\n{V}")
    
    # S = Q @ K.T / sqrt(D)
    S = (Q @ K.T) / math.sqrt(D)
    print(f"\nS (attention scores):\n{S}")
    
    # Softmax
    A = softmax_rows(S)
    print(f"\nA (attention weights):\n{A}")
    
    # 输出
    output = A @ V
    print(f"\nOutput:\n{output}")
    
    # 使用函数计算
    result = attention(X, Wq, Wk, Wv, D)
    print(f"\nFunction result:\n{result}")
    
    assert np.allclose(result, output), "手动计算和函数结果不一致"
    print("\n✓ 手动验证通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始 Attention 测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_softmax()
        test_attention_basic()
        test_solution_simple()
        test_attention_dimensions()
        test_manual_example()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
