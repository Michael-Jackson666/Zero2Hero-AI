"""
测试注意力模块
演示 ScaledDotProductAttention 和 MultiHeadAttention 的使用
"""
import torch
from ScaledDotProductAttention import ScaledDotProductAttention
from MultiHeadAttention import MultiHeadAttention


def test_scaled_dot_product_attention():
    """测试缩放点积注意力"""
    print("=" * 60)
    print("测试 ScaledDotProductAttention")
    print("=" * 60)
    
    # 参数设置
    B, H, T_q, T_k, d_k = 2, 4, 5, 6, 8
    
    # 创建随机输入
    Q = torch.randn(B, H, T_q, d_k)
    K = torch.randn(B, H, T_k, d_k)
    V = torch.randn(B, H, T_k, d_k)
    
    # 创建掩码 (可选)
    mask = torch.ones(B, 1, T_q, T_k)
    # 将第一个样本的最后两个 key 位置屏蔽
    mask[0, :, :, -2:] = 0
    
    # 创建注意力模块
    attn = ScaledDotProductAttention(dropout=0.1)
    
    # 前向传播
    out, attn_weights = attn(Q, K, V, mask)
    
    # 输出结果
    print(f"输入形状:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    print(f"  mask: {mask.shape}")
    print(f"\n输出形状:")
    print(f"  out: {out.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    
    # 验证注意力权重求和为 1
    print(f"\n注意力权重求和 (应该为1):")
    print(f"  样本0, 头0, 位置0: {attn_weights[0, 0, 0].sum().item():.4f}")
    print(f"  样本1, 头2, 位置3: {attn_weights[1, 2, 3].sum().item():.4f}")
    
    # 显示被屏蔽位置的注意力权重 (应该接近0)
    print(f"\n被屏蔽位置的注意力权重 (应该接近0):")
    print(f"  样本0, 头0, 位置0, key位置4: {attn_weights[0, 0, 0, 4].item():.6f}")
    print(f"  样本0, 头0, 位置0, key位置5: {attn_weights[0, 0, 0, 5].item():.6f}")
    print()


def test_multi_head_attention():
    """测试多头注意力"""
    print("=" * 60)
    print("测试 MultiHeadAttention")
    print("=" * 60)
    
    # 参数设置
    B, T_q, T_k, d_model, num_heads = 2, 5, 6, 32, 4
    
    # 创建随机输入
    x_q = torch.randn(B, T_q, d_model)
    x_kv = torch.randn(B, T_k, d_model)
    
    # 创建掩码 (可选)
    mask = torch.ones(B, 1, T_q, T_k)
    # 屏蔽一些位置
    mask[0, :, :, -1] = 0  # 样本0的最后一个key位置
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    
    # 前向传播
    out, attn_weights = mha(x_q, x_kv, mask)
    
    # 输出结果
    print(f"参数设置:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (每个头的维度): {d_model // num_heads}")
    print(f"\n输入形状:")
    print(f"  x_q: {x_q.shape}")
    print(f"  x_kv: {x_kv.shape}")
    print(f"  mask: {mask.shape}")
    print(f"\n输出形状:")
    print(f"  out: {out.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    
    # 验证输出形状
    assert out.shape == (B, T_q, d_model), f"输出形状错误! 期望 {(B, T_q, d_model)}, 得到 {out.shape}"
    assert attn_weights.shape == (B, num_heads, T_q, T_k), f"注意力权重形状错误!"
    
    print(f"\n✅ 形状验证通过!")
    print()


def test_self_attention():
    """测试自注意力 (Q=K=V)"""
    print("=" * 60)
    print("测试自注意力 (Self-Attention)")
    print("=" * 60)
    
    # 参数设置
    B, T, d_model, num_heads = 2, 10, 64, 8
    
    # 创建随机输入
    x = torch.randn(B, T, d_model)
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    # 自注意力: Q=K=V
    out, attn_weights = mha(x, x, mask=None)
    
    print(f"自注意力配置:")
    print(f"  输入: {x.shape}")
    print(f"  Query = Key = Value (同一个输入)")
    print(f"\n输出:")
    print(f"  out: {out.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    
    # 注意力权重应该是对称的 (因为 Q=K)
    print(f"\n注意力权重统计:")
    print(f"  最小值: {attn_weights.min().item():.4f}")
    print(f"  最大值: {attn_weights.max().item():.4f}")
    print(f"  均值: {attn_weights.mean().item():.4f}")
    print()


def test_cross_attention():
    """测试交叉注意力 (Q ≠ K=V)"""
    print("=" * 60)
    print("测试交叉注意力 (Cross-Attention)")
    print("=" * 60)
    
    # 参数设置 (模拟 Decoder 查询 Encoder)
    B = 2
    T_decoder = 8  # Decoder 序列长度
    T_encoder = 12  # Encoder 序列长度
    d_model = 64
    num_heads = 8
    
    # Decoder 的查询
    decoder_hidden = torch.randn(B, T_decoder, d_model)
    # Encoder 的输出 (作为 Key 和 Value)
    encoder_output = torch.randn(B, T_encoder, d_model)
    
    # 创建多头注意力模块
    cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 交叉注意力: Q 来自 Decoder, K/V 来自 Encoder
    out, attn_weights = cross_attn(decoder_hidden, encoder_output, mask=None)
    
    print(f"交叉注意力配置:")
    print(f"  Decoder (Query): {decoder_hidden.shape}")
    print(f"  Encoder (Key/Value): {encoder_output.shape}")
    print(f"\n输出:")
    print(f"  out: {out.shape}")
    print(f"  attn_weights: {attn_weights.shape}")
    
    # 注意力权重形状应该是 (B, H, T_decoder, T_encoder)
    assert attn_weights.shape == (B, num_heads, T_decoder, T_encoder)
    print(f"\n✅ 交叉注意力形状验证通过!")
    print()


def test_with_causal_mask():
    """测试带因果掩码的注意力 (用于 Decoder)"""
    print("=" * 60)
    print("测试因果掩码 (Causal Mask)")
    print("=" * 60)
    
    # 参数设置
    B, T, d_model, num_heads = 1, 5, 32, 4
    
    # 创建输入
    x = torch.randn(B, T, d_model)
    
    # 创建因果掩码 (下三角)
    causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    
    print(f"因果掩码 (下三角, 1=可见, 0=屏蔽):")
    print(causal_mask[0, 0].numpy())
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    # 应用因果掩码
    out, attn_weights = mha(x, x, mask=causal_mask)
    
    print(f"\n注意力权重 (第一个头):")
    print(attn_weights[0, 0].detach().numpy())
    
    print(f"\n说明:")
    print(f"  - 对角线及以下为非零 (可以看到当前和之前的位置)")
    print(f"  - 对角线以上接近零 (不能看到未来位置)")
    print()


if __name__ == "__main__":
    # 运行所有测试
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_self_attention()
    test_cross_attention()
    test_with_causal_mask()
    
    print("=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)
