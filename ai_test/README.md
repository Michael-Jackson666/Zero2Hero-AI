# AI/算法工程师面试准备资料库

本仓库包含 AI/算法工程师面试准备的各类资料, 涵盖算法、深度学习、编程技巧等多个方面.

---

## 📁 目录结构

```
ai_test/
├── Transformer/          # Transformer 架构详解
├── ACM/                  # ACM 编程模式教程
├── 2025/                 # 2025年算法题目解答
├── Whiteboard_Coding/    # 白板编程练习
└── README.md            # 本文件
```

---

## 📚 各模块详细说明

### 1. Transformer/ - Transformer 完整教程

**目标受众**: 准备深度学习/NLP 面试的工程师

**核心内容**:
- ✅ **手撕 Transformer**: 从零实现所有关键模块
  - Scaled Dot-Product Attention (带 Mask)
  - Multi-Head Attention (MHA)
  - Position-wise Feed Forward Network (FFN)
  - Positional Encoding (正弦/余弦位置编码)
  - Encoder/Decoder Layer 完整实现
  - Transformer 总装 + 贪心解码

- ✅ **面试高频问题详解**:
  - 影响 Transformer 主要复杂度的因素有哪些?
  - Embedding 层是什么? 有什么作用?
  - 序列长度如何影响训练复杂度?

- ✅ **复杂度分析**:
  - 自注意力: $\mathcal{O}(T^2 \cdot d)$
  - 前馈网络: $\mathcal{O}(T \cdot d^2)$
  - 五大影响因素详解 (序列长度、模型维度、层数、批次大小、头数)

- ✅ **易错点清单**:
  - MHA 头部分割/合并的维度变换
  - Mask 取值约定 (1=可见 vs 1=遮挡)
  - Decoder 自注意力的下三角 Mask
  - Cross-Attention 的 Q/K/V 来源
  - 残差连接 + LayerNorm 的顺序 (Post-LN vs Pre-LN)

- ✅ **可视化图解**:
  - Scaled Dot-Product Attention 结构图
  - Multi-Head Attention 流程图
  - Transformer 整体架构图

**文件**:
- `tutorial.ipynb`: Jupyter Notebook 交互式教程
- 包含完整可运行代码 + 数学公式推导 + 面试答题模板

**适用场景**:
- 面试前快速复习 Transformer 核心原理
- 白板/在线编程中手撕 Transformer
- 理解现代大模型 (GPT/BERT/LLaMA) 的基础架构

---

### 2. ACM/ - ACM 编程模式完全教程

**目标受众**: 准备大厂机考/算法笔试的工程师

**核心内容**:
- ✅ **输入输出规范**: 
  - 标准输入 `input()` 的各种用法
  - 多组测试数据处理 (EOF、特殊标记)
  - 矩阵/图的输入模板

- ✅ **20 个常见场景**:
  - 单行单个数、多个数
  - 数组输入 (定长/不定长)
  - 矩阵输入
  - 多组测试 (无结束标志、0 0 结束)
  - 字符串处理
  - 浮点数格式化
  - 图的输入 (邻接表/邻接矩阵)
  - ... 等

- ✅ **常见陷阱**:
  - 忘记 `strip()`
  - 类型转换错误
  - 多余的输出提示
  - 输出格式不匹配
  - 浮点数精度问题
  - EOF 未处理
  - 空行处理

- ✅ **实战技巧**:
  - 快速调试模板
  - 使用 `sys.stdin` 提高效率
  - 列表推导式
  - 批量输出
  - 文件重定向测试

**文件**:
- `README.md`: 完整教程文档
- `PRACTICE_GUIDE.md`: 练习指南
- `learn_ai.py`: 示例代码
- `test*.txt`: 测试输入文件

**适用场景**:
- 大厂笔试 (字节、腾讯、阿里等)
- OJ 平台刷题 (牛客、LeetCode、洛谷)
- ACM 竞赛准备

---

### 3. 2025/ - 算法题目解答集

**目录结构**:
```
2025/
├── 9/   # 2025年9月题目
└── 10/  # 2025年10月题目
    ├── 17/
    │   └── alarm_detail.py  # 聚类警告问题
    └── ...
```

**题目特点**:
- ✅ **完整注释**: 每行代码都有详细中文注释
- ✅ **复杂度分析**: 时间/空间复杂度详细推导
- ✅ **求解思路**: 算法设计思路、数据结构选择理由
- ✅ **边界处理**: 各种边界情况的处理方法
- ✅ **优化建议**: 可能的优化方向

**示例题目**:
- **聚类警告** (`2025/10/17/alarm_detail.py`):
  - 问题: 使用余弦相似度对警告进行聚类
  - 数据结构: 并查集 (Union-Find)
  - 算法: 路径压缩 + 按秩合并
  - 复杂度: $\mathcal{O}(n^2 d)$, $n$ 为警告数, $d$ 为特征维度
  - 包含: 完整的求解思路说明、边界情况处理、优化方向

**适用场景**:
- 历年真题练习
- 算法模式总结
- 面试前刷题复习

---

### 4. Whiteboard_Coding/ - 白板编程练习

**说明**: 白板编程/在线编程面试准备

**可能包含**:
- 经典算法手撕 (排序、搜索、动态规划等)
- 数据结构实现 (链表、树、图等)
- 系统设计编程题
- 场景题代码实现

---

## 🎯 使用建议

### 面试前 1 周
1. **复习 Transformer**: 
   - 阅读 `Transformer/tutorial.ipynb`
   - 重点记忆"面试高频问题详解"部分
   - 手撕一遍完整代码

2. **熟悉 ACM 模式**:
   - 过一遍 `ACM/README.md` 的 20 个场景
   - 特别注意"常见陷阱"部分
   - 准备好输入输出模板

3. **刷历年真题**:
   - 做 `2025/` 下的题目
   - 重点理解"求解思路"部分
   - 总结常见算法模式

### 面试前 1 天
1. **快速复习**:
   - Transformer: 看"标准回答框架"和"面试加分回答模板"
   - ACM: 看"快速查询表"
   - 算法: 过一遍经典题目的思路

2. **准备模板**:
   - Transformer 各模块的代码框架
   - ACM 输入输出模板
   - 常用数据结构 (并查集、线段树等)

### 面试中
1. **Transformer 手撕**:
   - 先说思路 (Encoder-Decoder 结构)
   - 按模块实现 (Attention → MHA → FFN → Layer)
   - 强调关键点 (Mask、维度变换、残差连接)

2. **算法题**:
   - 明确输入输出格式
   - 讨论边界情况
   - 分析时间/空间复杂度
   - 写注释说明关键步骤

---

## 📖 学习路径

### 初级 (1-2周)
```
ACM 模式输入输出 → 基础数据结构 → 简单算法题
```

### 中级 (2-4周)
```
Transformer 基础理解 → 注意力机制 → 复杂度分析 → 中等算法题
```

### 高级 (4-8周)
```
手撕 Transformer → 优化方法 (FlashAttention 等) → 困难算法题 → 系统设计
```

---

## 🔧 开发环境

### Python 环境
```bash
# 推荐使用 Python 3.8+
python --version

# 安装常用库
pip install torch numpy jupyter notebook
```

### Jupyter Notebook
```bash
# 启动 Jupyter
jupyter notebook

# 打开 Transformer/tutorial.ipynb
```

### ACM 模式测试
```bash
# 方法1: 手动输入
python ACM/learn_ai.py

# 方法2: 文件重定向
python ACM/learn_ai.py < ACM/test1_input.txt

# 方法3: 输入输出重定向
python ACM/learn_ai.py < input.txt > output.txt
```

---

## 📝 贡献指南

欢迎补充更多面试相关内容!

### 建议补充的内容
- [ ] 更多 Transformer 变体 (BERT、GPT、T5 等)
- [ ] 计算机视觉模型 (CNN、ResNet、ViT 等)
- [ ] 强化学习算法
- [ ] 更多经典算法题解
- [ ] 系统设计案例
- [ ] 机器学习基础 (梯度下降、正则化等)

### 提交格式
1. 代码需包含详细中文注释
2. 复杂算法需附带时间/空间复杂度分析
3. 重要概念需配有数学公式和可视化图解
4. 面试题需提供"标准回答框架"和"加分回答模板"

---

## 🎓 推荐资源

### 在线平台
- **LeetCode**: https://leetcode.cn/ (算法题库)
- **牛客网**: https://www.nowcoder.com/ (大厂真题)
- **AcWing**: https://www.acwing.com/ (算法课程)
- **洛谷**: https://www.luogu.com.cn/ (OJ 平台)

### 学习资料
- **Transformer 原论文**: "Attention is All You Need" (Vaswani et al., 2017)
- **The Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **Stanford CS224N**: NLP with Deep Learning
- **李宏毅机器学习**: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php

### 书籍推荐
- 《动手学深度学习》(Dive into Deep Learning)
- 《算法导论》(Introduction to Algorithms)
- 《剑指 Offer》(面试题集)
- 《深度学习》(Ian Goodfellow)

---

## 📧 联系方式

如有问题或建议, 欢迎提 Issue 或 Pull Request!

---

## 📄 许可证

本项目仅用于学习和面试准备, 请勿用于商业用途.

---

## ⭐ Star History

如果这个仓库对你有帮助, 请给个 Star ⭐️!

---

**最后更新时间**: 2025年11月8日

**祝你面试顺利! 🚀**
