r"""
题目:
数据中心要进行散热，依赖很多因素，在实际场景中，需要细化到调低1度，调高0.5度，维持不变，调高0.5，调高1度等档位
对逻辑回归进行改造，将其输出更改为softmax可满足要求。
优化后模型：O = XW +b, P = softmax(O)
X \in R(m,n)表示m个样本，n个特征
W \in R(n,k)表示n个特征，k个类别
b \in R(1,k)表示k个类别的偏置
P \in R(m,k)表示预测的概率，取概率最大的作为输出档位
输入描述:   
    1.第一行数据schema，表示特征个数n，类别个数k，待预测样本数m，数据均为int
    2.后续多行是k个分类的训练原本（一行一个样本），和m条带预测样本（一行一个样本）；数据为float类型
输出描述:
    每个待预测样本所属的分类，一行输出一个样本的预测结果
示例1
输入
2 3 2 3 2 3
9 95
33 53
53 55
69 21
68 31
70 85 
80 83
25 70
45 30
79 86
输出
0
1
2
"""

import sys
from typing import List, Tuple

def read_ints(line: str) -> List[int]:
    return [int(x) for x in line.strip().split()] if line is not None else []

def read_floats(line: str) -> List[float]:
    return [float(x) for x in line.strip().split()] if line is not None else []


def solve():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    # 跳过开头可能的空行
    idx = 0
    while idx < len(data) and not data[idx].strip():
        idx += 1
    if idx >= len(data):
        return

    schema = read_ints(data[idx])
    idx += 1

    # 期望格式: n, k, c1..ck, m
    if len(schema) < 3:
        print("", end="")
        return
    n = schema[0]
    k = schema[1]

    if len(schema) != 2 + k + 1:
        # 输入不符合期望，尽量兼容另一种: n k m （无类别计数，则默认每类相同数量，需从数据行推断）
        if len(schema) == 3:
            m = schema[2]
            # 尝试从剩余行数推断每类样本数量（假设总行数 = 训练 + m）
            remaining = len(data) - idx
            train_total = remaining - m
            if train_total < 0 or train_total % k != 0:
                raise ValueError("无法从输入推断每类训练样本数量")
            per_class = train_total // k
            counts = [per_class] * k
        else:
            raise ValueError("首行应为: n k c1 c2 ... ck m 或 n k m")
    else:
        counts = schema[2:2 + k]
        m = schema[2 + k]

    train_total = sum(counts)

    # 读取训练样本（按类别顺序依次给出）
    train: List[List[float]] = []
    for _ in range(train_total):
        # 跳过空行
        while idx < len(data) and not data[idx].strip():
            idx += 1
        if idx >= len(data):
            raise ValueError("训练样本数量不足")
        row = read_floats(data[idx])
        idx += 1
        if len(row) != n:
            raise ValueError("训练样本维度与n不一致")
        train.append(row)

    # 读取待预测样本
    tests: List[List[float]] = []
    for _ in range(m):
        while idx < len(data) and not data[idx].strip():
            idx += 1
        if idx >= len(data):
            raise ValueError("待预测样本数量不足")
        row = read_floats(data[idx])
        idx += 1
        if len(row) != n:
            raise ValueError("预测样本维度与n不一致")
        tests.append(row)

    # 计算每个类别的均值向量 μ_c
    means = [[0.0 for _ in range(n)] for _ in range(k)]
    start = 0
    for c in range(k):
        cnt = counts[c]
        if cnt == 0:
            continue
        for i in range(cnt):
            x = train[start + i]
            for j in range(n):
                means[c][j] += x[j]
        for j in range(n):
            means[c][j] /= cnt
        start += cnt

    # 构造线性分类器: 对于每个类别 c，logit_c = x · μ_c + b_c， 其中 b_c = -0.5 * ||μ_c||^2
    bs = []
    for c in range(k):
        s = 0.0
        for j in range(n):
            s += means[c][j] * means[c][j]
        bs.append(-0.5 * s)

    # 预测
    out_lines: List[str] = []
    for x in tests:
        # 计算每个类别的打分（无需显式softmax，argmax logits 等价）
        best_c = 0
        best_score = -float('inf')
        for c in range(k):
            score = bs[c]
            # x · μ_c
            for j in range(n):
                score += x[j] * means[c][j]
            if score > best_score:
                best_score = score
                best_c = c
        out_lines.append(str(best_c))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    solve()