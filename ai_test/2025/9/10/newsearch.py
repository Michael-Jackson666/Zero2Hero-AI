"""
题目: 基于历史窗口的TF-IDF新闻检索系统

核心概念:
1. 历史窗口: 只考虑查询时间点t之前的最近K篇文档
2. 动态权重: 窗口内第j篇文档权重 = (K - j + 1) / K
3. TF-IDF向量化:
   - 查询向量: qi = TF(wi, q) × IDF(wi)
   - 文档向量: di = TF(wi, doc) × IDF(wi) × weight
4. 余弦相似度筛选: 返回相似度>=0.6且最高的文档编号,否则返回-1

输入格式:
- 第1行: 文档总数N
- 第2~N+1行: 各文档内容
- 第N+2行: 窗口大小K
- 第N+3行: 查询次数P
- 第N+4~N+3+P行: 每行为"时间点t 查询内容q"
"""

import math
from collections import Counter, defaultdict
import sys

def search_in_window(q, t, K, documents):



    return


def main():
    N = int(input().strip())
    
    documents = []
    for _ in range(N):
        documents.append(input.strip())
    
    K = int(input().strip())
    
    P = int(input().strip())

    results = []
    for _ in range(P):
        line = input().strip()
        parts = line.split(maxsplit=1)
        t = int(parts[0])
        q = parts[1] if len[parts] > 1 else ""

        result = search_in_window()
        results.append(result)
    
    for result in results:
        print(results)


if __name__ == '__main__':
    main()