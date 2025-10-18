"""
题目: 聚类警告

功能概述:
- 从标准输入读取多行报警记录, 每行包含: "alarm_id feature1 feature2 ... featureD"
- 使用余弦相似度对报警进行“相似度聚类”: 若两条报警的相似度 >= 阈值(0.95), 则视为同一簇 (使用并查集 Union-Find 合并)
- 输出最大簇的大小(簇内报警的数量)

输入格式示例:
    A1 0.1 0.2 0.3
    A2 0.1 0.2 0.31
    A3 0.9 0.8 0.7

输出:
    整数, 表示最大簇的大小。

健壮性:
- 自动忽略空行
- 若某行数据非法或维度不一致, 输出 0
"""
import sys
import math
from collections import Counter

def cosine_similarity(vector1, vector2):
    """计算两个向量的余弦相似度。

    参数:
        vector1, vector2: List[float] 或等价的可迭代数值序列, 维度需一致。

    返回:
        - 若维度不一致返回 -1.0
        - 若任一向量为全零(避免除零), 返回 0.0
        - 否则返回 [ -1.0, 1.0 ] 范围内的相似度值
    """
    if len(vector1) != len(vector2):
        return -1.0
    
    dot_product = sum(a * b for a , b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(a * a for a in vector2))

    # 防止除以 0 的情况: 只要任一向量范数为 0, 认为相似度为 0
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm2 * norm1)

class UnionFind:
    """并查集(不相交集合)结构, 用于高效地进行合并与连通查询。"""
    def __init__(self, n):
        # 初始时每个元素的父节点是自己, 秩(rank)用于按秩合并优化
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """查找元素 x 的根节点, 带路径压缩优化。"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """合并包含 x 与 y 的两个集合。"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # 按秩合并: 将较小秩的根挂到较大秩的根上, 以降低树的高度
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_size(self):
        """返回每个连通分量(簇)的大小列表。"""
        roots = [self.find(i) for i in range(len(self.parent))]
        counter = Counter(roots)
        return list(counter.values())

def main():
    """读取输入, 构建特征向量, 基于余弦相似度进行聚类, 输出最大簇大小。"""
    # 读取所有行, 并去除首尾空白字符
    lines = sys.stdin.read().strip().splitlines()

    if not lines:
        # 没有任何输入时, 定义输出为 0
        print(0)
        return
    
    alarm = []  # 存放 (alarm_id, feature_vector)
    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 2:
            # 至少需要一个 id 和一个特征
            print(0)
            return
        
        alarm_id = parts[0]
        try:
            vector = [float(x) for x in parts[1:]]
            alarm.append((alarm_id, vector))
        except ValueError:
            # 存在非数字特征
            print(0)
            return
        
    if not alarm:
        print(0)
        return 
    
    # 校验所有特征向量维度一致
    n = len(alarm)
    if n > 0:
        dim = len(alarm[0][1])
        for i in range(1, n):
            if len(alarm[i][1]) != dim:
                print(0)
                return
            
    uf = UnionFind(n)
    # 相似度阈值: 大于等于该值认为属于同一簇
    similar_upbound = 0.95

    # 两两计算相似度, 若满足阈值则合并
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(alarm[i][1], alarm[j][1])
            if sim >= similar_upbound:
                uf.union(i,j)

    # 统计所有簇的大小, 输出最大值
    all_size = uf.get_size()
    if not all_size:
        # 正常情况下, 有 n 个元素则至少有一个簇
        print(1)
        return
    
    max_size = max(all_size)
    print(max_size)


    return

if __name__ == "__main__":
    main()