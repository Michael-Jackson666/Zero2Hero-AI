"""
问题描述：聚类警告
本程序解决的是警告聚类问题。给定多个警告，每个警告包含一个ID和一个特征向量。
程序的目标是将相似度高于阈值的警告聚类在一起，并输出最大聚类的大小。
相似度使用余弦相似度计算，聚类使用并查集实现。

输入格式：
每行包含一个警告ID和一组浮点数（特征向量）
示例：
warning1 0.1 0.2 0.3
warning2 0.2 0.3 0.4

输出：
最大聚类的大小（整数）
"""
import sys
import math
from collections import Counter

def cosine_similarity(vector1, vector2):
    """
    计算两个向量之间的余弦相似度
    
    参数：
        vector1 (list): 第一个向量
        vector2 (list): 第二个向量
    
    返回：
        float: 余弦相似度值，范围在[-1, 1]之间
              如果向量长度不同，返回-1.0
              如果任一向量的范数为0，返回0.0
    """
    # 检查向量长度是否相同
    if len(vector1) != len(vector2):
        return -1.0
    
    # 计算点积和向量范数
    dot_product = sum(a * b for a , b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(a * a for a in vector2))

    # 处理零向量的情况
    if norm1 == 0 or norm2 == 2:
        return 0.0

    # 计算余弦相似度
    return dot_product / (norm2 * norm1)

class UnionFind:
    """
    并查集类实现，用于高效地维护元素之间的连接关系
    
    属性：
        parent (list): 存储每个节点的父节点
        rank (list): 存储每个节点的秩（树的高度的上界）
    """
    def __init__(self, n):
        """
        初始化大小为n的并查集
        
        参数：
            n (int): 元素个数
        """
        self.parent = list(range(n))  # 初始时每个元素的父节点是自己
        self.rank = [0] * n           # 初始时每个节点的秩为0
    
    def find(self, x):
        """
        查找元素x所属的集合的根节点，并进行路径压缩
        
        参数：
            x (int): 要查找的元素索引
            
        返回：
            int: 元素x所属集合的根节点
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        """
        合并元素x和y所属的集合
        
        参数：
            x (int): 第一个元素的索引
            y (int): 第二个元素的索引
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:  # 如果已经在同一个集合中
            return
        
        # 按秩合并，将较小的树连接到较大的树上
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1  # 秩相等时，新根节点的秩加1

    def get_size(self):
        """
        获取所有集合的大小
        
        返回：
            list: 包含所有集合大小的列表
        """
        roots = [self.find(i) for i in range(len(self.parent))]  # 获取所有元素的根节点
        counter = Counter(roots)  # 统计每个根节点出现的次数
        return list(counter.values())  # 返回所有集合的大小

def main():
    """
    主函数：处理输入数据，执行聚类，输出结果
    """
    # 读取输入数据
    lines = sys.stdin.read().strip().splitlines()

    # 处理空输入的情况
    if not lines:
        print(0)
        return
    
    # 解析每行数据，提取警告ID和特征向量
    alarm = []
    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 2:  # 确保每行至少包含ID和一个特征值
            print(0)
            return
        
        alarm_id = parts[0]
        try:
            vector = [float(x) for x in parts[1:]]  # 将特征值转换为浮点数
            alarm.append((alarm_id, vector))
        except ValueError:  # 处理特征值无法转换为浮点数的情况
            print(0)
            return
        
    # 处理没有有效警告的情况
    if not alarm:
        print(0)
        return 
    
    # 验证所有特征向量的维度是否相同
    n = len(alarm)
    if n > 0:
        dim = len(alarm[0][1])
        for i in range(1, n):
            if len(alarm[i][1]) != dim:
                print(0)
                return
            
    # 初始化并查集
    uf = UnionFind(n)
    similar_upbound = 0.95  # 相似度阈值

    # 计算所有警告对之间的相似度，并合并相似的警告
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(alarm[i][1], alarm[j][1])
            if sim >= similar_upbound:
                uf.union(i,j)

    # 获取所有聚类的大小
    all_size = uf.get_size()
    if not all_size:
        print(1)
        return
    
    # 输出最大聚类的大小
    max_size = max(all_size)
    print(max_size)

    return

if __name__ == "__main__":
    main()

"""
时间复杂度分析：
1. cosine_similarity函数：
   - 对于维度为d的向量，计算点积和范数的时间复杂度为O(d)

2. UnionFind类：
   - 初始化：O(n)，n为元素个数
   - find操作：平均O(α(n))，其中α(n)是阿克曼函数的反函数，实际上近似为常数
   - union操作：O(α(n))
   - get_size操作：O(n)

3. main函数：
   - 读取和处理输入：O(m)，m为输入行数
   - 验证维度：O(n)
   - 计算相似度和聚类：O(n²d)，其中n是警告数量，d是特征向量维度
   - 获取最大聚类大小：O(n)

总体时间复杂度：O(n²d)，主要瓶颈在于需要计算所有警告对之间的相似度

空间复杂度分析：
1. 存储输入数据：O(nd)，其中n是警告数量，d是特征向量维度
2. UnionFind数据结构：O(n)
3. 其他辅助空间：O(n)

总体空间复杂度：O(nd)

优化建议：
1. 如果警告数量很大，可以考虑使用近似算法或降维技术来减少计算量
2. 可以使用并行计算来加速相似度计算
3. 如果内存限制严格，可以考虑流式处理输入数据
"""