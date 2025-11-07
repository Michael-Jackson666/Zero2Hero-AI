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
时间复杂度分析:
1. cosine_similarity函数:
    - 对于维度为d的向量, 计算点积和范数的时间复杂度为O(d)

2. UnionFind类:
    - 初始化: O(n), n为元素个数
    - find操作: 平均O(α(n)), 其中α(n)是阿克曼函数的反函数, 实际上近似为常数
    - union操作: O(α(n))
    - get_size操作: O(n)

3. main函数:
    - 读取和处理输入: O(m), m为输入行数
    - 验证维度: O(n)
    - 计算相似度和聚类: O(n²d), 其中n是警告数量, d是特征向量维度
    - 获取最大聚类大小: O(n)

总体时间复杂度: O(n²d), 主要瓶颈在于需要计算所有警告对之间的相似度

空间复杂度分析:
1. 存储输入数据: O(nd), 其中n是警告数量, d是特征向量维度
2. UnionFind数据结构: O(n)
3. 其他辅助空间: O(n)

总体空间复杂度: O(nd)

优化建议:
1. 如果警告数量很大, 可以考虑使用近似算法或降维技术来减少计算量
2. 可以使用并行计算来加速相似度计算
3. 如果内存限制严格, 可以考虑流式处理输入数据

================================================================================
求解思路:

本题的核心是将相似的警告聚类在一起, 最终输出最大聚类的大小. 解决方案分为以下几个步骤:

1. 相似度计算 (Cosine Similarity):
   - 问题分析: 需要判断两个警告是否相似, 使用余弦相似度作为度量标准
   - 数学原理: 余弦相似度 = (向量A · 向量B) / (||向量A|| * ||向量B||)
   - 实现细节:
     * 计算两个向量的点积 (dot product)
     * 分别计算两个向量的L2范数 (欧几里得范数)
     * 相除得到余弦相似度值, 范围在[-1, 1]之间
     * 值越接近1表示越相似, 越接近-1表示越不相似
   - 边界情况处理:
     * 向量长度不同: 返回-1.0 (无法比较)
     * 零向量: 返回0.0 (避免除零错误)

2. 并查集 (Union-Find / Disjoint Set):
   - 数据结构选择: 使用并查集来高效维护警告之间的连通性关系
   - 核心思想: 将每个聚类看作一棵树, 树根代表该聚类的代表元素
   - 关键操作:
     
     a) 初始化 (Initialize):
        - 创建parent数组: 初始时每个元素的父节点是自己
        - 创建rank数组: 记录树的高度, 用于优化合并操作
        - 时间复杂度: O(n)
     
     b) 查找 (Find):
        - 目的: 找到元素所属集合的根节点
        - 路径压缩优化: 在查找过程中, 将路径上的所有节点直接连接到根节点
        - 示例: 如果 1->2->3->4, 查找1后变成 1->4, 2->4, 3->4
        - 时间复杂度: 平均O(α(n)), α(n)是阿克曼函数的反函数, 实际上接近常数
     
     c) 合并 (Union):
        - 目的: 将两个元素所属的集合合并
        - 按秩合并优化: 将深度小的树连接到深度大的树上, 保持树的平衡
        - 过程:
          1. 找到两个元素的根节点
          2. 比较两个根节点的秩
          3. 将秩小的根连接到秩大的根下
          4. 如果秩相等, 任选一个作为新根, 并将其秩+1
        - 时间复杂度: O(α(n))
     
     d) 统计集合大小 (Get Size):
        - 遍历所有元素, 找到它们的根节点
        - 使用Counter统计每个根节点出现的次数
        - 返回所有集合的大小列表
        - 时间复杂度: O(n)

3. 聚类过程 (Clustering):
   - 算法流程:
     Step 1: 读取并解析所有警告数据 (ID + 特征向量)
     Step 2: 验证所有特征向量的维度是否一致
     Step 3: 初始化并查集, 每个警告初始为独立的集合
     Step 4: 双重循环遍历所有警告对 (i, j), 其中i < j:
             - 计算警告i和警告j之间的余弦相似度
             - 如果相似度 >= 阈值(0.95), 则合并这两个警告到同一聚类
     Step 5: 统计所有聚类的大小
     Step 6: 输出最大聚类的大小

4. 时间复杂度瓶颈分析:
   - 主要瓶颈: 计算所有警告对之间的相似度
   - 警告对的数量: C(n,2) = n*(n-1)/2 ≈ O(n²)
   - 每次相似度计算: O(d), d为特征向量维度
   - 总时间复杂度: O(n²d)
   - 实际影响:
     * n=100, d=10: 约50,000次计算
     * n=1000, d=10: 约5,000,000次计算
     * n=10000, d=10: 约500,000,000次计算 (可能需要优化)

5. 为什么选择并查集?
   - 优势对比:
     * 邻接表/邻接矩阵: 需要O(n²)空间存储所有边, 且查找连通分量较慢
     * DFS/BFS: 每次需要O(n+m)时间遍历图, m为边数
     * 并查集: 空间O(n), 单次操作近乎O(1), 特别适合动态连接问题
   - 适用场景: 需要频繁地合并集合和查询元素是否在同一集合中

6. 边界情况处理:
   - 空输入: 输出0
   - 无效数据格式: 输出0
   - 特征值无法转换为浮点数: 输出0
   - 不同警告的特征向量维度不一致: 输出0
   - 所有警告互不相似: 输出1 (最大聚类大小为1)
   - 所有警告高度相似: 输出n (所有警告在同一聚类)

7. 算法正确性证明:
   - 传递性: 如果A与B相似, B与C相似, 则A和C应该在同一聚类中
   - 并查集保证: 通过union操作自动维护这种传递闭包关系
   - 例子: 
     * 警告1与警告2相似 -> union(1,2)
     * 警告2与警告3相似 -> union(2,3)
     * 结果: find(1)==find(2)==find(3), 三者在同一聚类中

8. 可能的优化方向:
   - 空间优化: LSH (Locality-Sensitive Hashing) 减少比较次数
   - 时间优化: 使用KD-Tree或Ball-Tree进行近邻搜索
   - 并行优化: 相似度计算可以并行化
   - 近似优化: 使用采样或者只比较最相似的top-k对
"""