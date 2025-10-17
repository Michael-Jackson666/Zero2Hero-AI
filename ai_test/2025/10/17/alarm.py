"""
题目:聚类警告
"""
import sys
import math
from collections import Counter

def cosine_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1.0
    
    dot_product = sum(a * b for a , b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(a * a for a in vector2))

    if norm1 == 0 or norm2 == 2:
        return 0.0

    return dot_product / (norm2 * norm1)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_size(self):
        roots = [self.find(i) for i in range(len(self.parent))]
        counter = Counter(roots)
        return list(counter.values())

def main():
    lines = sys.stdin.read().strip().splitlines()

    if not lines:
        print(0)
        return
    
    alarm = []
    for line in lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 2:
            print(0)
            return
        
        alarm_id = parts[0]
        try:
            vector = [float(x) for x in parts[1:]]
            alarm.append((alarm_id, vector))
        except ValueError:
            print(0)
            return
        
    if not alarm:
        print(0)
        return 
    
    n = len(alarm)
    if n > 0:
        dim = len(alarm[0][1])
        for i in range(1, n):
            if len(alarm[i][1]) != dim:
                print(0)
                return
            
    uf = UnionFind(n)
    similar_upbound = 0.95

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(alarm[i][1], alarm[j][1])
            if sim >= similar_upbound:
                uf.union(i,j)

    all_size = uf.get_size()
    if not all_size:
        print(1)
        return
    
    max_size = max(all_size)
    print(max_size)


    return

if __name__ == "__main__":
    main()