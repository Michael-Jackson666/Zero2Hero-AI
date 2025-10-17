r"""
题目:
给定n个基站的二维坐标，使用k-means算法将基站分为k类，再通过计算每个簇的轮廓系数
识别信号覆盖最差的簇，并在该簇中心新增基站以优化型号覆盖
算法过程:
1.前k个基站作为初始聚类中心，执行k-means算法，k-means结束条件：最大迭代次数100或所有簇中心移动距离都不大于1e-6
2.计算每个簇的轮廓系数，识别出轮廓系数最小的簇
3.输出该簇中心坐标(保留两位小数)，作为新增基站位置

输入:
第一行: n k 以空格分开，n\in[1,500],m\in[1,120]
接下来n行: 每行两个整数x,y表示一个基站坐标，x,\in[0,5000],y\in[0,3000]
输出:
新增基站坐标x,y，保留小数点后2位,用RoundingMode.HALF_EVEN四舍五入

例子1:
输入:
6 2
0 0
1 1
2 2
10 10
11 11
5 5
输出:
8.67,8.67

例子2:
输入:
4 2
0 0
0 1
1 0
10 10
输出:
0.33,0.33
"""

import sys
from decimal import Decimal, ROUND_HALF_EVEN
from typing import List, Tuple

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点之间的欧几里得距离"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def kmeans(points: List[Tuple[float, float]], k: int, max_iter: int = 100, tol: float = 1e-6) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    K-means聚类算法
    返回: (每个点的簇标签, 每个簇的中心)
    """
    n = len(points)
    
    # 初始化：前k个点作为初始中心
    centers = [points[i] for i in range(k)]
    labels = [0] * n
    
    for iteration in range(max_iter):
        # 分配每个点到最近的中心
        for i in range(n):
            min_dist = float('inf')
            best_cluster = 0
            for j in range(k):
                dist = distance(points[i], centers[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            labels[i] = best_cluster
        
        # 计算新的中心
        new_centers = []
        max_move = 0.0
        
        for j in range(k):
            # 找到属于簇j的所有点
            cluster_points = [points[i] for i in range(n) if labels[i] == j]
            
            if len(cluster_points) == 0:
                # 如果簇为空，保持原中心
                new_centers.append(centers[j])
            else:
                # 计算均值
                cx = sum(p[0] for p in cluster_points) / len(cluster_points)
                cy = sum(p[1] for p in cluster_points) / len(cluster_points)
                new_centers.append((cx, cy))
                
                # 计算中心移动距离
                move = distance(centers[j], new_centers[j])
                max_move = max(max_move, move)
        
        centers = new_centers
        
        # 检查收敛
        if max_move <= tol:
            break
    
    return labels, centers

def silhouette_coefficient(points: List[Tuple[float, float]], labels: List[int], k: int) -> List[float]:
    """
    计算每个簇的平均轮廓系数
    轮廓系数 = (b - a) / max(a, b)
    a: 簇内平均距离
    b: 到最近其他簇的平均距离
    """
    n = len(points)
    
    # 为每个簇计算轮廓系数
    cluster_silhouettes = []
    
    for cluster_id in range(k):
        cluster_indices = [i for i in range(n) if labels[i] == cluster_id]
        
        if len(cluster_indices) <= 1:
            # 簇只有一个点或为空，轮廓系数为0
            cluster_silhouettes.append(0.0)
            continue
        
        silhouettes = []
        
        for i in cluster_indices:
            # 计算a: 簇内平均距离
            a = 0.0
            for j in cluster_indices:
                if i != j:
                    a += distance(points[i], points[j])
            a /= (len(cluster_indices) - 1)
            
            # 计算b: 到最近其他簇的平均距离
            b = float('inf')
            for other_cluster in range(k):
                if other_cluster == cluster_id:
                    continue
                
                other_indices = [idx for idx in range(n) if labels[idx] == other_cluster]
                if len(other_indices) == 0:
                    continue
                
                avg_dist = sum(distance(points[i], points[j]) for j in other_indices) / len(other_indices)
                b = min(b, avg_dist)
            
            # 计算轮廓系数
            if b == float('inf'):
                s = 0.0
            else:
                s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
            
            silhouettes.append(s)
        
        # 簇的平均轮廓系数
        cluster_silhouettes.append(sum(silhouettes) / len(silhouettes))
    
    return cluster_silhouettes

def round_half_even(value: float, decimal_places: int = 2) -> str:
    """使用ROUND_HALF_EVEN模式四舍五入"""
    d = Decimal(str(value))
    rounded = d.quantize(Decimal(10) ** -decimal_places, rounding=ROUND_HALF_EVEN)
    return format(rounded, f'.{decimal_places}f')

def solve():
    lines = sys.stdin.read().strip().splitlines()
    
    # 读取n和k
    n, k = map(int, lines[0].split())
    
    # 读取基站坐标
    points = []
    for i in range(1, n + 1):
        x, y = map(float, lines[i].split())
        points.append((x, y))
    
    # 执行k-means聚类
    labels, centers = kmeans(points, k, max_iter=100)
    
    # 计算每个簇的轮廓系数
    silhouettes = silhouette_coefficient(points, labels, k)
    
    # 计算每个簇的大小
    cluster_sizes = [sum(1 for i in range(n) if labels[i] == c) for c in range(k)]
    
    # 找到轮廓系数最小的簇（信号覆盖最差），但排除单点簇
    valid_clusters = [i for i in range(k) if cluster_sizes[i] > 1]
    if len(valid_clusters) == 0:
        # 如果都是单点簇，选择第一个
        worst_cluster = 0
    else:
        worst_cluster = min(valid_clusters, key=lambda i: silhouettes[i])
    
    # 输出该簇的中心坐标
    cx, cy = centers[worst_cluster]
    cx_str = round_half_even(cx, 2)
    cy_str = round_half_even(cy, 2)
    
    print(f"{cx_str},{cy_str}")

if __name__ == "__main__":
    solve()