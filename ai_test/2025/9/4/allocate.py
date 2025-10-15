'''
题目描述：
大模型训练通常采用数据并行的方式，处理大规模数据集（样本），加速训练过程，具体的：
假设有 n 个 NPU, m 个样本，把 m 个样本分配给 n 个 NPU, 每个 NPU 上有一份完整模型，
各自计算自己的样本数据，其中 m≥n, 保证每个 NPU 至少分到一个样本，且样本不能切分，
一个样本必须完整的被分到 NPU 上，每个 NPU 的运行时间跟所分到的样本的长度和呈正相关。
如果每个 NPU 上的样本长度和相差较大，会形成木桶效应，执行快的 NPU 等待执行慢的 NPU,
最终执行时间由最大样本长度的 NPU 决定。

试着编写一段程序对样本进行均衡分配，设 n 个 NPU 上所得的最大的样本和为l_max, 使l_max最小, 即为求min(l_max).
'''

'''
输入描述

第一行为 1 个整数 n（0<n<1000），表示 NPU 的个数
第二行为 1 个整数 m（0<m<10000），表示样本的个数
第三行有 m 个处于区间 [1,100000] 之内的整数，表示 m 个样本中每个样本的长度

输出描述

输出 1 个整数（行尾没有空格），该数字表示 min(l_max)
'''

from typing import List
import heapq

def group_samples(group_num:int, sample_num:int, simple_lens: List[int]):
    n, m = group_num, sample_num
    a = simple_lens

    if m == 0:
        print(0)
        return
    
    load = [0] * n
    a.sort(reverse=True)

    ans = 0
    for x in a:
        cur = heapq.heappop(load)
        cur += x
        ans = max(ans, cur)
        heapq.heappush(load, cur)

    print(ans)

if __name__ == "__main__":
    n = int(input().strip())
    m = int(input().strip())
    lens = list(map(int, input().strip().split()))
    group_samples(n, m, lens)
