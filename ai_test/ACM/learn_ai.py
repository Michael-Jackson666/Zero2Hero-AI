"""
ACM模式输入输出 - 完整示例集合

本文件包含大厂机考中常见的ACM模式输入输出场景
每个示例都有详细注释和测试用例
"""

import sys
from collections import defaultdict, Counter


# ============================================================
# 示例1: 单行单个整数
# ============================================================
def example_1_single_integer():
    """
    题目: 读取一个整数n，输出n的平方
    
    输入示例:
    5
    
    输出示例:
    25
    """
    n = int(input())
    result = n * n
    print(result)


# ============================================================
# 示例2: 单行多个整数（空格分隔）
# ============================================================
def example_2_multiple_integers_one_line():
    """
    题目: 读取两个整数a和b，输出它们的和
    
    输入示例:
    3 7
    
    输出示例:
    10
    """
    # 方法1: split + map
    a, b = map(int, input().split())
    print(a + b)
    
    # 方法2: 列表推导
    # numbers = [int(x) for x in input().split()]
    # print(numbers[0] + numbers[1])


# ============================================================
# 示例3: 读取不定长数组（单行）
# ============================================================
def example_3_array_single_line():
    """
    题目: 读取一行整数，输出它们的和
    
    输入示例:
    1 2 3 4 5
    
    输出示例:
    15
    """
    arr = list(map(int, input().split()))
    print(sum(arr))


# ============================================================
# 示例4: 第一行是数组长度，第二行是数组内容
# ============================================================
def example_4_array_with_length():
    """
    题目: 第一行是n，第二行是n个整数，输出最大值
    
    输入示例:
    5
    3 1 4 1 5
    
    输出示例:
    5
    """
    n = int(input())
    arr = list(map(int, input().split()))
    print(max(arr))


# ============================================================
# 示例5: 多行输入，每行一个数据
# ============================================================
def example_5_multiple_lines():
    """
    题目: 第一行是n，接下来n行每行一个整数，输出所有数的和
    
    输入示例:
    3
    10
    20
    30
    
    输出示例:
    60
    """
    n = int(input())
    total = 0
    for _ in range(n):
        total += int(input())
    print(total)


# ============================================================
# 示例6: 多行输入，每行多个数据
# ============================================================
def example_6_matrix_input():
    """
    题目: 读取n行，每行m个整数，输出所有数的和
    
    输入示例:
    3 4
    1 2 3 4
    5 6 7 8
    9 10 11 12
    
    输出示例:
    78
    """
    n, m = map(int, input().split())
    total = 0
    for _ in range(n):
        row = list(map(int, input().split()))
        total += sum(row)
    print(total)


# ============================================================
# 示例7: 多组测试用例（无结束标志）
# ============================================================
def example_7_multiple_test_cases():
    """
    题目: 多组测试用例，每组两个整数，输出它们的和
    
    输入示例:
    1 2
    3 4
    5 6
    
    输出示例:
    3
    7
    11
    """
    try:
        while True:
            line = input().strip()
            if not line:  # 空行则跳过
                break
            a, b = map(int, line.split())
            print(a + b)
    except EOFError:
        pass


# ============================================================
# 示例8: 多组测试用例（0 0结束）
# ============================================================
def example_8_test_cases_with_terminator():
    """
    题目: 多组测试用例，每组两个整数，当读到0 0时结束
    
    输入示例:
    1 2
    3 4
    0 0
    
    输出示例:
    3
    7
    """
    while True:
        a, b = map(int, input().split())
        if a == 0 and b == 0:
            break
        print(a + b)


# ============================================================
# 示例9: 字符串处理
# ============================================================
def example_9_string_processing():
    """
    题目: 读取一行字符串，统计每个字符出现的次数
    
    输入示例:
    hello world
    
    输出示例:
    h:1 e:1 l:3 o:2 w:1 r:1 d:1
    """
    s = input().strip()
    counter = Counter(s.replace(' ', ''))  # 去除空格
    # 按字符顺序输出（或按字典序）
    result = ' '.join([f"{char}:{count}" for char, count in sorted(counter.items())])
    print(result)


# ============================================================
# 示例10: 读取整个输入（sys.stdin.read）
# ============================================================
def example_10_read_all_input():
    """
    题目: 读取所有输入，处理后输出
    适用于复杂的多行输入场景
    
    输入示例:
    3
    apple banana cherry
    orange grape
    kiwi
    
    输出示例:
    6
    """
    data = sys.stdin.read().strip()
    lines = data.split('\n')
    n = int(lines[0])
    
    word_count = 0
    for i in range(1, n + 1):
        words = lines[i].split()
        word_count += len(words)
    
    print(word_count)


# ============================================================
# 示例11: 二维数组/矩阵输入
# ============================================================
def example_11_2d_array():
    """
    题目: 读取一个n×n矩阵，输出对角线元素之和
    
    输入示例:
    3
    1 2 3
    4 5 6
    7 8 9
    
    输出示例:
    15
    """
    n = int(input())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    # 主对角线之和
    diagonal_sum = sum(matrix[i][i] for i in range(n))
    print(diagonal_sum)


# ============================================================
# 示例12: 输出格式化（多个结果）
# ============================================================
def example_12_formatted_output():
    """
    题目: 读取n个数，输出最小值、最大值、平均值
    
    输入示例:
    5
    1 2 3 4 5
    
    输出示例:
    1 5 3.0
    或
    min=1, max=5, avg=3.0
    """
    n = int(input())
    arr = list(map(int, input().split()))
    
    min_val = min(arr)
    max_val = max(arr)
    avg_val = sum(arr) / len(arr)
    
    # 方式1: 空格分隔
    print(min_val, max_val, avg_val)
    
    # 方式2: 格式化字符串
    # print(f"min={min_val}, max={max_val}, avg={avg_val:.1f}")


# ============================================================
# 示例13: 处理浮点数
# ============================================================
def example_13_floating_point():
    """
    题目: 读取两个浮点数，输出它们的和（保留2位小数）
    
    输入示例:
    3.14 2.86
    
    输出示例:
    6.00
    """
    a, b = map(float, input().split())
    result = a + b
    print(f"{result:.2f}")
    # 或者: print("%.2f" % result)


# ============================================================
# 示例14: 逗号分隔的输入
# ============================================================
def example_14_comma_separated():
    """
    题目: 读取逗号分隔的整数，输出它们的和
    
    输入示例:
    1,2,3,4,5
    
    输出示例:
    15
    """
    arr = list(map(int, input().split(',')))
    print(sum(arr))


# ============================================================
# 示例15: 混合输入（字符串+数字）
# ============================================================
def example_15_mixed_input():
    """
    题目: 第一行是名字，第二行是年龄，输出信息
    
    输入示例:
    Alice
    25
    
    输出示例:
    Alice is 25 years old
    """
    name = input().strip()
    age = int(input())
    print(f"{name} is {age} years old")


# ============================================================
# 示例16: 处理多个查询
# ============================================================
def example_16_multiple_queries():
    """
    题目: 第一行n和q，第二行n个数，接下来q行每行一个查询索引
    
    输入示例:
    5 3
    10 20 30 40 50
    0
    2
    4
    
    输出示例:
    10
    30
    50
    """
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    
    for _ in range(q):
        idx = int(input())
        print(arr[idx])


# ============================================================
# 示例17: 输出数组（空格分隔）
# ============================================================
def example_17_output_array():
    """
    题目: 读取n个数，输出它们的逆序
    
    输入示例:
    5
    1 2 3 4 5
    
    输出示例:
    5 4 3 2 1
    """
    n = int(input())
    arr = list(map(int, input().split()))
    arr.reverse()
    
    # 方式1: join
    print(' '.join(map(str, arr)))
    
    # 方式2: * 解包
    # print(*arr)


# ============================================================
# 示例18: 输出二维数组/矩阵
# ============================================================
def example_18_output_matrix():
    """
    题目: 读取并转置一个矩阵
    
    输入示例:
    2 3
    1 2 3
    4 5 6
    
    输出示例:
    1 4
    2 5
    3 6
    """
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    # 转置
    transposed = [[matrix[i][j] for i in range(n)] for j in range(m)]
    
    # 输出
    for row in transposed:
        print(' '.join(map(str, row)))


# ============================================================
# 示例19: 读取到文件结束（大数据量）
# ============================================================
def example_19_read_until_eof():
    """
    题目: 读取所有行，每行处理后输出
    
    输入示例:
    hello
    world
    python
    
    输出示例:
    HELLO
    WORLD
    PYTHON
    """
    try:
        while True:
            line = input().strip()
            print(line.upper())
    except EOFError:
        pass


# ============================================================
# 示例20: 综合示例 - 图的邻接表输入
# ============================================================
def example_20_graph_adjacency_list():
    """
    题目: 读取图的邻接表表示
    第一行: n个节点, m条边
    接下来m行: 每行两个数表示一条边
    
    输入示例:
    4 5
    0 1
    0 2
    1 2
    1 3
    2 3
    
    输出示例:
    节点0的邻居: [1, 2]
    节点1的邻居: [0, 2, 3]
    节点2的邻居: [0, 1, 3]
    节点3的邻居: [1, 2]
    """
    n, m = map(int, input().split())
    graph = defaultdict(list)
    
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)  # 无向图
    
    # 输出
    for node in range(n):
        neighbors = sorted(graph[node])
        print(f"节点{node}的邻居: {neighbors}")


# ============================================================
# 主函数 - 选择运行哪个示例
# ============================================================
def main():
    """
    主函数: 取消注释你想运行的示例
    """
    # 取消下面某一行的注释来测试对应的示例
    
    # example_1_single_integer()
    # example_2_multiple_integers_one_line()
    # example_3_array_single_line()
    # example_4_array_with_length()
    # example_5_multiple_lines()
    # example_6_matrix_input()
    # example_7_multiple_test_cases()
    # example_8_test_cases_with_terminator()
    # example_9_string_processing()
    # example_10_read_all_input()
    # example_11_2d_array()
    # example_12_formatted_output()
    # example_13_floating_point()
    # example_14_comma_separated()
    # example_15_mixed_input()
    # example_16_multiple_queries()
    # example_17_output_array()
    # example_18_output_matrix()
    # example_19_read_until_eof()
    example_20_graph_adjacency_list()


if __name__ == "__main__":
    main()