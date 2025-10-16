# ACM模式输入输出完全教程

## 目录
- [什么是ACM模式](#什么是acm模式)
- [核心输入方法](#核心输入方法)
- [核心输出方法](#核心输出方法)
- [20个常见场景](#20个常见场景)
- [常见陷阱](#常见陷阱)
- [实战技巧](#实战技巧)

---

## 什么是ACM模式

ACM模式是在线编程竞赛和大厂机考中常用的输入输出模式，特点：

1. **标准输入输出**: 使用 `input()` 读取，`print()` 输出
2. **无交互提示**: 不需要输出 "请输入..." 等提示信息
3. **格式严格**: 输出格式必须完全匹配题目要求
4. **多组测试**: 通常有多组测试用例
5. **本地测试**: 需要手动输入或使用文件重定向

---

## 核心输入方法

### 1. 基础输入

```python
# 读取单个整数
n = int(input())

# 读取单个浮点数
x = float(input())

# 读取单个字符串
s = input().strip()  # strip()去除首尾空白字符
```

### 2. 读取多个数据（同一行）

```python
# 读取两个整数
a, b = map(int, input().split())

# 读取三个整数
x, y, z = map(int, input().split())

# 读取不定长整数数组
arr = list(map(int, input().split()))

# 读取浮点数数组
nums = list(map(float, input().split()))

# 读取字符串数组
words = input().split()
```

### 3. 读取多行数据

```python
# 读取n行
n = int(input())
lines = []
for _ in range(n):
    line = input().strip()
    lines.append(line)

# 读取矩阵
n, m = map(int, input().split())
matrix = []
for _ in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)
```

### 4. 读取所有输入

```python
# 方法1: 逐行读取直到EOF
try:
    while True:
        line = input()
        # 处理line
except EOFError:
    pass

# 方法2: 一次性读取所有
import sys
data = sys.stdin.read()
lines = data.strip().split('\n')
```

### 5. 特殊分隔符

```python
# 逗号分隔
arr = list(map(int, input().split(',')))

# 多种分隔符
import re
arr = list(map(int, re.split('[,;\\s]+', input())))
```

---

## 核心输出方法

### 1. 基础输出

```python
# 输出单个值
print(42)
print("Hello")

# 输出多个值（空格分隔）
print(1, 2, 3)  # 输出: 1 2 3

# 不换行输出
print(x, end='')
print(y, end=' ')
```

### 2. 格式化输出

```python
# 保留小数位数
print(f"{3.14159:.2f}")  # 输出: 3.14
print("%.2f" % 3.14159)  # 输出: 3.14

# 宽度对齐
print(f"{42:5d}")   # 右对齐，宽度5
print(f"{42:<5d}")  # 左对齐，宽度5
print(f"{42:0>5d}") # 右对齐，用0填充

# 百分比
print(f"{0.85:.1%}")  # 输出: 85.0%
```

### 3. 输出数组

```python
arr = [1, 2, 3, 4, 5]

# 方法1: join
print(' '.join(map(str, arr)))  # 输出: 1 2 3 4 5

# 方法2: * 解包
print(*arr)  # 输出: 1 2 3 4 5

# 方法3: 循环
for x in arr:
    print(x, end=' ')
print()  # 换行

# 逗号分隔
print(','.join(map(str, arr)))  # 输出: 1,2,3,4,5
```

### 4. 输出矩阵

```python
matrix = [[1, 2, 3], [4, 5, 6]]

# 方法1
for row in matrix:
    print(' '.join(map(str, row)))

# 方法2
for row in matrix:
    print(*row)
```

---

## 20个常见场景

### 场景1: 单行单个数
```python
# 输入: 5
# 输出: 25
n = int(input())
print(n * n)
```

### 场景2: 单行多个数
```python
# 输入: 3 7
# 输出: 10
a, b = map(int, input().split())
print(a + b)
```

### 场景3: 数组输入（不定长）
```python
# 输入: 1 2 3 4 5
# 输出: 15
arr = list(map(int, input().split()))
print(sum(arr))
```

### 场景4: 先读长度，再读数组
```python
# 输入:
# 5
# 3 1 4 1 5
# 输出: 5
n = int(input())
arr = list(map(int, input().split()))
print(max(arr))
```

### 场景5: 多行输入
```python
# 输入:
# 3
# 10
# 20
# 30
# 输出: 60
n = int(input())
total = sum(int(input()) for _ in range(n))
print(total)
```

### 场景6: 矩阵输入
```python
# 输入:
# 3 4
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 输出: 78
n, m = map(int, input().split())
total = sum(sum(map(int, input().split())) for _ in range(n))
print(total)
```

### 场景7: 多组测试（无结束标志）
```python
# 输入:
# 1 2
# 3 4
# 5 6
# 输出:
# 3
# 7
# 11
try:
    while True:
        a, b = map(int, input().split())
        print(a + b)
except EOFError:
    pass
```

### 场景8: 多组测试（0 0结束）
```python
# 输入:
# 1 2
# 3 4
# 0 0
# 输出:
# 3
# 7
while True:
    a, b = map(int, input().split())
    if a == 0 and b == 0:
        break
    print(a + b)
```

### 场景9: 字符串处理
```python
# 输入: hello world
# 输出: d:1 e:1 h:1 l:3 o:2 r:1 w:1
from collections import Counter
s = input().replace(' ', '')
counter = Counter(s)
for char in sorted(counter):
    print(f"{char}:{counter[char]}", end=' ')
```

### 场景10: 读取所有输入
```python
# 输入:
# 3
# apple banana
# cherry
# date
# 输出: 4
import sys
lines = sys.stdin.read().strip().split('\n')
n = int(lines[0])
count = sum(len(lines[i].split()) for i in range(1, n + 1))
print(count)
```

### 场景11: 二维数组（对角线）
```python
# 输入:
# 3
# 1 2 3
# 4 5 6
# 7 8 9
# 输出: 15
n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
print(sum(matrix[i][i] for i in range(n)))
```

### 场景12: 多个结果输出
```python
# 输入:
# 5
# 1 2 3 4 5
# 输出: 1 5 3.0
n = int(input())
arr = list(map(int, input().split()))
print(min(arr), max(arr), sum(arr) / len(arr))
```

### 场景13: 浮点数处理
```python
# 输入: 3.14 2.86
# 输出: 6.00
a, b = map(float, input().split())
print(f"{a + b:.2f}")
```

### 场景14: 逗号分隔
```python
# 输入: 1,2,3,4,5
# 输出: 15
arr = list(map(int, input().split(',')))
print(sum(arr))
```

### 场景15: 混合输入
```python
# 输入:
# Alice
# 25
# 输出: Alice is 25 years old
name = input().strip()
age = int(input())
print(f"{name} is {age} years old")
```

### 场景16: 查询操作
```python
# 输入:
# 5 3
# 10 20 30 40 50
# 0
# 2
# 4
# 输出:
# 10
# 30
# 50
n, q = map(int, input().split())
arr = list(map(int, input().split()))
for _ in range(q):
    idx = int(input())
    print(arr[idx])
```

### 场景17: 输出数组
```python
# 输入:
# 5
# 1 2 3 4 5
# 输出: 5 4 3 2 1
n = int(input())
arr = list(map(int, input().split()))
print(*arr[::-1])
```

### 场景18: 输出矩阵
```python
# 输入:
# 2 3
# 1 2 3
# 4 5 6
# 输出:
# 1 4
# 2 5
# 3 6
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]
for j in range(m):
    print(*[matrix[i][j] for i in range(n)])
```

### 场景19: EOF处理
```python
# 输入:
# hello
# world
# python
# 输出:
# HELLO
# WORLD
# PYTHON
try:
    while True:
        print(input().upper())
except EOFError:
    pass
```

### 场景20: 图的输入
```python
# 输入:
# 4 5
# 0 1
# 0 2
# 1 2
# 1 3
# 2 3
# 输出: 邻接表
from collections import defaultdict
n, m = map(int, input().split())
graph = defaultdict(list)
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
for i in range(n):
    print(f"{i}: {sorted(graph[i])}")
```

---

## 常见陷阱

### 陷阱1: 忘记strip()
```python
# ❌ 错误
s = input()  # 可能包含末尾的换行符或空格

# ✅ 正确
s = input().strip()
```

### 陷阱2: 类型转换
```python
# ❌ 错误
arr = input().split()  # arr是字符串列表
print(arr[0] + arr[1])  # 字符串拼接，不是数值相加

# ✅ 正确
arr = list(map(int, input().split()))
print(arr[0] + arr[1])
```

### 陷阱3: 多余的输出
```python
# ❌ 错误
n = int(input("请输入一个数: "))  # 不要有提示信息

# ✅ 正确
n = int(input())
```

### 陷阱4: 输出格式不匹配
```python
# ❌ 错误（题目要求空格分隔）
print(f"[{a}, {b}, {c}]")

# ✅ 正确
print(a, b, c)
```

### 陷阱5: 浮点数精度
```python
# ❌ 错误
print(3.14159)  # 题目要求保留2位小数

# ✅ 正确
print(f"{3.14159:.2f}")
```

### 陷阱6: EOF未处理
```python
# ❌ 错误（会抛出EOFError异常）
while True:
    line = input()

# ✅ 正确
try:
    while True:
        line = input()
except EOFError:
    pass
```

### 陷阱7: 空行处理
```python
# ❌ 错误（空行会导致split()返回空列表）
a, b = map(int, input().split())

# ✅ 正确
line = input().strip()
if line:
    a, b = map(int, line.split())
```

---

## 实战技巧

### 技巧1: 快速调试模板
```python
def solve():
    # 你的解题代码
    pass

if __name__ == "__main__":
    # 本地测试时使用文件输入
    # import sys
    # sys.stdin = open('input.txt', 'r')
    
    solve()
```

### 技巧2: 使用sys.stdin提高效率
```python
import sys
input = sys.stdin.readline  # 读取速度更快

# 注意：readline()会保留换行符，需要strip()
n = int(input().strip())
```

### 技巧3: 列表推导式
```python
# 读取n行整数
arr = [int(input()) for _ in range(n)]

# 读取矩阵
matrix = [list(map(int, input().split())) for _ in range(n)]
```

### 技巧4: 批量输出
```python
results = []
for _ in range(n):
    # 计算结果
    results.append(result)

# 一次性输出
print('\n'.join(map(str, results)))
```

### 技巧5: 使用模板
```python
def main():
    # 读取输入
    n = int(input())
    arr = list(map(int, input().split()))
    
    # 处理逻辑
    result = solve(arr)
    
    # 输出结果
    print(result)

def solve(arr):
    # 你的算法逻辑
    return sum(arr)

if __name__ == "__main__":
    main()
```

### 技巧6: 常用导入
```python
import sys
from collections import defaultdict, Counter, deque
from itertools import combinations, permutations
import heapq
import bisect
import math
```

---

## 本地测试方法

### 方法1: 手动输入
```bash
python learn_ai.py
# 然后手动输入测试数据
```

### 方法2: 文件重定向
```bash
python learn_ai.py < input.txt
# 或
python learn_ai.py < input.txt > output.txt
```

### 方法3: 代码中读取文件
```python
import sys
sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')

# 你的代码
```

---

## 练习建议

1. **从简单开始**: 先掌握基础的输入输出格式
2. **分类练习**: 按场景类型逐个练习
3. **注意细节**: 特别关注输出格式（空格、换行、精度）
4. **多做题目**: LeetCode、牛客网、AcWing等平台
5. **总结模板**: 为常见场景建立代码模板

---

## 推荐练习平台

- **牛客网**: 大厂真题，ACM模式
- **AcWing**: 算法课程，输入输出规范
- **LeetCode中国**: 部分题目支持ACM模式
- **洛谷**: 丰富的算法题库

---

## 快速查询表

| 场景 | 输入方法 | 输出方法 |
|------|---------|---------|
| 单个整数 | `n = int(input())` | `print(n)` |
| 多个整数（一行） | `a, b = map(int, input().split())` | `print(a, b)` |
| 整数数组 | `arr = list(map(int, input().split()))` | `print(*arr)` |
| 矩阵 | `[list(map(int, input().split())) for _ in range(n)]` | 逐行print |
| 浮点数（2位小数） | `x = float(input())` | `print(f"{x:.2f}")` |
| 字符串 | `s = input().strip()` | `print(s)` |
| 多组测试（EOF） | `try...except EOFError` | 每组print一次 |

---

## 总结

掌握ACM模式的关键：
1. ✅ 熟练使用 `input()` 和 `split()`
2. ✅ 掌握 `map()` 和类型转换
3. ✅ 注意输出格式（空格、换行、精度）
4. ✅ 处理边界情况（EOF、空行）
5. ✅ 多练习，建立肌肉记忆

祝你机考顺利！🚀
