# ACM模式快速练习指南

## 如何使用本教程

### 第一步：理解代码结构
打开 `learn_ai.py`，浏览20个示例函数，每个都有详细注释。

### 第二步：逐个练习
在 `learn_ai.py` 的 `main()` 函数中，取消注释想要练习的示例：

```python
def main():
    # example_1_single_integer()  # 取消这行注释来练习示例1
    example_20_graph_adjacency_list()  # 当前激活的示例
```

### 第三步：测试运行

#### 方法1: 手动输入
```bash
cd /Users/jack/Desktop/ML/Universal-Translator/ai_test/ACM
python learn_ai.py
# 然后手动输入测试数据
```

#### 方法2: 使用测试文件
```bash
# 测试示例1
python learn_ai.py < test1_input.txt

# 测试示例20（图）
python learn_ai.py < test20_input.txt
```

---

## 每日练习计划（7天掌握ACM模式）

### Day 1: 基础输入输出
- ✅ 示例1: 单个整数
- ✅ 示例2: 多个整数（一行）
- ✅ 示例3: 数组输入
- ✅ 示例4: 带长度的数组

**练习**: 完成3道简单题

### Day 2: 多行输入
- ✅ 示例5: 多行单个数据
- ✅ 示例6: 矩阵输入
- ✅ 示例11: 二维数组

**练习**: 完成3道矩阵相关题

### Day 3: 多组测试用例
- ✅ 示例7: EOF结束
- ✅ 示例8: 特定标志结束
- ✅ 示例19: 持续读取

**练习**: 完成5组多测试用例题

### Day 4: 字符串处理
- ✅ 示例9: 字符统计
- ✅ 示例14: 特殊分隔符
- ✅ 示例15: 混合输入

**练习**: 完成3道字符串题

### Day 5: 格式化输出
- ✅ 示例12: 多结果输出
- ✅ 示例13: 浮点数格式化
- ✅ 示例17: 数组输出
- ✅ 示例18: 矩阵输出

**练习**: 完成3道格式化输出题

### Day 6: 高级场景
- ✅ 示例10: 读取所有输入
- ✅ 示例16: 查询操作
- ✅ 示例20: 图结构

**练习**: 完成2道复杂结构题

### Day 7: 综合实战
- 完成10道混合场景题
- 总结常见模板
- 记录易错点

---

## 常用代码模板

### 模板1: 单组输入输出
```python
def solve():
    n = int(input())
    arr = list(map(int, input().split()))
    # 处理逻辑
    result = sum(arr)
    print(result)

if __name__ == "__main__":
    solve()
```

### 模板2: 多组测试（EOF）
```python
def solve(a, b):
    return a + b

if __name__ == "__main__":
    try:
        while True:
            a, b = map(int, input().split())
            print(solve(a, b))
    except EOFError:
        pass
```

### 模板3: 多组测试（指定数量）
```python
def solve(arr):
    return max(arr)

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        n = int(input())
        arr = list(map(int, input().split()))
        print(solve(arr))
```

### 模板4: 矩阵输入
```python
def solve(matrix):
    # 处理逻辑
    return sum(sum(row) for row in matrix)

if __name__ == "__main__":
    n, m = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(n)]
    print(solve(matrix))
```

### 模板5: 图的邻接表
```python
from collections import defaultdict

def solve(graph, n):
    # 图算法逻辑
    pass

if __name__ == "__main__":
    n, m = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)  # 无向图
    solve(graph, n)
```

---

## 快速测试脚本

创建 `quick_test.py`:
```python
import subprocess
import sys

test_cases = {
    "test1": "5",
    "test2": "3 7",
    "test4": "5\n3 1 4 1 5",
    "test6": "3 4\n1 2 3 4\n5 6 7 8\n9 10 11 12",
}

for name, input_data in test_cases.items():
    print(f"\n{'='*40}")
    print(f"Testing: {name}")
    print(f"{'='*40}")
    print(f"Input:\n{input_data}")
    print(f"\nOutput:")
    
    result = subprocess.run(
        ["python", "learn_ai.py"],
        input=input_data,
        text=True,
        capture_output=True
    )
    print(result.stdout)
```

---

## 调试技巧

### 技巧1: 打印调试信息到stderr
```python
import sys
# 调试信息输出到stderr，不影响标准输出
print(f"Debug: n={n}", file=sys.stderr)
```

### 技巧2: 本地文件测试
```python
if __name__ == "__main__":
    import sys
    # 取消下面的注释以使用文件输入
    # sys.stdin = open('test_input.txt', 'r')
    # sys.stdout = open('test_output.txt', 'w')
    
    solve()
```

### 技巧3: 检查输入格式
```python
line = input().strip()
print(f"原始输入: '{line}'", file=sys.stderr)
print(f"分割后: {line.split()}", file=sys.stderr)
```

---

## 常见错误检查清单

在提交代码前，检查：

- [ ] 是否有多余的输入提示（如"请输入..."）
- [ ] 输出格式是否完全匹配（空格、换行、精度）
- [ ] 是否处理了所有测试用例（多组测试）
- [ ] 是否正确处理EOF异常
- [ ] 数据类型是否正确（int vs str）
- [ ] 是否使用了strip()去除空白字符
- [ ] 浮点数精度是否符合要求
- [ ] 是否有未使用的导入或调试代码

---

## 推荐学习路径

1. **阶段1（1-2天）**: 熟悉基础输入输出
   - 完成示例1-6
   - 每个示例手写3遍

2. **阶段2（2-3天）**: 掌握特殊场景
   - 完成示例7-15
   - 做10道牛客网题目

3. **阶段3（2-3天）**: 综合应用
   - 完成示例16-20
   - 做5道中等难度算法题

4. **阶段4（持续）**: 实战演练
   - 每天至少1道ACM模式题
   - 积累常用模板
   - 参加在线模拟考试

---

## 资源链接

### 在线练习平台
- [牛客网](https://www.nowcoder.com/) - 大厂真题
- [AcWing](https://www.acwing.com/) - 算法课程
- [洛谷](https://www.luogu.com.cn/) - 题库丰富
- [LeetCode中国](https://leetcode.cn/) - 部分支持ACM模式

### 推荐题单
1. 牛客网：大厂真题-输入输出专项
2. AcWing：基础算法课-第一章
3. 洛谷：入门题单

---

## 进阶建议

### 提升速度
1. 使用 `sys.stdin.readline` 代替 `input()`
2. 批量输出而不是逐个print
3. 使用列表推导式
4. 熟记常用模板

### 提升准确性
1. 仔细阅读题目的输出格式
2. 用样例测试
3. 考虑边界情况
4. 检查数据类型

### 建立代码库
创建自己的模板库，包括：
- 输入输出模板
- 常用数据结构（并查集、线段树等）
- 常用算法（排序、搜索、动态规划等）
- 数学工具函数

---

## 获得帮助

如果遇到问题：
1. 检查 `README_ACM_TUTORIAL.md` 中的常见陷阱
2. 查看具体示例的注释
3. 对比你的代码和示例代码
4. 在stderr输出调试信息
5. 使用文件重定向测试

祝你早日掌握ACM模式，斩获Offer！🎯
