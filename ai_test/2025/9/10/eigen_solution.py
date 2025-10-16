"""
项目: 滑动窗口特征提取

描述:
本项目旨在实现一个多尺寸滑动窗口的特征转换.
给定一个时间序列数据 input_array 和一组窗口大小 window_array.
对于 window_array 中的每个窗口大小 w, 在 input_array 上滑动窗口.
从每个窗口中提取5个特征: 均值(mean), 标准差(std), 最小值(min), 最大值(max), 和斜率(slope).

特征计算说明:
- 均值: 窗口内元素的平均值.
- 标准差: 样本标准差, ddof为0.
- 最小值: 窗口内的最小值.
- 最大值: 窗口内的最大值.
- 斜率: 使用线性最小二乘法回归计算. 时间索引 x = [0, 1, 2, ..., w-1].
  斜率 beta 计算公式为:
  beta = [n * sum(xy) - sum(x) * sum(y)] / [n * sum(x^2) - (sum(x))^2]

数值修约规则:
- 如果一个数的整数部分不为0, 则小数点后最多保留3位.
- 如果整数部分为0, 则最多保留4位有效数字.
- 采用四舍五入的方式.

输入:
- input_array: 一个一维数组, 表示时间序列数据.
- window_array: 一个一维数组, 包含多个窗口的大小.

输出:
- 一个二维数组, 形状为 (n, m), 其中:
  - n = len(input_array) - max(window_array) + 1
  - m = len(window_array) * 5
- 每一行对应一个滑动窗口的起始位置, 包含了所有窗口大小计算出的特征的拼接结果.
- 每个窗口大小对应的特征顺序为 [mean, std, min, max, slope].
"""

# 读取整个输入，提取前两个[]中的整数
import sys, re, math
sall = sys.stdin.read()

# 提取两个数组的内容
# 错误1修复: findll -> findall (拼写错误)
blocks = re.findall(r'\[([^\]]*)\]', sall)
if len(blocks) < 2:
    raise ValueError("")
    sys.exit(0)

def to_ints(t):
    """将字符串中的整数提取为整数列表"""
    return list(map(int, re.findall(r'-?\d+', t)))

a = to_ints(blocks[0])
wins = to_ints(blocks[1])

def fmt(x:float) -> str:
    """
    根据题目要求格式化数字
    - 整数部分不为0: 保留小数点后最多3位
    - 整数部分为0: 保留4位有效数字
    """
    if abs(x) >= 1:
        # 整数部分不为0，保留3位小数
        y = round(x, 3)
        s = f"{y:.3f}".rstrip('0').rstrip('.')
        return s if s else "0"
    elif x == 0:
        return "0"
    else:
        # 整数部分为0，保留4位有效数字
        abs_x = abs(x)
        decimal_places = -int(math.floor(math.log10(abs_x))) + 3
        return str(round(x, decimal_places))

from collections import defaultdict
def slide_min_max(arr, w):
    """滑动窗口最小值和最大值"""
    from collections import deque
    n = len(arr)
    if w > n:
        return [], []
    
    min_deque = deque()
    max_deque = deque()
    min_vals = []
    max_vals = []
    
    for i in range(n):
        # 移除不在窗口内的元素
        if min_deque and min_deque[0] <= i - w:
            min_deque.popleft()
        if max_deque and max_deque[0] <= i - w:
            max_deque.popleft()
        
        # 维护最小值双端队列
        while min_deque and arr[min_deque[-1]] >= arr[i]:
            min_deque.pop()
        min_deque.append(i)
        
        # 维护最大值双端队列
        while max_deque and arr[max_deque[-1]] <= arr[i]:
            max_deque.pop()
        max_deque.append(i)
        
        # 当窗口形成时，记录当前的最小值和最大值
        if i >= w - 1:
            min_vals.append(arr[min_deque[0]])
            max_vals.append(arr[max_deque[0]])
    
    return min_vals, max_vals

if not a or not wins:
    print("[]")
    sys.exit(0)

wmax = max(wins)
n = len(a) - wmax + 1
if n <=0:
    print("[]")
    sys.exit(0)

N = len(a)
# 前缀和、平方和、索引乘值的前缀
S = [0.0]*(N+1)
Q = [0.0]*(N+1)
A = [0.0]*(N+1)
for i, v in enumerate(a):
    S[i+1] = S[i] + v
    Q[i+1] = Q[i] + v*v
    A[i+1] = A[i] + v*i

feat = {}
for w in wins:
    m = N - w + 1
    mean = [0.0]*m
    std = [0.0]*m
    for i in range(m):
        sumy = S[i+w] - S[i]
        meany = sumy / w
        mean[i] = meany
        if w == 1:
            std[i] = 0.0
        else:
            ss = Q[i+w] - Q[i]
            # 错误2修复: mu -> meany (变量名错误)
            # 标准差计算: ddof=0，所以是除以w，不是w-1
            var = (ss - w * meany * meany) / w
            if var < 0:
                var = 0.0
            std[i] = math.sqrt(var)
    mn, mx = slide_min_max(a, w)
    sumx = w * (w - 1) / 2.0
    sumx2 = (w - 1) * w * (2 * w - 1) / 6.0
    D = w * sumx2 - sumx * sumx
    slope = [0.0] * m
    if D != 0:
        for i in range(m):
            sumy = S[i+w] - S[i]
            abssum = A[i+w] - A[i]
            sumxy = abssum - i * sumy  
            slope[i] = (w * sumxy - sumx * sumy) / D
    feat[w] = (mean, std, mn, mx, slope)


# 按对齐规则输出
out = []
for r in range(n):
    row = []
    for w in wins:
        idx = r + (wmax - w)
        mean, std, mn, mx, slope = feat[w]
        
        # 格式化输出，假设 fmt 是一个用于格式化数字的函数
        row.extend([fmt(mean[idx]), fmt(std[idx]), fmt(mn[idx]), fmt(mx[idx]),
                    fmt(slope[idx])])
    
    out.append("[" + ",".join(row) + "]")

print("\n".join(out))