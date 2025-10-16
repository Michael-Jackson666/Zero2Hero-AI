# 二叉树中序遍历第K个祖先节点查找

## 题目描述

给定一个二叉树的根节点，以及两个整数u和k。任务是找出节点u在二叉树**中序遍历**中的第k个祖先节点的值。

### 关键概念

1. **祖先节点**: 从根节点到目标节点路径上的所有节点
2. **中序遍历**: 左子树 → 根节点 → 右子树的遍历顺序
3. **第k个祖先**: 在中序遍历序列中，位于节点u**前面**的祖先节点中，从后往前数第k个

### 重要理解

这道题的难点在于理解"中序遍历中的第k个祖先"：
- 不是路径上的第k个祖先
- 而是在中序遍历中，位于u之前的祖先节点中的第k个
- 从靠近u的位置往前数

---

## 输入格式

```
第一行: 二叉树的层次遍历表示，空节点用'#'表示，节点用空格分隔
第二行: 两个整数 u k，表示目标节点和祖先偏移量
```

## 输出格式

```
一个整数，表示第k个祖先节点的值，不存在则返回-1
```

---

## 示例详解

### 示例1

**输入:**
```
30 15 45 7 20 35 50 # # 18 # # 40
40 3
```

**树的结构:**
```
          30
        /    \
       15     45
      /  \   /  \
     7   20 35  50
        /      \
       18       40
```

**中序遍历:**
```
7, 15, 18, 20, 30, 35, 40, 45, 50
索引: 0  1   2   3   4   5   6   7   8
```

**分析过程:**

1. **找到节点40的位置**: 
   - 在中序遍历中索引为6

2. **找到40的祖先路径**:
   - 路径: 30 → 45 → 35 → 40
   - 祖先节点(不含40本身): [30, 45, 35]

3. **筛选在中序遍历中位于40之前的祖先**:
   - 30: 索引4 < 6 ✓
   - 45: 索引7 > 6 ✗
   - 35: 索引5 < 6 ✓
   - 在40之前的祖先: [30(索引4), 35(索引5)]

4. **按中序位置排序**: 
   - [30(索引4), 35(索引5)]

5. **找第k=3个祖先**:
   - 从后往前数: 第1个是35, 第2个是30
   - 第3个不存在

**输出:** `-1`

---

### 示例2

**输入:**
```
30 15 45 7 20 35 50 # # 18 # # 40
40 1
```

**分析:**
- 在40之前的祖先: [30(索引4), 35(索引5)]
- 第1个祖先（最接近40）: 35

**输出:** `35`

---

### 示例3

**输入:**
```
10 5 15 2 7 12 20
7 2
```

**树的结构:**
```
        10
       /  \
      5    15
     / \   / \
    2   7 12 20
```

**中序遍历:**
```
2, 5, 7, 10, 12, 15, 20
索引: 0  1  2   3   4   5   6
```

**分析:**
- 节点7的位置: 索引2
- 7的祖先路径: [10, 5]
- 在7之前的祖先: 5(索引1)
- 第k=2个祖先不存在

**输出:** `-1`

---

## 算法思路

### 步骤1: 构建二叉树
从层次遍历表示构建二叉树，使用队列逐层处理。

### 步骤2: 中序遍历
递归进行中序遍历，获取完整的中序序列。

### 步骤3: 找到目标节点位置
在中序序列中找到节点u的索引。

### 步骤4: 找到祖先路径
使用DFS从根节点到目标节点，记录路径上的所有节点。

### 步骤5: 筛选祖先
筛选出在中序遍历中位于u之前的祖先节点。

### 步骤6: 排序并取第k个
按中序位置排序，从后往前取第k个。

---

## 代码实现

### 关键函数

#### 1. 构建树 `build_tree(level_order)`
```python
def build_tree(level_order):
    if not level_order or level_order[0] == '#':
        return None
    
    root = TreeNode(int(level_order[0]))
    queue = deque([root])
    i = 1
    
    while queue and i < len(level_order):
        node = queue.popleft()
        
        # 处理左子节点
        if i < len(level_order) and level_order[i] != '#':
            node.left = TreeNode(int(level_order[i]))
            queue.append(node.left)
        i += 1
        
        # 处理右子节点
        if i < len(level_order) and level_order[i] != '#':
            node.right = TreeNode(int(level_order[i]))
            queue.append(node.right)
        i += 1
    
    return root
```

#### 2. 中序遍历 `inorder_traversal(root)`
```python
def inorder_traversal(root):
    result = []
    
    def inorder(node):
        if node is None:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    
    inorder(root)
    return result
```

#### 3. 找路径 `find_path_to_node(root, target)`
```python
def find_path_to_node(root, target):
    def dfs(node, path):
        if node is None:
            return False
        
        path.append(node.val)
        
        if node.val == target:
            return True
        
        if dfs(node.left, path) or dfs(node.right, path):
            return True
        
        path.pop()
        return False
    
    path = []
    return path if dfs(root, path) else None
```

#### 4. 主逻辑 `find_kth_ancestor_in_inorder(root, u, k)`
```python
def find_kth_ancestor_in_inorder(root, u, k):
    # 1. 获取中序遍历
    inorder_seq = inorder_traversal(root)
    
    # 2. 找到u的位置
    if u not in inorder_seq:
        return -1
    u_index = inorder_seq.index(u)
    
    # 3. 找到祖先路径
    path = find_path_to_node(root, u)
    if path is None:
        return -1
    
    # 4. 筛选在u之前的祖先
    ancestors = path[:-1]  # 去掉u本身
    ancestors_before_u = []
    for ancestor in ancestors:
        ancestor_index = inorder_seq.index(ancestor)
        if ancestor_index < u_index:
            ancestors_before_u.append((ancestor_index, ancestor))
    
    # 5. 排序并取第k个
    ancestors_before_u.sort()
    if k <= 0 or k > len(ancestors_before_u):
        return -1
    
    return ancestors_before_u[-k][1]
```

---

## 复杂度分析

- **时间复杂度**: O(n)
  - 构建树: O(n)
  - 中序遍历: O(n)
  - 查找路径: O(n)
  - 筛选祖先: O(h)，h为树的高度
  
- **空间复杂度**: O(n)
  - 树的存储: O(n)
  - 递归栈: O(h)
  - 中序序列: O(n)

---

## 测试用例

### Test 1: 祖先不存在
```
输入:
30 15 45 7 20 35 50 # # 18 # # 40
40 3

输出: -1
```

### Test 2: 第1个祖先
```
输入:
30 15 45 7 20 35 50 # # 18 # # 40
40 1

输出: 35
```

### Test 3: 小树
```
输入:
10 5 15 2 7 12 20
7 2

输出: -1
```

---

## 运行方法

```bash
# 测试单个用例
python mlp_solution.py < test1.txt

# 批量测试
for i in 1 2 3; do
    echo "=== Test $i ==="
    python mlp_solution.py < test${i}.txt
done
```

---

## 易错点

1. **理解题意**: 不是路径上的第k个祖先，而是中序遍历中u之前的第k个祖先
2. **索引处理**: k从1开始计数，需要从后往前数
3. **边界情况**: 
   - 目标节点不存在
   - 祖先数量不足k个
   - 所有祖先都在u之后（右子树的情况）
4. **重复值**: 如果树中有重复值，`index()`可能找到错误的节点

---

## 总结

这道题考察了：
- 二叉树的构建（层次遍历 → 树结构）
- 中序遍历的实现
- DFS路径查找
- 列表操作和排序

关键是理解"中序遍历中的第k个祖先"这个概念！
