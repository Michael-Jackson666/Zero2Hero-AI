"""
题目：给定一个二叉树的根节点，以及两个整数u和k。
任务是找出节点u在二叉树中序遍历中的第k个祖先节点的值。
一个节点的祖先是指从根节点到该节点路径上的所有节点。
这里，"第k个祖先"是指在中序遍历中，位于节点u前面的所有祖先节点中的第k个位置祖先节点，
如果这样的祖先节点不存在，则返回-1.

输入:
    包含两行
    第一行是一个字符串，表示一个二叉树；
       空节点用'#'表示
       非空节点的值为整数
       节点之间用一个空格分隔
       树的层次遍历，比如"1 2 3 # # 4 5"表示根节点为1, 左子节点为2, 右子节点为3;
       2没有子节点；3的左子节点为4, 右子节点为5
    第二行包含两个整数u和k, 分别表示目标节点的值和要查找的祖先节点的偏移量，一个空格分隔这两个值。

输出:
    一个整数，表示节点u的第k个祖先节点的值，如果不存在则返回-1.

示例:
输入:
    30 15 45 7 20 35 50 # # 18 # # 40
    40 3
输出:
    -1
    
解释:
    中序遍历: 7, 15, 18, 20, 30, 35, 40, 45, 50
    节点40在中序遍历中的位置是索引6
    40的祖先路径: 30 -> 45 -> 35 -> 40
    在中序遍历中，40之前的祖先有: 30(索引4), 35(索引5)
    第k=3个祖先不存在，返回-1
"""

from collections import deque


class TreeNode:
    """二叉树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(level_order):
    """
    从层次遍历构建二叉树
    
    参数:
        level_order: 层次遍历的节点值列表，'#'表示空节点
    
    返回:
        根节点
    """
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


def inorder_traversal(root):
    """
    中序遍历二叉树
    
    参数:
        root: 根节点
    
    返回:
        中序遍历的节点值列表
    """
    result = []
    
    def inorder(node):
        if node is None:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)
    
    inorder(root)
    return result


def find_path_to_node(root, target):
    """
    找到从根节点到目标节点的路径
    
    参数:
        root: 根节点
        target: 目标节点值
    
    返回:
        路径列表（从根到目标），如果不存在返回None
    """
    if root is None:
        return None
    
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
    if dfs(root, path):
        return path
    return None


def find_kth_ancestor_in_inorder(root, u, k):
    """
    找到节点u在中序遍历中的第k个祖先节点
    
    参数:
        root: 根节点
        u: 目标节点值
        k: 要找的第k个祖先
    
    返回:
        第k个祖先节点的值，不存在返回-1
    """
    # 1. 获取中序遍历序列
    inorder_seq = inorder_traversal(root)
    
    # 2. 找到节点u在中序遍历中的位置
    if u not in inorder_seq:
        return -1
    
    u_index = inorder_seq.index(u)
    
    # 3. 找到从根节点到u的路径（所有祖先节点）
    path = find_path_to_node(root, u)
    
    if path is None or len(path) == 0:
        return -1
    
    # 4. 找出在中序遍历中位于u之前的祖先节点
    # 祖先节点是路径上的节点（除了u自己）
    ancestors = path[:-1]  # 去掉u本身
    
    # 5. 筛选出在中序遍历中位于u之前的祖先
    ancestors_before_u = []
    for ancestor in ancestors:
        ancestor_index = inorder_seq.index(ancestor)
        if ancestor_index < u_index:
            ancestors_before_u.append((ancestor_index, ancestor))
    
    # 6. 按照在中序遍历中的位置排序
    ancestors_before_u.sort()
    
    # 7. 获取第k个祖先（k从1开始计数）
    if k <= 0 or k > len(ancestors_before_u):
        return -1
    
    # 从后往前数第k个（最接近u的是第1个）
    return ancestors_before_u[-k][1]


def main():
    """主函数"""
    # 读取输入
    tree_str = input().strip()
    u, k = map(int, input().split())
    
    # 解析树的层次遍历
    level_order = tree_str.split()
    
    # 构建二叉树
    root = build_tree(level_order)
    
    # 查找第k个祖先
    result = find_kth_ancestor_in_inorder(root, u, k)
    
    # 输出结果
    print(result)


if __name__ == "__main__":
    main()
