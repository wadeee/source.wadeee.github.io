---
layout: post
title: Binary Tree(二分法)
published: true
date: 2016-10-08
---

> 你好，树先生

## 关于二叉树

二叉树是每个结点最多有两个子树的树结构。

+ 树节点结构

    ```python
    class TreeNode:
        def __init__(self, x):
            self.val = x
            self.left = None
            self.right = None
    ```

## LeetCode真题

### 144. Binary Tree Preorder Traversal

二叉树前序遍历
[查看原题](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
```

+ 方法一：iteratively

    ```python
    def preorderTraversal(self, root: 'TreeNode') -> 'List[int]':
        ans, stack = [], root and [root]
        while stack:
            node = stack.pop()
            if node:
                ans.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return ans
    ```

+ 方法二：recursively

    ```python
    def preorder_traversal(root):
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left) + \
            self.preorderTraversal(root.right)
    ```

### 589. N-ary Tree Preorder Traversal

N-叉树的前序遍历。N叉树和二叉树有个区别，就是N叉树不需要考虑子节点知否为空，做单独的判断。
[查看原题](https://leetcode.com/problems/n-ary-tree-preorder-traversal/)

+ 方法一：recursively.

    ```python
    def preorder(self, root):
        if not root:
            return []
        res = [root.val]
        for child in root.children:
            res += self.preorder(child)
        return res
    ```

+ 方法二：iteratively.

    ```python
    def preorder(self, root):
        res, stack = [], root and [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            stack.extend(reversed(node.children))
        return res
    ```
  

### 94. Binary Tree Inorder Traversal

中序遍历二叉树
[查看原题](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)

```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
```

+ 方法一：使用栈迭代。

    ```python
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        stack, ans = [], []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            ans.append(root.val)
            root = root.right
        return ans
    ```

+ 方法二：Morris Traversal.

    ```python
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        cur, ans = root, []
        while cur:
            if not cur.left:
                ans.append(cur.val)
                cur = cur.right
            else:
                pre = cur.left
                # 找到当前节点左子树中最右的右节点
                while pre.right and pre.right != cur:
                    pre = pre.right
                    
                if not pre.right:
                    # 找到最右的节点，连接到根节点
                    pre.right = cur
                    cur = cur.left
                # 恢复节点
                else:
                    pre.right = None
                    ans.append(cur.val)
                    cur = cur.right
                    
        return ans
    ```

### 145. Binary Tree Postorder Traversal

后序遍历二叉树
[查看原题](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)

```
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [3,2,1]
```

+ 方法一：根右左，再倒序。

    ```python
    def postorder_traversal(root):
        res, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.left)
                stack.append(node.right)
        return res[::-1]
    ```

+ 方法二：思想: 使用last作为判断是否该节点的右子树完成遍历，如果一个node.right已经刚刚遍历完毕，那么将last==node.right，否则将会寻找node.right。

    ```python
    def postorderTraversal(self, root):
        res, stack, node, last = [], [], root, None
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack[-1]
                if not node.right or last == node.right:
                    node = stack.pop()
                    res.append(node.val)
                    last, node = node, None
                else:
                    node = node.right    
        return res
    ```

+ 方法三：使用boolean判断一个节点是否被遍历过

    ```python
    def postorderTraversal(self, root):
        res, stack = [], [(root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    res.append(node.val)
                else:
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))                
        return res
    ```

+ 方法四：dfs.

    ```python
    def postorderTraversal(self, root: 'TreeNode') -> 'List[int]':
        ans = []
    
        def dfs(node):
            if not node:
                return 
            dfs(node.left)
            dfs(node.right)
            ans.append(node.val)
            
        dfs(root)
        return ans
    ```

### 590. N-ary Tree Postorder Traversal

N-叉树的后序遍历。
[查看原题](https://leetcode.com/problems/n-ary-tree-postorder-traversal/)

+ 方法一：recursively.

    ```python
    def postorder(self, root):
        if not root:
            return []
        return sum([self.postorder(child) for child in root.children], []) + [root.val]
    ```

+ 方法二：iteratively and reversed.

    ```python
    def postorder(self, root):
        res, stack = [], root and [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            stack.extend(node.children)
        return res[::-1]
    ```

+ 方法三：iteratively and flag.

    ```python
    def postorder(self, root):
        res, stack = [], root and [(root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                res.append(node.val)
            else:
                stack.append((node, True))
                stack.extend((n, False) for n in reversed(node.children))
        return res
    ```

### 100. Same Tree

判断相同的二叉树。
[查看原题](https://leetcode.com/problems/same-tree/description/)

```
Input:     1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

Output: true
```

+ 方法一：recursively

    ```python
    def isSameTree(self, p: 'TreeNode', q: 'TreeNode') -> 'bool':
        if p and q:
            return (p.val==q.val and self.isSameTree(p.left, q.left) and 
                    self.isSameTree(p.right, q.right))
        else:
            return p is q
    ```

+ 方法二：recursively, tuple

    ```python
    def is_same_tree(p, q):
        def t(n):
            return n and (n.val, t(n.left), t(n.right))  
        return t(p) == t(q)
    ```

+ 方法三：iteratively.

    ```python
    def isSameTree(self, p: 'TreeNode', q: 'TreeNode') -> 'bool':
        stack = [(p, q)]
        while stack:
            p1, p2 = stack.pop()
            if not p1 and not p2:
                continue
            if not p1 or not p2:
                return False
            if p1.val != p2.val:
                return False
            stack.append((p1.left, p2.left))
            stack.append((p1.right, p2.right))
        return True
    ```

### 101. Symmetric Tree

判断二叉树是否对称。
[查看原题](https://leetcode.com/problems/symmetric-tree/description/)

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

+ 方法一：recursively.

    ```python
    def isSymmetric(self, root: 'TreeNode') -> 'bool':
    
        def symmetric(p1, p2):
            if p1 and p2:
                return (p1.val == p2.val and symmetric(p1.left, p2.right) and 
                        symmetric(p1.right, p2.left))
            else:
                return p1 is p2
    
        if not root:
            return True
        return symmetric(root.left, root.right)
    ```

+ 方法二：iteratively.

    ```python
    def isSymmetric(self, root: 'TreeNode') -> 'bool':
        stack = root and [(root.left, root.right)]        
        while stack:
            p1, p2 = stack.pop()
            if not p1 and not p2: continue
            if not p1 or not p2: return False
            if p1.val != p2.val: return False
            stack.append((p1.left, p2.right))
            stack.append((p1.right, p2.left))
        return True
    ```


### 104. Maximum Depth of Binary Tree

二叉树最大深度。
[查看原题](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)

```
    3
   / \
  9  20
    /  \
   15   7
return 3
```

+ 方法一：recursively

    ```python
    def max_depth(root):
        if not root:
            return 0
        return max(max_depth(root.left), max_depth(root.right)) + 1
    ```

+ 方法二：iteratively. BFS with deque

    ```python
    def maxDepth(self, root: 'TreeNode') -> 'int':
        q = root and collections.deque([(root, 1)])
        d = 0
        while q:
            node, d = q.popleft()
            if node.right:
                q.append((node.right, d+1))
            if node.left:
                q.append((node.left, d+1))
        return d
    ```

### 559. Maximum Depth of N-ary Tree

N-叉树的最大深度。
[查看原题](https://leetcode.com/problems/maximum-depth-of-n-ary-tree/)

+ 方法一：BFS with deque.

    ```python
    def maxDepth(self, root: 'Node') -> 'int':
        q = root and collections.deque([(root, 1)])
        d = 0
        while q:
            node, d = q.popleft()
            for child in node.children:
                q.append((child, d + 1))
        return d
    ```

+ 方法二：BFS.

    ```python
    def maxDepth(self, root):
        q, level = root and [root], 0
        while q:
            q, level = [child for node in q for child in node.children], level+1
        return level
    ```

+ 方法三：recursively.

    ```python
    def maxDepth(self, root: 'Node') -> 'int':
        if not root:
            return 0
        return max(list(map(self.maxDepth, root.children)) or [0]) + 1
    ```


### 111. Minimum Depth of Binary Tree

求根节点到叶子节点的最小深度。
[查看原题](https://leetcode.com/problems/minimum-depth-of-binary-tree)

+ 方法一：recursively

    ```python
    def minDepth(self, root):
        if not root:
            return 0
        if root.left and root.right:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
            return self.minDepth(root.left) + self.minDepth(root.right) + 1
    ```

+ 方法二：对上述方法修改，更加Pythonic. 注意一点，Python3中要加list,否则max因为空值报错。

    ```python
    def minDepth(self, root: 'TreeNode') -> 'int':
        if not root: return 0
        d = list(map(self.minDepth, (root.left, root.right)))
        return 1 + (min(d) or max(d))
    ```

+ 方法三：迭代法，BFS

    ```python
    def minDepth(self, root: 'TreeNode') -> 'int':
        q = root and collections.deque([(root, 1)])
        d = 0
        while q:
            node, d = q.popleft()
            if not node.left and not node.right:
                return d
            if node.left:
                q.append((node.left, d+1))
            if node.right:
                q.append((node.right, d+1))
        return d
    ```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

根据前序遍历和中序遍历重建二叉树。
[查看原题](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)

```
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
```

+ 方法一：切片。

    ```python
    def buildTree(preorder, inorder):
        if preorder == []:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)
        cut = inorder.index(root_val)
        root.left = buildTree(preorder[1:cut+1], inorder[:cut])
        root.right = buildTree(preorder[cut+1:], inorder[cut+1:])
        return root
    ```

+ 方法二：上述方法在极端情况下，如只有左子树的情况，由于index会将时间复杂度上升到O(n²)，而且切片产生了一些不必要的内存，pop和reverse是为了增加效率。

    ```python
    def buildTree(self, preorder: 'List[int]', inorder: 'List[int]') -> 'TreeNode':
        def build(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(preorder.pop())
                root.left = build(root.val)
                inorder.pop()
                root.right = build(stop)
                return root
        preorder.reverse()
        inorder.reverse()
        return build(None)
    ```

### 572. Subtree of Another Tree

判断是否是树的子结构。
[查看原题](https://leetcode.com/problems/subtree-of-another-tree/description/)

+ 思路：这道题是遍历加判断相同树的结合。这里采用前序遍历和递归判断相同树。

    ```python
    def isSubtree(self, s: 'TreeNode', t: 'TreeNode') -> 'bool':
    
        def is_same(s, t):
            if s and t:
                return (s.val==t.val and is_same(s.left, t.left) and 
                        is_same(s.right, t.right))
            else:
                return s is t
    
        stack = s and [s]
        while stack:
            node = stack.pop()
            if node:
                if is_same(node, t):
                    return True
                stack.append(node.right)
                stack.append(node.left)
        return False
    ```

### 102. Binary Tree Level Order Traversal

分层遍历二叉树。
[查看原题](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

+ 注意：循环条件要加上root，以防止root is None

    ```python
    def levelOrder(self, root: 'TreeNode') -> 'List[List[int]]':
        ans, level = [], root and [root]
        while level:
            ans.append([n.val for n in level])
            level = [k for n in level for k in (n.left, n.right) if k]
        return ans
    ```

### 103. Binary Tree Zigzag Level Order Traversal

之字形打印二叉树。
[查看原题](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/)

+ 解法

    ```python
    def zigzagLevelOrder(self, root: 'TreeNode') -> 'List[List[int]]':
        ans, level, order = [], root and [root], 1
        while level:
            ans.append([n.val for n in level][::order])
            order *= -1
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return ans
    ```

### 107. Binary Tree Level Order Traversal II

和102题不同的是，从下到上分层打印。
[查看原题](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

+ 方法一：将结果倒序输出。

    ```python
    def levelOrderBottom(self, root):
        res, level = [], [root]
        while root and level:
            res.append([n.val for n in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return res[::-1]
    ```

+ 方法二：也可以从前面插入元素。

    ```python
    def levelOrderBottom(self, root):
        res, level = [], [root]
        while root and level:
            res.insert(0, [n.val for n in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return res
    ```

### 429. N-ary Tree Level Order Traversal

分层打印N叉树。
[查看原题](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)

+ 方法一：将结果倒序输出。

    ```python
    def levelOrder(self, root: 'Node') -> 'List[List[int]]':
        ans, level = [], root and [root]
        while level:
            ans.append([n.val for n in level])
            level = [k for n in level for k in n.children if k]
        return ans
    ```

### 637. Average of Levels in Binary Tree

遍历一个二叉树，求每层节点的平均值，按照节点不为空的个数。
[查看原题](https://leetcode.com/problems/average-of-levels-in-binary-tree/)

```
Input:
    3
   / \
  9  20
    /  \
   15   7
Output: [3, 14.5, 11]
Explanation:
The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on level 2 is 11. Hence return [3, 14.5, 11].
```

+ 解法

    ```python
    def averageOfLevels(self, root: 'TreeNode') -> 'List[float]':
        ans, level = [], root and [root]
        while level:
            ans.append(sum(n.val for n in level) / len(level))
            level = [k for n in level for k in (n.left, n.right) if k]
        return ans
    ```

### 515. Find Largest Value in Each Tree Row

找到树每层的最大值。
[查看原题](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)

+ BFS.

    ```python
    def largestValues(self, root: TreeNode) -> List[int]:
        ans, levels = [], root and [root]
        while levels:
            ans.append(max(x.val for x in levels))
            levels = [k for n in levels for k in (n.left, n.right) if k]
        return ans
    ```

### 987. Vertical Order Traversal of a Binary Tree

垂直遍历二叉树，从左到右，从上到下，如果节点具有相同位置，按照值从小到大。
[查看原题](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)

```
Input: [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation: 
The node with value 5 and the node with value 6 have the same position according to the given scheme.
However, in the report "[1,5,6]", the node value of 5 comes first since 5 is smaller than 6.
```

+ dfs. 通过建立一个字典数组，将对应的节点使用深度优先遍历初始化数组。然后按照x, y, val三个优先级进行排序。

    ```python
    def verticalTraversal(self, root: 'TreeNode') -> 'List[List[int]]':
        seen = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
    
        def dfs(node, x=0, y=0):
            if node:
                seen[x][y].append(node.val)
                dfs(node.left, x-1, y+1)
                dfs(node.right, x+1, y+1)
    
        dfs(root)
        ans = []
        for x in sorted(seen):
            inner = []
            for y in sorted(seen[x]):
                inner.extend(sorted(n for n in seen[x][y]))
            ans.append(inner)
        return ans
    ```

### 257. Binary Tree Paths

打印二叉树从根节点到叶子节点全部路径。
[查看原题](https://leetcode.com/problems/binary-tree-paths/description/)

```
Input:

   1
 /   \
2     3
 \
  5

Output: ["1->2->5", "1->3"]

Explanation: All root-to-leaf paths are: 1->2->5, 1->3
```

+ 方法一：iteratively。思路：采用前序遍历二叉树，使用tuple保存节点当前路径，如果是叶子节点，则添加到结果中。开始老是想着用'->'.join()，这样反而麻烦，直接使用字符串保存就好。

    ```python
    def binaryTreePaths(self, root: 'TreeNode') -> 'List[str]':
        ans, stack = [], root and [(root, str(root.val))]
        while stack:
            n, p = stack.pop()
            if not n.left and not n.right:
                ans.append(p)
            if n.right:
                stack.append((n.right, p+'->'+str(n.right.val)))
            if n.left:
                stack.append((n.left, p+'->'+str(n.left.val)))
        return ans
    ```

+ 方法二：dfs.

    ```python
    def binaryTreePaths(self, root: 'TreeNode') -> 'List[str]':
        ans = []
        def dfs(n, path):
            if n:
                path.append(str(n.val))
                if not n.left and not n.right:
                    ans.append('->'.join(path))
                dfs(n.left, path)
                dfs(n.right, path)
                path.pop()
        dfs(root, [])
        return ans
    ```

+ 方法三：recursively

    ```python
    def binaryTreePaths(self, root): 
        if not root:
            return []
        return [str(root.val) + '->' + path
                for kid in (root.left, root.right) if kid
                for path in self.binaryTreePaths(kid)] or [str(root.val)]
    ```

### 257. Binary Tree Paths

求字典顺序最小的路径，路径指叶子节点到根节点的路径。0对应a，1对应b。
[查看原题](https://leetcode.com/problems/smallest-string-starting-from-leaf/)

```
Input: [0,1,2,3,4,3,4]
Output: "dba"
```

+ 方法一：先列出所有根到叶子的路径，再reverse求最小值。

    ```python
    def smallestFromLeaf(self, root: 'TreeNode') -> 'str':
        OFFSET = ord('a')
        stack = root and [(root, chr(root.val+OFFSET))]
        ans = '~'
        while stack:
            n, p = stack.pop()
            if not n.left and not n.right:
                ans = min(ans, p[::-1])
            if n.right:
                stack.append((n.right, p+chr(n.right.val+OFFSET)))
            if n.left:
                stack.append((n.left, p+chr(n.left.val+OFFSET)))
        return ans
    ```

+ 方法二：dfs. 递归计算完左右节点，然后再将根节点pop掉。

    ```python
    def smallestFromLeaf(self, root: 'TreeNode') -> 'str':
        self.ans = '~'
        
        def dfs(node, A):
            if node:
                A.append(chr(node.val + ord('a')))
                if not node.left and not node.right:
                    self.ans = min(self.ans, ''.join(reversed(A)))
                dfs(node.left, A)
                dfs(node.right, A)
                A.pop()
            
        dfs(root, [])
        return self.ans
    ```

### 112. Path Sum

判断是否具有从根节点到叶子节点上的值和为sum。
[查看原题](https://leetcode.com/problems/path-sum/description/)

+ 方法一：recursively

    ```python
    def hasPathSum(self, root: 'TreeNode', total: 'int') -> 'bool':
        if not root:
            return False
        elif (not root.left and not root.right and 
            root.val==total):
            return True
        else:
            return (self.hasPathSum(root.left, total-root.val) or 
                    self.hasPathSum(root.right, total-root.val))
    ```

+ 方法二：iteratively

    ```python
    def hasPathSum(self, root: 'TreeNode', total: 'int') -> 'bool':
        stack = root and [(root, total)]
        while stack:
            n, t = stack.pop()
            if not n.left and not n.right and n.val==t:
                return True
            if n.right:
                stack.append((n.right, t-n.val))
            if n.left:
                stack.append((n.left, t-n.val))
        return False
    ```

### 113. Path Sum II

上题的升级版，要求二维数组返回所有路径。
[查看原题](https://leetcode.com/problems/path-sum-ii/description/)

```
sum = 22

      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1

[
   [5,4,11,2],
   [5,8,4,5]
]
```

+ 方法一：iteratively. 举一反三。

    ```python
    def pathSum(self, root: 'TreeNode', total: 'int') -> 'List[List[int]]':
        stack = root and [(root, [root.val], total)]
        ans = []
        while stack:
            n, v, t = stack.pop()
            if not n.left and not n.right and n.val==t:
                ans.append(v)
            if n.right:
                stack.append((n.right, v+[n.right.val], t-n.val))
            if n.left:
                stack.append((n.left, v+[n.left.val], t-n.val))
        return ans
    ```

+ 方法二：recursively. 先找出所有路径，再过滤，实际上和257题一样。不过这并没有把这道题的特性涵盖进去。

    ```python
    def pathSum(self, root, sum_val):
        paths = self.all_paths(root)
        return [path for path in paths if sum(path)==sum_val]
        
    def all_paths(self, root):
        if not root:
            return []
        return [[root.val]+path
                for kid in (root.left, root.right) if kid
                for path in self.all_paths(kid)] or [[root.val]]
    ```

+ 方法三：recursively.

    ```python
    def pathSum(self, root, sum):
        if not root:
            return []
        val, *kids = root.val, root.left, root.right
        if any(kids):
            return [[val] + path
                    for kid in kids if kid
                    for path in self.pathSum(kid, sum-val)]
        return [[val]] if val==sum else []
    ```

### 297. Serialize and Deserialize Binary Tree

序列化反序列化二叉树。
[查看原题](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/)

+ 解法

    ```python
    class Codec:
    
        def serialize(self, root):
            if not root:
                return '$'
            return (str(root.val) + ',' + self.serialize(root.left) + 
                    ',' + self.serialize(root.right))
            
        def deserialize(self, data):
            nodes = data.split(',')[::-1]
            return self.deserialize_tree(nodes)
        
        def deserialize_tree(self, nodes):
            val = nodes.pop()
            if val == '$':
                return None
            root = TreeNode(val)
            root.left = self.deserialize_tree(nodes)
            root.right = self.deserialize_tree(nodes)
            return root
    ```

### 110. Balanced Binary Tree

判断是否是平衡二叉树。
[查看原题](https://leetcode.com/problems/balanced-binary-tree/description/)

+ 方法一：递归+递归。

    ```python
    def isBalanced(self, root):
        if not root:
            return True
        return self.isBalanced(root.left) and self.isBalanced(root.right) and \
               abs(self.max_depth(root.left)-self.max_depth(root.right)) <= 1
        
    def max_depth(self, root):
        if not root:
            return 0
        return max(self.max_depth(root.left), self.max_depth(root.right)) + 1
    ```

+ 方法二：dfs. 算深度的时候判断左右是否深度超过1. 这里变量不能把self去掉，否则`[1,2,2,3,3,null,null,4,4]`会错误的返回`True`而不是`False`。

    ```python
    def isBalanced(self, root: 'TreeNode') -> 'bool':
        self.balanced = True
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            if not self.balanced or abs(left - right) > 1:
                self.balanced = False
            return max(left, right) + 1
        
        dfs(root)
        return self.balanced
    ```

### 108. Convert Sorted Array to Binary Search Tree

将有序数组转换成二叉搜索树。
[查看原题](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

```
Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5],
      0
     / \
   -3   9
   /   /
 -10  5
```

+ 方法一：递归。

    ```python
    def sortedArrayToBST(self, nums):
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
    ```

+ 方法二：不使用切片。

    ```python
    def sortedArrayToBST(self, nums: 'List[int]') -> 'TreeNode':
        
        def convert(lo, hi):
            if lo > hi:
                return None
            mid = (lo+hi) // 2
            root = TreeNode(nums[mid])
            root.left = convert(lo, mid-1)
            root.right = convert(mid+1, hi)
            return root
    
        return convert(0, len(nums)-1)
    ```

### 235. Lowest Common Ancestor of a Binary Search Tree

寻找二叉搜索树的最小公共祖先。
[查看原题](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree)

+ 方法一：iteratively.

    ```python
    def lowestCommonAncestor(self, root, p, q):
        while (root.val-p.val) * (root.val-q.val) > 0:
            root = (root.left, root.right)[root.val < p.val]
        return root
    ```

+ 方法二：recursively.

    ```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if (root.val-p.val) * (root.val-q.val) <= 0:
            return root
        return self.lowestCommonAncestor(
            (root.left, root.right)[root.val < p.val], p, q)
    ```

### 404. Sum of Left Leaves

求一个二叉树所有左叶子节点的和。
[查看原题](https://leetcode.com/problems/sum-of-left-leaves/description/)

+ 方法一：iteratively.这里使用了tuple记录是否为左叶子节点。

    ```python
    def sumOfLeftLeaves(self, root: 'TreeNode') -> 'int':
        ans, stack = 0, root and [(root, False)]
        while stack:
            n, isleft = stack.pop()
            if n:
                if not n.left and not n.right and isleft:
                    ans += n.val
                stack.append((n.right, False))
                stack.append((n.left, True))
        return ans
    ```

+ 方法二：recursively.

    ```python
    def sumOfLeftLeaves(self, root: 'TreeNode') -> 'int':
        if not root:
            return 0
        if (root.left and not root.left.left and not root.left.right):
            return root.left.val + self.sumOfLeftLeaves(root.right)
        else:
            return (self.sumOfLeftLeaves(root.left) + 
                    self.sumOfLeftLeaves(root.right))
    ```

### 938. Range Sum of BST

给两个节点的值，求二叉搜索树在这两个值之间的节点和。每个节点的值唯一。
[查看原题](https://leetcode.com/contest/weekly-contest-110/problems/range-sum-of-bst/)

```
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
```

+ 方法一：先前序遍历了一下，再根据条件求和。

    ```python
    def rangeSumBST(self, root, L, R):
        traverse, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                traverse.append(node.val)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        return sum([x for x in traverse if L <= x <= R])

    ```

+ 方法二：利用二叉搜索树的特性。

    ```python
    def rangeSumBST(self, root: 'TreeNode', L: 'int', R: 'int') -> 'int':
        ans, stack = 0, root and [root]
        while stack:
            node = stack.pop()
            if node.val > L and node.left:
                stack.append(node.left)
            if node.val < R and node.right:
                stack.append(node.right)
            if L <= node.val <= R:
                ans += node.val
        return ans
    ```

### 530. Minimum Absolute Difference in BST

求二叉搜索树任意两个节点的最小差。
[查看原题](https://leetcode.com/problems/minimum-absolute-difference-in-bst/)

```
Input:

   1
    \
     3
    /
   2

Output:
1

Explanation:
The minimum absolute difference is 1, which is the difference between 2 and 1 (or between 2 and 3).
```

+ 解法

    ```python
    def getMinimumDifference(self, root: 'TreeNode') -> 'int':
    
        def inorder(n):
            if not n:
                return []
            return inorder(n.left) + [n.val] + inorder(n.right)
    
        nums = inorder(root)
        # return min(nums[i+1]-nums[i] for i in range(len(nums)-1))
        return min(b-a for a, b in zip(nums, nums[1:]))

    ```

### 783. Minimum Distance Between BST Nodes

二叉搜索树两个节点的最小值。和530是一道题。
[查看原题](https://leetcode.com/problems/minimum-distance-between-bst-nodes/)

```
Input: root = [4,2,6,1,3,null,null]
Output: 1
Explanation:
Note that root is a TreeNode object, not an array.

The given tree [4,2,6,1,3,null,null] is represented by the following diagram:

          4
        /   \
      2      6
     / \    
    1   3  

while the minimum difference in this tree is 1, it occurs between node 1 and node 2, also between node 3 and node 2.
```

+ 方法一：递归 + 生成器， 遍历了两次。

    ```python
    def minDiffInBST(self, root: 'TreeNode') -> 'int':
        
        def inorder(node):
            if not node:
                return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        
        t = inorder(root)
        return min(t[x]-t[x-1] for x in range(1, len(t)))
    ```

+ 方法二：一次遍历，没有保存整个遍历数组，效率高。

    ```python
    def minDiffInBST(self, root: TreeNode) -> int:
        ans, last, stack = float('inf'), float('-inf'), []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            ans, last = min(ans, root.val-last), root.val
            root = root.right
        return ans
    ```

+ 方法三：一次递归。

    ```python
    class Solution:
        pre = float('-inf')
        ans = float('inf')
        
        def minDiffInBST(self, root: 'TreeNode') -> 'int':
            if root.left:
                self.minDiffInBST(root.left)
            self.ans = min(self.ans, root.val-self.pre)
            self.pre = root.val
            if root.right:
                self.minDiffInBST(root.right)
            return self.ans
    ```

### 538. Convert BST to Greater Tree

二叉搜索树转换。使得节点的值等于所有比它大的节点的和。
[查看原题](https://leetcode.com/problems/convert-bst-to-greater-tree/)

```
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

+ 方法一：recursively。这里使用了一个变量来保存当前的累加和，然后递归中采用先右后左的方式。

    ```python
    def convertBST(self, root: 'TreeNode') -> 'TreeNode':
        self.sum_val = 0
    
        def convert(node):
            if node:
                convert(node.right)
                self.sum_val += node.val
                node.val = self.sum_val 
                convert(node.left)
    
        convert(root)
        return root
    ```

+ 方法二：iteratively。94题中的中序遍历迭代方式不能实现，因为迭代时改变了根节点。

    ```python
    def convertBST(self, root):
        stack = [(root, False)]
        sum_val = 0
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    node.val += sum_val
                    sum_val = node.val
                else:
                    stack.append((node.left, False))
                    stack.append((node, True))
                    stack.append((node.right, False))
        return root
    ```

### 958. Check Completeness of a Binary Tree

判断二叉树是否是完整二叉树。完整二叉树为：除了最后一层所有节点不能为空，最后一层节点全部去靠左。
[查看原题](https://leetcode.com/problems/check-completeness-of-a-binary-tree/)

```
Input: [1,2,3,4,5,6]
Output: true
Explanation: Every level before the last is full (ie. levels with node-values {1} and {2, 3}), and all nodes in the last level ({4, 5, 6}) are as far left as possible.

Input: [1,2,3,4,5,null,7]
Output: false
Explanation: The node with value 7 isn't as far left as possible.
```

+ 方法一：采用分层遍历的方式，判断每层的节点是否是2**level。最后一层采用切片的方式判断最左原则。

    ```python
    class Solution:
        def isCompleteTree(self, root):
            if not root:
                return True
            levels = [root]
            last_full = True
            level = 0
            while levels:
                value_nodes = [n for n in levels if n]
                if value_nodes != levels[:len(value_nodes)]:
                    return False
                else:
                    print(len(levels), 2**level)
                    if len(levels) != 2**level:
                        if not last_full:
                            return False
                        last_full = False
                    
                levels = [kid for n in levels if n for kid in (n.left, n.right)]
                level += 1
            return True
    ```

+ 方法二：遇见第一个None时，后面如果再有非None的值就不是玩整树了。

    ```python
    def isCompleteTree(self, root: 'TreeNode') -> 'bool':
        i, bfs = 0, [root]
        while bfs[i]:
            bfs.append(bfs[i].left)
            bfs.append(bfs[i].right)
            i += 1
        return not any(bfs[i:])
    ```

### 543. Diameter of Binary Tree

求二叉树的最大直径，即任意两节点的长度。
[查看原题](https://leetcode.com/problems/diameter-of-binary-tree/)

```
          1
         / \
        2   3
       / \     
      4   5    
Return **3**, which is the length of the path [4,2,1,3] or [5,2,1,3].
```

+ recursively, 使用一个实例变量计算了最大值。

    ```python
    def diameterOfBinaryTree(self, root: 'TreeNode') -> 'int':
        self.diameter = 0
    
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.diameter = max(self.diameter, left+right)
            return max(left, right) + 1
    
        dfs(root)
        return self.diameter
    ```

### 965. Univalued Binary Tree

判断一个二叉树是否所有节点具有相同的值。
[查看原题](https://leetcode.com/problems/univalued-binary-tree/)

+ 方法一：recursively。

    ```python
    def isUnivalTree(self, root: 'TreeNode') -> 'bool':
        def dfs(node):
            return (not node or root.val==node.val and 
                    dfs(node.left) and dfs(node.right))
        return dfs(root)
    ```

+ 方法二：iteratively.常规写法。

    ```python
    def isUnivalTree(self, root: 'TreeNode') -> 'bool':
        r_val, stack = root.val, [root]
        while stack:
            n = stack.pop()
            if n:
                if n.val != r_val:
                    return False
                stack.append(n.right)
                stack.append(n.left)
        return True
    ```

+ 方法三：前序遍历，生成器方法。

    ```python
    def isUnivalTree(self, root: 'TreeNode') -> 'bool':
        
        def bfs(node):
            if node:
                yield node.val
                yield from bfs(node.left)
                yield from bfs(node.right)
                
        it = bfs(root)
        root_val = next(it)
        for val in it:
            if val != root_val:
                return False
        return True
    ```

### 563. Binary Tree Tilt

返回一个二叉树整个树的倾斜度。所有节点倾斜度的总和。节点的倾斜度等于左子树和右子树所有和差的绝对值。
[查看原题](https://leetcode.com/problems/binary-tree-tilt/)

```
Input: 
         1
       /   \
      2     3
Output: 1
Explanation: 
Tilt of node 2 : 0
Tilt of node 3 : 0
Tilt of node 1 : |2-3| = 1
Tilt of binary tree : 0 + 0 + 1 = 1
```

+ 方法一：recursively. 这里用tuple记录了节点总和和倾斜度总和。

    ```python
    def findTilt(self, root):
        self.res = 0
        _, top_res = self.sum_and_diff(root)
        return self.res + top_res
    
    def sum_and_diff(self, node):
        if not node:
            return 0, 0
        l_sum, l_diff = self.sum_and_diff(node.left)
        r_sum, r_diff = self.sum_and_diff(node.right)
        self.res += l_diff + r_diff
        return node.val+l_sum+r_sum, abs(l_sum-r_sum)
    ```

+ 方法二: 想了一会后序遍历的迭代法，没想出来，貌似需要维护很多的变量。这里还是优化一下方法一。

    ```python
    def findTilt(self, root: 'TreeNode') -> 'int':
    
        def dfs(node):
            if not node:
                return 0, 0
            l_sum, l_diff = dfs(node.left)
            r_sum, r_diff = dfs(node.right)
            return (node.val + l_sum + r_sum, 
                    abs(l_sum-r_sum) + l_diff + r_diff)
    
        return dfs(root)[1]
    ```

### 606. Construct String from Binary Tree

根据二叉树重建字符串，使用()表示嵌套关系。
[查看原题](https://leetcode.com/problems/construct-string-from-binary-tree/)

```
Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     
Output: "1(2(4))(3)"

Input: Binary tree: [1,2,3,null,4]
       1
     /   \
    2     3
     \  
      4 
Output: "1(2()(4))(3)"
```

+ recursively. 左右节点有一点区别，在于如果左节点为空，右节点不为空，要保留左节点的括号。

    ```python
    def tree2str(self, t):
        if not t: return ''
        left = '({})'.format(self.tree2str(t.left)) if (t.left or t.right) else ''
        right = '({})'.format(self.tree2str(t.right)) if t.right else ''
        return '{}{}{}'.format(t.val, left, right)
    ```

### 617. Merge Two Binary Trees

合并两个二叉树，相同位置的节点值相加，空节点算0。
[查看原题](https://leetcode.com/problems/merge-two-binary-trees/)

+ 方法一：recursively. 

    ```python
    def mergeTrees(self, t1, t2):
        if not t1:
            return t2
        if not t2:
            return t1
        t = TreeNode(t1.val+t2.val)
        t.left = self.mergeTrees(t1.left, t2.left)
        t.right = self.mergeTrees(t1.right, t2.right)
        return t
    ```

+ 方法二：iteratively.

    ```python
    def mergeTrees(self, t1, t2):
        if not t1 and not t2:
            return []
        t = TreeNode(0)
        stack = [(t, t1, t2)]
        while stack:
            n, n1, n2 = stack.pop()
            if n1 or n2:
                n.val = (n1.val if n1 else 0) + (n2.val if n2 else 0)
                if (n1 and n1.right) or (n2 and n2.right):
                    n.right = TreeNode(None)
                    stack.append((n.right, n1.right if n1 else None, n2.right if n2 else None))
                if (n1 and n1.left) or (n2 and n2.left):
                    n.left = TreeNode(None)
                    stack.append((n.left, n1.left if n1 else None, n2.left if n2 else None))
        return t
    ```

### 653. Two Sum IV - Input is a BST

判断二叉树中是否有两个节点相加为k。
[查看原题](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/)

```
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True
```

+ preorder + set.

    ```python
    def findTarget(self, root, k):
        seen, stack = set(), root and [root]
        while stack:
            node = stack.pop()
            if node:
                if k-node.val in seen:
                    return True
                seen.add(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return False
    ```

### 669. Trim a Binary Search Tree

根据范围修剪二叉搜索树，注意是二叉搜索树，不是普通的二叉树。
[查看原题](https://leetcode.com/problems/trim-a-binary-search-tree/)

```
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
```

+ recursively.

    ```python
    def trimBST(self, root, L, R):
        def trim_node(node):
            if not node:
                return None
            elif node.val > R:
                return trim_node(node.left)
            elif node.val < L:
                return trim_node(node.right)
            else:
                node.left = trim_node(node.left)
                node.right = trim_node(node.right)
                return node
        return trim_node(root)
    ```

### 671. Second Minimum Node In a Binary Tree

找出二叉树中第二小的节点值。左右子节点同时存在或同时不存在，根节点小于等于任意子节点。
[查看原题](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/)

```
Input: 
    2
   / \
  2   5
     / \
    5   7

Output: 5
Explanation: The smallest value is 2, the second smallest value is 5.
```

+ 方法一：先放到set里.

    ```python
    def findSecondMinimumValue(self, root: 'TreeNode') -> 'int':
        self.uniques = set()
    
        def dfs(node):
            if node:
                self.uniques.add(node.val)
                dfs(node.left)
                dfs(node.right)
    
        dfs(root)
        min1, ans = root.val, float('inf')
        for v in self.uniques:
            if min1 < v < ans:
                ans = v
        return ans if ans < float('inf') else -1
    ```

+ 方法二： iteratively.

    ```python
    def findSecondMinimumValue(self, root):
        min1 = root.val if root else -1
        res = float('inf')
        stack = root and [root]
        while stack:
            node = stack.pop()
            if node:
                if min1 < node.val < res:
                    res = node.val
                stack.extend([node.right, node.left])
        return res if res < float('inf') else -1
    ```

### 687. Longest Univalue Path

相同节点最长路径，路径长度按照两个节点之间的边距，也就是节点数-1。
[查看原题](https://leetcode.com/problems/longest-univalue-path/)

```
              5
             / \
            4   5
           / \   \
          1   1   5
output: 2
```

+ 解法

    ```python
    def longestUnivaluePath(self, root):
        self.res = 0
        def traverse(node):
            if not node:
                return 0
            left_len, right_len = traverse(node.left), traverse(node.right)
            left = (left_len+1) if node.left and node.left.val==node.val else 0
            right = (right_len+1) if node.right and node.right.val==node.val else 0
            self.res = max(self.res, left + right)
            return max(left, right)
        traverse(root)
        return self.res
    ```

### 700. Search in a Binary Search Tree

在二叉搜索树中搜索节点。
[查看原题](https://leetcode.com/problems/search-in-a-binary-search-tree/)

```
Given the tree:
        4
       / \
      2   7
     / \
    1   3

And the value to search: 2
```

+ 方法一：recursively.

    ```python
    def searchBST(self, root: 'TreeNode', val: 'int') -> 'TreeNode':
        if root:
            if val == root.val:
                return root
            return self.searchBST(
                (root.left, root.right)[root.val < val], val)
    ```

+ 方法二：iteratively.

    ```python
    def searchBST(self, root: 'TreeNode', val: 'int') -> 'TreeNode':
        node = root
        while node and node.val != val:
            node = (node.left, node.right)[node.val < val]
        return node
    ```

### 872. Leaf-Similar Trees

叶子相近的树，只从左到右遍历叶子节点的顺序相同的两棵树。
[查看原题](https://leetcode.com/problems/leaf-similar-trees/)

+ 方法一：前序遍历+生成器。空间复杂度过高，beats 1%。

    ```python
    def leafSimilar(self, root1: 'TreeNode', root2: 'TreeNode') -> 'bool':
    
        def leaves(root):
            stack = root and [root]
            while stack:
                node = stack.pop()
                if node:
                    if not node.right and not node.left:
                        yield node.val
                    stack.append(node.right)
                    stack.append(node.left)
    
        leaves1 = leaves(root1)
        leaves2 = leaves(root2)
        return all(
            a==b for a, b in itertools.zip_longest(leaves1, leaves2))
    ```

+ 方法二：dfs.

    ```python
    def leafSimilar(self, root1: 'TreeNode', root2: 'TreeNode') -> 'bool':
    
        def dfs(node):
            if node:
                if not node.left and not node.right:
                    yield node.val
                yield from dfs(node.left)
                yield from dfs(node.right)
    
        return all(
            a==b for a, b in itertools.zip_longest(dfs(root1), dfs(root2)))
    ```

### 897. Increasing Order Search Tree

根据中序遍历建立一个只有右子树的二叉树。要求在原树上修改。
[查看原题](https://leetcode.com/problems/increasing-order-search-tree/)

```
Example 1:
Input: [5,3,6,2,4,null,8,1,null,null,null,7,9]

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]

 1
  \
   2
    \
     3
      \
       4
```

+ 方法一：iteratively.

    ```python
    def increasingBST(self, root: TreeNode) -> TreeNode:
        ans = head = TreeNode(0)
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            head.right = TreeNode(root.val)
            root, head = root.right, head.right
        return ans.right
    ```

+ 方法二：生成器。

    ```python
    def increasingBST(self, root: 'TreeNode') -> 'TreeNode':
        
        def inorder(node):
            if node:
                yield from inorder(node.left)
                yield node.val
                yield from inorder(node.right)
                
        ans = head = TreeNode(0)
        for v in inorder(root):
            head.right = TreeNode(v)
            head = head.right
        return ans.right
    ```

+ 方法三：题中有个要求在原树上修改，所以以上两种方法其实不符合要求，这里使用递归实现。

    ```python
    def increasingBST(self, root: 'TreeNode', tail=None) -> 'TreeNode':
        if not root: return tail
        res = self.increasingBST(root.left, root)
        root.left = None
        root.right = self.increasingBST(root.right, tail)
        return res
    ```

### 993. Cousins in Binary Tree

表弟节点指两个节点在同一深度，并且父节点不同。判断两个节点是否是表弟节点。树中节点值唯一。
[查看原题](https://leetcode.com/problems/cousins-in-binary-tree/)

+ 用dict记录。

    ```python
    def isCousins(self, root: 'TreeNode', x: 'int', y: 'int') -> 'bool':
        parent, depth = {}, {}
        
        def dfs(node, par=None):
            if node:
                parent[node.val] = par
                depth[node.val] = depth[par] + 1 if par else 0
                dfs(node.left, node.val)
                dfs(node.right, node.val)
                
        dfs(root)
        return depth[x] == depth[y] and parent[x] != parent[y]
    ```

### 230. Kth Smallest Element in a BST

二叉搜索树的第K小节点值。
[查看原题](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

```
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
```

+ 方法一：生成器前序遍历。

    ```python
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        def inorder(node):
            if node:
                yield from inorder(node.left)
                yield node.val
                yield from inorder(node.right)
                
        for n in inorder(root):
            if k == 1:
                return n
            else:
                k -= 1
    ```

+ 方法二：迭代。

    ```python
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
    ```

### 98. Validate Binary Search Tree

验证一个树是否是二叉搜索树。
[查看原题](https://leetcode.com/problems/validate-binary-search-tree/)

```
    5
   / \
  1   4
     / \
    3   6
Output: false
Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
             is 5 but its right child's value is 4.
```

+ 中序遍历即可。

    ```python
    def isValidBST(self, root: TreeNode) -> bool:
        stack, last = [], float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= last:
                return False
            last = root.val
            root = root.right
        return True
    ```

### 109. Convert Sorted List to Binary Search Tree

将有序链表转成平衡二叉搜索树。
[查看原题](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/)

```
Given the sorted linked list: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
```

+ 方法一：先遍历链表，再二分递归创建树。

    ```python
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        inorder = []
        while head:
            inorder.append(head.val)
            head = head.next
        lo, hi = 0, len(inorder)-1
        
        def build_tree(lo, hi):
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            root = TreeNode(inorder[mid])
            root.left = build_tree(lo, mid-1)
            root.right = build_tree(mid+1, hi)
            return root
            
        return build_tree(lo, hi)
    ```

+ 方法二：这个方法很棒。先遍历一遍找到链表的长度；然后递归去构建树，共享一个`head`可变对象。

    ```python
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        
        def find_size(head):
            h, count = head, 0
            while h:
                h = h.next
                count += 1
            return count
        lo, hi = 0, find_size(head)
        
        def form_bst(lo, hi):
            nonlocal head
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            left = form_bst(lo, mid-1)
            root = TreeNode(head.val)
            head = head.next
            root.left = left
            right = form_bst(mid+1, hi)
            root.right = right
            return root
        
        return form_bst(lo, hi-1)
    ```

### 1008. Construct Binary Search Tree from Preorder Traversal

根据前序遍历重建二叉搜索树。
[查看原题](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/)

```
Input: [8,5,1,7,10,12]
Output: [8,5,10,1,7,null,12]
```

+ recursively.

    ```python
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
         if not preorder: return None
         root = TreeNode(preorder[0])
         i = bisect.bisect(preorder, root.val)
         root.left = self.bstFromPreorder(preorder[1:i])
         root.right = self.bstFromPreorder(preorder[i:])
         return root
    ```

### 236. Lowest Common Ancestor of a Binary Tree

二叉树两个节点的最小公共祖先。
[查看原题](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

+ 方法一: 递归，是用mid表示当前节点是否是其中的一个。

    ```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.ans = None
        
        def dfs(node):
            if not node:
                return False
            left = dfs(node.left)
            right = dfs(node.right)
            mid = node in (p, q)
            if mid + left + right >= 2:
                self.ans = node
            return mid or left or right
        dfs(root)
        return self.ans
    ```

+ 方法二：递归，思想如果是两个节点中的一个，就返回这个节点。

    ```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        return root if left and right else left or right
    ```

+ 方法三：参考了257的dfs解法。需要注意的是一定要加`list(path)`，否则由于可变对象的问题，会导致最后结果为`[]`。

    ```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ans = []
        def dfs(n, path):
            if n:
                path.append(n)
                if n in (p, q):
                    ans.append(list(path))   # must use list, or you will get []
                    if len(ans) == 2:		 # optimized
                        return 
                dfs(n.left, path)
                dfs(n.right, path)
                path.pop()
        dfs(root, [])
        return next(a for a, b in list(zip(*ans))[::-1] if a==b)
    ```

### 654. Maximum Binary Tree

根据数组建立一个树，要求根节点为数组最大的树。
[查看原题](https://leetcode.com/problems/maximum-binary-tree/)

+ 解法

    ```python
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        v = max(nums)
        root = TreeNode(v)
        i = nums.index(v)
        root.left = self.constructMaximumBinaryTree(nums[:i])
        root.right = self.constructMaximumBinaryTree(nums[i+1:])
        return root
    ```

### 513. Find Bottom Left Tree Value

寻找二叉树最底层的最左节点。
[查看原题](https://leetcode.com/problems/find-bottom-left-tree-value/)

+ 方法一：根据分层遍历改编。

    ```python
    def findBottomLeftValue(self, root: TreeNode) -> int:
        ans, levels = None, root and [root]
        while levels:
            ans = levels[0].val
            levels = [k for n in levels for k in (n.left, n.right) if k]
        return ans
    ```

+ 方法二：双端队列，BFS.

    ```python
    def findBottomLeftValue(self, root: TreeNode) -> int:
        q = collections.deque([root])
        while q:
            node = q.pop()
            if node.right:
                q.appendleft(node.right)
            if node.left:
                q.appendleft(node.left)
        return node.val
    ```

+ 方法三：循环时改变迭代对象，这种方式个人觉得不好。不过好在是在遍历之前添加到末端。

    ```python
    def findBottomLeftValue(self, root: TreeNode) -> int:
        queue = [root]
        for node in queue:
            queue += (x for x in (node.right, node.left) if x)
        return node.val
    ```

### 814. Binary Tree Pruning

剪掉树中不包含1的子树。
[查看原题](https://leetcode.com/problems/binary-tree-pruning/)

+ recursively.

    ```python
    def pruneTree(self, root: TreeNode) -> TreeNode:
        
        def dfs(node):
            if not node:
                return True
            left = dfs(node.left)
            right = dfs(node.right)
            if left:
                node.left = None
            if right:
                node.right = None
            
            return node.val==0 and left and right
        dfs(root)
        return root
    ```

### 199. Binary Tree Right Side View

二叉树从右向左看时，从上到下的节点。
[查看原题](https://leetcode.com/problems/binary-tree-right-side-view/)

+ 方法一：和分层遍历思想相同。

    ```python
    def rightSideView(self, root: TreeNode) -> List[int]:
        ans, levels = [], root and [root]
        while levels:
            ans.append(levels[-1].val)
            levels = [k for n in levels for k in (n.left, n.right) if k]
        return ans
    ```

+ 方法二：dfs. 从右到左深度遍历，用一个深度变量控制是否是第一个最右节点。

    ```python
    def rightSideView(self, root: TreeNode) -> List[int]:
        ans = []
        def dfs(n, depth):
            if n:
                if depth == len(ans):
                    ans.append(n.val)
                dfs(n.right, depth+1)
                dfs(n.left, depth+1)
        dfs(root, 0)
        return ans
    ```

### 662. Maximum Width of Binary Tree

二叉树的最大宽度。
[查看原题](https://leetcode.com/problems/maximum-width-of-binary-tree/)

+ 方法一：常规队列写法。需要注意的是，每层遍历要用最右边的减去最左边的才是宽度。

    ```python
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        queue = [(root, 0, 0)]
        ans = cur_depth = left = 0
        for node, depth, pos in queue:
            if node:
                queue.append((node.left, depth+1, pos*2))
                queue.append((node.right, depth+1, pos*2+1))
                if cur_depth != depth:
                    cur_depth = depth
                    left = pos
                ans = max(pos-left+1, ans)
        return ans
    ```

+ 方法二：按照分层顺序将所有节点编号，从1开始，`enumerate`其实就是计算`2*pos`, `2*pos+1`。

    ```python
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        levels = [(1, root)]
        width = 0
        while levels:
            width = max(levels[-1][0] - levels[0][0] + 1, width)
            levels = [k
                      for pos, n in levels
                      for k in enumerate((n.left, n.right), 2 * pos)
                      if k[1]]
        return width
    ```

### 222. Count Complete Tree Nodes

统计完整树的节点个数。
[查看原题](https://leetcode.com/problems/count-complete-tree-nodes/)

+ 二分法。比较左子树的深度和右子树的深度，如果相同则表明左子树为满树，右子树为完整树。如果不同则表明左子树为完整树，右子树为满树。

    ```python
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        left, right = self.depth(root.left), self.depth(root.right)
        if left == right:
            return 2 ** left + self.countNodes(root.right)
        else:
            return 2 ** right + self.countNodes(root.left)
        
    def depth(self, node):
        if not node:
            return 0
        return 1 + self.depth(node.left)
    ```

### 1022. Sum of Root To Leaf Binary Numbers

计算所有根到叶子节点路径二进制数表示的的和。
[查看原题](https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/)

```
Input: [1,0,1,0,1,0,1]
Output: 22
Explanation: (100) + (101) + (110) + (111) = 4 + 5 + 6 + 7 = 22
```

+ 思路和 257.Binary Tree Paths一样。

    ```python
    def sumRootToLeaf(self, root: TreeNode) -> int:
        self.ans = 0
        def dfs(n, path):
            if n:
                path.append(str(n.val))
                if not n.left and not n.right:
                    self.ans += int(''.join(path), 2)
                dfs(n.left, path)
                dfs(n.right, path)
                path.pop()
                
        dfs(root, [])
        return self.ans % (10**9 + 7)
    ```

### 1026. Maximum Difference Between Node and Ancestor

祖先和其子节点的最大差绝对值。
[查看原题](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/)

+ 方法一：周赛时写的dfs. 380ms. 瓶颈在于每次都求一次最大值和最小值。

    ```python
    def maxAncestorDiff(self, root: TreeNode) -> int:
        self.ans = float('-inf')
        
        def dfs(n, p):
            if n:
                if p:
                    max_diff = max(abs(max(p)-n.val), abs(min(p)-n.val))
                    self.ans = max(self.ans, max_diff)
                p.append(n.val)
                dfs(n.left, p)
                dfs(n.right, p)
                p.pop()
        
        dfs(root, [])
        return self.ans
    ```

+ 方法二：改良了一下，使用p记录一个当前的最大值和最小值。52ms.

    ```python
    def maxAncestorDiff(self, root: TreeNode) -> int:
        self.ans = float('-inf')
        
        def dfs(n, p):
            if n:
                if p:
                    mx, mn = p[-1]
                    self.ans = max(self.ans, max(mx-n.val, n.val-mn))
                    p.append((max(mx, n.val), min(mn, n.val)))
                else:
                    p.append((n.val, n.val))
                dfs(n.left, p)
                dfs(n.right, p)
                p.pop() 
        dfs(root, [])
        return self.ans
    ```

### 1038. Binary Search Tree to Greater Sum Tree

二叉搜索树转成一颗规则的树，从右根左的顺序累加节点值。
[查看原题](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/)

+ 方法一：使用栈。

    ```python
    def bstToGst(self, root: TreeNode) -> TreeNode:
        head = root
        stack, total = [], 0
        while stack or root:
            while root:
                stack.append(root)
                root = root.right
            root = stack.pop()
            total += root.val
            root.val = total
            root = root.left
        return head
    ```

+ 方法二：Lee神的递归方式。

    ```python
    class Solution:
        val = 0
        def bstToGst(self, root: TreeNode) -> TreeNode:
            if root.right: self.bstToGst(root.right)
            root.val = self.val = self.val + root.val
            if root.left: self.bstToGst(root.left)
            return root
    ```

### 1080. Insufficient Nodes in Root to Leaf Paths

计算所有的根到叶子节点的路径，如果路径和小于给定值，则剪掉这个树枝。
[查看原题](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/)

+ recursively.

    ```python
    def sufficientSubset(self, root: TreeNode, limit: int) -> TreeNode:
        if not root:
            return None
        if not root.left and not root.right:
            return root if root.val >= limit else None
        root.left = self.sufficientSubset(root.left, limit-root.val)
        root.right = self.sufficientSubset(root.right, limit-root.val)
        return root if root.left or root.right else None
    ```

### 1161. Maximum Level Sum of a Binary Tree

求最节点和最大层的层数。
[查看原题](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/)

+ 分层遍历

    ```python
    def maxLevelSum(self, root: TreeNode) -> int:
        lvsum = []
        level = [root]
        while level:
            lvsum.append(sum(n.val for n in level))
            level = [k for n in level for k in (n.left, n.right) if k]
        return lvsum.index(max(lvsum)) + 1
    ```

### 1104. Path In Zigzag Labelled Binary Tree

之字形树的目标节点路径。
[查看原题](https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/)

+ 方法一：迭代，此题纯粹是数学题，这里先假设非之字形的树，找到规律，然后知道每层的节点数再相减。

    ```python
    def pathInZigZagTree(self, label: int) -> List[int]:
        
        ans = []
        n = 0
        while 2 ** n <= label:
            n += 1
        
        while n > 0 and label >= 1:
            ans.append(label)
            org_lable = label // 2
            label = 2**(n-1)-1-org_lable+2**(n-2)
            n -= 1
        return ans[::-1]
    ```

+ 方法二：Lee神的递归。原理一样，层数n是通过查2的幂求的。

    ```python
    def pathInZigZagTree(self, x):
        return self.pathInZigZagTree(3 * 2 ** (len(bin(x)) - 4) - 1 - x / 2) + [x] if x > 1 else [1]
    ```

### 1110. Delete Nodes And Return Forest

给定一个树，删除指定的一些节点，然后删除的节点的左右子树成为单独的根节点。返回所有的树。
[查看原题](https://leetcode.com/problems/delete-nodes-and-return-forest/)

```
Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]
```

+ 递归。做着题的时候有个误区：在当前节点被删除后，找到其在父节点对应的位置，然后置为空。实际上应该讲根节点删除的状态保留，在下一层处理。

    ```python
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        ans = []
        to_del = set(to_delete)
        
        def helper(root, is_root):
            
            if not root:
                return None
            is_del = root.val in to_del
            root.left = helper(root.left, is_del)
            root.right = helper(root.right, is_del)
            if not is_del and is_root:
                ans.append(root)
            return None if is_del else root
        
        helper(root, True)
        return ans
    ```
