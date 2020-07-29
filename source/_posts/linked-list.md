---
layout: post
title: Linked List(链表)
published: true
date: 2016-09-23
---

> 用时间换取空间

## 关于链表

链表是一种物理存储单元上非连续、非顺序的存储结构，数据元素的逻辑顺序是通过链表中的指针链接次序实现的。

## LeetCode真题

### 2. Add Two Numbers

两个链表相加。
[查看原题](https://leetcode.com/problems/add-two-numbers/description/)

```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

+ 注意边界处理

    ```python
    def addTwoNumbers(l1, l2):
        l = head = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            l.next = ListNode(val)
            l = l.next
        return head.next
    ```


### 445. Add Two Numbers II

跟上题类似，只不过是进位方式不同。
[查看原题](https://leetcode.com/problems/add-two-numbers-ii/)

```
Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
```

+ 方法一：先reverse再相加，最后再reverse。

    ```python
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        def reverse(head):
            prev = None
            while head:
                head.next, prev, head = prev, head, head.next
            return prev
        
        ans = head = ListNode(0)
        l1, l2 = reverse(l1), reverse(l2)
        carry = 0
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            head.next = ListNode(val)
            head = head.next
        return reverse(ans.next)
    ```

+ 方法二：由于Python int没有限制，所以可以遍历相加，再从尾到头还原节点。

    ```python
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        v1 = v2 = 0
        while l1:
            v1 = v1*10 + l1.val
            l1 = l1.next
        while l2:
            v2 = v2*10 + l2.val
            l2 = l2.next
        val = v1 + v2
        tail, head = None, None
        while val > 0:
            head = ListNode(val % 10)
            head.next = tail
            tail = head
            val //= 10
        return head if head else ListNode(0)
    ```
  

### 21. Merge Two Sorted Lists

合并两个有序链表。
[查看原题](https://leetcode.com/problems/merge-two-sorted-lists/description/)

```
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```

+ 方法1：iteratively 迭代

    ```python
    def mergeTwoLists(l1, l2):
        l = head = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                l.next, l1 = l1, l1.next
            else:
                l.next, l2 = l2, l2.next
            l = l.next
        l.next = l1 or l2
        return head.next
    ```

+ 方法2：recursively 递归

    ```python
    def mergeTwoLists(l1, l2):
        # 判断是否存在None
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = mergeTwoLists(l1, l2.next)
            return l2
    ```

### 23. Merge k Sorted Lists

合并k个有序列表。
[查看原题](https://leetcode.com/problems/merge-k-sorted-lists/)

```
Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6
```

+ 方法一：Brute Force. 时间效率O(NlogN)

    ```python
    def searchRange(self, nums, target):
    
        left_idx = self.search_edge(nums, target, True)
        if left_idx == len(nums) or nums[left_idx] != target:
            return [-1, -1]
        return [left_idx, self.search_edge(nums, target, False)-1]
        
    def search_edge(self, nums, target, left):
        l, r = 0, len(nums)
        while l < r:
            mid = (l+r) // 2
            if nums[mid] > target or (left and nums[mid]==target):
                r = mid
            else:
                l = mid + 1
        return l
    ```

+ 方法二：优先级队列。本来优先级就没有方法一快，再加上Python3中的比较符机制不同，导致要实现__lt__方法，就更慢了。不过理论时间复杂度是比方法一小的。Time: O(Nlogk)

    ```python
    class CmpNode:
        
        def __init__(self, node):
            self.node = node
            
        def __lt__(self, other):
            return self.node.val < other.node.val
        
    class Solution:
            
        def mergeKLists(self, lists: List[ListNode]) -> ListNode:
            from queue import PriorityQueue
            head = h = ListNode(0)
            q = PriorityQueue()
            for l in lists:
                if l:
                    q.put(CmpNode(l))
            while not q.empty():
                to_add = q.get().node
                h.next = to_add
                h = h.next
                if to_add.next:
                    q.put(CmpNode(to_add.next))
            return head.next
    ```


+ 方法三：规避ListNode的比较，以解决上述问题。只要加上该链表在原数组中的索引位置，就一定不会重复，从而忽略对ListNode的比较。

    ```python
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        from queue import PriorityQueue
        q = PriorityQueue()
        for idx, l in enumerate(lists):
            if l:
                q.put((l.val, idx, l))
        h = head = ListNode(0)
        while not q.empty():
            val, idx, node = q.get()
            h.next = node
            h, node = h.next, node.next
            if node:
                q.put((node.val, idx, node))
        return head.next
    ```

+ 方法四：俩俩合并。Time: O(Nlogk)

    ```python
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        def merge_both(l1, l2):
            if not l1 or not l2:
                return l1 or l2
            if l1.val <= l2.val:
                l1.next = merge_both(l1.next, l2)
                return l1
            else:
                l2.next = merge_both(l1, l2.next)
                return l2
            
        pairs = list(lists)
        while len(pairs) > 1:
            n = len(pairs)
            if n & 1 == 1:
                pairs.append(None)
            pairs = [merge_both(pairs[i*2], pairs[i*2+1])
                     for i in range(((n+1)//2))]
        return pairs[0] if pairs else None
    ```


### 141. Linked List Cycle

判断一个链表是否有环。
[查看原题](https://leetcode.com/problems/linked-list-cycle/description/)

+ 经典的一道题，看成两个人在赛跑，如果有环，快的人会和慢的人相遇

    ```python
    def hasCycle(self, head):
        slow = fast = head:
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if fast is slow:
                return True
        return False
    ```

### 142. Linked List Cycle II

求链表中环的入口节点。
[查看原题](https://leetcode.com/problems/guess-number-higher-or-lower/description/)


+ 首先判断此链表是否有环。然后在相交点和头结点一起走，一定会在入口相遇。

    ```python
    def detectCycle(self, head):        
        fast = slow = head
        # 检测是否有环
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:
                break
        else:
            return None
        # 找出入口节点
        while head is not slow:
            head, slow = head.next, slow.next
        return head
    ```


### 206. Reverse Linked List

倒置一个链表。
[查看原题](https://leetcode.com/problems/reverse-linked-list/description/)

```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

+ 方法一： iteratively

    ```python
    def reverseList(head):
        prev = None
        while head:
            cur = head
            head = head.next
            cur.next = prev
            prev = cur
        return prev
    ```

+ 方法二：使用一行赋值

    ```python
    def reverseList(self, head):
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev
    ```

+ 方法三：递归

    ```python
    def reverseList(self, head, prev=None):
        if not head:
          return prev
      
        cur, head.next = head.next, prev
        return self.reverseList(cur, head)
    ```

### 92. Reverse Linked List II

跟上题不同的是，只倒置指定区间的部分。
[查看原题](https://leetcode.com/problems/reverse-linked-list-ii/)

```
Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
```

+ iteratively

    ```python
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
    
        root = h = ListNode(0)
        h.next = head
        
        for _ in range(m-1):
            h = h.next
        cur_head = h
        p1 = p2 = cur_head.next
        for _ in range(n-m):
            p2 = p2.next
        prev = p2.next if p2 else None
        if p2:
            p2.next = None
        while p1:
            p1.next, prev, p1 = prev, p1, p1.next
        cur_head.next = prev
        return root.next
    ```

### 160. Intersection of Two Linked Lists

两个链表求相交。
[查看原题](https://leetcode.com/problems/intersection-of-two-linked-lists/description/)

+ 解法

    ```python
    def getIntersectionNode(self, headA, headB):
        p1, p2 = headA, headB
        while p1 is not p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return p1
    ```


### 138. Copy List with Random Pointer

深拷贝一个复杂链表，链表多包含了一个随机指针。
[查看原题](https://leetcode.com/problems/copy-list-with-random-pointer/description/)

+ 第一次迭代的过程委托给了defaultdict，通过创建一个默认的对象，再去修改它的label值。

    ```python
    def copyRandomList(self, head):
        from collections import defaultdict
        cp = defaultdict(lambda: RandomListNode(0))
        cp[None] = None
        n = head
        while n:
            cp[n].label = n.label
            cp[n].next = cp[n.next]
            cp[n].random = cp[n.random]
            n = n.next
        return cp[head]
    ```

### 237. Delete Node in a Linked List

在链表中删除节点。给定的节点不是尾节点。
[查看原题](https://leetcode.com/problems/delete-node-in-a-linked-list/description/)

```
Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
```

+ 这道题关键在于复制

    ```python
    def deleteNode(self, node):
        node.val = node.next.val  # 4->1->1->9
        node.next = node.next.next  # 4->1->9

    ```

### 203. Remove Linked List Elements

删除链表中值为val的元素。
[查看原题](https://leetcode.com/problems/remove-linked-list-elements/)

+ 方法一：遍历head并构建新的ListNode。

    ```python
    def removeElements(self, head, val):
        l = res = ListNode(0)
        while head:
            if head.val != val:
                l.next = ListNode(head.val)
                l = l.next
            head = head.next
        return res.next
    ```

+ 方法二：更喜欢这个方法。

    ```python
    def removeElements(self, head: 'ListNode', val: 'int') -> 'ListNode':
        l = ListNode(0)
        l.next, ans = head, l
        while l and l.next:
            if l.next.val == val:
                l.next = l.next.next
            else:
                l = l.next
        return ans.next
    ```


### 83. Remove Duplicates from Sorted List

删除有序链表中重复的节点。
[查看原题](https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/)

+ 解法

    ```python
    def delete_duplicates(head):
        root = head
        while head and head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        return root
    ```


### 82. Remove Duplicates from Sorted List II

和上题不同的是，重复的节点要全部删除。
[查看原题](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

```
Input: 1->2->3->3->4->4->5
Output: 1->2->5
```

+ 解法

    ```python
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        prev = ans = ListNode(0)        
        prev.next = h = head
        
        while h and h.next:
            remove = False
            while h.next and h.val == h.next.val:
                h.next = h.next.next
                remove = True
            if remove:
                prev.next = h.next
            else:
                prev = prev.next
            h = h.next
        return ans.next
    ```

### 876. Middle of the Linked List

链表中点，如果偶数个，则返回第二个节点。
[查看原题](https://leetcode.com/problems/middle-of-the-linked-list/)

```
Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).
Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.

Output: Node 4 from this list (Serialization: [4,5,6])
Since the list has two middle nodes with values 3 and 4, we return the second one.
```

+ 解法

    ```python
    def middleNode(self, head: 'ListNode') -> 'ListNode':
        fast = slow = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        return slow
    ```

### 234. Palindrome Linked List

判断一个链表是否是回文链表。
[查看原题](https://leetcode.com/problems/palindrome-linked-list/)

```
Input: 1->2->2->1
Output: true
```

+ 方法一：此题为倒置链表和快慢指针的总和应用。

    ```python
    def isPalindrome(self, head: 'ListNode') -> 'bool':
        rev = None
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow.next, rev, slow = rev, slow, slow.next
        if fast:
            slow = slow.next
        while rev and rev.val == slow.val:
            rev, slow = rev.next, slow.next
        return rev is None
    ```

+ 方法二：上述方法有一个缺点就是改变了原始的head，这里进行一些改进。

    ```python
    def isPalindrome(self, head):
        rev = None
        fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, head = head, rev, head.next
        tail = head.next if fast else head
        isPali = True
        while rev:
            isPali = isPali and rev.val == tail.val
            head, head.next, rev = rev, head, rev.next
            tail = tail.next
        return isPali
    ```


### 24. Swap Nodes in Pairs

成对转换链表。
[查看原题](https://leetcode.com/problems/swap-nodes-in-pairs/)

```
Given 1->2->3->4, you should return the list as 2->1->4->3.
```

+ 解法

    ```python
    def swapPairs(self, head: ListNode) -> ListNode:
        prev, prev.next = self, head
        while prev.next and prev.next.next:
            a = prev.next    # current
            b = a.next
            prev.next, b.next, a.next = b, a, b.next
            prev = a
        return self.next
    ```


### 19. Remove Nth Node From End of List

删除倒数第N个节点。
[查看原题](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

```
Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
```

+ 解法

    ```python
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        root = slow = fast = ListNode(0)
        slow.next = head
        while n >= 0 and fast:
            fast = fast.next
            n -= 1
        while fast:
            slow, fast = slow.next, fast.next
        slow.next = slow.next.next if slow.next else None
        return root.next
    ```


### 328. Odd Even Linked List 

重排链表，使奇数位节点在前，偶数位节点在后，就地排序。
[查看原题](https://leetcode.com/problems/odd-even-linked-list/)

```
Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
```

+ 解法

    ```python
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        odd = head
        even_h = even = head.next
        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next
            even = even.next
        odd.next = even_h
        return head
    ```


### 148. Sort List

给链表排序。
[查看原题](https://leetcode.com/problems/sort-list/)

```
Input: 4->2->1->3
Output: 1->2->3->4
```

+ 解法

    ```python
    def sortList(self, head: ListNode) -> ListNode:
        
        def merge_both(l1, l2):
            l = h = ListNode(0)
            while l1 and l2:
                if l1.val <= l2.val:
                    l.next, l1 = l1, l1.next
                else:
                    l.next, l2 = l2, l2.next
                l = l.next
            l.next = l1 or l2
            return h.next
        
        def merge_sort(h):
            if not h or not h.next:
                return h
            slow = fast = h
            prev = None
            while fast and fast.next:
                prev, slow, fast = slow, slow.next, fast.next.next
            prev.next = None
            left = merge_sort(h)
            right = merge_sort(slow)
            return merge_both(left, right)
        
        return merge_sort(head)
    ```


### 817. Linked List Components

链表的组件。给定一个集合G，然后根据是否在G中分成若干部分，求连起来在G中的部分的个数。
[查看原题](https://leetcode.com/problems/linked-list-components/)

```
Input: 
head: 0->1->2->3->4
G = [0, 3, 1, 4]
Output: 2
Explanation: 
0 and 1 are connected, 3 and 4 are connected, so [0, 1] and [3, 4] are the two connected components.
```

+ 解法

    ```python
    def numComponents(self, head: ListNode, G: List[int]) -> int:
        SET_G = set(G)
        h = head
        count = 0
        while h:
            if h.val in SET_G:
                if (h.next and h.next.val not in SET_G or 
                    not h.next):
                    count += 1
            h = h.next
        return count
    ```


### 86. Partition List

链表分区，将比x小的节点放到前面，其余节点放到后面，并保持原有顺序。
[查看原题](https://leetcode.com/problems/partition-list/)

```
Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
```

+ 解法

    ```python
    def partition(self, head: ListNode, x: int) -> ListNode:
        lt = letter = ListNode(0)
        gt = greater = ListNode(0)
        h = head
        while h:
            if h.val < x:
                lt.next = h
                lt = h
            else:
                gt.next = h
                gt = h
            h = h.next
        gt.next = None   # important !!
        lt.next = greater.next
        return letter.next
    ```


### 61. Rotate List

向右旋转链表k次。
[查看原题](https://leetcode.com/problems/rotate-list/)

```
Input: 0->1->2->NULL, k = 4
Output: 2->0->1->NULL
Explanation:
rotate 1 steps to the right: 2->0->1->NULL
rotate 2 steps to the right: 1->2->0->NULL
rotate 3 steps to the right: 0->1->2->NULL
rotate 4 steps to the right: 2->0->1->NULL
```

+ 解法

    ```python
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        n, cur, prev = 0, head, None
        while cur:
            n += 1
            prev, cur = cur, cur.next
        
        if n==0 or k%n==0:
            return head
        k = k % n
        tail = head
        for _ in range(n-k-1):
            tail = tail.next
        ans, tail.next, prev.next = tail.next, None, head
        return ans
    ```


### 725. Split Linked List in Parts

按部分拆分链表。如果不能整除，要保证前面部分的大。
[查看原题](https://leetcode.com/problems/split-linked-list-in-parts/)

```
Input: 
root = [1, 2, 3], k = 5
Output: [[1],[2],[3],[],[]]
Input: 
root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
```

+ 解法

    ```python
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        n, cur = 0, root
        ans = []
        while cur:
            n += 1
            cur = cur.next
        parts, remain = divmod(n, k)
        h = root
        for i in range(k):
            head = h
            for i in range(parts-1+(i<remain)):
                h = h.next
            if h:
                h.next, h = None, h.next
            ans.append(head)                            
        return ans
    ```


### 143. Reorder List

链表头尾捡取直至结束。
[查看原题](https://leetcode.com/problems/reorder-list/)

```
Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
```

+ 解法

    ```python
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        
        tail, slow.next = slow.next, None
        def reverse(node):
            prev = None
            while node:
                node.next, prev, node = prev, node, node.next
            return prev
        tail = reverse(tail)
        h = head
        while h and tail:
            h.next, tail.next, tail, h = tail, h.next, tail.next, h.next
    ```


### 1030. Next Greater Node In Linked List

链表中下一个比当前节点大的值。和503题类似。
[查看原题](https://leetcode.com/problems/next-greater-node-in-linked-list/)

```
Input: [2,7,4,3,5]
Output: [7,0,5,5,0]
```

+ 解法

    ```python
    def nextLargerNodes(self, head: ListNode) -> List[int]:
        ans, stack = [], []
        while head:
            while stack and stack[-1][1] < head.val:
                ans[stack.pop()[0]] = head.val
            stack.append((len(ans), head.val))
            ans.append(0)
            head = head.next
        return ans
    ```


### 1171. Remove Zero Sum Consecutive Nodes from Linked List

移除相连和为0的节点。像祖玛一样，连续地删除。答案不唯一。
[查看原题](https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/)

```
Input: head = [1,2,-3,3,1]
Output: [3,1]
Note: The answer [1,2,1] would also be accepted.
```

+ 解法

    ```python
    def removeZeroSumSublists(self, head: ListNode) -> ListNode:
        p = dummy = ListNode(0)
        dummy.next = head
        s = 0
        s_sum = [s]
        vals = {}
        while p:
            s += p.val
            s_sum.append(s)
            if s not in vals:
                vals[s] = p
            else:
                vals[s].next = p.next
                s_sum.pop() # remove cur, keep the last
                while s_sum[-1] != s:
                    vals.pop(s_sum.pop())
            p = p.next
        return dummy.next
    ```

