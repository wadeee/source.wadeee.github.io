---
layout: post
title: Binary Search(二分法)
published: true
date: 2016-10-08
---

> 在单调中寻找，在起伏中失效

## 关于二分法

单调区间中寻找特定元素的高效算法。

## 使用核心

+ 区间单调

    + 在单调模型上求目标解，非单调模型不可使用

+ 时间效率`O(logn)`

+ 核心代码

    ```python
    def binarysearch(array, target):
        head = 0
        tail = len(array) - 1
        while head < tail:
            mid = (head + tail) >> 1
            if array[mid] < target:
                head = mid + 1
            else:
                tail = mid
        return head
    ```

## 技巧

+ 题意反推

    很多需要用二分法的题目，会在数据范围上暴露信息。
    
    比如`(0 < M <= 100000000)`，
    这种数据范围一般会和时间复杂度为`O(logn)`的算法有关系，
    快排堆排线段树等等。

## LeetCode真题

### 704. Binary Search

用二分法在有序数组中查找元素。
[查看原题](https://leetcode.com/problems/binary-search/description/)

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

+ 方法一：实现原理。

    ```python
    def binary_search(nums, target):
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l+r) // 2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else:
                return mid
        return -1
    ```

+ 方法二：使用标准库。

    ```python
    def search(self, nums, target):
        from bisect import bisect_left 
        index = bisect_left(nums, target)
        return index if index < len(nums) and nums[index] == target else -1
    ```

### 35. Search Insert Position

给定一个target，插入到一个有序数组中，假定数组中无重复元素。
[查看原题](https://leetcode.com/problems/search-insert-position/description/)

```
Input: [1,3,5,6], 5
Output: 2
```

+ 方法一：实现原理。

    ```python
    def binary_insert(nums, target):
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (l+r) // 2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else:
                return mid
        return l
    ```

+ 方法二：使用标准库。

    ```python
    def searchInsert(self, nums, target):
        from bisect import bisect_left
        return bisect_left(nums, target)
    ```
  

### 153. Find Minimum in Rotated Sorted Array

通过一个排序数组旋转后的结果，找出最小元素。
[查看原题](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/)

```
Input: [3,4,5,1,2] 
Output: 1
```

+ 思路：通过二分法不断缩小范围，由于mid是整除，最后l==mid，并且nums[mid] > nums[r]的。

    ```python
    def find_min(nums):
        l, r = 0, len(nums)-1
        if nums[l] < nums[r]:
            return nums[l]
        while l <= r:
            mid = (l+r) // 2
            if nums[mid] > nums[l]:
                l = mid
            elif nums[mid] < nums[r]:
                r = mid
            else:
                return nums[r]
    ```

### 34. Find First and Last Position of Element in Sorted Array

有序数组中查找数组，返回数字的索引范围。
[查看原题](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

+ 解法

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


### 278. First Bad Version

找出提交版本中的bad version。
[查看原题](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

```
Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version.
```

+ 解法

    ```python
    def firstBadVersion(self, n):
        l, r = 1, n
        while l <= r:
            mid = (l+r) // 2
            if isBadVersion(mid):
                r = mid - 1
            else:
                l = mid + 1
        return l
    ```

### 374. Guess Number Higher or Lower

猜数游戏1~n，每猜一次会告诉你答案是更小还是更大。
[查看原题](https://leetcode.com/problems/guess-number-higher-or-lower/description/)

```
def guess(num):
    return
    -1 : My number is lower
     1 : My number is higher
     0 : Congrats! You got it!
     
Input: n = 10, pick = 6
Output: 6
```

+ 方法一：实现原理。

    ```python
    def guessNumber(self, n):
        l, r = 1, n
        while l <= r:
            mid = (l+r) // 2
            if guess(mid) == -1:
                r = mid - 1
            elif guess(mid) == 1:
                l = mid + 1
            else:
                return mid
    ```

+ 方法二：使用标准库。核心思想为将guess返回的结果转为一个数组，然后使用二分法查找。

    ```python
    def guessNumber(self, n):
        from bisect import bisect, bisect_left
        class C:
            def __getitem__(self, x):
                return -guess(x)
        # return bisect(C(), -1, 1, n)
        return bisect_left(C(), 0, 1, n)
    ```

    解析：以n=10, pick=6为例。实际上C class相当于:
    
    ```
    ary = map(lambda x: -guess(x), range(1, n+1))
    ary.insert(0, None)
    # ary = [None, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1]
    return bisect(ary, -1, 1, n)
    ```

    而索引又是从1开始，所以这里在前面添加了一个None，实际上将题转为了查找ary的0，问题便迎刃而解。
    值得注意的是，如果使用了map，会导致空间，时间复杂度增加，而使用class的方法，并没有求出整个的list，
    所以效率更高。


### 744. Find Smallest Letter Greater Than Target

找出比目标大的最小字母，没有的返回首字母
[查看原题](https://leetcode.com/problems/find-smallest-letter-greater-than-target/)

```
Input:
letters = ["c", "f", "j"]
target = "d"
Output: "f"

Input:
letters = ["c", "f", "j"]
target = "g"
Output: "j"

Input:
letters = ["c", "f", "j"]
target = "j"
Output: "c"
```

+ 方法一：实现原理。

    ```python
    def nextGreatestLetter(self, letters: 'List[str]', target: 'str') -> 'str':
        lo, hi = 0, len(letters)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            if letters[mid] > target:
                hi = mid -1
            elif letters[mid] <= target:
                lo = mid + 1
        return letters[lo % len(letters)]
    ```

+ 方法二：使用库。

    ```python
    def nextGreatestLetter(self, letters: 'List[str]', target: 'str') -> 'str':
        index = bisect.bisect(letters, target)
        return letters[index % len(letters)]
    ```


### 852. Peak Index in a Mountain Array

找到数组中的峰值。假设峰值一定存在。
[查看原题](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

```
Input: [0,2,1,0]
Output: 1
```

+ 方法一：线性枚举O(n)。

    ```python
    def peakIndexInMountainArray(self, A: 'List[int]') -> 'int':
        for i in range(1, len(A)-1):
            if A[i] > A[i+1]:
                return i
    ```

+ 方法二：max函数

    ```python
    def peakIndexInMountainArray(self, A: 'List[int]') -> 'int':
        return A.index(max(A))
    ```

+ 方法三：二分法

    ```python
    def peakIndexInMountainArray(self, A: 'List[int]') -> 'int':
        lo, hi = 0, len(A)-1
        while lo < hi:
            mid = (lo + hi) // 2
            if A[mid] > A[mid+1]:
                hi = mid
            else:
                lo = mid + 1
        return lo
    ```

+ 方法四：黄金分割法，应用在单峰函数求极值，速度比二分法要快。

    ```python
    def peakIndexInMountainArray(self, A: 'List[int]') -> 'int':
        
        def gold1(i, j):
            return i + int(round((j-i) * 0.382))
        def gold2(i, j):
            return i + int(round((j-i) * 0.618))
        
        l, r = 0, len(A) - 1
        x1, x2 = gold1(l, r), gold2(l, r)
        while x1 < x2:
            if A[x1] < A[x2]:
                l = x1
                x1 = x2
                x2 = gold1(x1, r)
            else:
                r = x2
                x2 = x1
                x1 = gold2(l, x2)
        return x1
    ```


### 1014. Capacity To Ship Packages Within D Days

n天内轮船运送的最小容量
[查看原题](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

```
Input: weights = [1,2,3,4,5,6,7,8,9,10], D = 5
Output: 15
Explanation: 
A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.
```

+ 二分结果快速出解

    ```python
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        lo, hi = max(weights), sum(weights)
        while lo <= hi:
            mid, days, cur = (lo + hi) // 2, 1, 0
            for w in weights:
                if cur+w > mid:
                    days += 1
                    cur = 0
                cur += w
            if days > D:
                lo = mid + 1
            else:
                hi = mid - 1
            # print(lo, mid, hi)
        return lo
    ```


### 875. Koko Eating Bananas

这道题思路和1014一样。不同的是，如果当前堆的香蕉小于吃的速度，那么也不能吃下一堆。
[查看原题](https://leetcode.com/problems/koko-eating-bananas/)

```
Input: piles = [3,6,7,11], H = 8
Output: 4
```

+ 二分结果快速出解

    ```python
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        lo, hi = 1, max(piles) 
        while lo <= hi:
            mid = (lo + hi ) >> 1
            # needs = sum(math.ceil(p/mid) for p in piles)   # slower
            needs = sum((p-1)//mid+1 for p in piles)
            if needs > H:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo
    ```

### 1145. Binary Tree Coloring Game

二叉树染色游戏。两个人轮流给二叉树染色，每次只能染相邻位的节点，给定第一个人染色的位置，问第二个人是否能够必胜。
[查看原题](https://leetcode.com/problems/binary-tree-coloring-game/)

```
Input: piles = [3,6,7,11], H = 8
Output: 4
```

+ 关键的一点需要想明白，从第一个人染色的地方，有三个分支，如果有一个分支可以大于整个节点的一半，那么第二个人选择这个分支，就能赢得比赛

    ```python
    def btreeGameWinningMove(self, root: TreeNode, n: int, x: int) -> bool:
        count = [0, 0]
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            if node.val == x:
                count[0] = left
                count[1] = right
            return left + right + 1
    
        dfs(root)
        return max(max(count), n - sum(count) - 1) > n // 2
    ```
