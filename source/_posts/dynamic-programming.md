---
layout: post
title: Dynamic Programming(动态规划)
published: true
date: 2016-11-13
---

> 你一定听过基于连通性状态压缩的动态规划问题

## 关于动态规划

动态规划（Dynamic Programming，DP）是运筹学的一个分支，是求解决策过程最优化的过程。20世纪50年代初，美国数学家贝尔曼（R.Bellman）等人在研究多阶段决策过程的优化问题时，提出了著名的最优化原理，从而创立了动态规划。动态规划的应用极其广泛，包括工程技术、经济、工业生产、军事以及自动化控制等领域，并在背包问题、生产经营问题、资金管理问题、资源分配问题、最短路径问题和复杂系统可靠性问题等中取得了显著的效果 [1]  。

## LeetCode真题

### 70. Climbing Stairs

爬楼梯，一次可以爬一阶或两阶楼梯，爬上n阶楼梯有多少种方法？
[查看原题](https://leetcode.com/problems/climbing-stairs/description/)

+ 解法

    ```python
    def fibonacci(n):
        a = b = 1
        for _ in range(n-1):
            a, b = b, a+b
        return b
    ```


### 746. Min Cost Climbing Stairs

楼梯上每层写了到达该层的卡路里，求上到顶层消耗的最小卡路里。
[查看原题](https://leetcode.com/problems/min-cost-climbing-stairs/)

```
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.

Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
```

+ 到达一层有两种选择，一种是上一层，一种是上两层。

    ```python
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        f1 = f2 = 0
        for x in reversed(cost):
            f1, f2 = min(f1, f2) + x, f1
        return min(f1, f2)
    ```


### 121. Best Time to Buy and Sell Stock

买入卖出最大收益。原题
[查看原题](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

+ 其实就是求最高峰点和前面最低谷点的差。

    ```python
    def maxProfit(self, prices: List[int]) -> int:
        ans, min_buy = 0, float('inf')
        for price in prices:
            if price < min_buy:
                min_buy = price
            elif price-min_buy > ans:
                ans = price - min_buy
        return ans
    ```

### 122. Best Time to Buy and Sell Stock II

买入卖出，允许多次交易。
[查看原题](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/)

```
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
```

+ 比较每两天的价格，如果是涨价了，那就把收益计算进去，否则不出手交易。

    ```python
    def max_profit(prices):
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]
        return profit
    ```

+ 方法二：弗洛伊德算法，这个时间复杂度为O(N^3)，space: O(N^2)但是代码简单。

    ```python
    def findTheCity(self, n: int, edges: List[List[int]], maxd: int) -> int:
        dis = [[float('inf')] * n for _ in range(n)]
        for i, j, w in edges:
            dis[i][j] = dis[j][i] = w
        for i in range(n):
            dis[i][i] = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j])
        ans = {sum(d<=maxd for d in dis[i]): i for i in range(n)}  # 这里id大的会将小的覆盖
        return ans[min(ans)]
    ```


### Best Time to Buy and Sell Stock III

最多允许交易两次。
[查看原题](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)


+ 先从左到右按照一次交易计算每天的利润。然后按照从右到左，判断如果进行第二次交易，最大的利润。

    ```python
    def maxProfit(self, prices: List[int]) -> int:
        min_buy = float('inf')
        profits = []
        max_profit = 0
        for p in prices:
            min_buy = min(min_buy, p)
            max_profit = max(max_profit, p-min_buy)
            profits.append(max_profit)
        
        max_profit = 0
        total_profit = 0
        max_sell = float('-inf')
        for i in range(len(prices)-1, -1, -1):
            max_sell = max(max_sell, prices[i])
            max_profit = max(max_profit, max_sell-prices[i])
            total_profit = max(total_profit, max_profit+profits[i])
        return total_profit
    ```

### 198. House Robber

抢劫房子问题。不能连续抢劫两个挨着的房间。
[查看原题](https://leetcode.com/problems/house-robber)

```
Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.
```

+ 解法

    ```python
    def rob(self, nums):
        last, now = 0, 0
        for num in nums:
            last, now = now, max(last+num, now)
        return now
    ```


### 213. House Robber II

与上题不同的是，所有的房子连成一个环。
[查看原题](https://leetcode.com/problems/house-robber-ii/)

```
Input: [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
             because they are adjacent houses.
```

+ 注意nums长度为1的情况。

    ```python
    def rob(self, nums: List[int]) -> int:
        
        def robber(nums):
            last = now = 0
            for num in nums:
                last, now = now, max(last+num, now)
            return now
        return max(robber(nums[:-1]), robber(nums[len(nums)!=1:]))
    ```

### 303. Range Sum Query - Immutable

给定一个数组，计算索引i, j之间的和。
[查看原题](https://leetcode.com/problems/range-sum-query-immutable/)

```
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
```

+ 思路：如果单纯采用切片计算，效率过低，题中要求sumRange调用多次。所以这里采用动态规划。

    ```python
    class NumArray:
        
        def __init__(self, nums):
            # self.sum_item = [0]
            # for num in nums:
            #     self.sum_item.append(self.sum_item[-1] + num)
            from itertools import accumulate
            from operator import add
            self.sum_item = list(accumulate(nums, add))
    
        def sumRange(self, i, j):
            # return self.sum_item[j+1] - self.sum_item[i] 
            return self.sum_item[j] - self.sum_item[i-1] if i > 0 else self.sum_item[j]
    ```

### 91. Decode Ways

将数字翻译成字母有多少种方式。
[查看原题](https://leetcode.com/problems/decode-ways/)

```
Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

+ 解法

    ```python
    def numDecodings(self, s: str) -> int:
        # w tells the number of ways
        # v tells the previous number of ways
        # d is the current digit
        # p is the previous digit
        v, w, p = 0, int(s>''), ''
        for d in s:
            v, w, p = w, int(d>'0')*w + (9<int(p+d)<27)*v, d
        return w
    ```


### 62. Unique Paths

一个矩阵中，从左上走到右下有多少种不同走法，每次只能向右或向下移动。
[查看原题](https://leetcode.com/problems/unique-paths/)


+ 方法一：构建二维矩阵。

    ```python
    def uniquePaths(self, m: int, n: int) -> int:
        g = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if i==0 or j==0:
                    g[i][j] = 1
                else:
                    g[i][j] = g[i-1][j] + g[i][j-1]
            
        return g[-1][-1]
    ```

+ 方法二：二维数组时没有必要的，仔细观察发现每层都是累计的关系，accumulate为此而生。

    ```python
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * m
        for _ in range(n-1):
            row = itertools.accumulate(row)
        return list(row)[-1]
    ```

### 63. Unique Paths II

和62一样，不同的是中间加了障碍1。
[查看原题](https://leetcode.com/problems/unique-paths-ii/)

```
Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2
```

+ 解法

    ```python
    def uniquePathsWithObstacles(self, g: List[List[int]]) -> int:
        R, C = len(g), len(g[0])
        if g[0][0] == 1:
            return 0
        g[0][0] = 1
        for i in range(1, C):
            g[0][i] = int(g[0][i-1]==1 and g[0][i]==0)
    
        for j in range(1, R):
            g[j][0] = int(g[j-1][0]==1 and g[j][0]==0)
    
        for i in range(1, R):
            for j in range(1, C):
                if g[i][j] == 0:
                    g[i][j] = g[i-1][j] + g[i][j-1]
                else:
                    g[i][j] = 0
        return g[-1][-1]
    ```

### 120. Triangle

三角形从上到下最小路径。
[查看原题](https://leetcode.com/problems/triangle/)

```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
i.e., 2 + 3 + 5 + 1 = 11
```

+ 错位相加大法。

    ```python
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        from functools import reduce
        def combine_rows(lower_row, upper_row):
            return [upper + min(lower_left, lower_right)
                    for upper, lower_left, lower_right in 
                    zip(upper_row, lower_row, lower_row[1:])]
        return reduce(combine_rows, triangle[::-1])[0]
    ```

### 931. Minimum Falling Path Sum

和120相似，不过形状变成了矩形。
[查看原题](https://leetcode.com/problems/minimum-falling-path-sum/)

```
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: 12
Explanation: 
The possible falling paths are:
```

+ 方法一：常规写法。

    ```python
    def minFallingPathSum(self, A: List[List[int]]) -> int:
        R, C = len(A), len(A[0])
        for i in range(R-2, -1, -1):
            for j in range(C):
                path = slice(max(0, j-1), min(C, j+2))
                A[i][j] += min(A[i+1][path])
        return min(A[0])
    ```

+ 方法二：错位计算的方式，这个比120三角形的要复杂一点。需要填充无穷大来使生效。

    ```python
    def minFallingPathSum(self, A: List[List[int]]) -> int:
        from functools import reduce
        padding = [float('inf')]
        def combine_rows(lower_row, upper_row):
            return [upper + min(lower_left, lower_mid, lower_right)
                    for upper, lower_left, lower_mid, lower_right in
                    zip(upper_row, lower_row[1:]+padding, lower_row, padding+lower_row[:-1])]
        return min(reduce(combine_rows, A[::-1]))
    ```

### 1289. Minimum Falling Path Sum II

上题变形，每行找到非自己那列的元素。
[查看原题](https://leetcode.com/problems/minimum-falling-path-sum-ii/)


+ 用堆记录2个最小的值。

    ```python
    def minFallingPathSum(self, arr: List[List[int]]) -> int:
        m, n = len(arr), len(arr[0])
        for i in range(1, m):
            r = heapq.nsmallest(2, arr[i-1])
            for j in range(n):
                arr[i][j] += r[1] if arr[i-1][j]==r[0] else r[0]
        return min(arr[-1])
    ```

### 279. Perfect Squares

完美平方，找出n的最少的能被几个平方数相加。
[查看原题](https://leetcode.com/problems/perfect-squares/)

```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```

+ f(n)表示n最少的个数。f(n)=min(f(n-1²), f(n-2²)...f(0)) + 1

    ```python
    class Solution:
        _dp = [0]
        def numSquares(self, n: int) -> int:
            dp = self._dp
            while len(dp) <= n:
                # dp.append(min(dp[len(dp)-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1)
                dp.append(min(dp[-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1)
            return dp[n]
    ```

### 5. Longest Palindromic Substring

最长回文子字符串。
[查看原题](https://leetcode.com/problems/longest-palindromic-substring/)

```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

+ 马拉车算法。Time: O(n).

    ```python
    def longestPalindrome(self, s: str) -> str:
        # Transform S into T.
        # For example, S = "abba", T = "^#a#b#b#a#$".
        # ^ and $ signs are sentinels appended to each end to avoid bounds checking
        T = '#'.join('^{}$'.format(s))
        n = len(T)
        P = [0] * n
        C = R = 0
        for i in range (1, n-1):
            P[i] = (R > i) and min(R - i, P[2*C - i]) # equals to i' = C - (i-C)
            # Attempt to expand palindrome centered at i
            while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
                P[i] += 1
    
            # If palindrome centered at i expand past R,
            # adjust center based on expanded palindrome.
            if i + P[i] > R:
                C, R = i, i + P[i]
    
        # Find the maximum element in P.
        maxLen, centerIndex = max((n, i) for i, n in enumerate(P))
        return s[(centerIndex  - maxLen)//2: (centerIndex  + maxLen)//2]
    ```

### 1024. Video Stitching

影片剪辑，给定n组影片段，求能够拼出0~T完整影片所使用的最小段数。
[查看原题](https://leetcode.com/problems/video-stitching/)

```
Input: clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
Output: 3
Explanation: 
We take the clips [0,2], [8,10], [1,9]; a total of 3 clips.
Then, we can reconstruct the sporting event as follows:
We cut [1,9] into segments [1,2] + [2,8] + [8,9].
Now we have segments [0,2] + [2,8] + [8,10] which cover the sporting event [0, 10].
```

+ 解法

    ```python
    def videoStitching(self, clips: List[List[int]], T: int) -> int:
        end, end2, cnt = -1, 0, 0   # end 表示上一段最后截止点，end2表示当前可以最大延伸的最远地点。
        for s, e in sorted(clips):
            if end2 >= T or s > end2:   # 完成或者接不上了
                break
            elif end < s <= end2:       # 续1s
                cnt += 1
                end = end2
            end2 = max(end2, e)
        return cnt if end2 >= T else -1
    ```

### 1048. Longest String Chain

每个字符添加任意一个字符，可以组成一个字符串链。
[查看原题](https://leetcode.com/problems/longest-string-chain/)

+ 解法

    ```python
    def longestStrChain(self, words: List[str]) -> int:
        words2 = {i:set() for i in range(1, 17)}
        for word in words:
            words2[len(word)].add(word)
        dp = collections.defaultdict(lambda : 1)
        for k in range(2, 17):
            for w in words2[k]:
                for i in range(k):
                    prev = w[:i] + w[i+1:]
                    if prev in words2[k-1]:
                        # dp[w] = max(dp[w], dp[prev]+1)
                        dp[w] = dp[prev] + 1
        return max(dp.values() or [1])
    ```

### 1143. Longest Common Subsequence

最长公共子串的长度。
[查看原题](https://leetcode.com/problems/longest-common-subsequence/)

```
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
```

+ 方法一：递归

    ```python
    import functools
    class Solution:
        def longestCommonSubsequence(self, text1: str, text2: str) -> int:
            @functools.lru_cache(None)
            def helper(i,j):
                if i<0 or j<0:
                    return 0
                if text1[i]==text2[j]:
                    return helper(i-1,j-1)+1
                return max(helper(i-1,j),helper(i,j-1))
            return helper(len(text1)-1,len(text2)-1)
    ```

+ 方法二：迭代。dp(i,j) means the longest common subsequence of text1[:i] and text2[:j].

    ```python
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1, n2 = len(text1), len(text2)
        dp = [[0]*(n2+1) for i in range(n1+1)]
        for i in range(n1):
            for j in range(n2):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[-1][-1]
    ```

### 1312. Minimum Insertion Steps to Make a String Palindrome

将一个字符串变为回文串，最小插入字母步数。
[查看原题](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)

```
Input: s = "zzazz"
Output: 0
Explanation: The string "zzazz" is already palindrome we don't need any insertions.
Input: s = "leetcode"
Output: 5
Explanation: Inserting 5 characters the string becomes "leetcodocteel".
```

+ 和1134，Longest Common Subsequence一样，当这个字符串和他倒序的公共子串越多，需要添加的字母就越少。

    ```python
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * (n+1) for _ in range(n+1)]
        for i in range(n):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j] + 1 if s[i] == s[~j] else max(dp[i+1][j], dp[i][j+1])
        return n - dp[n][n]
    ```

### 221. Maximal Square

最大的正方形岛屿面积。
[查看原题](https://leetcode.com/problems/maximal-square/)

```
Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4
```

+ 方法一：此题看似和最大岛屿面积相似，但解法完全不同。

    ```python
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * (n+1) for _ in range(m+1)]
        max_side = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1]) + 1
                    max_side = max(max_side, dp[i+1][j+1])
        return max_side**2
    ```

### 1340. Jump Game V

跳跃游戏，可以向左右d范围内矮的地方跳下。
[查看原题](https://leetcode.com/problems/jump-game-v/)

```
Input: arr = [6,4,14,6,8,13,9,7,10,6,12], d = 2
Output: 4
Explanation: You can start at index 10. You can jump 10 --> 8 --> 6 --> 7 as shown.
Note that if you start at index 6 you can only jump to index 7. You cannot jump to index 5 because 13 > 9. You cannot jump to index 4 because index 5 is between index 4 and 6 and 13 > 9.
Similarly You cannot jump from index 3 to index 2 or index 1.
```

+ 解法

    ```python
    def maxJumps(self, arr: List[int], d: int) -> int:
        n = len(arr)
        ans = [0] * n
    
        def jump(i):
            if ans[i]: return ans[i]
            ans[i] = 1
            for di in (-1, 1):
                for j in range(i+di, i+d*di+di, di):
                    if not (0<=j<n and arr[j]<arr[i]): break
                    ans[i] = max(ans[i], jump(j)+1)
            return ans[i]
    ```

### 1301. Number of Paths with Max Score

左上到右下，最大值，路径中存在障碍，并且需要返回路径的个数。
[查看原题](https://leetcode.com/problems/number-of-paths-with-max-score/)

```
Input: board = ["E23","2X2","12S"]
Output: [7,1]
```

+ 解法

    ```python
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        n, mod = len(board), 10**9+7
        dp = [[[float('-inf'), 0] for j in range(n+1)] for i in range(n+1)]
        dp[n-1][n-1] = [0, 1]
        for x in range(n)[::-1]:
            for y in range(n)[::-1]:
                if board[x][y] in 'XS': continue
                for i, j in ((0, 1), (1, 0), (1, 1)):
                    if dp[x][y][0] < dp[x+i][y+j][0]:
                        dp[x][y] = [dp[x+i][y+j][0], 0]
                    if dp[x][y][0] == dp[x+i][y+j][0]:
                        dp[x][y][1] += dp[x+i][y+j][1]
                dp[x][y][0] += int(board[x][y]) if x or y else 0
        return [dp[0][0][0] if dp[0][0][1] else 0, dp[0][0][1] % mod]
    ```

### 1277. Count Square Submatrices with All Ones

矩阵中最多有多少个1构成的正方形。
[查看原题](https://leetcode.com/problems/count-square-submatrices-with-all-ones/)

```
Input: matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There is  1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.
```

+ 解法

    ```python
    def countSquares(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        # dp[i][j] 表示以i, j为右下点时，正方形的个数。
        dp = [[0] * (n) for i in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return sum(map(sum, dp))
    ```

### 1269. Number of Ways to Stay in the Same Place After Some Steps

回到原点的走法一共有多少种，一次只能向右，向左或者停留，要求始终保持在数组范围。
[查看原题](https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

```
Input: steps = 3, arrLen = 2
Output: 4
Explanation: There are 4 differents ways to stay at index 0 after 3 steps.
Right, Left, Stay
Stay, Right, Left
Right, Stay, Left
Stay, Stay, Stay
```

+ 找到状态转移方程，`dp[p][s] = dp[p-1][s-1] + dp[p][s-1] + dp[p+1, s-1]`p代表位置，s代表步数。首部添加0方便求和。注意t+3这个范围。

    ```python
    def countSquares(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        # dp[i][j] 表示以i, j为右下点时，正方形的个数。
        dp = [[0] * (n) for i in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return sum(map(sum, dp))
    ```

### 338. Counting Bits

返回从0到num的数中，每个数二进制中含有1的个数。
[查看原题](https://leetcode.com/problems/counting-bits/)

```
Input: 5
Output: [0,1,1,2,1,2]
```

+ dp 。`f[i]=f[i//2]+i&1`

    ```python
    def countSquares(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        # dp[i][j] 表示以i, j为右下点时，正方形的个数。
        dp = [[0] * (n) for i in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return sum(map(sum, dp))
    ```

### 1262. Greatest Sum Divisible by Three

最多的元素和能被3整除。
[查看原题](https://leetcode.com/problems/greatest-sum-divisible-by-three/)

```
Input: nums = [3,6,5,1,8]
Output: 18
Explanation: Pick numbers 3, 6, 1 and 8 their sum is 18 (maximum sum divisible by 3).
```

+ 解法

    ```python
    def maxSumDivThree(self, nums: List[int]) -> int:
        # dp[pos][mod]
        # # 0  1  2
        # 0 3  0  0
        # 1 9  0  0 
        # 2 9  0  14
        # 3 15 10 14
        # 4 18 22 23
        dp = [0] * 3
        for a in nums:
            for j in dp[:]:
                dp[(j+a) % 3] = max(dp[(j+a) % 3], j+a)
        return dp[0]
    ```

### 72. Edit Distance

两个单词，将a变成b的最小步数，可以添加、删除，替换一个字母。
[查看原题](https://leetcode.com/problems/greatest-sum-divisible-by-three/)

```
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
```

+ 解法

    ```python
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
            
        for i in range(m):
            for j in range(n):
                if word1[i] == word2[j]:
                    dp[i+1][j+1] = dp[i][j]
                else:
                    dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1], dp[i][j]) + 1
        return dp[-1][-1]
    ```

### 518. Coin Change 2

找钱问题，给你几种面值的硬币，不限制每种硬币的个数，问组成多少钱有多少种方法。
[查看原题](https://leetcode.com/problems/coin-change-2/)

```
Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

+ 背包问题

    ```python
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in coins:
            for j in range(1, amount + 1):
               if j >= i:
                   dp[j] += dp[j - i]
        return dp[amount]
    ```

### 1220. Count Vowels Permutation

元音字母的全排列，根据指定规则的，求全排列的个数。
[查看原题](https://leetcode.com/problems/count-vowels-permutation/)

```
Input: n = 2
Output: 10
Explanation: All possible strings are: "ae", "ea", "ei", "ia", "ie", "io", "iu", "oi", "ou" and "ua".
```

+ 背包问题

    ```python
    def countVowelPermutation(self, n: int) -> int:
        a = e = i = o = u = 1
        for _ in range(n-1):
            a, e, i, o, u = e, a+i, a+e+o+u, i+u, a
        return (a+e+i+o+u) % (10**9+7)
    ```

### 368. Largest Divisible Subset

最大的整除子集。
[查看原题](https://leetcode.com/problems/largest-divisible-subset/)

```
Input: [1,2,3]
Output: [1,2] (of course, [1,3] will also be ok)
```

+ 背包问题

    ```python
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        S = {-1: set()}
        for x in sorted(nums):
            S[x] = max((S[d] for d in S if x % d == 0), key=len) | {x}
        return list(max(S.values(), key=len))
    ```

### 5456. Kth Ancestor of a Tree Node

找出一个树节点的k个祖先。
[查看原题](https://leetcode.com/problems/kth-ancestor-of-a-tree-node/)

```
Input:
["TreeAncestor","getKthAncestor","getKthAncestor","getKthAncestor"]
[[7,[-1,0,0,1,1,2,2]],[3,1],[5,2],[6,3]]

Output:
[null,1,0,-1]

Explanation:
TreeAncestor treeAncestor = new TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2]);

treeAncestor.getKthAncestor(3, 1);  // returns 1 which is the parent of 3
treeAncestor.getKthAncestor(5, 2);  // returns 0 which is the grandparent of 5
treeAncestor.getKthAncestor(6, 3);  // returns -1 because there is no such ancestor
```

+ 用的倍增法，binary lifting.

    ```python
    class TreeAncestor:
    
        step = 15
        def __init__(self, n, A):
            A = dict(enumerate(A))
            jump = [A]
            for s in range(self.step):
                B = {}
                for i in A:
                    if A[i] in A:
                        B[i] = A[A[i]]
                jump.append(B)
                A = B
            self.jump = jump
            print(jump)
    
        def getKthAncestor(self, x: int, k: int) -> int:
            step = self.step
            while k > 0 and x > -1:
                if k >= 1 << step:
                    x = self.jump[step].get(x, -1)
                    k -= 1 << step
                else:
                    step -= 1
            return x
    ```

### 1477. Find Two Non-overlapping Sub-arrays Each With Target Sum

找到数组中等于目标值的两个不重叠子数组的最小长度和。
[查看原题](https://leetcode.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/)

```
Input: arr = [3,2,2,4,3], target = 3
Output: 2
Explanation: Only two sub-arrays have sum = 3 ([3] and [3]). The sum of their lengths is 2.
```

+ 方法一：看了提示后使用了前后遍历法做出来的。其实有一次遍历的方式。这个方法看了挺长时间，才明白，实际上记录了一个以end为结尾的前面的所有元素最好的长度是多少。

    ```python
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        prefix = {0: -1}
        best_till = [math.inf] * len(arr)
        ans = best = math.inf
        for i, curr in enumerate(itertools.accumulate(arr)):
            # print(i, curr)
            if curr - target in prefix:
                end = prefix[curr - target]
                if end > -1:
                    ans = min(ans, i - end + best_till[end])
                best = min(best, i - end)
                # print('\t', best, i-end, best_till, ans)
            best_till[i] = best
            prefix[curr] = i
        return -1 if ans == math.inf else ans
    ```

### 494. Target Sum

给你一组数，用+或-连接起来最后等于target，问有多少种填法。
[查看原题](https://leetcode.com/problems/target-sum/)

```
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
```

+ 解法

    ```python
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        n = len(nums)
    
        self.memo = {}
    
        # @functools.lru_cache(maxsize=2**17)
        def dfs(i, total):
            # print(i, total)
            if (i, total) in self.memo:
                return self.memo[(i, total)]
            if i == n:
                return total==S
            ans = dfs(i+1, total+nums[i]) + dfs(i+1, total-nums[i])
            self.memo[(i, total)] = ans
            # return dfs(i+1, total+nums[i]) + dfs(i+1, total-nums[i])
            return ans
    
        return dfs(0, 0)
    ```

### 174. Dungeon Game

地牢游戏，从左上走到右下，每次只能像右或者向下，格子里会扣血和加血，问最少需要多少血，全程保持血量为1以上。
[查看原题](https://leetcode.com/problems/dungeon-game/)

+ 解法

    ```python
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        R, C = len(dungeon), len(dungeon[0])
        dp = [[0] * C for _ in range(R)] 
        for i in range(R-1, -1, -1):
            for j in range(C-1, -1, -1):
                if i == R-1 and j == C-1:
                    dp[i][j] = max(1, 1 - dungeon[i][j])
                elif i == R-1:
                    dp[i][j] = max(1, dp[i][j+1] - dungeon[i][j])
                elif j == C-1:
                    dp[i][j] = max(1, dp[i+1][j] - dungeon[i][j])
                else:
                    dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
        return dp[0][0]
    ```

### 96. Unique Binary Search Trees

不重复的二叉搜索树，1~n节点。
[查看原题](https://leetcode.com/problems/unique-binary-search-trees/)

```
Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

+ 状态转移方程式这样的G(n)表示n个节点能组成的二叉搜索树节点个数。F(i, n)表示有n个节点时，以i为root的个数。`G(n) = F(1, n) + F(2, n) + ... + F(n, n)`. `F(3, 7)=G(2)*G(4)`即`F(i, n) = G(i-1) * G(n-i)`, 所以最后`G(n) = G(0) * G(n-1) + G(1) * G(n-2) + … + G(n-1) * G(0)`

    ```python
    def numTrees(self, n: int) -> int:
        G = [0] * (n+1)
        G[0] = G[1] = 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1]*G[i-j]
        return G[n]
    ```
