---
layout: post
title: Stack(堆栈)
published: true
date: 2016-10-23
---

> 先进后出，后进先出

## 关于堆栈

栈（stack）又名堆栈，它是一种运算受限的线性表。限定仅在表尾进行插入和删除操作的线性表。这一端被称为栈顶，相对地，把另一端称为栈底。向一个栈插入新元素又称作进栈、入栈或压栈，它是把新元素放到栈顶元素的上面，使之成为新的栈顶元素；从一个栈删除元素又称作出栈或退栈，它是把栈顶元素删除掉，使其相邻的元素成为新的栈顶元素。

## LeetCode真题

### 1021. Remove Outermost Parentheses

删除最外层的括号。
[查看原题](https://leetcode.com/problems/remove-outermost-parentheses/)

```
Input: "(()())(())"
Output: "()()()"
The input string is "(()())(())", with primitive decomposition "(()())" + "(())".
After removing outer parentheses of each part, this is "()()" + "()" = "()()()".
Input: "(()())(())(()(()))"
Output: "()()()()(())"
The input string is "(()())(())(()(()))", with primitive decomposition "(()())" + "(())" + "(()(()))".
After removing outer parentheses of each part, this is "()()" + "()" + "()(())" = "()()()()(())".
Input: "()()"
Output: ""
```

+ 解法

    ```python
    def removeOuterParentheses(self, S: str) -> str:
        ans, opened = [], 0
        for s in S:
            if s == '(' and opened > 0:
                ans.append(s)
            if s == ')' and opened > 1:
                ans.append(s)
            opened += 1 if s=='(' else -1
        return ''.join(ans)
    ```


### 1047. Remove All Adjacent Duplicates In String

每两个相邻的相同字符串可以消掉。类似于连连看。
[查看原题](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

+ 解法

    ```python
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for c in S:
            if stack and stack[-1] == c:
                stack.pop()
            else:
                stack.append(c)
        return ''.join(stack)
    ```


### 1172. Dinner Plate Stacks

有这样一堆栈，每个栈有个容量，可以在指定索引下删除某个栈，每次push时需要在最左的不满的栈。
[查看原题](https://leetcode.com/problems/dinner-plate-stacks/)

```
Input: 
["DinnerPlates","push","push","push","push","push","popAtStack","push","push","popAtStack","popAtStack","pop","pop","pop","pop","pop"]
[[2],[1],[2],[3],[4],[5],[0],[20],[21],[0],[2],[],[],[],[],[]]
Output: 
[null,null,null,null,null,null,2,null,null,20,21,5,4,3,1,-1]
```

+ 核心思想在于维护一个堆，记录不满的栈。以便插入时可以找到该索引。

    ```python
    class DinnerPlates:
        def __init__(self, capacity: int):
            self.c = capacity
            self.q = []
            self.emp = []
        
        def push(self, val: int) -> None:
            if self.emp:
                index = heapq.heappop(self.emp)
                self.q[index].append(val)
            else:
                if self.q and len(self.q[-1])!=self.c:
                    self.q[-1].append(val)
                else:
                    self.q.append([val])
        
        def pop(self) -> int:
            while self.q:
                if self.q[-1]:
                    return self.q[-1].pop()
                self.q.pop()
            return -1
        
        def popAtStack(self, index: int) -> int:
            heapq.heappush(self.emp, index)
            if self.q[index]:
                return self.q[index].pop()
            else:
                return -1
    ```

### 901. Online Stock Span

实时找出数据流中连续的比不大于当前值的个数。
[查看原题](https://leetcode.com/problems/online-stock-span/)

```
Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
Output: [null,1,1,1,2,1,4,6]
Explanation: 
First, S = StockSpanner() is initialized.  Then:
S.next(100) is called and returns 1,
S.next(80) is called and returns 1,
S.next(60) is called and returns 1,
S.next(70) is called and returns 2,
S.next(60) is called and returns 1,
S.next(75) is called and returns 4,
S.next(85) is called and returns 6.

Note that (for example) S.next(75) returned 4, because the last 4 prices
(including today's price of 75) were less than or equal to today's price.
```

+ <=可以累加。

    ```python
    class StockSpanner:
    
        def __init__(self):
            self.stack = []
    
        def next(self, price: int) -> int:
            cnt = 1
            while self.stack and self.stack[-1][0] <= price:
                cnt += self.stack.pop()[1]
            self.stack.append((price, cnt))
            return cnt
    ```


### 1299. Replace Elements with Greatest Element on Right Side

根据数组重新生成一个数组，每个元素对应原数组右侧最大的数字。
[查看原题](https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/)

```
Input: arr = [17,18,5,4,6,1]
Output: [18,6,6,6,1,-1]
```

+ 方法一：栈

    ```python
    def replaceElements(self, arr: List[int]) -> List[int]:
        stack = [-1]
        for i in range(len(arr)-1, 0, -1):
            if arr[i] < stack[-1]:
                stack.append(stack[-1])
            else:
                stack.append(arr[i])
        return stack[::-1]
    ```

+ 方法二：Lee215.

    ```python
    def replaceElements(self, arr: List[int], mx=-1) -> List[int]:
        for i in range(len(arr)-1, -1, -1):
            arr[i], mx = mx, max(mx, arr[i])
        return arr
    ```

### 1249. Minimum Remove to Make Valid Parentheses

删除字符串中多余的括号。
[查看原题](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)

```
Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
```

+ 解法

    ```python
    def minRemoveToMakeValid(self, s: str) -> str:
        stack, cur = [], ''
        for c in s:
            if c == '(':
                stack.append(cur)
                cur = ''
            elif c == ')':
                if stack:
                    cur = '{}({})'.format(stack.pop(), cur)
            else:
                cur += c
                
        while stack:
            cur = stack.pop() + cur
        return cur
    ```


### 1190. Reverse Substrings Between Each Pair of Parentheses

将括号内的字符串反转。
[查看原题](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

```
Input: s = "(u(love)i)"
Output: "iloveu"
Explanation: The substring "love" is reversed first, then the whole string is reversed.
```

+ 和1249一样的解法。

    ```python
    def reverseParentheses(self, s: str) -> str:
        stack, cur = [], ''
        for c in s:
            if c == '(':
                stack.append(cur)
                cur = ''
            elif c == ')':
                if stack:
                    cur = '{}{}'.format(stack.pop(), cur[::-1])
            else:
                cur += c
        return cur
    ```

### 1209. Remove All Adjacent Duplicates in String II

将字符串中连续的k的字母全部删除，返回剩余的字符串。
[查看原题](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/)

```
Input: s = "abcd", k = 2
Output: "abcd"
Explanation: There's nothing to delete.
```

+ 相同字符的记录个数。

    ```python
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = [['#', 0]]
        for c in s:
            if stack[-1][0] == c:
                stack[-1][1] += 1
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append([c, 1])
        return ''.join(c*k for c, k in stack)
    ```

### 1475. Final Prices With a Special Discount in a Shop

找出数组中后面元素比当前元素小的差。
[查看原题](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/)

```
Input: prices = [8,4,6,2,3]
Output: [4,2,4,2,3]
Explanation: 
For item 0 with price[0]=8 you will receive a discount equivalent to prices[1]=4, therefore, the final price you will pay is 8 - 4 = 4. 
For item 1 with price[1]=4 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 4 - 2 = 2. 
For item 2 with price[2]=6 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 6 - 2 = 4. 
For items 3 and 4 you will not receive any discount at all.
```

+ 解法

    ```python
    def finalPrices(self, A: List[int]) -> List[int]:
        stack = []
        for i, a in enumerate(A):
            while stack and A[stack[-1]] >= a:
                A[stack.pop()] -= a
            stack.append(i)
        return A
    ```


### 739. Daily Temperature

找出比当前元素之后的大的值，计算索引差，没有则是0。
[查看原题](https://leetcode.com/problems/daily-temperatures/)

+ 刚刚做完1475的题，一样的解法。

    ```python
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        ans = [0] * len(T)
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                j = stack.pop()
                ans[j] = i - j
            stack.append(i)
        return ans
    ```
