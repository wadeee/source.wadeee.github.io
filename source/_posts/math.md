---
layout: post
title: Math(数学)
published: true
date: 2016-10-18
---

> Math is the base.


## LeetCode真题

### 7. Reverse Integer

倒置一个整数， 此答案忽略了原题中的范围判断。
[查看原题](https://leetcode.com/problems/reverse-integer/description/)

```
Input: -123
Output: -321
```

+ 方法一：str

    ```python
    def reverse_int(x):
        if x >= 0:
            return int(str(x)[::-1])
        else:
            return -int(str(x)[:0:-1])
    ```

+ 方法二：math

    ```python
    def reverse(self, x: int) -> int:
        sign = 1 if x >= 0 else -1
        ans, tail = 0, abs(x)
        while tail:
            ans = ans*10 + tail%10
            tail //= 10
        return ans * sign if ans < 2**31 else 0
    ```

### 9. Palindrome Number

判断一个数是否是回文数，这里把负数认为是不符合条件的。
[查看原题](https://leetcode.com/problems/palindrome-number/description/)

+ 方法一：str

    ```python
    def is_palindrome(x):
        return str(x) == str(x)[::-1]
    ```

+ 方法二：math

    ```python
    def is_palindrome(x):
        l, r = x, 0
        while l > 0:
            r = r*10 + l%10
            l //= 10
        return r == x
    ```
  

### 13. Roman to Integer

罗马数字转换整型。
[查看原题](https://leetcode.com/problems/roman-to-integer/description/)

```
Input: [3,4,5,1,2] 
Output: 1
```

+ 解法

    ```python
    def roman_to_int(s):
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 
                 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        for i in range(len(s)):
            if i == len(s)-1 or roman[s[i]] >= roman[s[i+1]]
                total += roman[s[i]]
            else:
                total -= roman[s[i]]
        return total
    ```

### 69. Sqrt(x)

实现开方，返回整数部分。
[查看原题](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

```
Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
```

+ 牛顿迭代法

    ```python
    def my_sqrt(x):
        r = x
        while r**2 > x:
            r = (r+x//r) // 2
        return r
    ```


### 367. Valid Perfect Square

判断一个数是不是某个数的平方。
[查看原题](https://leetcode.com/problems/valid-perfect-square/)

```
Input: 16
Output: true
```

+ 方法一：牛顿迭代法。同69。

    ```python
    def isPerfectSquare(self, num):
        r = num
        while r**2 > num:
            r = (r + num // r) // 2
        return r**2 == num
    ```

### 171. Excel Sheet Column Number

excel表格列表数字转换，二十六进制。
[查看原题](https://leetcode.com/problems/excel-sheet-column-number/description/)

```
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28     A -> 1
```

+ 解法

    ```python
    def titleToNumber(self, s: str) -> int:
        OFFSET = ord('A')-1
        return sum((ord(x)-OFFSET)*26**i for i, x in enumerate(s[::-1]))
    ```

### 168. Excel Sheet Column Title

excel转换，数字转字母。十进制->26进制。
[查看原题](https://leetcode.com/problems/excel-sheet-column-title)

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

+ 解法

    ```python
    def convertToTitle(self, n):
        res = ''
        while n:
            res = chr((n-1)%26+65) + res
            # n //= 26
            n = (n-1) // 26
        return res
    ```


### 172. Factorial Trailing Zeroes

求n的阶乘末尾有几个0。
[查看原题](https://leetcode.com/problems/factorial-trailing-zeroes/description/)

```
Input: 5
Output: 1
Explanation: 5! = 120, one trailing zero.
```

+ 思路：每一对2和5可以产生一个0，在n的阶乘中，5比2多，所以问题变成求5的个数，而25这种数有两个5，所以递归求解

    ```python
    def trailing_zeroes(n):
        return 0 if n == 0 else n//5 + trailing_zeroes(n//5)
    ```

### 204. Count Primes

求小于n的整数中，有多少个质数。
[查看原题](https://leetcode.com/problems/count-primes/description/)

+ 解法

    ```python
    def countPrimes(self, n):
        is_prime = [False]*2 + [True]*(n-2)
        for i in range(2, int(n ** 0.5)+1):
            if is_prime[i]:
                is_prime[i*i:n:i] = [False] * len(is_prime[i*i:n:i])
        return sum(is_prime)
    ```


### 50. Pow(x, n)

实现pow函数。
[查看原题](https://leetcode.com/problems/powx-n/description/)

```
Input: 2.00000, 10
Output: 1024.00000

Input: 2.00000, -2
Output: 0.25000 .
```

+ 说明：常规方法在Leetcode 上内存会爆掉。

    ```python
    class Solution(object):
    
        def myPow(self, x, n):
            if n < 0:
                return 1 / self.pow_with_unsigned(x, -n)
            else:
                return self.pow_with_unsigned(x, n)
                  
        def pow_with_unsigned(self, x, n):
            if n == 1:
                return x
            if n == 0:
                return 1
            
            res = self.pow_with_unsigned(x, n >> 1)
            res *= res
            
            if n & 1 == 1:
                res *= x
                
            return res
    ```

### 233. Number of Digit One

1~n数字中1的个数。
[查看原题](https://leetcode.com/problems/number-of-digit-one/description/)

+ 解法

    ```python
    def countDigitOne(self, n):    
        countr, i = 0, 1
        while i <= n:
            divider = i * 10
            countr += (n // divider) * i + min(max(n % divider - i + 1, 0), i)
            i *= 10
        return countr
    ```

### 263. Ugly Number

判断一个数是否是丑数。
[查看原题](https://leetcode.com/problems/ugly-number-ii/description/)

+ 根据定义实现。< num是为了判断num=0的情况。

    ```python
    def isUgly(self, num):
        for f in 2, 3, 5:
            while num % f == 0 < num:
                num //= f
        return num == 1
    ```

### 264. Ugly Number II

输出第n个丑数。
[查看原题](https://leetcode.com/problems/ugly-number-ii/)

+ 解法

    ```python
    def nthUglyNumber(self, n):
        q = [1]
        t2, t3, t5 = 0, 0, 0
        for i in range(n-1):
            a2, a3, a5 = q[t2]*2, q[t3]*3, q[t5]*5
            to_add = min(a2, a3, a5)
            q.append(to_add)
            if a2 == to_add:
                t2 += 1
            if a3 == to_add:
                t3 += 1
            if a5 == to_add:
                t5 += 1
        return q[-1]
    ```

### 67.Add Binary

实现二进制加法。
[查看原题](https://leetcode.com/problems/add-binary/description/)

```
Input: a = "11", b = "1"
Output: "100"
```

+ 方法一：按照加法的二进制思想来计算，不过Runtime大约100ms。后来试着将list comprehension拆成一个for循环，也并没有提高速度。居然beats只有4%，难道大部分人都用的bin。讨论区简单翻了了一下，没有找到一个高效的pythonic的方法。

    ```python
    def addBinary(self, a, b):
        if len(a) > len(b):
            b = b.zfill(len(a))
        else:
            a = a.zfill(len(b))
        
        while int(b):
            sum_not_carry = ''.join([str(int(a[i]) ^ int(b[i])) for i in range(len(a))])
            carry = ''.join([str(int(a[i]) & int(b[i])) for i in range(len(a))])
            a, b = "0"+sum_not_carry, carry+'0'
        return a.lstrip('0') if a != '0' else '0'
    ```

### 202. Happy Number

判断是否是欢乐数。进行所有位的平方和运算，最后为1的是欢乐数。
[查看原题](https://leetcode.com/problems/happy-number/)

```
Input: 19
Output: true
Explanation: 
1**2 + 9**2 = 82
8**2 + 2**2 = 68
6**2 + 8**2 = 100
1**2 + 0**2 + 0**2 = 1
```

+ 思路，使用一个字典映射0~9的平方值，然后如果死循环的话，各位数的和一定存在着一种循环，所以用一个set来判断是否重复。

    ```python
    def isHappy(self, n):
        squares = {str(k): k**2 for k in range(0, 10)}
        sum_digit = set()
        while n != 1:
            n = sum(squares[digit] for digit in str(n))
            if n in sum_digit:
                return False
            else:
                sum_digit.add(n)
        return True
    ```

### 231. Power of Two

判断一个数是否是2的n次方。思路也就是判断这个数的二进制形式是否只有一个’1’。
[查看原题](https://leetcode.com/problems/power-of-two)


+ 方法一：二进制统计1。

    ```python
    def isPowerOfTwo(self, n):
        return n > 0 and bin(n).count('1') == 1
    ```

+ 方法三：如果一个数n的二进制只有一个1，那么n&(n-1)一定为0。

    ```python
    def isPowerOfTwo(self, n):
        return n > 0 and (n&n-1) == 0
    ```

### 342. Power of Four

判断一个数是否是4的n次方。
[查看原题](https://leetcode.com/problems/power-of-four/)


+ 方法一：从简单入手通过231题，了解到了2的n次方特点是，二进制形式只有一个’1’，那么4的n次方就是不但只有一个’1’，后面还跟了偶数个’0’。

    ```python
    def isPowerOfFour(self, num):
        # return num > 0 and (num & num-1)==0 and bin(num)[2:].count('0')&1==0
        return num > 0 and (num & num-1)==0 and len(bin(num))&1==1
    ```

+ 方法三：也可以使用正则。

    ```python
    def isPowerOfFour(self, num):
        import re
        return bool(re.match(r'^0b1(00)*$',bin(num)))
    ```

### 292. Nim Game

说，有这么一堆石头，一次只能拿1~3个，拿到最后一个石头的人获胜。求n堆石头，你先拿是否可以获胜。
[查看原题](https://leetcode.com/problems/nim-game/)


+ 思路：找规律，发现只有最后剩4个石头的时候，此时轮到谁，谁输。

    ```python
    def canWinNim(self, n):
        return n % 4 != 0
    ```

### 400. Nth Digit

找出无限整数序列中的第n个数字。
[查看原题](https://leetcode.com/problems/nth-digit/description/)

```
Input:
11
Output:
0
Explanation:
The 11th digit of the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... is a 0, which is part of the number 10.
```

+ 思路，根据n的位数，将无限序列分为几个范围。
    1. 寻找范围。寻找n处于哪个范围，是1~9，还是10~99，例如n=15。则需要跳过1~9的范围，而这个范围有size*step个数字，所以问题变成在10~99范围上寻找第15-1*9=6个数。
    2. 定位数字。10~99范围中是从10开始的，每一个数都有两位数字，所以最终数字为10+(6-1)//2，因为索引从0开始，所以需要-1。
    3. 定位数字的位。上一步找到了数字为12，对size求余就可以知道，'12'[(6-1)%2]='2'。

    ```python
    def findNthDigit(self, n):
        start, step, size = 1, 9, 1
        while n > size * step:
            n, start, step, size = n-size*step, start*10, step*10, size+1
        return int(str(start + (n-1)//size)[(n-1) % size])
    ```

### 415. Add Stings

给定两个字符串表示的数字，把它们相加，这两个数的长度小于5100，不能使用任何BitIntegr库或是直接将其转换为整数。ps: 题中要求不将输入直接转换成int，所以我个人认为int还是可以使用的，有一些答案中是使用了ord来做运算。
[查看原题](https://leetcode.com/problems/add-strings/)

+ 使用zip_longest。

    ```python
    def addStrings(self, num1, num2):
        from itertools import zip_longest
        nums = list(zip_longest(num1[::-1], num2[::-1], fillvalue='0'))
        carry, res = 0, ''
        for digits in nums:
            d1, d2 = map(int, digits)
            carry, val = divmod(d1+d2+carry, 10)
            res = res + str(val)
        res = res if carry==0 else res+str(carry)        
        return res[::-1]
    ```

### 492. Construct the Rectangle

给定一个面积，求组成这个面积的长高差最小。
[查看原题](https://leetcode.com/problems/construct-the-rectangle/)

```
Input: 4
Output: [2, 2]
Explanation: The target area is 4, and all the possible ways to construct it are [1,4], [2,2], [4,1]. 
But according to requirement 2, [1,4] is illegal; according to requirement 3,  [4,1] is not optimal compared to [2,2]. So the length L is 2, and the width W is 2.
```

+ 解法

    ```python
    def constructRectangle(self, area):
        import math
        w = int(math.sqrt(area))
        while area % w != 0:
            w -= 1
        return [area//w, w]
    ```

### 504. Base 7

10进制转7进制。
[查看原题](https://leetcode.com/problems/base-7/)

```
Input: 100
Output: "202"
Input: -7
Output: "-10"
```

+ 需要注意负数。

    ```python
    def convertToBase7(self, num: int) -> str:
        if num == 0: return '0'
        n, ans = abs(num), ''
        while n:
            n, val = divmod(n, 7)
            ans = str(val) + ans
        return ans if num > 0 else '-'+ans
    ```

### 970. Powerful Integers

求满足x^i+y^j <= bound的所有和。
[查看原题](https://leetcode.com/contest/weekly-contest-118/problems/powerful-integers/)

```
Input: x = 2, y = 3, bound = 10
Output: [2,3,4,5,7,9,10]
Explanation: 
2 = 2^0 + 3^0
3 = 2^1 + 3^0
4 = 2^0 + 3^1
5 = 2^1 + 3^1
7 = 2^2 + 3^1
9 = 2^3 + 3^0
10 = 2^0 + 3^2
```

+ 这题难得地方在于两个循环的临界值，貌似我这样写也不是最优解，原题的Solution中给定了2**18>bound的最大值。所以两个范围都是18。

    ```python
    def powerfulIntegers(self, x, y, bound):
        res = set()
        imax = self.get_max(x, bound) + 1
        jmax = self.get_max(y, bound) + 1
        for i in range(imax):
            for j in range(jmax):
                if x**i + y**j <= bound:
                    res.add(x**i+y**j)
        return list(res)
    
        def get_max(self, n, bound):
            for i in range(bound//n + 1):
                if n ** i >= bound:
                    return i
            return bound//n + 1
    ```

### 973. K Closest Points to Origin

求离原点最近的K个坐标点。
[查看原题](https://leetcode.com/contest/weekly-contest-119/problems/k-closest-points-to-origin/)

```
Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
```

+ easy

    ```python
    def kClosest(self, points, K):
        res = sorted(points, key=lambda x: x[0]**2 + x[1]**2)
        return res[:K]
    ```

### 976. Largest Perimeter Triangle

给定一个边长数组，求能组成的三角形的最长周长。
[查看原题](https://leetcode.com/contest/weekly-contest-119/problems/largest-perimeter-triangle/)

+ 就是长度为3的滑动窗口。

    ```python
    def largestPerimeter(self, A):
        res = sorted(A, reverse=True)
        for i in range(len(res)-2):
            if sum(res[i+1:i+3]) > res[i]:
                return sum(res[i:i+3])
        return 0
    ```

### 628. Maximum Product of Three Numbers

数组中三个数的最大乘积。元素范围[-1000, 1000]。
[查看原题](https://leetcode.com/problems/maximum-product-of-three-numbers/)

```
Input: [1,2,3,4]
Output: 24
```

+ 方法一：排序。在正数个数大于等于3的时候，显然最大的三个数就可以产生最大的乘积。而当正数个数不够的时候，那么必须需要两个最小的负数（即绝对值最大），和一个最大的正数。

    ```python
    def maximumProduct(self, nums):
        ary = sorted(nums)
        return max((ary[0]*ary[1]*ary[-1], ary[-3]*ary[-2]*ary[-1]))
    ```

+ 方法二：使用heapq.

    ```python
    def maximumProduct(self, nums):
        import heapq
        from operator import mul
        from functools import reduce
        three_max = heapq.nlargest(3, nums)
        two_min = heapq.nsmallest(2, nums)
        return max(reduce(mul, three_max), reduce(mul, two_min + three_max[:1]))
    ```

### 728. Self Dividing Numbers

自整除数字，一个数字能够被本身的每个数字整除，并且不能有0，求某个范围内所有的数。
[查看原题](https://leetcode.com/problems/self-dividing-numbers/)

```
Input: 
left = 1, right = 22
Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
```

+ Brute Force. 此题强行使用列表生成式没有意义。

    ```python
    def selfDividingNumbers(self, left, right):
        res = []
        for i in range(left, right+1):
            for char in str(i):
                if int(char)==0 or i % int(char)!=0:
                    break
            else:
                res.append(i)
        return res
    ```

### 836. Rectangle Overlap

矩形是否重叠，矩形的边平行于坐标轴。
[查看原题](https://leetcode.com/problems/rectangle-overlap/)

```
Input: rec1 = [0,0,2,2], rec2 = [1,1,3,3]
Output: true
```

+ 解法

    ```python
    def isRectangleOverlap(self, rec1: 'List[int]', rec2: 'List[int]') -> 'bool':
        return rec2[0] < rec1[2] and rec1[0] < rec2[2] and \
               rec2[1] < rec1[3] and rec1[1] < rec2[3]
    ```

### 991. Broken Calculator

坏掉的计算器，只能*2或者-1，使X变为Y。
[查看原题](https://leetcode.com/problems/broken-calculator/)

```
Input: X = 5, Y = 8
Output: 2
Explanation: Use decrement and then double {5 -> 4 -> 8}.
```

+ 如果从X到Y问题会变得复杂，不确定什么时候该*2或者是-1。所以逆向思维从Y变成X。因为如果Y是奇数，那么必定在+1操作后要/2，这里将其合并。

    ```python
    def brokenCalc(self, X: 'int', Y: 'int') -> 'int':
        return X - Y if X >= Y else 1+(Y&1)+self.brokenCalc(X, (Y+1)//2)
    ```

### 908. Smallest Range I

给定一个数组，和一个K，数组里的数加上-k<=x<=k的任意一个数字后，求数组最大数和最小数的，最小差。
[查看原题](https://leetcode.com/problems/smallest-range-i/)

```
Input: A = [0,10], K = 2
Output: 6
Explanation: B = [2,8]
```

+ 解法

    ```python
    def smallestRangeI(self, A: 'List[int]', K: 'int') -> 'int':
        return max(max(A) - min(A) - 2*K, 0)
    ```

### 949. Largest Time for Given Digits

给定四个数字，返回能生成的最大时间。24小时制。
[查看原题](https://leetcode.com/problems/largest-time-for-given-digits/)

```
Input: [1,2,3,4]
Output: "23:41"
```

+ 解法

    ```python
    def largestTimeFromDigits(self, A: 'List[int]') -> 'str':
        p = itertools.permutations(A)
        return max(['{}{}:{}{}'.format(*d) for d in p 
                    if d[:2] < (2, 4) and d[2] < 6] or [''])
    ```

### 914. X of a Kind in a Deck of Cards

有这样一堆数字卡牌，问是否存在一个X>=2，使得将同样数字的卡牌分为每X个一组，并且刚好所有的卡牌分完。
[查看原题](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards)

+ 使用Counter来统计每个数字的个数，然后求这些数字的最大公约数是否大于等于2，这里思路卡了一下，因为没想到最大公约数可以通过reduce来计算，没考虑到是可以累积的。

    ```python
    def hasGroupsSizeX(self, deck):
        from collections import Counter
        from math import gcd
        from functools import reduce
        return reduce(gcd, Counter(deck).values()) >= 2
    ```

### 470. Implement Rand10() Using Rand7()

使用rand7实现rand10
[查看原题](https://leetcode.com/problems/implement-rand10-using-rand7/)

```
Input: 3
Output: [8,1,10]
```

+ 解法

    ```python
    def rand10(self):
        while True:
            x = (rand7()-1)*7 + rand7()-1
            if x < 40: 
                return x%10 + 1
    ```

### 1006. Clumsy Factorial

将一个阶乘的式子用*/+-替代，给出结果。
[查看原题](https://leetcode.com/problems/clumsy-factorial/)

```
Input: 10
Output: 12
Explanation: 12 = 10 * 9 / 8 + 7 - 6 * 5 / 4 + 3 - 2 * 1
```

+ 解法

    ```python
    def clumsy(self, N: int) -> int:
        op = itertools.cycle(['*', '//', '+', '-'])
        return eval(''.join(str(n)+next(op) if n!=1 else str(n) 
                            for n in range(N, 0, -1)))
    ```

### 1022. Smallest Integer Divisible by K

最小的由1组成的能被K整除。
[查看原题](https://leetcode.com/problems/smallest-integer-divisible-by-k/)

```
Input: 2
Output: -1
Explanation: There is no such positive integer N divisible by 2.
```

+ 如果有2或5的质因数，那么不能整除。

    ```python
    def smallestRepunitDivByK(self, K: int) -> int:
        if K % 2 == 0 or K % 5 == 0: return -1
        r = 0
        for N in range(1, K + 1):
            r = (r * 10 + 1) % K
            if not r: return N
    ```

### 1028. Convert to Base -2

10进制转成-2进制。
[查看原题](https://leetcode.com/problems/convert-to-base-2/)


+ 在二进制上加一个负号。

    ```python
    def baseNeg2(self, N: int) -> str:
        ans = []
        while N:
            ans.append(N & 1)
            N = -(N >> 1)
        return ''.join(map(str, ans[::-1] or [0]))
    ```

### 313. Super Ugly Number

根据指定的质数序列，找出第n个超级丑数。
[查看原题](https://leetcode.com/problems/super-ugly-number/)

+ 解法

    ```python
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        import heapq as hq
        uglies = [1]
        
        def gen_ugly(prime):
            for ugly in uglies:
                yield ugly * prime
                
        merged = hq.merge(*map(gen_ugly, primes))
        while len(uglies) < n:
            ugly = next(merged)
            if ugly != uglies[-1]:
                uglies.append(ugly)
        return uglies[-1]
    ```

### 869. Reordered Power of 2

重新排列一个数字的各位数，判断是否能组成2的幂。
[查看原题](https://leetcode.com/problems/reordered-power-of-2/)

+ 2的幂是指数上升的，所以，在范围内的数一共也没有几个。那么使用Counter来判断是否能组成这个数。

    ```python
    def reorderedPowerOf2(self, N: int) -> bool:
        c = Counter(str(N))
        return any(c==Counter(str(1<<i)) for i in range(0, 30))
    ```

### 1025. Divisor Game

两个人做游戏，黑板上有个数N，每次找到一个0 <x<N的数，并且N能被x整除，然后替换这个N，直到找不出这样x，就输了。问给出这样一个数N，第一个人是否能赢。
[查看原题](https://leetcode.com/problems/divisor-game/)

+ 只要N为偶数就能赢

    ```python
    def divisorGame(self, N: int) -> bool:
        return N & 1 == 0
    ```

### 1037. Valid Boomerang

验证三个坐标点是否共线。
[查看原题](https://leetcode.com/problems/valid-boomerang/)

+ 需要注意的是，除数为0 的情况，所以这里改成了乘法。

    ```python
    def isBoomerang(self, points: List[List[int]]) -> bool:
        return (points[1][1]-points[0][1])*(points[2][0]-points[1][0]) != \
               (points[2][1]-points[1][1])*(points[1][0]-points[0][0])
    ```

### 1041. Robot Bounded In Circle

一个面向北的机器人进行三种操作，一种是前进，或者向左向右转。问一系列的操作中，无限循环时，机器人是否在绕圈。
[查看原题](https://leetcode.com/problems/robot-bounded-in-circle/)

+ 在一次之后，如果面向的不再是北，那么最后将会绕圈。

    ```python
    def isRobotBounded(self, instructions: str) -> bool:
        x, y, dx, dy = 0, 0, 0, 1
        for inst in instructions:
            if inst == 'G': x, y = x+dx, y+dy
            elif inst == 'L': dx, dy = -dy, dx
            elif inst == 'R': dx, dy = dy, -dx
        return (x == y == 0) or (dx, dy) != (0, 1)
    ```

### 1137. N-th Tribonacci Number

三个数的斐波那契数列。
[查看原题](https://leetcode.com/problems/n-th-tribonacci-number/)

```
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4
```

+ 解法

    ```python
    def tribonacci(self, n: int) -> int:
        a, b, c = 1, 0, 0
        for _ in range(n):
            a, b, c = b, c, a+b+c
        return c
    ```

### 1073. Adding Two Negabinary Numbers

两个-2进制的数相加。
[查看原题](https://leetcode.com/problems/adding-two-negabinary-numbers/)

+ 转成十进制相加，再转回-2进制。

    ```python
    def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
        
        def to_ten(arr):
            return sum(d*(-2)**i for i, d in enumerate(arr[::-1]))
        
        def to_neg_binary(n):
            if not n:
                return '0'
            ans = ''
            while n:
                remainder = n % (-2)
                ans += str(abs(remainder))
                n //= -2
                n += (remainder < 0)
            return ans[::-1]
        
        return to_neg_binary(to_ten(arr1) + to_ten(arr2))
    ```

### 1154. Day of the Year

根据输入的日期，返回它是一年中的第几天。
[查看原题](https://leetcode.com/problems/ordinal-number-of-date/)

+ 使用了datetime库，开始还自己手动减

    ```python
    def dayOfYear(self, date: str) -> int:
        import datetime
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        return date.timetuple().tm_yday
    ```

### 1155. Number of Dice Rolls With Target Sum

扔一个f面的 骰子d次，结果为target的次数。
[查看原题](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/)

```
Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.
```

+ 解法

    ```python
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        last_p = collections.defaultdict(int)
        last_p.update({d: 1 for d in range(1, f+1)})
        for i in range(2, d+1):
            new_p = collections.defaultdict(int)
            for j in range(i, i*f+1):
                new_p[j] = sum(last_p[j-k] for k in range(1, f+1))
            last_p = new_p
        return last_p[target] % (10**9+7)
    ```

### 1093. Statistics from a Large Sample

统计大量的样本数据，求最小值，最大值，平均值，众数。
[查看原题](https://leetcode.com/problems/statistics-from-a-large-sample/)

```
Input: count = [0,1,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Output: [1.00000,3.00000,2.37500,2.50000,3.00000]
```

+ 中位数的求法这里没想到，使用二分可以完美的解决奇偶问题。

    ```python
    def sampleStats(self, count: List[int]) -> List[float]:
        n = sum(count)
        mi = next(i for i in range(255) if count[i]) * 1.0
        ma = next(i for i in range(255, -1, -1) if count[i]) * 1.0
        mean = sum(i * val for i, val in enumerate(count)) * 1.0 / n
        mode = count.index(max(count)) * 1.0
        cc = list(itertools.accumulate(count))
        left = bisect.bisect(cc, (n-1)//2)
        right = bisect.bisect(cc, n//2)
        median = (left + right) / 2.0
        return mi, ma, mean, median, mode
    ```

### 1103. Distribute Candies to People

发糖果，按照顺序每个人比上一人多一颗，发到最后再循环。
[查看原题](https://leetcode.com/problems/distribute-candies-to-people/)

```
Input: candies = 7, num_people = 4
Output: [1,2,3,1]
Explanation:
On the first turn, ans[0] += 1, and the array is [1,0,0,0].
On the second turn, ans[1] += 2, and the array is [1,2,0,0].
On the third turn, ans[2] += 3, and the array is [1,2,3,0].
On the fourth turn, ans[3] += 1 (because there is only one candy left), and the final array is [1,2,3,1].
```

+ 解法

    ```python
    def distributeCandies(self, candies: int, n: int) -> List[int]:
        ans = [0] * n
        cur = 1
        while candies > 0:
            ans[cur%n-1] += min(candies, cur)
            candies -= cur
            cur += 1
        return ans
    ```

### 1109. Corporate Flight Bookings

通过给定的一些区间，确定每天的座位数。
[查看原题](https://leetcode.com/problems/corporate-flight-bookings/)

```
Input: bookings = [[1,2,10],[2,3,20],[2,5,25]], n = 5
Output: [10,55,45,25,25]
```

+ 记录变化的状态，然后累加求结果。

    ```python
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        ans = [0] * (n+1)
        for s, e, v in bookings:
            ans[s-1] += v
            ans[e] -= v
        return list(itertools.accumulate(ans))[:-1]
    ```

### 1175. Prime Arrangements

质数排列。
[查看原题](https://leetcode.com/problems/prime-arrangements/)

```
Input: n = 5
Output: 12
Explanation: For example [1,2,5,4,3] is a valid permutation, but [5,2,3,4,1] is not because the prime number 5 is at index 1.
```

+ 解法

    ```python
    def numPrimeArrangements(self, n: int) -> int:
        
        def countPrimes(n):
            is_prime = [False]*2 + [True]*(n-2)
            for i in range(2, int(n ** 0.5)+1):
                if is_prime[i]:
                    is_prime[i*i:n:i] = [False] * len(is_prime[i*i:n:i])
            return sum(is_prime)
        c = countPrimes(n+1)
        ans = math.factorial(c) * math.factorial(n-c)
        
        return ans % (10**9+7)
    ```

### 1360. Number of Days Between Two Dates

计算两个日期之间的天数。
[查看原题](https://leetcode.com/problems/number-of-days-between-two-dates/)

```
Input: date1 = "2020-01-15", date2 = "2019-12-31"
Output: 15
```

+ 方法一：简单的datetime模块方式。

    ```python
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        from datetime import datetime
        d1 = datetime.strptime(date1, '%Y-%m-%d')
        d2 = datetime.strptime(date2, '%Y-%m-%d')
        return abs((d2-d1).days)
    ```

+ 方法二：有个公式，如果将1月二月看成是13月和14月，那么月份转化天数有个公式(153 * m + 8) // 5

    ```python
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        def f(date):
            y, m, d = map(int, date.split('-'))
            if m < 3:
                m += 12
                y -= 1
            return 365 * y + y // 4 + y // 400 - y // 100 + d + (153 * m + 8) // 5
    
        return abs(f(date1) - f(date2))
    ```

### 1363. Largest Multiple of Three

组成的最大的3的倍数。
[查看原题](https://leetcode.com/problems/largest-multiple-of-three/)

```
Input: digits = [8,1,9]
Output: "981"
```

+ 方法一：简单的datetime模块方式。

    ```python
    def largestMultipleOfThree(self, A):
        total = sum(A)
        count = collections.Counter(A)
        A.sort(reverse=1)
    
        def f(i):
            if count[i]:
                A.remove(i)
                count[i] -= 1
            if not A: return ''
            if not any(A): return '0'
            if sum(A) % 3 == 0: return ''.join(map(str, A))
    
        if total % 3 == 0:
            return f(-1)
        if total % 3 == 1 and count[1] + count[4] + count[7]:
            return f(1) or f(4) or f(7)
        if total % 3 == 2 and count[2] + count[5] + count[8]:
            return f(2) or f(5) or f(8)
        if total % 3 == 2:
            return f(1) or f(1) or f(4) or f(4) or f(7) or f(7)
        return f(2) or f(2) or f(5) or f(5) or f(8) or f(8)
    ```
