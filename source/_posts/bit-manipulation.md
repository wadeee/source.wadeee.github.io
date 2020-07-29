---
layout: post
title: Bit manipulation(位运算)
published: true
date: 2016-09-28
---

> 这才是计算机运算的本质

## 关于位运算

位运算就是直接对整数在内存中的二进制位进行操作。

## LeetCode真题

### 191. Number of 1 Bits

计算数字的二进制中有多少个1。
[查看原题](https://leetcode.com/problems/number-of-1-bits/description/)

```
Input: 11
Output: 3
Explanation: Integer 11 has binary representation 00000000000000000000000000001011
```

+ 方法一：常规解法，使用1与n作与运算，如果不是0说明，含有一个1。

    ```python
    def hamming_weight(n):
        bits, mask = 0, 1
        for _ in range(32):
            if n&mask != 0:
                bits += 1
            mask <<= 1
        return bits
    ```

+ 方法二：关键点是，一个数n和n-1的与运算操作，相当于去掉了最右面的1。

    ```python
    def hamming_weigth(n):
        bits = 0
        while n:
            bits += 1
            n = (n-1) & n
        return bits
    ```


### 136. Single Number

找出数组中不重复的元素。其它元素出现两次。
[查看原题](https://leetcode.com/problems/single-number/description/)

```
Input: [4,1,2,1,2]
Output: 4
```

+ 解

    ```python
    def single_num(nums):
        return reduce(lambda x, y: x ^ y, nums)
    ```

### 137. Single Number II

找出数组中出现一次的元素，其它元素出现三次。
[查看原题](https://leetcode.com/problems/single-number-ii/description/)

```
Input: [2,2,3,2]
Output: 3
```

+ 方法一：找出单独元素每一位的值。如果把所有数字的二进制每一位加起来，如果某一位可以被3整除，则表示单独元素的该位为0，否则为1。以下使用count来表示每一位1的个数。假设count%3!=0为True，说明该元素i位为1，然后是用|=更新ans在第i个位置的值，这里也可以使用+=，但是效率稍慢。convert的作用是因为python中的int是个对象，且没有最大限制，不是在第32位使用1来表示负数。

    ```python
    def singleNumber(self, nums, n=3):
        ans = 0
        for i in range(32):
            count = 0
            for num in nums:
                if ((num >> i) & 1):
                    count += 1
            ans |= ((count%n!=0) << i)
        return self.convert(ans)
    
    def convert(self, x):
        if x >= 2**31:
            x -= 2**32
        return x
    ```

+ 方法2：状态机解法

    ```python
    def singleNumber(self, nums):
        ones, twos = 0, 0;
        for i in range(len(nums)):
            ones = (ones ^ nums[i]) & ~twos
            twos = (twos ^ nums[i]) & ~ones
        return ones
    ```

### 260. Single Number III

找出数组中两个唯一出现一次的元素，其余元素均出现两次。
[查看原题](https://leetcode.com/problems/single-number-iii/description/)

```
Input:  [1,2,1,3,2,5]
Output: [3,5]
```

+ 思想：将这两个元素分到两个组，由于这两个数不相等，所以亦或结果不为0，也就是说二进制中至少有一位1，记为第n位。我们以第n位是否为1，把数组分为两个子数组。

    ```python
    def singleNumber(self, nums):
        total_xor = self.get_xor(nums)
        mask = 1
        while total_xor&mask == 0:
            mask <<= 1
        p1 = [num for num in nums if num&mask==0]
        p2 = [num for num in nums if num&mask!=0]
        return [self.get_xor(p1), self.get_xor(p2)]
        
    def get_xor(self, nums):
        from functools import reduce
        return reduce(lambda x, y: x ^ y, nums)
    ```


### 371. Sum of Two Integers

不用加减乘除做加法。
[查看原题](https://leetcode.com/problems/sum-of-two-integers/description/)

+ 实际上加法分为三个步骤
  
  相加但不进位，1^0=1，1^1=0，0^0=0，所以第一步用异或。
  只求进位的结果，只有两个1才会进位，所以用&，然后左移1位，表示要进的位。
  把前两步的结果再重复1，2步，直到没有进位产生，即b=0。

    ```python
    def getSum(self, a, b):
        # 32 bits integer max
        MAX = 0x7FFFFFFF  # 2**31-1
        # 32 bits interger min  
        MIN = 0x80000000  # -2**31
        # mask to get last 32 bits
        mask = 0xFFFFFFFF  # 2*32-1
        while b != 0:
            # ^ get different bits and & gets double 1s, << moves carry
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return a if a <= MAX else ~(a ^ mask)
    ```

### 190. Reverse Bits

返回一个数的二进制的倒序的十进制。
[查看原题](https://leetcode.com/problems/reverse-bits/)

```
Input: 43261596
Output: 964176192
Explanation: 43261596 represented in binary as 00000010100101000001111010011100, 
             return 964176192 represented in binary as 00111001011110000010100101000000.
```

+ 方法一：使用原生库。ljust表示在右侧补’0’。或者使用format来补0。

    ```python
    def reverseBits(self, n):
        return int(bin(n)[:1:-1].ljust(32, '0'), 2)
        # return int('{:0<32s}'.format(bin(n)[:1:-1]), 2)
    ```

+ 方法二：自己实现进制转换，使用位运算优化。

    ```python
    def reverseBits(self, n):
        code = 0
        for _ in range(32):
            code = (code<<1) + (n&1)
            n >>= 1
        return code
    ```


### 389. Find the Difference

s和t两个由小写字母组成的字符串，t是由s打乱顺序并再随机添加一个小写字母组成。
[查看原题](https://leetcode.com/problems/find-the-difference/description/)

+ 方法一：使用Collection。

    ```python
    def findTheDifference(self, s, t):
        from collections import Counter
        return next((Counter(t) - Counter(s)).elements())
    ```

+ 方法二：使用异或。

    ```python
    def findTheDifference(self, s, t):
        from operator import xor
        from functools import reduce
        return chr(reduce(xor, map(ord, s+t)))
    ```

### 401. Binary Watch

有这样一个二进制的手表，输入一个n，表示有几个亮着的灯，返回所有可能出现的时间。时间范围为12小时制，即hours(0-11)，minutes(0-59)。
[查看原题](https://leetcode.com/problems/binary-watch/description/)

```
Input: n = 1
Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
```

+ 遍历所有可能的时间，找到符合条件的。因为表中的数组都是二进制，所以’1’的个数就是亮灯的个数。

    ```python
    def readBinaryWatch(self, num):
        return ['{:d}:{:0>2d}'.format(h, m)
                for h in range(12) for m in range(60)
                if (bin(h)+bin(m)).count('1') == num]
    ```

### 405. Convert a Number to Hexadecimal

把一个32位有符号的整数转换成16进制。
[查看原题](https://leetcode.com/problems/convert-a-number-to-hexadecimal/)

```
Input:
26
Output:
"1a"

Input:
-1
Output:
"ffffffff"
```

+ 解法

    ```python
    def toHex(self, num):
        return ''.join(['0123456789abcdef'[(num >> 4 * i) & 15]
                         for i in range(8)])[::-1].lstrip('0') or '0'
    ```


### 461. Hamming Distance

求两个正数的原码中不同位的个数。
[查看原题](https://leetcode.com/problems/hamming-distance/)

```
Input: x = 1, y = 4
Output: 2
Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
The above arrows point to positions where the corresponding bits are different.
```

+ 解法

    ```python
    def hammingDistance(self, x, y):
        return bin(x ^ y).count('1')
    ```

### 476. Number Complement

给定一个正数，求其原码的按位取反后的数。
[查看原题](https://leetcode.com/problems/number-complement/)

```
Input: 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
```

+ 方法一：其实就是求101和111的异或。所以先找到111。

    ```python
    def findComplement(self, num):
        i = 1
        while i <= num:
            i <<= 1
        return (i-1) ^ num
    ```

+ 方法二：更少的位移。核心思想还是找到111。比如一个8位数，最高代表符号：1000000，先将其右移1位，使得左边两位都变成1。然后再右移2位，使得左边四位变成1，以此类推，8位数最多移动3次就可以得到1111111，32位则还需要再移动2次。

    ```python
    def findComplement(self, num):
        mask = num
        for i in range(5):
            mask |= mask >> (2**i)
        return num ^ mask
    ```

### 693. Binary Number with Alternating Bits

二进制是否是交替的0和1。
[查看原题](https://leetcode.com/problems/binary-number-with-alternating-bits/)

```
Input: 5
Output: True
Explanation:
The binary representation of 5 is: 101
```

+ 方法一：除2法。

    ```python
    def hasAlternatingBits(self, n):
        n, cur = divmod(n, 2)
        while n:
            if cur == n % 2:
                return False
            n, cur = divmod(n, 2)
        return True
    ```

+ 方法二：异或。

    ```python
    def hasAlternatingBits(self, n):
        if not n:
            return False
        num = n ^ (n >> 1)
        return not (num & num+1)
    ```


### 762. Prime Number of Set Bits in Binary Representation

求某范围的所有自然数中，二进制中1的个数是质数的个数。
[查看原题](https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation/)

```
Input: L = 10, R = 15
Output: 5
Explanation:
10 -> 1010 (2 set bits, 2 is prime)
11 -> 1011 (3 set bits, 3 is prime)
12 -> 1100 (2 set bits, 2 is prime)
13 -> 1101 (3 set bits, 3 is prime)
14 -> 1110 (3 set bits, 3 is prime)
15 -> 1111 (4 set bits, 4 is not prime)
```

+ 方法一：direct.

    ```python
    def countPrimeSetBits(self, L: 'int', R: 'int') -> 'int':
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        # ans = 0
        # for num in range(L, R+1):
        #     if bin(num)[2:].count('1') in primes:
        #         ans += 1
        # return ans
        return sum(bin(n)[2:].count('1') in primes for n in range(L, R+1))
    ```

+ 方法二：位运算。p 的2，3，5，7。。位是1，其余是0，这样在右移后，可&1就可以判断这个数是否是质数。

    ```python
    def countPrimeSetBits(self, L: 'int', R: 'int') -> 'int':
        p = int('10100010100010101100', 2)
        return sum(p >> bin(i).count('1') & 1 for i in range(L, R+1))
    ```

### 868. Binary Gap

二进制两个1的最大距离。
[查看原题](https://leetcode.com/problems/binary-gap/)

```
Input: 22
Output: 2
Explanation: 
22 in binary is 0b10110.
In the binary representation of 22, there are three ones, and two consecutive pairs of 1's.
The first consecutive pair of 1's have distance 2.
The second consecutive pair of 1's have distance 1.
The answer is the largest of these two distances, which is 2.
```

+ 列表生成式。

    ```python
    def binaryGap(self, N: 'int') -> 'int':
        one = [i for i, v in enumerate(bin(N)) if v == '1']
        # return max([one[i+1] - one[i] for i in range(len(one)-1)] or [0])
        return max([b-a for a, b in zip(one, one[1:])] or [0])
    ```

### 268. Missing Number

0~n中缺失的数字。
[查看原题](https://leetcode.com/problems/missing-number/description/)

+ 方法一：数学公式。

    ```python
    def missingNumber(self, nums):
        n = len(nums)
        expected_sum = n*(n+1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
    ```

+ 方法二：XOR.

    ```python
    def missingNumber(self, nums: 'List[int]') -> 'int':
        missing = len(nums)
        for i, num in enumerate(nums):
            missing ^= i ^ num
        return missing
    ```

### 1012. Complement of Base 10 Integer

非负数的反码。
[查看原题](https://leetcode.com/problems/complement-of-base-10-integer/)

+ 解法

    ```python
    def bitwiseComplement(self, N: int) -> int:
        mask = 1
        while mask < N:
            mask = (mask << 1) + 1
        # return mask - N
        return N ^ mask
    ```


### 1404. Number of Steps to Reduce a Number in Binary Representation to One

几下操作可以将其变为1。偶数除以2，奇数+1.
[查看原题](https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/)

```
Input: s = "1101"
Output: 6
Explanation: "1101" corressponds to number 13 in their decimal representation.
Step 1) 13 is odd, add 1 and obtain 14. 
Step 2) 14 is even, divide by 2 and obtain 7.
Step 3) 7 is odd, add 1 and obtain 8.
Step 4) 8 is even, divide by 2 and obtain 4.  
Step 5) 4 is even, divide by 2 and obtain 2. 
Step 6) 2 is even, divide by 2 and obtain 1.
```

+ 解法

    ```python
    def numSteps(self, s: str) -> int:
        i, mid_0 = 0, 0
        for j in range(1, len(s)):
            if s[j] == '1':
                mid_0 += j - i - 1
                i = j
        if i == 0: return len(s) - 1
        return mid_0 + 1 + len(s)
    ```


### 201. Bitwise AND of Numbers Range

范围内的数字求与运算和。
[查看原题](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

```
Input: [5,7]
Output: 4
```

+ 解法

    ```python
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        i = 0 
        while m != n:
            m >>= 1
            n >>= 1
            i += 1
        return n << i
    ```


### 1442. Count Triplets That Can Form Two Arrays of Equal XOR

数组中找出两段的异或和相等。
[查看原题](https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/)

+ 找规律

    ```python
    def countTriplets(self, arr: List[int]) -> int:
        m = list(itertools.accumulate([0] + arr, operator.xor))
        count = 0
        for i in range(len(m)):
            for j in range(i+1, len(m)):
                if m[i] == m[j]:
                    count += j-i-1
        return count
    ```


### 1238. Circular Permutation in Binary Representation

返回指定为位数的二进制环，每两个数的二进制只有1位不同。
[查看原题](https://leetcode.com/problems/circular-permutation-in-binary-representation/)

```
Input: n = 2, start = 3
Output: [3,2,0,1]
Explanation: The binary representation of the permutation is (11,10,00,01). 
All the adjacent element differ by one bit. Another valid permutation is [3,1,0,2]
```

+ 我想了半天这道题，以为和二进制无关，是个数学题，没想到最后还得用异或来解决。这是个 gray code的问题，有一个公式。

    ```python
    def circularPermutation(self, n, start):
        return [start ^ i ^ i >> 1 for i in range(1 << n)]
    ```
