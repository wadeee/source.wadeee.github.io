---
layout: post
title: Hash Table(哈希表)
published: true
date: 2016-08-02
---

> 加速为你找到目标

## 关于哈希表

哈希表（Hash table），是根据关键码值(Key value)而直接进行访问的数据结构。
也就是说，它通过把关键码值映射到表中一个位置来访问记录，以加快查找的速度。
这个映射函数叫做散列函数，存放记录的数组叫做散列表。

## LeetCode真题

### 1. Two Sum

给定一个数组，找出数组两个元素相加为目标值，假定只有唯一解。
[查看原题](https://leetcode.com/problems/two-sum/description/)

```
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

+ 解法

    ```python
    def two_sum(nums, target):
        buff_dict = {}
        for i, num in enumerate(nums):
            if num not in buff_dict:
                buff_dict[target-num] = i
            else:
                return [buff_dict[num], i]

    ```


### 720. Longest Word in Dictionary

字典中的最长单词，找出一个列表中的一个单词，该单词的子单词也必须在字典中。相同长度的单词，返回字典序最前的一个。
[查看原题](https://leetcode.com/problems/longest-word-in-dictionary/)

```
Input: 
words = ["w","wo","wor","worl", "world"]
Output: "world"
Explanation: 
The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".

```

+ 解法：Brute Force.

    ```python
    def longestWord(self, words):
        res = ''
        wordset = set(words)
        for word in words:
            if len(word)>len(res) or len(word)==len(res) and word<res:
                if all(word[:k] in wordset for k in range(1, len(word))):
                    res = word         
        return res
    ```


### 748. Shortest Completing Word

最短的完整匹配单词。包含licensePlate中的所有字母，大小写不敏感。假设答案一定存在。
[查看原题](https://leetcode.com/problems/shortest-completing-word/)

```
Input: licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]
Output: "steps"
Explanation: The smallest length word that contains the letters "S", "P", "S", and "T".
Note that the answer is not "step", because the letter "s" must occur in the word twice.
Also note that we ignored case for the purposes of comparing whether a letter exists in the word.
```

+ 解法

    ```python
    def shortestCompletingWord(self, licensePlate: 'str', words: 'List[str]') -> 'str':
        ans = ''
        lp = ''.join(x for x in licensePlate.lower() if x.isalpha())
        for w in words:
            temp = list(w.lower())
            for l in lp:
                if l in temp:
                    temp.remove(l)
                else:
                    break
            else:
                if len(w)<len(ans) or ans=='':
                    ans = w
        return ans
    ```

### 811. Subdomain Visit Count

子域名访问量。给定一个三级或二级域名列表，统计所有三级、二级和顶级域名的访问量。
[查看原题](https://leetcode.com/problems/subdomain-visit-count/)

```
https://leetcode.com/problems/subdomain-visit-count/
```

+ 解法

    ```python
    def subdomainVisits(self, cpdomains: 'List[str]') -> 'List[str]':
        ans = collections.defaultdict(int)
        for domain in cpdomains:
            count, d = domain.split()
            count = int(count)
            frags = d.split('.')
            for i in range(len(frags)):
                ans['.'.join(frags[i:])] += count
        return ['{} {}'.format(c, d) for d, c in ans.items()]
    ```


### 884. Uncommon Words from Two Sentences

求两句话中的单词，在本句中出现一次，并不在另一句中的单词。也就是在两句中出现一次。
[查看原题](https://leetcode.com/problems/uncommon-words-from-two-sentences/)

```
Input: A = "this apple is sweet", B = "this apple is sour"
Output: ["sweet","sour"]
```

+ counter

    ```python
    def uncommonFromSentences(self, A: 'str', B: 'str') -> 'List[str]':
        from collections import Counter
        count = Counter((A + ' ' + B).split())
        return [word for word, c in count.items() if c == 1]
    ```

### 1010. Pairs of Songs With Total Durations Divisible by 60

和能被60整除的为一对，求有多少对。
[查看原题](https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

```
Input: [30,20,150,100,40]
Output: 3
Explanation: Three pairs have a total duration divisible by 60:
(time[0] = 30, time[2] = 150): total duration 180
(time[1] = 20, time[3] = 100): total duration 120
(time[1] = 20, time[4] = 40): total duration 60
```

+ 首先判断此链表是否有环。然后在相交点和头结点一起走，一定会在入口相遇。

    ```python
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        c = collections.defaultdict(int)
        ans = 0
        for t in time:
            # ans += c[(60-t%60)%60]
            ans += c[-t % 60]
            c[t%60] += 1
        return ans
    ```


### 1138. Alphabet Board Path

小写字母排列的键盘，要打出目标字母需要移动的操作。
[查看原题](https://leetcode.com/problems/alphabet-board-path/)

```
Input: target = "leet"
Output: "DDR!UURRR!!DDD!"
```

+ 此题需要注意z，然后按照一个优先的顺序移动即可。另外使用字典可以快速定位坐标，而不用每个字符做比较

    ```python
    def alphabetBoardPath(self, target: str) -> str:
        import string
        m = {c: (i//5, i%5) for i, c in enumerate(string.ascii_lowercase)}
        ans = ''
        x0 = y0 = 0
        for c in target:
            x, y = m[c]
            if y < y0: ans += 'L' * (y0-y)
            if x < x0: ans += 'U' * (x0-x)
            if y > y0: ans += 'R' * (y-y0)
            if x > x0: ans += 'D' * (x-x0)
            x0, y0 = x, y
            ans += '!'
        return ans
    ```

### 1072. Flip Columns For Maximum Number of Equal Rows

二维数组，翻转某几列可以最多使多少行内的元素都相同。
[查看原题](https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/)

```
Input: [[0,1],[1,1]]
Output: 1
Explanation: After flipping no values, 1 row has all values equal.

Input: [[0,0,0],[0,0,1],[1,1,0]]
Output: 2
Explanation: After flipping values in the first two columns, the last two rows have equal values.
```

+ 方法一：核心思想在于找到每行的模式，具有相同模式的行，最终可变成同样的数值。

    ```python
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        c = collections.Counter()
        for row in matrix:
            c[tuple([x for x in row])] += 1
            c[tuple([1-x for x in row])] += 1
        return max(c.values())
    ```

+ 方法二：使用异或。方法一中其实有多余的部分，模式与反模式都求了出来，其实没有必要。

    ```python
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        return max(collections.Counter(tuple(r ^ row[0] for r in row) for row in matrix).values())
    ```

### 1160. Find Words That Can Be Formed by Characters

找出能被目标字符串组成的子串长度和。
[查看原题](https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/)

+ 解法

    ```python
    def countCharacters(self, words: List[str], chars: str) -> int:
        ma = collections.Counter(chars)
        return sum(len(w) for w in words if not collections.Counter(w)-ma)
    ```
