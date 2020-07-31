---
layout: post
title: Graph(图)
published: true
date: 2016-11-02
---

> 从Floyd开始学图

## 关于图

图论中的图是由若干给定的点及连接两点的线所构成的图形，这种图形通常用来描述某些事物之间的某种特定关系，用点代表事物，用连接两点的线表示相应两个事物间具有这种关系。

## LeetCode真题

### 990. Satisfiability of Equality Equations

满足所有方程式，判断是否存在变量的值满足所有的等式与不等式。
[查看原题](https://leetcode.com/problems/satisfiability-of-equality-equations/)

```
Input: ["a==b","b!=a"]
Output: false
Explanation: If we assign say, a = 1 and b = 1, then the first equation is satisfied, but not the second.  There is no way to assign the variables to satisfy both equations.

Input: ["a==b","b==c","a==c"]
Output: true

Input: ["a==b","b!=c","c==a"]
Output: false
```

+ union find. 并查集。find方法可以想象成一个链表，返回的是链表末尾key,val相等的元素。同时建立连接关系。如a==b, b==c时fc={'a': 'b', 'b': 'c', 'c': 'c'}比较a!=c时就会最终找到fc['a'] == 'c'；如a==b, c==a时，fc={'a': 'b', 'b': 'b', 'c': 'b'}。

    ```python
    def equationsPossible(self, equations: 'List[str]') -> 'bool':
        equations.sort(key=lambda e: e[1] == '!')
        uf = {a: a for a in string.ascii_lowercase}
    
        def find(x):
            if x != uf[x]: 
                uf[x] = find(uf[x])
            return uf[x]
    
        for a, e, _, b in equations:
            if e == "=":
                uf[find(a)] = find(b)
            else:
                if find(a) == find(b):
                    return False
        return True
    ```


### 997. Find the Town Judge

找到小镇审判长。审判长被除自己以外的所有人信任，并且不信任任何人。根据信任列表找出审判长。
[查看原题](https://leetcode.com/problems/find-the-town-judge/)

```
Input: N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
Output: 3
Input: N = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1
Input: N = 3, trust = [[1,3],[2,3]]
Output: 3
```

+ 方法一：brute force.

    ```python
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        if not trust:
            return N
        a, b = zip(*trust)           
        candidates = collections.Counter(b)
        villages = set(a)
        for c, votes in candidates.most_common():
            if votes < N - 1:
                return -1
            if c not in villages:
                return c
        return -1
    ```

+ 方法二：定向图。

    ```python
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        count = [0] * (N + 1)
        for i, j in trust:
            count[i] -= 1
            count[j] += 1
            print(count)
        for i in range(1, N + 1):
            if count[i] == N - 1:
                return i
        return -1
    ```


### 133. Clone Graph

深拷贝一个简单环。
[查看原题](https://leetcode.com/problems/clone-graph/)

+ 解法

    ```python
    def cloneGraph(self, node: 'Node') -> 'Node':
        cp = collections.defaultdict(lambda: Node(0, []))
        nodes = [node]
        seen = set()
        while nodes:
            n = nodes.pop()
            cp[n].val = n.val
            cp[n].neighbors = [cp[x] for x in n.neighbors]
            nodes.extend(x for x in n.neighbors if x not in seen)
            seen.add(n)
        return cp[node]
    ```

### 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

找到距离范围内邻居最少的城市。
[查看原题](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

```
Input: n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4
Output: 3
Explanation: The figure above describes the graph. 
The neighboring cities at a distanceThreshold = 4 for each city are:
City 0 -> [City 1, City 2] 
City 1 -> [City 0, City 2, City 3] 
City 2 -> [City 0, City 1, City 3] 
City 3 -> [City 1, City 2] 
Cities 0 and 3 have 2 neighboring cities at a distanceThreshold = 4, but we have to return city 3 since it has the greatest number.
```

+ 方法一：狄克斯特拉算法。这里没想到用一个堆来维持最小的距离。

    ```python
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        g = collections.defaultdict(list)
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
            
        def count_neighbor(city):
            heap = [(0, city)]
            dist = {}
            
            while heap:
                cur_w, u = heapq.heappop(heap)
                if u in dist:
                    continue
                if u != city:
                    dist[u] = cur_w
                for v, w in g[u]:
                    if v in dist:
                        continue
                    if cur_w + w <= distanceThreshold:
                        heapq.heappush(heap, (cur_w+w, v))
            return len(dist)
        
        return min(range(n), key=lambda x: (count_neighbor(x), -x))
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


### 1267. Count Servers that Communicate

找到2个以上的服务器连接个数，服务器可以在一行或是一列算是连接上。
[查看原题](https://leetcode.com/problems/count-servers-that-communicate/)

```
Input: grid = [[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
Output: 4
Explanation: The two servers in the first row can communicate with each other. The two servers in the third column can communicate with each other. The server at right bottom corner can't communicate with any other server.
```

+ 行列累计求和，但是只是用来判断而不是累加，然后遍历所有的元素。

    ```python
    def countServers(self, g: List[List[int]]) -> int:
        X, Y = tuple(map(sum, g)), tuple(map(sum, zip(*g)))
        return sum(X[i]+Y[j]>2 for i in range(len(g)) for j in range(len(g[0])) if g[i][j])
    ```

### 886. Possible Bipartition

将不喜欢的人放在两组中，根据关系是否能将其分为2组。
[查看原题](https://leetcode.com/problems/possible-bipartition/)

```
Input: N = 4, dislikes = [[1,2],[1,3],[2,4]]
Output: true
Explanation: group1 [1,4], group2 [2,3]

Input: N = 3, dislikes = [[1,2],[1,3],[2,3]]
Output: false
```

+ 方法一：dfs。等同于在一个无向图中，寻找一个奇数边的环。

    ```python
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        g = [[] for _ in range(N+1)]
        for a, b in dislikes:
            g[a].append(b)
            g[b].append(a)
    
        seen = set()
        def dfs(i, p, p_len):
            seen.add(i)
            p[i] = p_len
            for nxt in g[i]:
                if nxt not in seen:
                    if dfs(nxt, p, p_len+1):
                        return True
                elif nxt in p and (p_len-p[nxt])&1==0:
                    return True
            p.pop(i)
            return False
    
        p = {}
        for i in range(1, N+1):
            if i not in seen and dfs(i, p, 0):
                return False
        return True
    ```


### 207. Course Schedule

课程调度，课程有依赖关系，问是否能完成所有的课程。
[查看原题](https://leetcode.com/problems/course-schedule/)

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
```

+ 方法一：dfs。注意这里状态要用3中，1表示遍历过，-1表示正在遍历，0表未遍历。这样可以避免重复的遍历。

    ```python
    def canFinish(self, n: int, prerequisites: List[List[int]]) -> bool:
        g = [[] for _ in range(n)]
        for a, b in prerequisites:
            g[a].append(b)
            
        seen = [0] * n
        
        def dfs(i):
            if seen[i] in {1, -1}: return seen[i]==1
            seen[i] = -1
            if any(not dfs(j) for j in g[i]): return False
            seen[i] = 1
            return True
    ```

### 1462. Course Schedule IV

和上题差不多，问题不一样，问的是根据给定的依赖关系，判断两节课是否有依赖。
[查看原题](https://leetcode.com/problems/course-schedule-iv/)

+ bfs. 拓扑排序。

    ```python
    def checkIfPrerequisite(self, n: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        g = collections.defaultdict(list)
        degree = [0] * n
        pres = [set() for _ in range(n)]
        for u, v in prerequisites:
            g[u].append(v)
            degree[v] -= 1
            pres[v].add(u)
        bfs = [i for i in range(n) if degree[i]==0]
        for i in bfs:
            for j in g[i]:
                degree[j] += 1
                pres[j] |= pres[i]
                if degree[j] == 0:
                    bfs.append(j)
        return [a in pres[b] for a, b in queries]
    ```

### 1466. Reorder Routes to Make All Paths Lead to the City Zero

有一个有向图，两个节点之间只有一条边，要求所有的边指向0，需要改多少条边的方向。
[查看原题](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)

```
Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).
```

+ 解法

    ```python
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        g1 = collections.defaultdict(list)
        g2 = collections.defaultdict(list)
        for u, v in connections:
            g1[u].append(v)
            g2[v].append(u)
        
        seen = set()
        def dfs(i):
            seen.add(i)
            ans = 0
            for j in g1[i]:
                if j not in seen:
                    ans += 1 + dfs(j)
                    
            for k in g2[i]:
                if k not in seen:
                    ans += dfs(k)
                    
            return ans
        return dfs(0)
    ```


### 1210. Minimum Moves to Reach Target with Rotations

一个占位2格的蛇，从左上走到右下需要最少的步数，可以旋转。
[查看原题](https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/)

```
Input: grid = [[0,0,0,0,0,1],
               [1,1,0,0,1,0],
               [0,0,0,0,1,1],
               [0,0,1,0,1,0],
               [0,1,1,0,0,0],
               [0,1,1,0,0,0]]
Output: 11
Explanation:
One possible solution is [right, right, rotate clockwise, right, down, down, down, down, rotate counterclockwise, right, down].
```

+ 将蛇的横竖状态记录，这样一个点也能表示。

    ```python
    def minimumMoves(self, g: List[List[int]]) -> int:
        n = len(g)
        q, seen, target = [(0, 0, 0, 0)], set(), (n-1, n-2, 0)
        for r, c, dr, step in q:
            if (r, c, dr) == target: return step
            if (r, c, dr) not in seen:
                seen.add((r, c, dr))
                if dr:
                    if c+1<n and g[r][c+1]==g[r+1][c+1]==0:
                        q += [(r, c+1, 1, step+1), (r, c, 0, step+1)]
                    if r+2<n and g[r+2][c]==0:
                        q += [(r+1, c, 1, step+1)]
                else:
                    if r+1<n and g[r+1][c]==g[r+1][c+1]==0:
                        q += [(r+1, c, 0, step+1), (r, c, 1, step+1)]
                    if c+2<n and g[r][c+2]==0:
                        q += [(r, c+1, 0, step+1)]
        return -1
    ```


### 1202. Smallest String With Swaps

给定一组pairs表明索引对可以互换，求这个字符串能换的最小值时多少，同一对可以进行多次互换。
[查看原题](https://leetcode.com/problems/smallest-string-with-swaps/)

```
Input: s = "dcab", pairs = [[0,3],[1,2]]
Output: "bacd"
Explaination: 
Swap s[0] and s[3], s = "bcad"
Swap s[1] and s[2], s = "bacd"
```

+ union-find

    ```python
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n):
                self.p = list(range(n))
            def union(self, x, y):
                self.p[self.find(x)] = self.find(y)
            def find(self, x):
                if x!=self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x, y in pairs:
            uf.union(x, y)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for comp_id in m.keys(): 
            m[comp_id].sort(reverse=True)
        for i in range(len(s)): 
            res.append(m[uf.find(i)].pop())
        return ''.join(res)
    ```

### 787. Cheapest Flights Within K Stops

经过K个站点的最便宜的航班。
[查看原题](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

+ 狄克斯特拉算法，只不过多了一个条件，经过K个站点。不需要用seen记录已经去过的点，因为该点可能有更少步数的到达方式。

    ```python
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        g = collections.defaultdict(list)
        for u, v, w in flights:
            g[u].append((v, w))
        
        q = [(0, src, 0)]
        heapq.heapify(q)
        while q:
            p, city, step = heapq.heappop(q)
            if city == dst:
                return p
            for v, w in g[city]:
                if step < K+1:
                    heapq.heappush(q, (p+w, v, step+1))
        return -1
    ```
