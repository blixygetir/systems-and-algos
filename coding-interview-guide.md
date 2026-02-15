# Complete Coding Interview Preparation Guide
> For CS Undergraduates targeting FAANG & Top Tech Companies

---

## Table of Contents
1. [Data Structures Deep Dive](#1-data-structures-deep-dive)
2. [Algorithm Patterns & Tricks](#2-algorithm-patterns--tricks)
3. [Time & Space Complexity Mastery](#3-time--space-complexity-mastery)
4. [Red Flags to Avoid](#4-red-flags-to-avoid)
5. [Interview Tips & Tricks](#5-interview-tips--tricks)
6. [Problem-Solving Framework](#6-problem-solving-framework)
7. [Language-Specific Tips](#7-language-specific-tips)
8. [Resources & Practice Plan](#8-resources--practice-plan)

---

## 1. Data Structures Deep Dive

### Arrays
**When to use:** Contiguous memory, index-based access, fixed or dynamic size

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(1) |
| Search (unsorted) | O(n) |
| Search (sorted) | O(log n) |
| Insert at end | O(1) amortized |
| Insert at position | O(n) |
| Delete | O(n) |

**Tricks:**
- Two-pointer technique for sorted arrays
- Prefix sum for range queries: `prefix[i] = prefix[i-1] + arr[i]`
- Kadane's algorithm for max subarray: `maxEndingHere = max(arr[i], maxEndingHere + arr[i])`
- For in-place operations, iterate backwards to avoid overwriting

**Common Patterns:**
```
// Two Sum pattern (using HashMap)
for each num in array:
    complement = target - num
    if complement in hashmap:
        return [hashmap[complement], i]
    hashmap[num] = i

// Sliding Window Template
left = 0
for right in range(len(arr)):
    // expand window
    while (window is invalid):
        // shrink from left
        left += 1
    // update result
```

---

### Linked Lists
**When to use:** Frequent insertions/deletions, unknown size, no random access needed

| Operation | Time Complexity |
|-----------|----------------|
| Access | O(n) |
| Search | O(n) |
| Insert at head | O(1) |
| Insert at tail | O(1) with tail pointer, O(n) without |
| Delete | O(1) if node given, O(n) to find |

**Tricks:**
- **Dummy head node**: Simplifies edge cases (empty list, single node)
  ```
  dummy = ListNode(0)
  dummy.next = head
  // ... operations
  return dummy.next
  ```
- **Two pointers (slow/fast)**: Cycle detection, find middle, find nth from end
  ```
  // Find middle
  slow = fast = head
  while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
  // slow is at middle
  
  // Detect cycle (Floyd's algorithm)
  while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
      if slow == fast:
          return True  // cycle exists
  ```
- **Reverse in-place**:
  ```
  prev = None
  curr = head
  while curr:
      next_temp = curr.next
      curr.next = prev
      prev = curr
      curr = next_temp
  return prev
  ```

**Red Flags:**
- Forgetting to handle null/empty list
- Losing reference to nodes (always save `next` before modifying)
- Not updating tail pointer after operations

---

### Stacks
**When to use:** LIFO operations, matching brackets, undo functionality, DFS

| Operation | Time Complexity |
|-----------|----------------|
| Push | O(1) |
| Pop | O(1) |
| Peek | O(1) |

**Tricks:**
- **Monotonic Stack**: Find next greater/smaller element in O(n)
  ```
  // Next Greater Element
  result = [-1] * len(arr)
  stack = []  // stores indices
  for i in range(len(arr)):
      while stack and arr[stack[-1]] < arr[i]:
          result[stack.pop()] = arr[i]
      stack.append(i)
  ```
- **Expression evaluation**: Use two stacks (operators + operands)
- **Min Stack**: Keep parallel stack tracking minimums
  ```
  push(x):
      stack.push(x)
      if minStack.empty() or x <= minStack.top():
          minStack.push(x)
  
  pop():
      if stack.top() == minStack.top():
          minStack.pop()
      stack.pop()
  ```

---

### Queues
**When to use:** FIFO operations, BFS, scheduling, buffering

| Operation | Time Complexity |
|-----------|----------------|
| Enqueue | O(1) |
| Dequeue | O(1) |
| Peek | O(1) |

**Variants:**
- **Deque (Double-ended queue)**: Insert/remove from both ends O(1)
- **Priority Queue/Heap**: See Heaps section
- **Circular Queue**: Fixed size, wrap around

**Tricks:**
- **Sliding Window Maximum** using Deque:
  ```
  // Keep deque in decreasing order
  for i in range(len(arr)):
      // Remove indices outside window
      while deque and deque[0] < i - k + 1:
          deque.popleft()
      // Remove smaller elements
      while deque and arr[deque[-1]] < arr[i]:
          deque.pop()
      deque.append(i)
      if i >= k - 1:
          result.append(arr[deque[0]])
  ```

---

### Hash Maps / Hash Tables
**When to use:** O(1) lookups, counting, caching, deduplication

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Search | O(1) | O(n) |

**Tricks:**
- **Two Sum pattern**: Store complement
- **Frequency counting**: `count[item] = count.get(item, 0) + 1`
- **Grouping/Anagrams**: Use sorted string or character count as key
  ```
  // Group anagrams
  key = tuple(sorted(word))  // or
  key = tuple(count of each char)  // O(1) for fixed alphabet
  groups[key].append(word)
  ```
- **Subarray Sum equals K**:
  ```
  count = 0
  prefix_sum = 0
  sum_count = {0: 1}
  for num in arr:
      prefix_sum += num
      if prefix_sum - k in sum_count:
          count += sum_count[prefix_sum - k]
      sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
  ```

**Red Flags:**
- Hash collisions in worst case → O(n)
- Mutable objects as keys (don't use lists as keys in Python)
- Not handling missing keys

---

### Trees

#### Binary Trees
| Operation | Average | Worst (skewed) |
|-----------|---------|----------------|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |

**Traversal Patterns:**
```
// Inorder (Left, Root, Right) - gives sorted order for BST
def inorder(node):
    if node:
        inorder(node.left)
        visit(node)
        inorder(node.right)

// Preorder (Root, Left, Right) - useful for copying/serialization
def preorder(node):
    if node:
        visit(node)
        preorder(node.left)
        preorder(node.right)

// Postorder (Left, Right, Root) - useful for deletion
def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        visit(node)

// Level Order (BFS)
def levelOrder(root):
    queue = [root]
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.pop(0)
            visit(node)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
```

**Tricks:**
- **Height/Depth calculation**:
  ```
  def height(node):
      if not node: return 0
      return 1 + max(height(node.left), height(node.right))
  ```
- **Diameter** (longest path): Track max of (leftHeight + rightHeight) at each node
- **LCA (Lowest Common Ancestor)**:
  ```
  def lca(root, p, q):
      if not root or root == p or root == q:
          return root
      left = lca(root.left, p, q)
      right = lca(root.right, p, q)
      if left and right: return root
      return left or right
  ```
- **Validate BST**: Pass min/max bounds down
  ```
  def isValidBST(node, min_val, max_val):
      if not node: return True
      if node.val <= min_val or node.val >= max_val:
          return False
      return isValidBST(node.left, min_val, node.val) and \
             isValidBST(node.right, node.val, max_val)
  ```

#### Binary Search Tree (BST)
- **Inorder traversal gives sorted order**
- Search/Insert/Delete: O(log n) average, O(n) worst
- Use for: Ordered data, range queries, predecessor/successor

#### Balanced Trees (AVL, Red-Black)
- Guarantee O(log n) operations
- Self-balancing through rotations
- Usually use library implementations (TreeMap, TreeSet)

#### Tries (Prefix Trees)
**When to use:** Autocomplete, spell checker, prefix matching, word games

```
class TrieNode:
    def __init__(self):
        self.children = {}  # or [None] * 26 for lowercase only
        self.is_end = False

class Trie:
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**Complexity:** O(m) where m = length of word

---

### Heaps / Priority Queues
**When to use:** K largest/smallest, median finding, scheduling, Dijkstra's

| Operation | Time Complexity |
|-----------|----------------|
| Insert (push) | O(log n) |
| Extract min/max | O(log n) |
| Peek | O(1) |
| Build heap | O(n) |

**Tricks:**
- **K largest elements**: Use min-heap of size K
  ```
  import heapq
  heap = []
  for num in arr:
      heapq.heappush(heap, num)
      if len(heap) > k:
          heapq.heappop(heap)
  // heap[0] is kth largest
  ```
- **K smallest elements**: Use max-heap of size K (negate values)
- **Merge K sorted lists**:
  ```
  heap = [(list[0], 0, list) for list in lists if list]
  heapify(heap)
  while heap:
      val, idx, lst = heappop(heap)
      result.append(val)
      if idx + 1 < len(lst):
          heappush(heap, (lst[idx+1], idx+1, lst))
  ```
- **Running Median**: Two heaps (max-heap for lower half, min-heap for upper half)

**Heap Property:**
- Min-heap: parent ≤ children
- Max-heap: parent ≥ children
- Complete binary tree stored as array
- Parent of i: (i-1)//2
- Children of i: 2i+1, 2i+2

---

### Graphs
**Representations:**

| Type | Space | Edge Check | Iterate Neighbors |
|------|-------|------------|-------------------|
| Adjacency Matrix | O(V²) | O(1) | O(V) |
| Adjacency List | O(V+E) | O(degree) | O(degree) |

**When to use which:**
- **Dense graph** (E ≈ V²): Matrix
- **Sparse graph** (E << V²): List (most real-world graphs)

**BFS Template:**
```python
def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    level = 0
    
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1
```
**Use for:** Shortest path (unweighted), level-order traversal

**DFS Template:**
```python
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Iterative DFS
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
```
**Use for:** Path finding, cycle detection, topological sort, connected components

**Topological Sort (DAG only):**
```python
# Kahn's Algorithm (BFS)
def topological_sort(graph, n):
    indegree = [0] * n
    for node in graph:
        for neighbor in graph[node]:
            indegree[neighbor] += 1
    
    queue = deque([i for i in range(n) if indegree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []  # empty if cycle
```

**Cycle Detection:**
```python
# Directed Graph - DFS with colors (0=white, 1=gray, 2=black)
def has_cycle(graph, node, color):
    color[node] = 1  # visiting
    for neighbor in graph[node]:
        if color[neighbor] == 1:  # back edge
            return True
        if color[neighbor] == 0 and has_cycle(graph, neighbor, color):
            return True
    color[node] = 2  # done
    return False

# Undirected Graph - track parent
def has_cycle_undirected(graph, node, visited, parent):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if has_cycle_undirected(graph, neighbor, visited, node):
                return True
        elif neighbor != parent:
            return True
    return False
```

**Dijkstra's Algorithm (weighted, non-negative):**
```python
def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, node = heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = dist[node] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heappush(heap, (new_dist, neighbor))
    
    return dist
```
**Complexity:** O((V + E) log V)

**Union-Find (Disjoint Set):**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```
**Use for:** Connected components, cycle detection, Kruskal's MST
**Complexity:** O(α(n)) ≈ O(1) per operation with optimizations

---

## 2. Algorithm Patterns & Tricks

### Two Pointers
**When to use:** Sorted arrays, finding pairs, removing duplicates, partitioning

```python
# Two Sum II (sorted array)
left, right = 0, len(arr) - 1
while left < right:
    sum = arr[left] + arr[right]
    if sum == target:
        return [left, right]
    elif sum < target:
        left += 1
    else:
        right -= 1

# Remove duplicates in-place
def removeDuplicates(nums):
    if not nums: return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read-1]:
            nums[write] = nums[read]
            write += 1
    return write

# Container with most water
left, right = 0, len(height) - 1
max_area = 0
while left < right:
    area = min(height[left], height[right]) * (right - left)
    max_area = max(max_area, area)
    if height[left] < height[right]:
        left += 1
    else:
        right -= 1
```

### Sliding Window
**When to use:** Subarray/substring problems, fixed or variable window size

```python
# Fixed size window - max sum of k elements
def maxSumSubarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable size window - smallest subarray with sum >= target
def minSubArrayLen(target, nums):
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        current_sum += nums[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0

# Longest substring with at most K distinct characters
def lengthOfLongestSubstringKDistinct(s, k):
    left = 0
    char_count = {}
    max_len = 0
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

### Binary Search
**When to use:** Sorted data, search space reduction, optimization problems

```python
# Standard binary search
def binarySearch(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow!
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find first occurrence (lower bound)
def lowerBound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# Find last occurrence (upper bound - 1)
def upperBound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left

# Binary search on answer (optimization)
# Example: Minimum capacity to ship packages in D days
def shipWithinDays(weights, D):
    left, right = max(weights), sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if canShip(weights, D, mid):  # feasibility check
            right = mid
        else:
            left = mid + 1
    return left
```

**Trick:** Use binary search on answer when:
- Answer is in a range [min, max]
- You can check feasibility in O(n) or O(n log n)
- Monotonic property: if x works, all values > x (or < x) also work

### Dynamic Programming
**When to use:** Overlapping subproblems, optimal substructure

**Framework:**
1. Define state: What information do I need?
2. Define recurrence: How do states relate?
3. Define base cases
4. Determine order of computation
5. Optimize space if possible

**Common Patterns:**

```python
# 1D DP - Fibonacci / Climbing Stairs
def climbStairs(n):
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1

# 2D DP - Grid paths
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# Knapsack (0/1)
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]  # don't take
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][W]

# Space-optimized Knapsack
def knapsack_optimized(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):  # reverse!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]

# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Longest Increasing Subsequence - O(n log n)
def lengthOfLIS(nums):
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

# Edit Distance
def editDistance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # delete
                                   dp[i][j-1],      # insert
                                   dp[i-1][j-1])    # replace
    return dp[m][n]
```

**Memoization vs Tabulation:**
- **Memoization (top-down)**: Natural recursion, compute only needed states
- **Tabulation (bottom-up)**: Iterative, better cache performance, easier space optimization

### Backtracking
**When to use:** Generate all solutions, constraints satisfaction, puzzles

```python
# Template
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])  # copy!
        return
    for choice in choices:
        if is_valid(choice):
            path.append(choice)
            backtrack(path, updated_choices)
            path.pop()  # undo choice

# Permutations
def permute(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result

# Combinations
def combine(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, [])
    return result

# Subsets
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# N-Queens
def solveNQueens(n):
    result = []
    def is_valid(board, row, col):
        for i in range(row):
            if board[i] == col or \
               board[i] - i == col - row or \
               board[i] + i == col + row:
                return False
        return True
    
    def backtrack(row, board):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if is_valid(board, row, col):
                board.append(col)
                backtrack(row + 1, board)
                board.pop()
    
    backtrack(0, [])
    return result
```

### Greedy
**When to use:** Local optimal = global optimal, interval scheduling, Huffman coding

**Prove greedy works by:**
1. Greedy choice property: Local optimal leads to global optimal
2. Optimal substructure: Optimal solution contains optimal solutions to subproblems

```python
# Activity Selection / Meeting Rooms
def maxMeetings(intervals):
    intervals.sort(key=lambda x: x[1])  # sort by end time
    count = 0
    end = float('-inf')
    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish
    return count

# Jump Game
def canJump(nums):
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True

# Gas Station
def canCompleteCircuit(gas, cost):
    total_tank = current_tank = 0
    start = 0
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        if current_tank < 0:
            start = i + 1
            current_tank = 0
    return start if total_tank >= 0 else -1
```

### Divide and Conquer
**When to use:** Problem can be split into independent subproblems

```python
# Merge Sort
def mergeSort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergeSort(arr[:mid])
    right = mergeSort(arr[mid:])
    return merge(left, right)

# Quick Select (Kth largest)
def quickSelect(arr, k):
    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    if k <= len(right):
        return quickSelect(right, k)
    elif k <= len(right) + len(mid):
        return pivot
    else:
        return quickSelect(left, k - len(right) - len(mid))
```

---

## 3. Time & Space Complexity Mastery

### Quick Reference

| Complexity | Name | Example |
|------------|------|---------|
| O(1) | Constant | Array access, hash lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Single loop, linear search |
| O(n log n) | Linearithmic | Merge sort, efficient sorts |
| O(n²) | Quadratic | Nested loops, bubble sort |
| O(n³) | Cubic | 3 nested loops, Floyd-Warshall |
| O(2ⁿ) | Exponential | Subsets, recursive Fibonacci |
| O(n!) | Factorial | Permutations |

### Calculation Tricks

**1. Drop Constants and Lower Order Terms**
```
O(2n + 100) → O(n)
O(n² + n) → O(n²)
O(n/2) → O(n)
```

**2. Different Variables = Different Terms**
```
O(a + b) stays O(a + b), NOT O(n)
O(a * b) stays O(a * b)
```

**3. Loops**
```python
# Simple loop: O(n)
for i in range(n):
    # O(1) operation

# Nested loops: O(n * m)
for i in range(n):
    for j in range(m):
        # O(1) operation

# Dependent loops: Calculate exactly
for i in range(n):
    for j in range(i):  # 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 = O(n²)
        # O(1) operation

# Loop with multiplication: O(log n)
i = 1
while i < n:
    i *= 2  # log₂(n) iterations
```

**4. Recursion - Master Theorem**
For T(n) = aT(n/b) + f(n):
- If f(n) = O(n^c) where c < log_b(a): T(n) = O(n^(log_b(a)))
- If f(n) = O(n^c) where c = log_b(a): T(n) = O(n^c * log n)
- If f(n) = O(n^c) where c > log_b(a): T(n) = O(f(n))

**Common Recurrences:**
```
T(n) = T(n/2) + O(1)         → O(log n)      [Binary Search]
T(n) = T(n/2) + O(n)         → O(n)          [Binary Search + linear work]
T(n) = 2T(n/2) + O(1)        → O(n)          
T(n) = 2T(n/2) + O(n)        → O(n log n)    [Merge Sort]
T(n) = 2T(n-1) + O(1)        → O(2ⁿ)         [Recursive Fibonacci]
T(n) = T(n-1) + O(n)         → O(n²)         [Selection Sort]
T(n) = T(n-1) + O(1)         → O(n)          [Linear recursion]
```

**5. Space Complexity**
```python
# O(1) - constant extra space
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total

# O(n) - linear extra space
def copy_array(arr):
    return arr[:]

# O(n) - recursion stack
def recursive_sum(arr, i):
    if i == len(arr):
        return 0
    return arr[i] + recursive_sum(arr, i + 1)

# O(log n) - recursion stack for balanced divide
def binary_search_recursive(arr, target, low, high):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        return binary_search_recursive(arr, target, low, mid - 1)
```

**6. Amortized Analysis**
- Dynamic array append: O(1) amortized (occasional O(n) resize)
- Hash table operations: O(1) amortized

### Common Complexities by Data Structure

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Sorted Array | O(1) | O(log n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1)* | O(1)* |
| Stack/Queue | O(n) | O(n) | O(1) | O(1) |
| Hash Table | N/A | O(1)† | O(1)† | O(1)† |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | O(n) | O(n) | O(log n) | O(log n) |

*If at known position, †Amortized average

### Sorting Algorithm Complexities

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |
| Radix Sort | O(nk) | O(nk) | O(nk) | O(n+k) | Yes |

---

## 4. Red Flags to Avoid

### During Problem Solving

❌ **Jumping to code immediately**
✅ Clarify requirements, discuss approach, then code

❌ **Silence during thinking**
✅ Think aloud, explain your thought process

❌ **Ignoring edge cases**
✅ Ask about: empty input, single element, duplicates, negative numbers, overflow

❌ **Not testing your code**
✅ Trace through with examples after writing

❌ **Arguing with interviewer's hints**
✅ Consider their suggestions - they're trying to help

❌ **Over-engineering the first solution**
✅ Start with brute force, then optimize

❌ **Getting stuck silently**
✅ Explain what you're thinking, ask for hints

### Code Quality Red Flags

❌ **Magic numbers**
```python
# Bad
if len(arr) > 100:

# Good  
MAX_SIZE = 100
if len(arr) > MAX_SIZE:
```

❌ **Poor variable names**
```python
# Bad
def f(a, b):
    c = []
    for i in a:
        if i > b:
            c.append(i)
    return c

# Good
def filter_above_threshold(numbers, threshold):
    result = []
    for num in numbers:
        if num > threshold:
            result.append(num)
    return result
```

❌ **Not handling edge cases in code**
```python
# Bad
def get_first(arr):
    return arr[0]

# Good
def get_first(arr):
    if not arr:
        return None  # or raise exception
    return arr[0]
```

❌ **Integer overflow (especially in Java/C++)**
```java
// Bad - can overflow
int mid = (left + right) / 2;

// Good
int mid = left + (right - left) / 2;
```

❌ **Modifying collection while iterating**
```python
# Bad
for item in items:
    if should_remove(item):
        items.remove(item)  # Bug!

# Good
items = [item for item in items if not should_remove(item)]
# Or
items_to_remove = [item for item in items if should_remove(item)]
for item in items_to_remove:
    items.remove(item)
```

### Behavioral Red Flags

❌ Blaming others or making excuses
❌ Being arrogant or dismissive
❌ Not asking any questions
❌ Giving up too easily
❌ Not admitting when you don't know something
❌ Speaking negatively about previous employers

---

## 5. Interview Tips & Tricks

### Before the Interview

1. **Practice on paper/whiteboard** - You often won't have IDE support
2. **Mock interviews** - Get comfortable explaining while coding
3. **Review your resume** - Be ready to discuss every project in detail
4. **Prepare questions for interviewer** - Shows genuine interest

### During the Interview

**The UMPIRE Method:**
1. **U**nderstand the problem
2. **M**atch to known patterns
3. **P**lan your approach
4. **I**mplement the solution
5. **R**eview/test your code
6. **E**valuate complexity

**Time Management (45 min coding interview):**
- 5 min: Clarify problem, ask questions
- 5 min: Discuss approach, get buy-in
- 25 min: Code the solution
- 5 min: Test and fix bugs
- 5 min: Discuss optimizations

**Communication Phrases:**
```
"Let me make sure I understand the problem..."
"Can I assume the input is always valid?"
"Let me think about edge cases..."
"My initial approach would be... but let me also consider..."
"The time complexity is O(n) because..."
"I see a bug here, let me fix it..."
"One optimization we could make is..."
```

**If You're Stuck:**
1. Re-read the problem
2. Work through a simple example manually
3. Think about related problems you've solved
4. Consider brute force first
5. Ask for a hint (this is okay!)

**Testing Your Code:**
```
1. Normal case
2. Edge cases:
   - Empty input
   - Single element
   - All same elements
   - Already sorted (for sorting problems)
   - Reverse sorted
   - Very large input (think about overflow)
   - Negative numbers
   - Zero
```

### After Coding

**Questions to Ask:**
- "Would you like me to optimize this further?"
- "Should I handle additional edge cases?"
- "Are there any test cases you'd like me to walk through?"

**Discussing Trade-offs:**
```
"This solution is O(n) time but O(n) space. 
We could reduce space to O(1) by... but that would increase time to O(n²)."
```

---

## 6. Problem-Solving Framework

### Step 1: Understand (5 minutes)
- Repeat the problem in your own words
- Ask clarifying questions:
  - Input format and constraints?
  - Can there be duplicates?
  - Is the input sorted?
  - What to return if no solution?
  - Can I modify the input?

### Step 2: Examples (3 minutes)
- Work through 2-3 examples by hand
- Include edge cases
- Verify understanding with interviewer

### Step 3: Approach (5 minutes)
- Start with brute force
- Identify bottlenecks
- Think about data structures that could help
- Consider known algorithms/patterns
- Discuss time/space complexity
- Get interviewer buy-in before coding

### Step 4: Code (20-25 minutes)
- Write clean, readable code
- Use meaningful variable names
- Add brief comments for complex logic
- Handle edge cases
- Think about helper functions

### Step 5: Test (5 minutes)
- Trace through your code with examples
- Focus on edge cases
- Fix any bugs you find
- Don't just say "it works" - actually trace it

### Step 6: Optimize (if time permits)
- Discuss possible improvements
- Consider time/space trade-offs

---

## 7. Language-Specific Tips

### Python
```python
# Built-in functions
sorted(arr)                    # Returns new sorted list
arr.sort()                     # Sorts in-place
sorted(arr, key=lambda x: x[1])  # Custom sort
sorted(arr, reverse=True)      # Descending

# Collections
from collections import defaultdict, Counter, deque
counter = Counter(arr)         # Frequency map
dd = defaultdict(list)         # Auto-initialize
queue = deque()                # Efficient queue

# Heap (min-heap by default)
import heapq
heapq.heappush(heap, item)
heapq.heappop(heap)
heapq.heapify(list)
# For max-heap, negate values

# Bisect (binary search)
from bisect import bisect_left, bisect_right
pos = bisect_left(arr, x)      # First position >= x
pos = bisect_right(arr, x)     # First position > x

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in arr if x % 2 == 0]

# String operations
s.split()                      # Split by whitespace
''.join(chars)                 # Join list to string
s[::-1]                        # Reverse string

# Useful tricks
float('inf'), float('-inf')    # Infinity
sys.maxsize                    # Max int
a, b = b, a                    # Swap
x = x or default               # Default value
```

### Java
```java
// Arrays
Arrays.sort(arr);
Arrays.binarySearch(arr, target);
Arrays.fill(arr, value);
Arrays.copyOf(arr, newLength);

// Collections
List<Integer> list = new ArrayList<>();
Set<Integer> set = new HashSet<>();
Map<String, Integer> map = new HashMap<>();
Queue<Integer> queue = new LinkedList<>();
Deque<Integer> deque = new ArrayDeque<>();
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

// Sorting with comparator
Collections.sort(list, (a, b) -> a - b);  // ascending
Arrays.sort(arr, (a, b) -> a[0] - b[0]);  // by first element

// String operations
String[] parts = s.split(" ");
String joined = String.join(" ", parts);
StringBuilder sb = new StringBuilder();
sb.append("text");
String result = sb.toString();

// Useful methods
Math.max(a, b);
Math.min(a, b);
Math.abs(x);
Integer.MAX_VALUE;
Integer.MIN_VALUE;
```

### C++
```cpp
// Vectors
vector<int> v;
v.push_back(x);
v.pop_back();
sort(v.begin(), v.end());
sort(v.begin(), v.end(), greater<int>());  // descending

// Maps and Sets
unordered_map<string, int> map;
unordered_set<int> set;
map<int, int> orderedMap;  // sorted by key

// Priority Queue
priority_queue<int> maxHeap;
priority_queue<int, vector<int>, greater<int>> minHeap;

// Algorithms
lower_bound(v.begin(), v.end(), x);  // iterator to first >= x
upper_bound(v.begin(), v.end(), x);  // iterator to first > x
binary_search(v.begin(), v.end(), x);  // bool

// Strings
string s;
s.substr(start, length);
s.find("pattern");  // returns position or string::npos

// Useful
INT_MAX, INT_MIN
LLONG_MAX, LLONG_MIN
```

---

## 8. Resources & Practice Plan

### Recommended Practice Platforms

| Platform | Best For |
|----------|----------|
| LeetCode | Most comprehensive, company-tagged problems |
| HackerRank | Structured learning tracks |
| Codeforces | Competitive programming |
| InterviewBit | Structured interview prep |
| AlgoExpert | Video explanations |

### Must-Do LeetCode Problems (by category)

**Arrays & Hashing:**
- Two Sum (#1)
- Contains Duplicate (#217)
- Product of Array Except Self (#238)
- Top K Frequent Elements (#347)
- Group Anagrams (#49)

**Two Pointers:**
- Valid Palindrome (#125)
- Container With Most Water (#11)
- 3Sum (#15)
- Trapping Rain Water (#42)

**Sliding Window:**
- Best Time to Buy and Sell Stock (#121)
- Longest Substring Without Repeating Characters (#3)
- Minimum Window Substring (#76)

**Binary Search:**
- Binary Search (#704)
- Search in Rotated Sorted Array (#33)
- Find Minimum in Rotated Sorted Array (#153)
- Median of Two Sorted Arrays (#4)

**Linked Lists:**
- Reverse Linked List (#206)
- Merge Two Sorted Lists (#21)
- Linked List Cycle (#141)
- LRU Cache (#146)

**Trees:**
- Maximum Depth of Binary Tree (#104)
- Validate Binary Search Tree (#98)
- Lowest Common Ancestor (#236)
- Binary Tree Level Order Traversal (#102)
- Serialize and Deserialize Binary Tree (#297)

**Graphs:**
- Number of Islands (#200)
- Clone Graph (#133)
- Course Schedule (#207)
- Word Ladder (#127)

**Dynamic Programming:**
- Climbing Stairs (#70)
- Coin Change (#322)
- Longest Increasing Subsequence (#300)
- Longest Common Subsequence (#1143)
- Word Break (#139)
- Edit Distance (#72)

**Backtracking:**
- Subsets (#78)
- Permutations (#46)
- Combination Sum (#39)
- N-Queens (#51)

### Weekly Study Plan (12 weeks)

| Week | Focus | Daily Problems |
|------|-------|----------------|
| 1-2 | Arrays, Strings, Hashing | 3 easy |
| 3-4 | Two Pointers, Sliding Window | 2 easy, 1 medium |
| 5-6 | Trees, Binary Search | 2 medium |
| 7-8 | Graphs (BFS/DFS) | 2 medium |
| 9-10 | Dynamic Programming | 2 medium |
| 11-12 | Mixed Review, Hard problems | 2 medium, 1 hard |

### Books
1. "Cracking the Coding Interview" by Gayle Laakmann McDowell
2. "Elements of Programming Interviews" (EPI)
3. "Introduction to Algorithms" (CLRS) - for deep understanding
4. "Designing Data-Intensive Applications" - for system design

### YouTube Channels
- NeetCode (LeetCode explanations)
- Back To Back SWE
- Abdul Bari (algorithms)
- Tushar Roy (DP)
- Gaurav Sen (system design)

---

## Quick Reference Card

### Pattern Recognition
| If you see... | Think... |
|--------------|----------|
| "Top/Least K" | Heap |
| "Sorted array" | Binary Search |
| "All permutations/subsets" | Backtracking |
| "Tree" | BFS/DFS, recursion |
| "Graph" | BFS/DFS, Union-Find |
| "Linked List" | Two pointers |
| "Recursion banned" | Stack |
| "Maximum/Minimum subarray" | DP, Kadane's |
| "Common strings" | Map, Trie |
| "String transformation" | BFS |
| "Most/Least recently used" | LinkedHashMap |
| "Intervals" | Sort + Greedy |
| "Optimization" | DP or Binary Search on Answer |
| "Shortest path" | BFS (unweighted), Dijkstra (weighted) |

### Complexity Cheat Sheet
```
n ≤ 10        → O(n!) or O(n^6)      Brute force
n ≤ 20        → O(2^n)               Backtracking
n ≤ 500       → O(n³)                Triple loop, DP
n ≤ 10,000    → O(n²)                Nested loops
n ≤ 1,000,000 → O(n log n)           Sorting
n ≤ 100,000,000 → O(n)               Linear scan
n > 100,000,000 → O(log n) or O(1)   Math, Binary search
```

---

**Good luck with your interviews! Remember: Practice consistently, think out loud, and never give up.**
