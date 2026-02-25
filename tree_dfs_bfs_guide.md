# Tree Data Structures: DFS vs BFS Traversals

## Overview
Trees are hierarchical data structures with a root node and child nodes connected by edges. Each node has at most one parent (except root). Understanding **DFS (Depth-First Search)** and **BFS (Breadth-First Search)** is critical for tree/graph problems in coding interviews.

---

## Trees Fundamentals

### Types of Trees
- **Binary Tree**: Each node has at most 2 children
- **Binary Search Tree (BST)**: Left child < Parent < Right child
- **N-ary Tree**: Each node can have multiple children
- **Balanced Trees**: AVL, Red-Black (maintain height balance)

### Key Properties
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to a node
- **Leaf Node**: Node with no children

---

## DFS vs BFS: Complete Comparison

| Aspect | DFS | BFS |
|--------|-----|-----|
| **Order** | Goes deep first, explores one path fully before backtracking | Goes level by level, explores all neighbors at current depth |
| **Data Structure** | Stack (recursion or explicit) | Queue |
| **Space Complexity** | O(h) where h = height of tree | O(w) where w = max width of tree |
| **Time Complexity** | O(n) | O(n) |
| **Use Case** | Deep exploration, backtracking, path finding | Shortest path, level-order, closest node |

---

## When to Use DFS (Interview Perspective)

### ✅ Choose DFS When:

1. **Finding any path/solution in tree** - checking if target exists
   - Example: "Does value X exist in tree?"
   
2. **Topological Sorting** - dependencies, task scheduling
   - Example: Course prerequisites, build systems
   
3. **Cycle Detection** - especially in directed graphs
   - Example: Detect circular dependencies
   
4. **Backtracking Problems** - exploring all possibilities
   - Examples: Permutations, N-Queens, Sudoku, word search
   
5. **Inorder/Preorder/Postorder Traversals** - visiting nodes in specific order
   - Inorder on BST gives sorted order
   
6. **Space Constrained Environments** - O(h) vs O(w) can be huge difference
   - In wide trees, BFS uses exponentially more space
   
7. **Tree Properties** - validate BST, check ancestors, etc.

### Interview Answer Template:
> "I'd use DFS because we need to explore the entire tree/path and space efficiency matters. DFS recursively goes deep, which is elegant for tree structures and uses O(h) space instead of O(w)."

---

## When to Use BFS (Interview Perspective)

### ✅ Choose BFS When:

1. **Shortest Path in Unweighted Graph** - guaranteed shortest
   - Example: Number of steps to reach target
   
2. **Level-Order Traversal** - process nodes by levels
   - Examples: Binary tree level order, connect nodes at same level
   
3. **Word Ladder / Minimum Moves** - find minimum steps to solution
   - Example: "Ladder" problems, matrix distance problems
   
4. **Finding Closest Node** - proximity-based search
   - Examples: Social networks (degrees of separation), nearest facility
   
5. **Checking Tree Balance** - easier level-by-level
   - Example: Is tree balanced/symmetric
   
6. **Bipartite Graph Checking** - easier to color level by level

### Interview Answer Template:
> "BFS guarantees finding the shortest path in unweighted graphs because it explores level by level. For problems needing the closest/minimum result, BFS is optimal."

---

## Code Examples

### DFS - Recursive (Preorder)
```python
def dfs_recursive(node):
    if not node:
        return
    
    print(node.val)              # Preorder: visit before children
    dfs_recursive(node.left)
    dfs_recursive(node.right)
```

### DFS - Recursive (Inorder - for BST)
```python
def dfs_inorder(node):
    if not node:
        return
    
    dfs_inorder(node.left)
    print(node.val)              # Inorder: visit between children
    dfs_inorder(node.right)
```

### DFS - Iterative with Stack
```python
def dfs_iterative(root):
    if not root:
        return
    
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
        
        # Push right first so left is processed first (LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

### BFS - Queue (Level Order)
```python
from collections import deque

def bfs(root):
    if not root:
        return
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        print(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

### BFS - Level Order with Levels
```python
def level_order_with_levels(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):  # Process all nodes at current level
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

---

## Classic Interview Problems & Approach

| Problem | Best Approach | Why | Example LeetCode |
|---------|---------------|-----|------------------|
| Binary tree level-order | BFS | Process per level, clear structure | 102, 107 |
| Validate BST | DFS (Inorder) | Inorder traversal gives sorted sequence | 98 |
| Word ladder | BFS | Shortest path in unweighted graph | 127 |
| Permutations/Combinations | DFS/Backtracking | Explore all branches exhaustively | 46, 77 |
| Detect cycle in graph | DFS | Mark visited, find back edge | 207 |
| Closest value in BST | BFS | Level-by-level finds closest first | 270 |
| Path sum / paths | DFS | Backtrack and accumulate | 112, 113 |
| Symmetric tree | BFS or DFS | Compare mirror nodes | 101 |
| Lowest common ancestor | DFS | Recursive structure matches problem | 236 |
| Number of islands | DFS | Explore connected components | 200 |

---

## Interview Tips

### ✅ Best Practices
- **Ask clarifying questions**: "Do I need the optimal/shortest path?" → hints BFS
- **Mention space complexity**: Shows you think beyond just time complexity
- **Code iteratively first**: Recursive DFS is elegant but iterative shows depth understanding
- **Discuss trade-offs**: "DFS is space-efficient but BFS guarantees shortest path"
- **Draw the tree**: Visualizing helps explain your approach

### ❌ Common Mistakes
- Using BFS for "any solution" problems (wastes space)
- Using DFS for shortest path without mentioning the gap
- Forgetting to mark visited nodes (causes infinite loops)
- Not handling null nodes properly
- Confusing tree traversals with graph traversals

---

## Space Complexity Deep Dive

### Why Space Matters

**Scenario: Wide vs Deep Tree**

```
Wide Tree (BFS nightmare):
              1
    /    /    |    \    \
   2    3     4     5     6
  /|\  /|\   /|\   /|\   /|\
 ...  ...   ...   ...   ...
 
Height = 3, but Width at level 3 = 243 nodes
BFS Queue size: O(243) 
DFS Stack size: O(3)
```

```
Deep Tree (BFS advantage):
         1
         |
         2
         |
         3
        /|\
       ...
       
Height = 1000, Width = 3
BFS Queue size: O(3)
DFS Stack size: O(1000)
```

---

## Key Takeaways for Interviews

1. **DFS = Deep Exploration**: Use for finding any solution, backtracking, cycle detection
2. **BFS = Shortest Path**: Use for minimum steps, level-order, proximity
3. **Space Matters**: DFS is O(height), BFS is O(width) - can differ exponentially
4. **Recursion vs Iteration**: Both valid DFS approaches, show you understand the stack
5. **Problem Signals**:
   - "shortest path" / "minimum" → BFS
   - "all solutions" / "backtrack" → DFS
   - "level order" / "by levels" → BFS
   - "any path" / "check existence" → DFS

