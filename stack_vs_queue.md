# Stack vs Queue: Differences, Use Cases, and Why

## Overview

**Stack** and **Queue** are fundamental data structures that differ in how elements are added and removed. Both are linear data structures but with opposite ordering principles.

---

## Core Differences

### Stack (LIFO - Last In, First Out)

- **Principle**: Last element added is the first element removed
- **Operations**:
  - `push(x)`: Add element to the top
  - `pop()`: Remove element from the top
  - `peek()`: View top element without removing it
- **Time Complexity**: O(1) for push, pop, and peek

**Visual Example**:
```
Push 1, 2, 3:
  [3] ← top
  [2]
  [1]

Pop removes 3, then 2, then 1
```

### Queue (FIFO - First In, First Out)

- **Principle**: First element added is the first element removed
- **Operations**:
  - `enqueue(x)`: Add element to the rear
  - `dequeue()`: Remove element from the front
  - `peek()`: View front element without removing it
- **Time Complexity**: O(1) for enqueue, dequeue, and peek

**Visual Example**:
```
Enqueue 1, 2, 3:
[1] [2] [3]
↑front    ↑rear

Dequeue removes 1, then 2, then 3
```

---

## Side-by-Side Comparison

| Aspect | Stack | Queue |
|--------|-------|-------|
| **Order** | LIFO (Last In, First Out) | FIFO (First In, First Out) |
| **Insert** | push() - adds to top | enqueue() - adds to rear |
| **Remove** | pop() - removes from top | dequeue() - removes from front |
| **Access** | Only top element | Only front element |
| **Real-world analogy** | Plate stack, Undo button | Bank queue, Task scheduler |

---

## When to Use Stack

### Use Cases:

1. **Function Call Stack**
   - Tracking function calls and returns in programs
   - Managing recursion and call sequences

2. **Undo/Redo Functionality**
   - Text editors, drawing apps, code editors
   - Each action is pushed; undo pops the stack

3. **Expression Evaluation**
   - Converting infix to postfix notation
   - Evaluating mathematical expressions

4. **Parenthesis Matching**
   - Validating balanced parentheses, brackets, braces
   - Compiler syntax checking

5. **Depth-First Search (DFS)**
   - Graph and tree traversal
   - Backtracking algorithms

6. **Syntax Parsing**
   - HTML/XML tag matching
   - Parsing compiler statements

### Why Stack?
- **Natural fit** for problems where you need the most recent item
- **Efficient** when order of reversal is needed
- **Memory efficient** - simple array or linked list implementation

---

## When to Use Queue

### Use Cases:

1. **Task Scheduling**
   - Job queues in background processors
   - Print queue management
   - Task execution in order of arrival

2. **Breadth-First Search (BFS)**
   - Level-order tree/graph traversal
   - Finding shortest path in unweighted graphs

3. **Message Processing**
   - Message brokers and event systems
   - Asynchronous task processing
   - Real-time systems

4. **Cache Management**
   - LRU (Least Recently Used) caches
   - Page replacement in operating systems

5. **Ticket/Service Counters**
   - Customer service systems
   - Resource allocation fairness
   - Ensures FIFO ordering

6. **Network Packet Routing**
   - Data transmission buffers
   - I/O request scheduling

### Why Queue?
- **Fairness** - ensures items are processed in arrival order
- **Natural** for sequential processing and task management
- **Load balancing** - distributes work evenly

---

## Practical Decision Guide

### Choose Stack if:
- You need **reverse order** processing
- Working with **recursive problems** or **backtracking**
- Building **undo/redo** systems
- **Most recent item** should be processed first
- You need **DFS** traversal

### Choose Queue if:
- You need **FIFO ordering** (fairness principle)
- Building **task schedulers** or **job processors**
- Implementing **BFS** or level-order traversal
- Working with **message/event systems**
- **First arrival** should be processed first

---

## Implementation Example (Python)

### Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[-1]

# Example: Undo functionality
stack = Stack()
stack.push("action1")
stack.push("action2")
stack.push("action3")
print(stack.pop())  # action3 (undo)
```

### Queue Implementation
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft()
    
    def peek(self):
        return self.items[0]

# Example: Task processing
queue = Queue()
queue.enqueue("task1")
queue.enqueue("task2")
queue.enqueue("task3")
print(queue.dequeue())  # task1 (process in order)
```

---

## Key Takeaways

| Factor | Stack | Queue |
|--------|-------|-------|
| **Best for reversal** | ✓ | ✗ |
| **Best for fairness** | ✗ | ✓ |
| **Recursion/DFS** | ✓ | ✗ |
| **BFS/Level order** | ✗ | ✓ |
| **Undo/Redo** | ✓ | ✗ |
| **Task scheduling** | ✗ | ✓ |

Both are **O(1)** for basic operations and are essential building blocks for more complex data structures and algorithms. Choose based on the **ordering requirement** of your problem.

---

## Additional Resources

- **Stack**: Used in compilers, operating systems, web browsers (back button)
- **Queue**: Used in operating systems (process scheduling), networks (packet queues), databases (query queues)
- Both are foundational for understanding **Advanced Data Structures** like priority queues, deques, and circular queues
