# Dynamic Programming: Tabulation (Bottom-Up)

**Dynamic Programming (DP)** is just: *"Don't solve the same problem twice — save the answer and reuse it."*

---

## Tabulation in 4 Steps

Think of it like filling out a spreadsheet, cell by cell, until you reach the answer.

1. **Define what each cell means** — Create an array where `dp[i]` = *"the answer to the problem of size `i`"*
2. **Fill in the base cases** — These are the trivially small problems you already know the answer to.
3. **Write the formula** — Each cell is computed from cells you've *already filled in*.
4. **Loop from small → big**, filling the table.

---

## Full Example: Climbing Stairs

> *You can climb 1 or 2 steps at a time. How many ways to reach step `n`?*

```python
def climb(n):
    # Step 1: Create the table
    dp = [0] * (n + 1)

    # Step 2: Base cases (things we know for sure)
    dp[0] = 1  # 1 way to stay on the ground
    dp[1] = 1  # 1 way to reach step 1

    # Step 3 & 4: Fill from small to big using the formula
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        #        ↑ came from 1 below  ↑ came from 2 below

    return dp[n]

print(climb(5))  # → 8
```

**What's happening visually:**

```
Step:  0   1   2   3   4   5
dp:  [ 1 , 1 , 2 , 3 , 5 , 8 ]
              ↑               ↑
          1+1=2          5+3=8 ← your answer!
```

---

## The Mental Model

Think of it like dominoes:

1. **Set up the first few dominoes** (base cases)
2. **Each domino knocks over the next** (the formula)
3. **The last domino to fall is your answer**

You never go backward, you never redo work — you just march forward, one cell at a time.

---

## Why Tabulation Over Memoization?

| | Tabulation (Bottom-Up) | Memoization (Top-Down) |
|---|---|---|
| **Stack overflow** | ✅ No recursion, no risk | ❌ Deep recursion can crash |
| **Overhead** | ✅ Simple loop, no call overhead | ❌ Recursive call + cache lookup overhead |
| **Space optimization** | ✅ Can often reduce to O(1) space | ❌ Hard to optimize space |
| **Predictability** | ✅ Solves all subproblems in order | ❌ Solves only needed subproblems |

### Space Optimization Example

Tabulation lets you keep only what you need:

```python
# O(1) space Fibonacci
def fib(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

> **When memoization is still better:** when only a sparse subset of subproblems is actually needed (e.g., some grid/graph problems), memoization avoids computing unnecessary states.
