# LeetCode 6 大模板骨架代码

> **核心思想**：先掌握骨架，再套题目。不要题海战术，要模板思维。

---

## 📋 模板清单

1. **Hash / Counting**（频次、去重、映射）：49、1、242
2. **Two Pointers / Sliding Window**（双指针/滑动窗口）：3、76、209
3. **DFS/BFS 图遍历**（visited）：200、133、695
4. **Binary Search / Lower Bound**（二分查找）：704、33、34
5. **DP 一维滚动**（动态规划）：70、198、322
6. **Trie / DFS**（前缀树/深度搜索）：208、211

---

## 1️⃣ Hash / Counting（频次、去重、映射）

### 模板 1.1：频次统计

```python
def template_frequency_count(nums):
    """频次统计模板"""
    # 1. 创建哈希表
    count = {}  # 或者 defaultdict(int)
    
    # 2. 统计频次
    for num in nums:
        count[num] = count.get(num, 0) + 1
        # 或者：count[num] += 1 (如果使用 defaultdict)
    
    # 3. 使用频次
    for num, freq in count.items():
        # 处理逻辑
        pass
    
    return result
```

**经典题目**：
- **49. Group Anagrams**：按频次分组
- **242. Valid Anagram**：比较两个字符串的字符频次
- **1. Two Sum**：用哈希表查找补数

### 模板 1.2：去重/查找

```python
def template_hash_lookup(nums):
    """哈希查找模板（Two Sum 类型）"""
    # 1. 创建哈希表存储已遍历的元素
    seen = {}  # value -> index
    
    # 2. 遍历数组
    for i, num in enumerate(nums):
        target = ...  # 目标值（可能是补数、差值等）
        
        # 3. 查找目标值
        if target in seen:
            return [seen[target], i]  # 找到结果
        
        # 4. 将当前元素加入哈希表
        seen[num] = i
    
    return []  # 未找到
```

**经典题目**：
- **1. Two Sum**：`target = complement = target - num`
- **219. Contains Duplicate II**：`target = num`（查找相同值）

### 模板 1.3：分组/映射

```python
def template_grouping(items):
    """分组映射模板"""
    # 1. 创建分组字典
    groups = {}  # 或者 defaultdict(list)
    
    # 2. 按 key 分组
    for item in items:
        key = ...  # 分组的 key（可能是排序后的字符串、特征值等）
        
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
        # 或者：groups[key].append(item) (如果使用 defaultdict)
    
    # 3. 返回分组结果
    return list(groups.values())
```

**经典题目**：
- **49. Group Anagrams**：`key = ''.join(sorted(s))`
- **249. Group Shifted Strings**：`key = 偏移特征`

---

## 2️⃣ Two Pointers / Sliding Window（双指针/滑动窗口）

### 模板 2.1：双指针（左右指针）

```python
def template_two_pointers(nums):
    """双指针模板（左右指针）"""
    left, right = 0, len(nums) - 1
    
    while left < right:
        # 1. 计算当前状态
        current = nums[left] + nums[right]  # 或其他计算
        
        # 2. 根据条件移动指针
        if current == target:
            return [left, right]  # 找到结果
        elif current < target:
            left += 1  # 增大值
        else:
            right -= 1  # 减小值
    
    return []  # 未找到
```

**经典题目**：
- **1. Two Sum**（排序后）：`current = nums[left] + nums[right]`
- **15. 3Sum**：固定一个数，双指针找另外两个
- **11. Container With Most Water**：`current = min(height[left], height[right]) * (right - left)`

### 模板 2.2：滑动窗口（固定窗口）

```python
def template_sliding_window_fixed(s, k):
    """滑动窗口模板（固定窗口大小）"""
    n = len(s)
    if n < k:
        return 0
    
    # 1. 初始化窗口
    window = s[:k]
    # 计算初始窗口的状态
    current_sum = sum(window)  # 或其他计算
    
    max_sum = current_sum
    
    # 2. 滑动窗口
    for i in range(k, n):
        # 移除左边界
        left = s[i - k]
        current_sum -= left  # 或其他操作
        
        # 添加右边界
        right = s[i]
        current_sum += right  # 或其他操作
        
        # 更新结果
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**经典题目**：
- **209. Minimum Size Subarray Sum**：找最短窗口
- **643. Maximum Average Subarray I**：固定窗口求最大值

### 模板 2.3：滑动窗口（可变窗口）

```python
def template_sliding_window_variable(s):
    """滑动窗口模板（可变窗口大小）"""
    left = 0
    window = {}  # 窗口内的元素计数
    max_len = 0
    
    # 1. 扩展右边界
    for right in range(len(s)):
        char = s[right]
        
        # 添加元素到窗口
        window[char] = window.get(char, 0) + 1
        
        # 2. 收缩左边界（当窗口不满足条件时）
        while not is_valid(window):  # 自定义条件
            left_char = s[left]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
            left += 1
        
        # 3. 更新结果
        max_len = max(max_len, right - left + 1)
    
    return max_len

def is_valid(window):
    """判断窗口是否满足条件"""
    # 例如：不同字符数 <= k
    return len(window) <= k
```

**经典题目**：
- **3. Longest Substring Without Repeating Characters**：`is_valid = len(window) == right - left + 1`
- **76. Minimum Window Substring**：`is_valid = 所有目标字符都在窗口中`
- **424. Longest Repeating Character Replacement**：`is_valid = (窗口大小 - 最多字符数) <= k`

---

## 3️⃣ DFS/BFS 图遍历（visited）

### 模板 3.1：DFS 递归（图的连通分量）

```python
def template_dfs_graph(graph):
    """DFS 递归模板（图的连通分量）"""
    visited = set()
    result = 0
    
    def dfs(node):
        # 1. 标记已访问
        visited.add(node)
        
        # 2. 处理当前节点
        # process(node)
        
        # 3. 递归访问邻居
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    # 4. 遍历所有节点
    for node in graph:
        if node not in visited:
            dfs(node)
            result += 1  # 连通分量计数
    
    return result
```

**经典题目**：
- **200. Number of Islands**：二维网格的连通分量
- **695. Max Area of Island**：连通分量的大小
- **133. Clone Graph**：图的深拷贝

### 模板 3.2：DFS 递归（二维网格）

```python
def template_dfs_grid(grid):
    """DFS 递归模板（二维网格）"""
    m, n = len(grid), len(grid[0])
    visited = set()
    
    def dfs(i, j):
        # 1. 边界检查
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        
        # 2. 访问检查
        if (i, j) in visited or grid[i][j] == 0:  # 或其他条件
            return
        
        # 3. 标记已访问
        visited.add((i, j))
        
        # 4. 处理当前格子
        # process(grid[i][j])
        
        # 5. 递归访问四个方向
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    # 6. 遍历所有格子
    for i in range(m):
        for j in range(n):
            if (i, j) not in visited and grid[i][j] == 1:  # 或其他条件
                dfs(i, j)
                # 处理连通分量
```

**经典题目**：
- **200. Number of Islands**：`grid[i][j] == '1'`
- **695. Max Area of Island**：`grid[i][j] == 1`，返回面积

### 模板 3.3：BFS 队列（层次遍历）

```python
from collections import deque

def template_bfs_graph(graph, start):
    """BFS 队列模板"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        # 1. 取出队列头部
        node = queue.popleft()
        
        # 2. 处理当前节点
        result.append(node)
        
        # 3. 将邻居加入队列
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

**经典题目**：
- **133. Clone Graph**：图的克隆
- **200. Number of Islands**：也可以用 BFS
- **127. Word Ladder**：最短路径

---

## 4️⃣ Binary Search / Lower Bound（二分查找）

### 模板 4.1：标准二分查找

```python
def template_binary_search(nums, target):
    """标准二分查找模板"""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        # 1. 计算中点（防止溢出）
        mid = left + (right - left) // 2
        
        # 2. 比较中点值
        if nums[mid] == target:
            return mid  # 找到目标
        elif nums[mid] < target:
            left = mid + 1  # 目标在右半部分
        else:
            right = mid - 1  # 目标在左半部分
    
    return -1  # 未找到
```

**经典题目**：
- **704. Binary Search**：标准二分查找
- **35. Search Insert Position**：查找插入位置

### 模板 4.2：查找左边界（Lower Bound）

```python
def template_binary_search_left_bound(nums, target):
    """二分查找左边界模板（Lower Bound）"""
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid  # 关键：不 mid - 1
    
    # left 是第一个 >= target 的位置
    return left if left < len(nums) and nums[left] == target else -1
```

**经典题目**：
- **34. Find First and Last Position of Element in Sorted Array**：找第一个位置
- **35. Search Insert Position**：查找插入位置

### 模板 4.3：查找右边界（Upper Bound）

```python
def template_binary_search_right_bound(nums, target):
    """二分查找右边界模板（Upper Bound）"""
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] <= target:
            left = mid + 1  # 关键：等于时也向右
        else:
            right = mid
    
    # left - 1 是最后一个 <= target 的位置
    return left - 1 if left > 0 and nums[left - 1] == target else -1
```

**经典题目**：
- **34. Find First and Last Position of Element in Sorted Array**：找最后一个位置

### 模板 4.4：旋转数组二分查找

```python
def template_binary_search_rotated(nums, target):
    """旋转数组二分查找模板"""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # 判断左半部分是否有序
        if nums[left] <= nums[mid]:
            # 左半部分有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # 右半部分有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

**经典题目**：
- **33. Search in Rotated Sorted Array**：旋转数组查找
- **81. Search in Rotated Sorted Array II**：有重复元素

---

## 5️⃣ DP 一维滚动（动态规划）

### 模板 5.1：一维 DP（线性 DP）

```python
def template_dp_1d(n):
    """一维 DP 模板"""
    # 1. 定义 DP 数组
    dp = [0] * (n + 1)
    
    # 2. 初始化
    dp[0] = ...  # 初始值
    dp[1] = ...  # 初始值
    
    # 3. 状态转移
    for i in range(2, n + 1):
        dp[i] = ...  # 状态转移方程
        # 例如：dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

**经典题目**：
- **70. Climbing Stairs**：`dp[i] = dp[i-1] + dp[i-2]`
- **198. House Robber**：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
- **322. Coin Change**：`dp[i] = min(dp[i], dp[i-coin] + 1)`

### 模板 5.2：一维 DP（滚动数组优化）

```python
def template_dp_1d_optimized(n):
    """一维 DP 模板（滚动数组优化空间）"""
    # 1. 只需要两个变量（或少量变量）
    prev2 = ...  # dp[i-2]
    prev1 = ...  # dp[i-1]
    
    # 2. 状态转移
    for i in range(2, n + 1):
        current = ...  # 状态转移方程
        # 例如：current = prev1 + prev2
        
        # 3. 滚动更新
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**经典题目**：
- **70. Climbing Stairs**：`current = prev1 + prev2`
- **198. House Robber**：`current = max(prev1, prev2 + nums[i])`

### 模板 5.3：背包 DP（一维）

```python
def template_dp_knapsack(weights, values, capacity):
    """背包 DP 模板（一维）"""
    # 1. 定义 DP 数组：dp[i] 表示容量为 i 的最大价值
    dp = [0] * (capacity + 1)
    
    # 2. 遍历物品
    for i in range(len(weights)):
        weight = weights[i]
        value = values[i]
        
        # 3. 倒序遍历容量（避免重复使用）
        for j in range(capacity, weight - 1, -1):
            dp[j] = max(dp[j], dp[j - weight] + value)
    
    return dp[capacity]
```

**经典题目**：
- **322. Coin Change**：完全背包，`dp[j] = min(dp[j], dp[j-coin] + 1)`
- **416. Partition Equal Subset Sum**：0-1 背包

---

## 6️⃣ Trie / DFS（前缀树/深度搜索）

### 模板 6.1：Trie 基础结构

```python
class TrieNode:
    """Trie 节点"""
    def __init__(self):
        self.children = {}  # 字符 -> TrieNode
        self.is_end = False  # 是否是单词结尾

class Trie:
    """Trie 树模板"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        """搜索单词（完全匹配）"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        """搜索前缀"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**经典题目**：
- **208. Implement Trie (Prefix Tree)**：标准 Trie 实现
- **211. Design Add and Search Words Data Structure**：支持通配符的 Trie

### 模板 6.2：Trie + DFS（通配符搜索）

```python
class WordDictionary:
    """Trie + DFS 模板（支持通配符）"""
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word):
        """添加单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        """搜索单词（支持 '.' 通配符）"""
        def dfs(node, index):
            # 1. 到达单词结尾
            if index == len(word):
                return node.is_end
            
            char = word[index]
            
            # 2. 处理通配符
            if char == '.':
                # 尝试所有可能的字符
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # 3. 精确匹配
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        
        return dfs(self.root, 0)
```

**经典题目**：
- **211. Design Add and Search Words Data Structure**：支持 '.' 通配符
- **212. Word Search II**：Trie + DFS 回溯

### 模板 6.3：Trie + DFS（多模式匹配）

```python
def template_trie_dfs_multiple(words, board):
    """Trie + DFS 多模式匹配模板"""
    # 1. 构建 Trie
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    result = []
    m, n = len(board), len(board[0])
    
    def dfs(i, j, node, path):
        # 1. 边界检查
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        
        char = board[i][j]
        
        # 2. 检查字符是否在 Trie 中
        if char not in node.children:
            return
        
        # 3. 移动到下一个节点
        next_node = node.children[char]
        path += char
        
        # 4. 检查是否找到单词
        if next_node.is_end:
            result.append(path)
            next_node.is_end = False  # 避免重复
        
        # 5. 标记已访问（回溯）
        board[i][j] = '#'  # 临时标记
        dfs(i + 1, j, next_node, path)
        dfs(i - 1, j, next_node, path)
        dfs(i, j + 1, next_node, path)
        dfs(i, j - 1, next_node, path)
        board[i][j] = char  # 恢复
        
        # 6. 优化：删除已匹配的节点（可选）
    
    # 7. 遍历所有起点
    for i in range(m):
        for j in range(n):
            dfs(i, j, trie.root, "")
    
    return result
```

**经典题目**：
- **212. Word Search II**：在二维网格中搜索多个单词

---

## 🎯 使用建议

### 1. 背诵骨架
- 每天复习 1-2 个模板
- 能够闭眼写出骨架代码
- 理解每个步骤的作用

### 2. 套用题目
- 看到题目先识别模板类型
- 将题目转换为模板结构
- 填充具体逻辑

### 3. 练习顺序
1. **Hash/Counting**：最容易，先掌握
2. **Two Pointers**：理解双指针思想
3. **DFS/BFS**：图遍历基础
4. **Binary Search**：边界处理是难点
5. **DP**：状态转移是关键
6. **Trie**：相对复杂，最后掌握

---

## 📚 相关题目快速索引

| 模板 | 题目编号 | 难度 | 关键点 |
|------|---------|------|--------|
| Hash | 1, 49, 242 | Easy-Medium | 频次统计、查找 |
| Two Pointers | 3, 76, 209 | Medium-Hard | 窗口维护 |
| DFS/BFS | 200, 133, 695 | Medium | 访问标记 |
| Binary Search | 704, 33, 34 | Easy-Medium | 边界处理 |
| DP | 70, 198, 322 | Easy-Medium | 状态转移 |
| Trie | 208, 211, 212 | Medium-Hard | 树结构+DFS |

---

**记住**：先掌握骨架，再套题目。不要题海战术，要模板思维！
