# AI 推理系统面试 LeetCode 刷题指南

## 🎯 基于今天学习内容的刷题推荐

根据你今天学的：**KV Cache、Batching、调度器、优先级队列、贪心算法**，以下是针对性的 LeetCode 题目。

---

## 📚 必刷题目（5-8 题）

### 1. **LeetCode 621: Task Scheduler（任务调度器）** ⭐⭐⭐

**为什么重要**：
- 直接对应 **batching 和 TFFT 的权衡问题**
- 需要在执行时间和吞吐量之间平衡
- 使用优先级队列管理任务

**核心考点**：
- 贪心算法
- 优先级队列（堆）
- 等待时间 vs 吞吐量的权衡

**题目链接**：https://leetcode.com/problems/task-scheduler/

**关键思路**：
```python
# 伪代码（对应 batching 问题）
def leastInterval(tasks, n):
    # 统计任务频率
    freq = Counter(tasks)
    
    # 使用优先级队列（最大堆）
    heap = [-count for count in freq.values()]
    heapq.heapify(heap)
    
    # 模拟调度过程
    time = 0
    while heap:
        # 贪心：每次选择频率最高的任务
        # 对应：每次选择优先级最高的请求加入 batch
        ...
```

**面试回答**：
> "这题和 batching 调度很像：我们需要在等待时间（冷却时间）和吞吐量（完成任务数）之间平衡。就像 SGLang 的调度器需要在 TFFT（延迟）和吞吐量之间权衡一样。"

---

### 2. **LeetCode 253: Meeting Rooms II（会议室问题 II）** ⭐⭐⭐

**为什么重要**：
- 对应 **资源分配问题**（GPU 内存、KV Cache）
- 需要在有限资源下分配多个请求
- 使用贪心算法 + 优先级队列

**核心考点**：
- 区间调度
- 资源分配
- 最小堆的应用

**题目链接**：https://leetcode.com/problems/meeting-rooms-ii/

**关键思路**：
```python
# 伪代码（对应 GPU 资源分配）
def minMeetingRooms(intervals):
    # 按开始时间排序（对应请求到达时间）
    intervals.sort(key=lambda x: x[0])
    
    # 使用最小堆存储结束时间（对应资源释放时间）
    heap = []
    
    for start, end in intervals:
        # 如果最早结束的会议已经结束，复用房间
        # 对应：如果 GPU 资源已释放，分配给新请求
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # 分配新房间（对应：分配 GPU 资源）
        heapq.heappush(heap, end)
    
    return len(heap)  # 需要的最大资源数
```

**面试回答**：
> "这题和 GPU 资源分配很像：我们需要在有限的 GPU 内存（会议室）下，分配多个请求（会议）。就像 SGLang 的调度器需要在有限的 KV Cache 内存下分配多个请求一样。"

---

### 3. **LeetCode 358: Rearrange String k Distance Apart（重排字符串 k 距离）** ⭐⭐⭐

**为什么重要**：
- 对应 **间隔调度问题**
- 需要在约束条件下安排任务
- 使用优先级队列避免冲突

**核心考点**：
- 贪心算法
- 优先级队列
- 间隔约束

**题目链接**：https://leetcode.com/problems/rearrange-string-k-distance-apart/

**关键思路**：
```python
# 伪代码（对应 batch 间隔调度）
def rearrangeString(s, k):
    # 统计字符频率
    freq = Counter(s)
    
    # 使用最大堆（按频率排序）
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)
    
    # 使用队列存储等待中的字符（对应等待队列）
    wait_queue = deque()
    
    result = []
    while heap or wait_queue:
        # 如果等待队列有元素且可以复用，加入堆
        if wait_queue and len(result) - wait_queue[0][1] >= k:
            heapq.heappush(heap, wait_queue.popleft()[0])
        
        # 选择频率最高的字符（对应优先级最高的请求）
        if heap:
            count, char = heapq.heappop(heap)
            result.append(char)
            
            # 如果还有剩余，加入等待队列（对应 waiting_queue）
            if count + 1 < 0:
                wait_queue.append(((count + 1, char), len(result) - 1))
```

**面试回答**：
> "这题和调度器的 waiting_queue 很像：我们需要在间隔约束下安排任务，使用优先级队列选择最优任务，用等待队列管理暂时不能执行的任务。"

---

### 4. **LeetCode 767: Reorganize String（重构字符串）** ⭐⭐

**为什么重要**：
- 简化版的间隔调度问题
- 练习优先级队列的基本应用
- 对应 **请求调度中的优先级问题**

**核心考点**：
- 贪心算法
- 优先级队列
- 冲突避免

**题目链接**：https://leetcode.com/problems/reorganize-string/

**关键思路**：
```python
# 伪代码（对应优先级调度）
def reorganizeString(s):
    # 统计频率
    freq = Counter(s)
    
    # 最大堆（按频率排序）
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)
    
    result = []
    prev = None
    
    while heap:
        # 选择频率最高的（对应优先级最高的请求）
        count, char = heapq.heappop(heap)
        result.append(char)
        
        # 如果前一个字符还有剩余，重新加入堆
        # 对应：如果请求还没完成，重新加入 waiting_queue
        if prev:
            heapq.heappush(heap, prev)
        
        # 更新 prev（对应：标记当前处理的请求）
        prev = (count + 1, char) if count + 1 < 0 else None
```

**面试回答**：
> "这题练习了优先级队列的基本应用，和调度器选择优先级最高的请求加入 batch 的逻辑很像。"

---

### 5. **LeetCode 239: Sliding Window Maximum（滑动窗口最大值）** ⭐⭐⭐

**为什么重要**：
- 对应 **动态批处理超时机制**
- 滑动窗口 + 超时机制
- 使用双端队列优化

**核心考点**：
- 滑动窗口
- 双端队列（Deque）
- 时间窗口管理

**题目链接**：https://leetcode.com/problems/sliding-window-maximum/

**关键思路**：
```python
# 伪代码（对应动态批处理超时）
def maxSlidingWindow(nums, k):
    # 使用双端队列存储索引（对应请求索引）
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # 移除窗口外的元素（对应超时的请求）
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # 移除小于当前元素的元素（对应优先级低的请求）
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        # 添加当前元素（对应新请求）
        dq.append(i)
        
        # 如果窗口大小达到 k，记录最大值（对应 batch 已满）
        if i >= k - 1:
            result.append(nums[dq[0]])
```

**面试回答**：
> "这题和动态批处理超时机制很像：我们需要在滑动窗口内选择最优元素，窗口大小对应 batch 大小，超时对应窗口滑动。"

---

### 6. **LeetCode 295: Find Median from Data Stream（数据流的中位数）** ⭐⭐⭐

**为什么重要**：
- 对应 **实时统计和监控**
- 使用两个堆（最大堆 + 最小堆）
- 练习堆的高级应用

**核心考点**：
- 两个堆的配合
- 实时统计
- 数据结构设计

**题目链接**：https://leetcode.com/problems/find-median-from-data-stream/

**关键思路**：
```python
# 伪代码（对应实时监控）
class MedianFinder:
    def __init__(self):
        # 最大堆：存储较小的一半（对应低优先级请求）
        self.max_heap = []
        # 最小堆：存储较大的一半（对应高优先级请求）
        self.min_heap = []
    
    def addNum(self, num):
        # 平衡两个堆（对应平衡不同优先级的请求）
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        
        if len(self.max_heap) < len(self.min_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        # 返回中位数（对应监控指标）
        ...
```

**面试回答**：
> "这题练习了两个堆的配合使用，和调度器中管理不同优先级请求的逻辑很像。可以用来实时监控系统的性能指标（如 TFFT、吞吐量）。"

---

### 7. **LeetCode 480: Sliding Window Median（滑动窗口中位数）** ⭐⭐⭐⭐

**为什么重要**：
- 结合滑动窗口和堆的应用
- 对应 **时间窗口内的统计和监控**
- 难度适中，综合性强

**核心考点**：
- 滑动窗口
- 两个堆的配合
- 实时统计

**题目链接**：https://leetcode.com/problems/sliding-window-median/

**关键思路**：
```python
# 伪代码（对应时间窗口监控）
def medianSlidingWindow(nums, k):
    # 使用两个堆维护窗口内的中位数
    max_heap = []
    min_heap = []
    
    def add(num):
        # 添加元素到合适的堆
        ...
    
    def remove(num):
        # 从堆中移除元素（对应超时的请求）
        ...
    
    result = []
    for i in range(len(nums)):
        add(nums[i])
        
        # 移除窗口外的元素（对应超时）
        if i >= k:
            remove(nums[i - k])
        
        # 计算中位数（对应监控指标）
        if i >= k - 1:
            result.append(get_median())
```

**面试回答**：
> "这题综合了滑动窗口和堆的应用，可以用来监控时间窗口内的性能指标，比如最近 1 分钟的平均 TFFT。"

---

### 8. **LeetCode 1834: Single-Threaded CPU（单线程 CPU）** ⭐⭐⭐

**为什么重要**：
- 直接对应 **任务调度问题**
- 优先级队列 + 时间管理
- 最接近实际调度器的实现

**核心考点**：
- 任务调度
- 优先级队列
- 时间管理

**题目链接**：https://leetcode.com/problems/single-threaded-cpu/

**关键思路**：
```python
# 伪代码（对应调度器实现）
def getOrder(tasks):
    # 按到达时间排序（对应请求到达时间）
    tasks = sorted([(start, duration, idx) for idx, (start, duration) in enumerate(tasks)])
    
    # 优先级队列：按执行时间排序，相同则按索引排序
    # 对应：按优先级排序，相同则按到达时间排序
    heap = []
    
    time = 0
    result = []
    i = 0
    
    while i < len(tasks) or heap:
        # 添加所有已到达的任务到堆（对应 waiting_queue）
        while i < len(tasks) and tasks[i][0] <= time:
            heapq.heappush(heap, (tasks[i][1], tasks[i][2]))
            i += 1
        
        if heap:
            # 选择执行时间最短的任务（对应优先级最高的请求）
            duration, idx = heapq.heappop(heap)
            result.append(idx)
            time += duration
        else:
            # 如果没有任务，跳到下一个任务到达时间
            time = tasks[i][0]
```

**面试回答**：
> "这题最接近实际调度器的实现：我们需要管理等待队列（waiting_queue），使用优先级队列选择最优任务，处理时间管理。和 SGLang 的调度器逻辑非常相似！"

---

## 🎯 刷题策略

### 优先级排序

1. **必刷（核心）**：
   - LeetCode 621（Task Scheduler）- 最直接对应 batching 问题
   - LeetCode 253（Meeting Rooms II）- 资源分配问题
   - LeetCode 1834（Single-Threaded CPU）- 最接近调度器实现

2. **推荐刷（加深理解）**：
   - LeetCode 358（Rearrange String k Distance Apart）- 间隔调度
   - LeetCode 239（Sliding Window Maximum）- 滑动窗口 + 超时
   - LeetCode 295（Find Median from Data Stream）- 实时统计

3. **可选刷（进阶）**：
   - LeetCode 767（Reorganize String）- 简化版优先级调度
   - LeetCode 480（Sliding Window Median）- 综合应用

### 刷题建议

1. **先理解题目**：
   - 不要急着写代码
   - 先理解题目和调度问题的对应关系
   - 画出流程图，对应到 SGLang 的调度逻辑

2. **写代码时思考**：
   - 这个数据结构对应什么？（堆 = 优先级队列）
   - 这个操作对应什么？（添加 = 请求入队，弹出 = 请求出队）
   - 这个约束对应什么？（超时 = batch_wait_timeout_s）

3. **面试准备**：
   - 每道题都要能说出和调度问题的对应关系
   - 准备"这题和 batching/TFFT/KV Cache 的关系"的回答
   - 练习画图解释（调度器流程图）

### 面试回答模板

```
1. 理解题目：这题和 [调度问题] 很像
2. 对应关系：
   - [数据结构] 对应 [调度器组件]
   - [算法] 对应 [调度策略]
   - [约束] 对应 [系统限制]
3. 解决方案：
   - 使用 [算法/数据结构]
   - 时间复杂度：O(...)
   - 空间复杂度：O(...)
4. 实际应用：
   - 在 SGLang 中，[具体应用场景]
   - 解决了 [什么问题]
```

---

## 📝 总结

**核心知识点对应**：

| LeetCode 题目 | 对应知识点 | 难度 |
|--------------|-----------|------|
| 621: Task Scheduler | Batching vs TFFT 权衡 | ⭐⭐⭐ |
| 253: Meeting Rooms II | 资源分配（GPU/KV Cache） | ⭐⭐⭐ |
| 1834: Single-Threaded CPU | 调度器实现 | ⭐⭐⭐ |
| 358: Rearrange String k Distance | 间隔调度、等待队列 | ⭐⭐⭐ |
| 239: Sliding Window Maximum | 动态批处理超时 | ⭐⭐⭐ |
| 295: Find Median from Data Stream | 实时监控 | ⭐⭐⭐ |
| 480: Sliding Window Median | 时间窗口统计 | ⭐⭐⭐⭐ |
| 767: Reorganize String | 优先级调度基础 | ⭐⭐ |

**刷题目标**：
- ✅ 理解每道题和调度问题的对应关系
- ✅ 能够用调度器的术语解释题目
- ✅ 能够说出在实际系统中的应用

**加油！这些题目刷完后，你对调度系统的理解会更深入！** 🚀


