# AI Infra LLM 面试 Python 刷题清单 - 2 个月版本（SGLang 核心技术导向）

## 📋 计划概览

**时间范围**：2 个月（8 周）  
**目标岗位**：AI Infra LLM 工程师（SGLang、MiniMax、vLLM 等）  
**重点方向**：基于 SGLang 核心技术栈的算法训练

**SGLang 核心技术映射**：
- **RadixAttention (Radix Tree)** → Trie/Prefix Tree 相关题目
- **Zero-overhead CPU Scheduler** → 调度系统、CPU-GPU 重叠
- **Prefill-Decode Disaggregation** → 区间调度、资源分配
- **Continuous Batching** → 动态批处理、优先级队列
- **Chunked Prefill** → 分块处理、滑动窗口
- **Router (文本级别近似匹配)** → Trie、字符串匹配、模糊查询
- **KV Cache 管理** → 内存池、LRU/LFU、Paged Attention
- **Speculative Decoding** → 推测执行、并发控制

**总体策略**（按 SGLang 技术栈优先级）：
- **第一优先级**：Radix Tree & Prefix Matching（SGLang 核心创新）
- **第二优先级**：调度系统（Zero-overhead Scheduler、Prefill-Decode 分离）
- **第三优先级**：内存池管理（KV Cache、Paged Attention）
- **第四优先级**：连续批处理（Continuous Batching、Chunked Prefill）
- **第五优先级**：路由与负载均衡（Router、文本级别匹配）
- **第六优先级**：并发与异步（CPU-GPU 重叠、多线程协调）
- **第七优先级**：统计与监控（QPS、延迟、吞吐量）
- **第八优先级**：性能优化（哈希表优化、双指针）

**时间分配**：
- 每天 2-3 小时
- **每道题：30 分钟**（理解 + 实现 + 优化）
- 每天可完成：4-6 道题
- 每周安排：13 道题（包括核心题和扩展题）
- **刷题前必读**：15-30 分钟（SGLang 相关文档）
- **每日快速回顾**：5-10 分钟（相关概念）
- **复习计划**：基于记忆曲线（第1天、第3天、第7天、第14天、第30天）

---

## 📚 刷题前必读文档（按周组织）

### Week 1 必读：Radix Tree & Prefix Matching（15-20 分钟）

**刷题前必读**（Week 1 开始前）：
1. **`06_RadixAttention详解.md`** - 10-15 分钟
   - 理解 RadixAttention 的核心概念
   - 理解 Radix Tree 的数据结构
   - 理解前缀缓存的原理

2. **`18_RadixAttention与Prefix_Matching关系详解.md`** - 5-10 分钟
   - 理解 RadixAttention 和 Prefix Matching 的关系
   - 理解 Router 和 Scheduler 的前缀匹配差异

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾 Radix Tree 的核心操作（insert、search、startsWith）
- Day 4-5: 回顾 Router 文本级别匹配 vs Scheduler token 级别匹配
- Day 6-7: 回顾前缀缓存的优化效果

---

### Week 2 必读：调度系统（20-25 分钟）

**刷题前必读**（Week 2 开始前）：
1. **`13_03_SGLang_Scheduler_技术变迁.md`** - 15-20 分钟
   - 理解 Zero-overhead CPU Scheduler 的核心思想
   - 理解 waiting_queue、new_batch、running_batch 的关系
   - 理解 CPU-GPU 重叠的原理

2. **`24_为什么攒Batch会让TTFT变大_排队等待详解.md`** - 5-10 分钟
   - 理解 Continuous Batching 的权衡
   - 理解 TTFT 和吞吐量的关系

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾调度系统的核心数据结构
- Day 4-5: 回顾 Prefill-Decode 分离的原理
- Day 6-7: 回顾资源分配的优化策略

---

### Week 3 必读：内存池管理（15-20 分钟）

**刷题前必读**（Week 3 开始前）：
1. **`25_KV_Cache详解与Decode带宽瓶颈.md`** - 10-15 分钟
   - 理解 KV Cache 的基本概念
   - 理解 Prefill vs Decode 的区别
   - 理解内存带宽瓶颈

2. **`06_RadixAttention详解.md`**（复习）- 5 分钟
   - 快速回顾 Radix Tree 在 KV Cache 中的应用

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾 KV Cache 的淘汰策略（LRU、LFU）
- Day 4-5: 回顾内存池管理的核心思想
- Day 6-7: 回顾 Paged Attention 的原理

---

### Week 4 必读：连续批处理（15-20 分钟）

**刷题前必读**（Week 4 开始前）：
1. **`07_多请求场景与批处理.md`** - 10-15 分钟
   - 理解多请求场景
   - 理解 Continuous Batching 的原理
   - 理解批处理的优化策略

2. **`08_process_batch具体实现详解.md`** - 5-10 分钟
   - 理解批处理的具体实现
   - 理解 Chunked Prefill 的原理

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾 Continuous Batching 的核心思想
- Day 4-5: 回顾 Chunked Prefill 的优化效果
- Day 6-7: 回顾批量合并的策略

---

### Week 5 必读：路由与负载均衡（15-20 分钟）

**刷题前必读**（Week 5 开始前）：
1. **`14_Cache_Aware_Router前缀匹配详解.md`** - 10-15 分钟
   - 理解 Router 的文本级别近似匹配
   - 理解 Cache-aware Routing 的原理

2. **`22_SGLang完整请求流程详解_纠正版.md`**（重点看 Router 部分）- 5-10 分钟
   - 理解 Router 在完整流程中的作用
   - 理解 Router 和 Worker 的交互

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾 Router 的文本级别匹配机制
- Day 4-5: 回顾 Cache-aware Routing 的优化效果
- Day 6-7: 回顾负载均衡的策略

---

### Week 6 必读：并发与异步（10-15 分钟）

**刷题前必读**（Week 6 开始前）：
1. **`13_03_SGLang_Scheduler_技术变迁.md`**（复习 CPU-GPU 重叠部分）- 10-15 分钟
   - 理解 Zero-overhead Scheduler 的并发机制
   - 理解 CPU-GPU 重叠的原理

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾多线程协调的核心概念
- Day 4-5: 回顾 CPU-GPU 重叠的优化效果
- Day 6-7: 回顾异步处理的策略

---

### Week 7 必读：统计与监控（10-15 分钟）

**刷题前必读**（Week 7 开始前）：
1. **`03_TTFT_为什么重要.md`** - 10-15 分钟
   - 理解 TTFT、TPOT 等关键指标
   - 理解延迟和吞吐量的关系

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾 QPS 统计的核心概念
- Day 4-5: 回顾延迟统计的方法
- Day 6-7: 回顾性能优化的策略

---

### Week 8 必读：系统设计（15-20 分钟）

**刷题前必读**（Week 8 开始前）：
1. **`22_SGLang完整请求流程详解_纠正版.md`**（完整复习）- 10-15 分钟
   - 完整理解 SGLang 的请求流程
   - 理解各个组件的协调

2. **`23_面试视角_云端推理常见系统与问题.md`** - 5-10 分钟
   - 理解面试常见问题
   - 理解系统设计的思路

**每日快速回顾**（每天刷题前 5 分钟）：
- Day 1-3: 回顾系统设计的核心思想
- Day 4-5: 回顾性能优化的常见模式
- Day 6-7: 综合复习所有技术栈

---

## 🔄 基于记忆曲线的复习计划

### 记忆曲线原理
- **第1天**：初次学习（100% 记忆）
- **第3天**：第一次复习（保留 80%）
- **第7天**：第二次复习（保留 60%）
- **第14天**：第三次复习（保留 40%）
- **第30天**：第四次复习（长期记忆）

### 复习策略

**每周复习安排**：
- **Day 1**：新内容学习 + 刷题
- **Day 3**：复习 Day 1 的内容（快速回顾 10-15 分钟）
- **Day 7**：复习 Week 1 的内容（完整回顾 20-30 分钟）
- **Day 14**：复习 Week 1-2 的内容（重点回顾 30-40 分钟）
- **Day 30**：复习所有内容（综合复习 1-2 小时）

**具体复习方法**：
1. **快速回顾**（10-15 分钟）：
   - 重新阅读必读文档的关键部分
   - 回顾本周刷过的题目
   - 总结核心算法模式

2. **完整回顾**（20-30 分钟）：
   - 完整阅读必读文档
   - 重新刷 2-3 道核心题目
   - 总结技术栈的核心思想

3. **综合复习**（1-2 小时）：
   - 回顾所有必读文档
   - 重新刷 5-10 道核心题目
   - 总结所有技术栈的关系

---

## 📅 8 周详细计划

### Week 1: Radix Tree & Prefix Matching（SGLang 核心创新）

**目标**：掌握 RadixAttention 的核心数据结构 - Radix Tree（Trie 变种）

**📖 刷题前必读**（Week 1 开始前，15-20 分钟）：
1. **`06_RadixAttention详解.md`** - 10-15 分钟
   - 重点：Radix Tree 数据结构、前缀缓存原理
2. **`18_RadixAttention与Prefix_Matching关系详解.md`** - 5-10 分钟
   - 重点：Router vs Scheduler 的前缀匹配差异

**🔄 复习计划**：
- **Day 3**：快速回顾 Radix Tree 核心操作（10 分钟）
- **Day 7**：完整回顾 Week 1 内容（20 分钟）
- **Day 14**：复习 Week 1-2（30 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **RadixAttention** 是 SGLang 的核心创新，使用 Radix Tree 实现前缀缓存
- Router 使用文本级别的近似前缀匹配来选择 Worker
- Scheduler 使用 token 级别的精确前缀匹配来查找 KV Cache

#### Day 1: Radix Tree 基础（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `06_RadixAttention详解.md` 中 Radix Tree 的核心概念
- 理解 Trie 树的基本操作（insert、search、startsWith）

- **LeetCode 208: Implement Trie (Prefix Tree)** - 30 分钟
  - **SGLang 关联**：Radix Tree 是 Trie 的变种，用于存储和匹配前缀
  - **核心考点**：Trie 树基础操作（insert、search、startsWith）、树形结构
  - **实际应用**：Router 文本级别匹配、Scheduler token 级别匹配、RadixAttention
  - **难度**：Medium

- **LeetCode 211: Design Add and Search Words** - 30 分钟
  - **SGLang 关联**：Router 的近似前缀匹配，支持通配符查询
  - **核心考点**：Trie + DFS、模糊匹配、递归回溯
  - **实际应用**：Router 近似匹配、Prefix Cache 优化
  - **难度**：Medium

- **LeetCode 212: Word Search II** - 30 分钟
  - **SGLang 关联**：多模式匹配，类似 Router 同时匹配多个 prefix
  - **核心考点**：Trie + DFS、多模式匹配、回溯
  - **实际应用**：多 prefix 匹配、批量路由决策
  - **难度**：Hard

#### Day 2: 字符串匹配（3 道题）
- **LeetCode 438: Find All Anagrams in a String** - 30 分钟
  - **SGLang 关联**：Router 的文本级别匹配，查找相似请求模式
  - **核心考点**：滑动窗口、哈希表计数、字符串匹配
  - **实际应用**：请求分组、Prefix 相似度计算
  - **难度**：Medium

- **LeetCode 76: Minimum Window Substring** - 30 分钟
  - **SGLang 关联**：滑动窗口的经典应用，用于文本匹配
  - **核心考点**：滑动窗口、哈希表、字符串
  - **实际应用**：文本匹配、Prefix 匹配
  - **难度**：Hard

- **LeetCode 3: Longest Substring Without Repeating Characters** - 30 分钟
  - **SGLang 关联**：滑动窗口基础，理解窗口维护
  - **核心考点**：滑动窗口、哈希表
  - **实际应用**：字符串处理基础
  - **难度**：Medium

#### Day 3: Prefix 匹配扩展（3 道题）
- **LeetCode 14: Longest Common Prefix** - 30 分钟
  - **SGLang 关联**：查找最长公共前缀，Radix Tree 的核心操作
  - **核心考点**：字符串处理、前缀匹配
  - **实际应用**：Prefix Cache 查找、共享前缀计算
  - **难度**：Easy

- **LeetCode 720: Longest Word in Dictionary** - 30 分钟
  - **SGLang 关联**：Trie 树应用，查找最长匹配
  - **核心考点**：Trie、DFS、字符串处理
  - **实际应用**：最长 prefix 匹配
  - **难度**：Medium

- **LeetCode 692: Top K Frequent Words** - 30 分钟
  - **SGLang 关联**：统计高频 prefix，优化缓存策略
  - **核心考点**：哈希表 + 堆、排序、字符串
  - **实际应用**：热点统计、Prefix Cache 优化
  - **难度**：Medium

#### Day 4-5: 复习与扩展（4 道题）
- **LeetCode 49: Group Anagrams** - 30 分钟
  - **SGLang 关联**：Router 根据 prefix 分组请求
  - **核心考点**：哈希表分组、Key 设计
  - **难度**：Medium

- **LeetCode 242: Valid Anagram** - 30 分钟
  - **SGLang 关联**：字符串匹配基础
  - **核心考点**：哈希表计数、字符串处理
  - **难度**：Easy

- **LeetCode 205: Isomorphic Strings** - 30 分钟
  - **SGLang 关联**：字符串映射，理解字符对应关系
  - **核心考点**：哈希表、字符串映射
  - **难度**：Easy

- **LeetCode 387: First Unique Character in a String** - 30 分钟
  - **SGLang 关联**：字符串处理基础
  - **核心考点**：哈希表、字符串遍历
  - **难度**：Easy

#### Day 6-7: 综合练习与总结
- **快速回顾**（5 分钟）：回顾 Radix Tree 核心操作、前缀缓存原理
- 复习本周 13 道题
- 理解 Radix Tree 在 SGLang 中的应用
- 思考 Router 和 Scheduler 的前缀匹配差异（文本级别 vs token 级别）
- 总结 Trie 和滑动窗口的常见模式

**📝 本周复习提醒**：
- **Day 3**（Week 1 Day 3）：快速回顾 Day 1 的 Radix Tree 基础（10 分钟）
- **Day 7**（Week 1 Day 7）：完整回顾 Week 1 所有内容（20 分钟）

---

### Week 2: 调度系统（Zero-overhead Scheduler & Prefill-Decode 分离）

**目标**：掌握 SGLang 调度系统的核心算法

**📖 刷题前必读**（Week 2 开始前，20-25 分钟）：
1. **`13_03_SGLang_Scheduler_技术变迁.md`** - 15-20 分钟
   - 重点：Zero-overhead Scheduler、waiting_queue、new_batch、running_batch
2. **`24_为什么攒Batch会让TTFT变大_排队等待详解.md`** - 5-10 分钟
   - 重点：Continuous Batching 的权衡、TTFT vs 吞吐量

**🔄 复习计划**：
- **Day 3**：快速回顾调度系统核心数据结构（10 分钟）
- **Day 7**：完整回顾 Week 2 内容（20 分钟）
- **Day 14**：复习 Week 1-2（30 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **Zero-overhead CPU Scheduler**：CPU 和 GPU 重叠执行，避免 GPU 空闲
- **Prefill-Decode Disaggregation**：Prefill 和 Decode 可以独立调度，提高资源利用率
- **Continuous Batching**：动态添加和移除请求，最大化批处理效率

#### Day 1: 调度核心（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `13_03_SGLang_Scheduler_技术变迁.md` 中 waiting_queue、new_batch、running_batch 的关系
- 理解 Zero-overhead Scheduler 的核心思想

- **LeetCode 621: Task Scheduler** - 30 分钟
  - **SGLang 关联**：Scheduler 的核心就是任务调度，需要合理安排 Prefill 和 Decode
  - **核心考点**：贪心算法、优先级队列、时间窗口调度
  - **实际应用**：Continuous Batching、GPU 资源调度、冷却时间
  - **难度**：Medium

- **LeetCode 253: Meeting Rooms II** - 30 分钟
  - **SGLang 关联**：Prefill-Decode 分离，需要分配不同的 GPU 资源
  - **核心考点**：区间调度、扫描线算法、贪心策略
  - **实际应用**：Prefill 和 Decode 的时序管理、多 Worker 任务分配
  - **难度**：Medium

- **LeetCode 252: Meeting Rooms** - 30 分钟
  - **SGLang 关联**：区间调度基础，判断是否有冲突
  - **核心考点**：区间排序、冲突检测
  - **实际应用**：资源冲突检测
  - **难度**：Easy

#### Day 2: 优先级队列（3 道题）
- **LeetCode 295: Find Median from Data Stream** - 30 分钟
  - **SGLang 关联**：Scheduler 需要实时统计延迟、吞吐量等指标
  - **核心考点**：双堆结构（大顶堆 + 小顶堆）、流式数据处理
  - **实际应用**：实时监控、动态统计、优先级调度
  - **难度**：Hard

- **LeetCode 703: Kth Largest Element in a Stream** - 30 分钟
  - **SGLang 关联**：流式数据 Top K 问题，类似优先级调度
  - **核心考点**：堆、流式处理
  - **实际应用**：Top K 请求选择
  - **难度**：Easy

- **LeetCode 215: Kth Largest Element in an Array** - 30 分钟
  - **SGLang 关联**：快速选择算法，用于优先级调度
  - **核心考点**：快速选择、堆、分治
  - **实际应用**：优先级调度、Top K 问题
  - **难度**：Medium

#### Day 3: 动态调度（3 道题）
- **LeetCode 767: Reorganize String** - 30 分钟
  - **SGLang 关联**：任务重排，类似请求重新调度
  - **核心考点**：贪心算法、优先级队列
  - **实际应用**：请求重排、避免冲突
  - **难度**：Medium

- **LeetCode 358: Rearrange String k Distance Apart** - 30 分钟
  - **SGLang 关联**：带距离约束的调度，类似 Prefill-Decode 时序约束
  - **核心考点**：贪心算法、优先级队列
  - **实际应用**：带约束的调度
  - **难度**：Hard

- **LeetCode 630: Course Schedule III** - 30 分钟
  - **SGLang 关联**：带截止时间的调度，类似请求的 SLA 约束
  - **核心考点**：贪心算法、优先级队列、排序
  - **实际应用**：SLA 约束的调度
  - **难度**：Hard

#### Day 4-5: 资源分配（4 道题）
- **LeetCode 135: Candy** - 30 分钟
  - **SGLang 关联**：资源分配问题，类似 GPU 资源分配
  - **核心考点**：贪心算法、数组处理
  - **实际应用**：资源分配优化
  - **难度**：Hard

- **LeetCode 455: Assign Cookies** - 30 分钟
  - **SGLang 关联**：贪心分配，类似请求分配到 Worker
  - **核心考点**：贪心算法、排序
  - **实际应用**：请求分配
  - **难度**：Easy

- **LeetCode 435: Non-overlapping Intervals** - 30 分钟
  - **SGLang 关联**：区间调度，选择不重叠的区间
  - **核心考点**：贪心算法、区间排序
  - **实际应用**：资源调度优化
  - **难度**：Medium

- **LeetCode 452: Minimum Number of Arrows to Burst Balloons** - 30 分钟
  - **SGLang 关联**：区间覆盖问题，类似资源覆盖
  - **核心考点**：贪心算法、区间排序
  - **实际应用**：资源覆盖优化
  - **难度**：Medium

#### Day 6-7: 综合练习与总结
- **快速回顾**（5 分钟）：回顾调度系统核心数据结构、CPU-GPU 重叠原理
- 复习本周 13 道题
- 理解 Zero-overhead Scheduler 的核心思想
- 思考 Prefill-Decode 分离如何提高资源利用率
- 总结调度算法的常见模式

**📝 本周复习提醒**：
- **Day 3**（Week 2 Day 3）：快速回顾 Day 1 的调度系统基础（10 分钟）
- **Day 7**（Week 2 Day 7）：完整回顾 Week 2 所有内容（20 分钟）
- **Day 14**（Week 2 Day 7）：复习 Week 1-2 所有内容（30 分钟）

---

### Week 3: 内存池管理（KV Cache & Paged Attention）

**目标**：掌握 SGLang KV Cache 管理的核心算法

**📖 刷题前必读**（Week 3 开始前，15-20 分钟）：
1. **`25_KV_Cache详解与Decode带宽瓶颈.md`** - 10-15 分钟
   - 重点：KV Cache 基本概念、Prefill vs Decode、内存带宽瓶颈
2. **`06_RadixAttention详解.md`**（复习）- 5 分钟
   - 重点：Radix Tree 在 KV Cache 中的应用

**🔄 复习计划**：
- **Day 3**：快速回顾 KV Cache 淘汰策略（10 分钟）
- **Day 7**：完整回顾 Week 3 内容（20 分钟）
- **Day 14**：复习 Week 1-3（40 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **KV Cache 管理**：需要高效分配和回收 GPU 内存
- **Paged Attention**：分页管理 KV Cache，类似操作系统的虚拟内存
- **内存池**：快速分配和回收，避免内存碎片

#### Day 1: 缓存基础（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `25_KV_Cache详解与Decode带宽瓶颈.md` 中 KV Cache 的基本概念
- 理解 Prefill vs Decode 的区别

- **LeetCode 146: LRU Cache** - 30 分钟
  - **SGLang 关联**：KV Cache 的淘汰策略，当内存不足时淘汰最久未使用的缓存
  - **核心考点**：哈希表 + 双向链表、O(1) 操作、内存管理
  - **实际应用**：KV Cache 淘汰、GPU 内存管理、Prefix Cache 管理
  - **难度**：Medium

- **LeetCode 460: LFU Cache** - 30 分钟
  - **SGLang 关联**：另一种缓存淘汰策略，基于访问频率
  - **核心考点**：复杂数据结构设计、频率统计、多层哈希表
  - **实际应用**：KV Cache 访问频率统计、缓存优化
  - **难度**：Hard

- **LeetCode 432: All O`one Data Structure** - 30 分钟
  - **SGLang 关联**：复杂数据结构设计，用于频率统计
  - **核心考点**：哈希表 + 双向链表、频率统计
  - **实际应用**：频率统计、缓存优化
  - **难度**：Hard

#### Day 2: 内存池管理（3 道题）
- **LeetCode 380: Insert Delete GetRandom O(1)** - 30 分钟
  - **SGLang 关联**：KV Cache 内存池需要快速分配和回收，类似 Paged Attention
  - **核心考点**：哈希表 + 动态数组、O(1) 随机访问、删除优化
  - **实际应用**：内存池管理、Paged Attention、随机采样
  - **难度**：Medium

- **LeetCode 381: Insert Delete GetRandom O(1) - Duplicates allowed** - 30 分钟
  - **SGLang 关联**：支持重复元素的内存池管理
  - **核心考点**：哈希表 + 动态数组、重复元素处理
  - **实际应用**：复杂内存池管理
  - **难度**：Hard

- **LeetCode 706: Design HashMap** - 30 分钟
  - **SGLang 关联**：理解哈希表的实现，KV Cache 管理的基础
  - **核心考点**：哈希表实现、冲突解决、数据结构设计
  - **实际应用**：底层数据结构设计、冲突处理
  - **难度**：Easy

#### Day 3: 缓存优化（3 道题）
- **LeetCode 355: Design Twitter** - 30 分钟
  - **SGLang 关联**：综合的系统设计题目，包含多个组件
  - **核心考点**：系统设计思维、多个数据结构组合、实时数据处理
  - **实际应用**：系统设计、多组件协调
  - **难度**：Medium

- **LeetCode 588: Design In-Memory File System** - 30 分钟
  - **SGLang 关联**：树形结构设计，类似 Radix Tree
  - **核心考点**：Trie 树、系统设计
  - **实际应用**：树形数据结构设计
  - **难度**：Hard

- **LeetCode 642: Design Search Autocomplete System** - 30 分钟
  - **SGLang 关联**：前缀匹配系统，类似 Router 的 prefix matching
  - **核心考点**：Trie + 优先级队列、前缀匹配
  - **实际应用**：Prefix 匹配系统
  - **难度**：Hard

#### Day 4-5: 扩展练习（4 道题）
- **LeetCode 1: Two Sum** - 30 分钟
  - **SGLang 关联**：最经典的哈希表应用，O(n²) → O(n) 优化
  - **核心考点**：哈希表、空间换时间、算法优化
  - **实际应用**：快速查找、优化思想
  - **难度**：Easy

- **LeetCode 217: Contains Duplicate** - 30 分钟
  - **SGLang 关联**：去重基础，理解哈希集合
  - **核心考点**：哈希集合、去重
  - **实际应用**：去重处理
  - **难度**：Easy

- **LeetCode 219: Contains Duplicate II** - 30 分钟
  - **SGLang 关联**：滑动窗口 + 哈希表，理解窗口维护
  - **核心考点**：滑动窗口、哈希表
  - **实际应用**：窗口维护
  - **难度**：Easy

- **LeetCode 220: Contains Duplicate III** - 30 分钟
  - **SGLang 关联**：滑动窗口 + 有序集合，理解复杂窗口
  - **核心考点**：滑动窗口、有序集合、桶排序
  - **实际应用**：复杂窗口维护
  - **难度**：Hard

#### Day 6-7: 综合练习与总结
- 复习本周 13 道题
- 理解 KV Cache 管理的核心思想
- 对比 LRU vs LFU vs Random 的优缺点
- 思考 Paged Attention 如何优化内存管理

---

### Week 4: 连续批处理（Continuous Batching & Chunked Prefill）

**目标**：掌握 SGLang 连续批处理和分块预填充的核心算法

**📖 刷题前必读**（Week 4 开始前，15-20 分钟）：
1. **`07_多请求场景与批处理.md`** - 10-15 分钟
   - 重点：多请求场景、Continuous Batching 原理、批处理优化策略
2. **`08_process_batch具体实现详解.md`** - 5-10 分钟
   - 重点：批处理具体实现、Chunked Prefill 原理

**🔄 复习计划**：
- **Day 3**：快速回顾 Continuous Batching 核心思想（10 分钟）
- **Day 7**：完整回顾 Week 4 内容（20 分钟）
- **Day 14**：复习 Week 1-4（50 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **Continuous Batching**：动态添加和移除请求，最大化批处理效率
- **Chunked Prefill**：将长序列分块处理，避免内存溢出
- **优先级队列**：根据前缀长度、延迟等指标动态调整优先级

#### Day 1: 滑动窗口（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `07_多请求场景与批处理.md` 中 Continuous Batching 的原理
- 理解 Chunked Prefill 的优化效果

- **LeetCode 239: Sliding Window Maximum** - 30 分钟
  - **SGLang 关联**：Chunked Prefill 中处理分块的最值统计
  - **核心考点**：单调队列/单调栈、滑动窗口、双端队列
  - **实际应用**：批量处理、Chunked Prefill、Batch 优先级维护
  - **难度**：Hard

- **LeetCode 480: Sliding Window Median** - 30 分钟
  - **SGLang 关联**：滑动窗口的中位数统计，类似动态统计
  - **核心考点**：滑动窗口、双堆、有序集合
  - **实际应用**：动态统计、中位数计算
  - **难度**：Hard

- **LeetCode 424: Longest Repeating Character Replacement** - 30 分钟
  - **SGLang 关联**：滑动窗口优化，理解窗口调整
  - **核心考点**：滑动窗口、哈希表、字符串
  - **实际应用**：窗口优化
  - **难度**：Medium

#### Day 2: 批量合并（3 道题）
- **LeetCode 23: Merge k Sorted Lists** - 30 分钟
  - **SGLang 关联**：合并多个 batch 的输出，类似多 GPU 合并结果
  - **核心考点**：堆/优先级队列、分治算法、链表操作
  - **实际应用**：多 GPU 合并结果、批量处理、Pipeline Parallelism
  - **难度**：Hard

- **LeetCode 21: Merge Two Sorted Lists** - 30 分钟
  - **SGLang 关联**：合并基础，理解合并逻辑
  - **核心考点**：链表合并、双指针
  - **实际应用**：合并基础
  - **难度**：Easy

- **LeetCode 88: Merge Sorted Array** - 30 分钟
  - **SGLang 关联**：数组合并，理解合并优化
  - **核心考点**：数组合并、双指针、从后往前
  - **实际应用**：数组合并优化
  - **难度**：Easy

#### Day 3: 热点统计（3 道题）
- **LeetCode 347: Top K Frequent Elements** - 30 分钟
  - **SGLang 关联**：统计最频繁的请求模式，优化 Prefix Cache
  - **核心考点**：哈希表 + 堆、桶排序、频率统计
  - **实际应用**：热点统计、Prefix Cache 优化、请求路由
  - **难度**：Medium

- **LeetCode 692: Top K Frequent Words** - 30 分钟
  - **SGLang 关联**：统计高频 prefix，优化缓存策略
  - **核心考点**：哈希表 + 堆、排序、字符串
  - **实际应用**：热点统计、Prefix Cache 优化
  - **难度**：Medium

- **LeetCode 973: K Closest Points to Origin** - 30 分钟
  - **SGLang 关联**：Top K 问题变种，理解优先级选择
  - **核心考点**：堆、快速选择、排序
  - **实际应用**：Top K 选择
  - **难度**：Medium

#### Day 4-5: 批处理扩展（4 道题）
- **LeetCode 56: Merge Intervals** - 30 分钟
  - **SGLang 关联**：区间合并，类似 batch 合并
  - **核心考点**：区间排序、合并、贪心
  - **实际应用**：区间合并
  - **难度**：Medium

- **LeetCode 57: Insert Interval** - 30 分钟
  - **SGLang 关联**：区间插入，类似请求插入 batch
  - **核心考点**：区间处理、插入逻辑
  - **实际应用**：区间插入
  - **难度**：Medium

- **LeetCode 986: Interval List Intersections** - 30 分钟
  - **SGLang 关联**：区间交集，理解区间操作
  - **核心考点**：双指针、区间处理
  - **实际应用**：区间交集
  - **难度**：Medium

- **LeetCode 1288: Remove Covered Intervals** - 30 分钟
  - **SGLang 关联**：区间覆盖，理解覆盖关系
  - **核心考点**：区间排序、覆盖检测
  - **实际应用**：区间覆盖
  - **难度**：Medium

#### Day 6-7: 综合练习与总结
- 复习本周 13 道题
- 理解 Continuous Batching 的核心思想
- 思考 Chunked Prefill 如何优化长序列处理
- 总结滑动窗口和批量处理的常见模式

---

### Week 5: 路由与负载均衡（Router & 文本级别匹配）

**目标**：掌握 SGLang Router 的核心算法

**📖 刷题前必读**（Week 5 开始前，15-20 分钟）：
1. **`14_Cache_Aware_Router前缀匹配详解.md`** - 10-15 分钟
   - 重点：Router 文本级别近似匹配、Cache-aware Routing 原理
2. **`22_SGLang完整请求流程详解_纠正版.md`**（重点看 Router 部分）- 5-10 分钟
   - 重点：Router 在完整流程中的作用、Router 和 Worker 的交互

**🔄 复习计划**：
- **Day 3**：快速回顾 Router 文本级别匹配机制（10 分钟）
- **Day 7**：完整回顾 Week 5 内容（20 分钟）
- **Day 14**：复习 Week 1-5（60 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **Router**：在多个 Worker 之间路由请求，使用文本级别的近似前缀匹配
- **负载均衡**：根据 Worker 的负载情况，选择最合适的 Worker
- **Cache-aware Routing**：将相似请求路由到同一 Worker，提高缓存命中率

#### Day 1: 路由分组（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `14_Cache_Aware_Router前缀匹配详解.md` 中 Router 的文本级别近似匹配
- 理解 Cache-aware Routing 的原理

- **LeetCode 49: Group Anagrams** - 30 分钟
  - **SGLang 关联**：Router 根据 prefix 分组请求，路由到相同 Worker
  - **核心考点**：哈希表分组、Key 设计、字符串处理
  - **实际应用**：Cache-aware Routing、请求分组、负载均衡
  - **难度**：Medium

- **LeetCode 249: Group Shifted Strings** - 30 分钟
  - **SGLang 关联**：字符串分组变种，理解分组模式
  - **核心考点**：哈希表分组、字符串处理
  - **实际应用**：分组模式
  - **难度**：Medium

- **LeetCode 609: Find Duplicate File in System** - 30 分钟
  - **SGLang 关联**：分组应用，理解分组场景
  - **核心考点**：哈希表分组、字符串处理
  - **实际应用**：分组应用
  - **难度**：Medium

#### Day 2: 前缀和（3 道题）
- **LeetCode 560: Subarray Sum Equals K** - 30 分钟
  - **SGLang 关联**：统计和查询的基础，用于路由决策
  - **核心考点**：前缀和、哈希表优化、数组问题
  - **实际应用**：统计请求累积延迟、Token 数统计、负载统计
  - **难度**：Medium

- **LeetCode 523: Continuous Subarray Sum** - 30 分钟
  - **SGLang 关联**：前缀和变种，理解模运算
  - **核心考点**：前缀和、哈希表、模运算
  - **实际应用**：前缀和变种
  - **难度**：Medium

- **LeetCode 974: Subarray Sums Divisible by K** - 30 分钟
  - **SGLang 关联**：前缀和 + 模运算，理解复杂前缀和
  - **核心考点**：前缀和、哈希表、模运算
  - **实际应用**：复杂前缀和
  - **难度**：Medium

#### Day 3: 负载均衡（3 道题）
- **LeetCode 325: Maximum Size Subarray Sum Equals k** - 30 分钟
  - **SGLang 关联**：前缀和 + 哈希表，理解最大子数组
  - **核心考点**：前缀和、哈希表
  - **实际应用**：最大子数组
  - **难度**：Medium

- **LeetCode 713: Subarray Product Less Than K** - 30 分钟
  - **SGLang 关联**：滑动窗口变种，理解窗口维护
  - **核心考点**：滑动窗口、双指针
  - **实际应用**：窗口维护
  - **难度**：Medium

- **LeetCode 209: Minimum Size Subarray Sum** - 30 分钟
  - **SGLang 关联**：滑动窗口，查找最小窗口
  - **核心考点**：滑动窗口、双指针
  - **实际应用**：最小窗口
  - **难度**：Medium

#### Day 4-5: 路由扩展（4 道题）
- **LeetCode 30: Substring with Concatenation of All Words** - 30 分钟
  - **SGLang 关联**：字符串匹配，理解复杂匹配
  - **核心考点**：滑动窗口、哈希表、字符串
  - **实际应用**：复杂匹配
  - **难度**：Hard

- **LeetCode 159: Longest Substring with At Most Two Distinct Characters** - 30 分钟
  - **SGLang 关联**：滑动窗口变种，理解字符限制
  - **核心考点**：滑动窗口、哈希表
  - **实际应用**：字符限制窗口
  - **难度**：Medium

- **LeetCode 340: Longest Substring with At Most K Distinct Characters** - 30 分钟
  - **SGLang 关联**：滑动窗口扩展，K 个不同字符
  - **核心考点**：滑动窗口、哈希表
  - **实际应用**：K 字符窗口
  - **难度**：Hard

- **LeetCode 395: Longest Substring with At Least K Repeating Characters** - 30 分钟
  - **SGLang 关联**：滑动窗口 + 分治，理解复杂窗口
  - **核心考点**：滑动窗口、分治、哈希表
  - **实际应用**：复杂窗口
  - **难度**：Medium

#### Day 6-7: 综合练习与总结
- 复习本周 13 道题
- 理解 Router 的文本级别匹配机制
- 思考 Cache-aware Routing 如何提高缓存命中率
- 总结路由和负载均衡的常见模式

---

### Week 6: 并发与异步（CPU-GPU 重叠 & 多线程协调）

**目标**：掌握 SGLang 并发和异步处理的核心算法

**📖 刷题前必读**（Week 6 开始前，10-15 分钟）：
1. **`13_03_SGLang_Scheduler_技术变迁.md`**（复习 CPU-GPU 重叠部分）- 10-15 分钟
   - 重点：Zero-overhead Scheduler 并发机制、CPU-GPU 重叠原理

**🔄 复习计划**：
- **Day 3**：快速回顾多线程协调核心概念（10 分钟）
- **Day 7**：完整回顾 Week 6 内容（20 分钟）
- **Day 14**：复习 Week 1-6（70 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **Zero-overhead CPU Scheduler**：CPU 和 GPU 重叠执行，避免 GPU 空闲
- **多线程协调**：Tokenizer、Scheduler、GPU 之间的协调
- **异步处理**：异步 tokenize、异步调度、异步推理

#### Day 1: 线程同步基础（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `13_03_SGLang_Scheduler_技术变迁.md` 中 CPU-GPU 重叠的原理
- 理解 Zero-overhead Scheduler 的并发机制

- **LeetCode 1114: Print in Order** - 30 分钟
  - **SGLang 关联**：推理系统中多个组件的协调（Tokenizer → Scheduler → GPU）
  - **核心考点**：线程同步、锁机制、条件变量
  - **实际应用**：多线程协调、状态机、CPU-GPU 重叠
  - **难度**：Easy

- **LeetCode 1115: Print FooBar Alternately** - 30 分钟
  - **SGLang 关联**：两个线程交替执行，类似 Prefill 和 Decode 的交替
  - **核心考点**：线程同步、信号量
  - **实际应用**：Prefill-Decode 交替、资源切换
  - **难度**：Medium

- **LeetCode 1116: Print Zero Even Odd** - 30 分钟
  - **SGLang 关联**：多线程协调，理解复杂同步
  - **核心考点**：线程同步、信号量、条件变量
  - **实际应用**：复杂同步
  - **难度**：Medium

#### Day 2: 多线程协作（3 道题）
- **LeetCode 1195: Fizz Buzz Multithreaded** - 30 分钟
  - **SGLang 关联**：多个 Worker 协调工作，类似多线程协作
  - **核心考点**：多线程协作、状态机、同步原语
  - **实际应用**：多 Worker 协调、并发控制、状态机
  - **难度**：Medium

- **LeetCode 1226: The Dining Philosophers** - 30 分钟
  - **SGLang 关联**：经典并发问题，理解死锁避免
  - **核心考点**：死锁避免、信号量、同步
  - **实际应用**：死锁避免
  - **难度**：Medium

- **LeetCode 1279: Traffic Light Controlled Intersection** - 30 分钟
  - **SGLang 关联**：资源控制，理解互斥访问
  - **核心考点**：互斥锁、资源控制
  - **实际应用**：资源控制
  - **难度**：Easy

#### Day 3: 并发控制（3 道题）
- **LeetCode 1188: Design Bounded Blocking Queue** - 30 分钟
  - **SGLang 关联**：有界阻塞队列，类似请求队列
  - **核心考点**：阻塞队列、条件变量、同步
  - **实际应用**：请求队列、阻塞队列
  - **难度**：Medium

- **LeetCode 1242: Web Crawler Multithreaded** - 30 分钟
  - **SGLang 关联**：多线程爬虫，理解并发处理
  - **核心考点**：多线程、同步、并发控制
  - **实际应用**：并发处理
  - **难度**：Medium

- **LeetCode 1196: How Many Apples Can You Put into the Basket** - 30 分钟
  - **SGLang 关联**：资源分配，理解贪心选择
  - **核心考点**：贪心算法、排序
  - **实际应用**：资源分配
  - **难度**：Easy

#### Day 4-5: 异步处理（4 道题）
- **LeetCode 1249: Minimum Remove to Make Valid Parentheses** - 30 分钟
  - **SGLang 关联**：字符串处理，理解括号匹配
  - **核心考点**：栈、字符串处理
  - **实际应用**：括号匹配
  - **难度**：Medium

- **LeetCode 20: Valid Parentheses** - 30 分钟
  - **SGLang 关联**：括号匹配基础
  - **核心考点**：栈、字符串处理
  - **实际应用**：括号匹配基础
  - **难度**：Easy

- **LeetCode 32: Longest Valid Parentheses** - 30 分钟
  - **SGLang 关联**：括号匹配扩展，理解动态规划
  - **核心考点**：动态规划、栈
  - **实际应用**：括号匹配扩展
  - **难度**：Hard

- **LeetCode 301: Remove Invalid Parentheses** - 30 分钟
  - **SGLang 关联**：括号匹配复杂版，理解回溯
  - **核心考点**：回溯、字符串处理
  - **实际应用**：复杂括号匹配
  - **难度**：Hard

#### Day 6-7: 综合练习与总结
- 复习本周 13 道题
- 理解 Zero-overhead Scheduler 的并发机制
- 思考 CPU-GPU 重叠如何提高系统效率
- 总结并发和多线程的常见模式

---

### Week 7: 统计与监控（QPS、延迟、吞吐量）

**目标**：掌握 SGLang 统计和监控的核心算法

**📖 刷题前必读**（Week 7 开始前，10-15 分钟）：
1. **`03_TTFT_为什么重要.md`** - 10-15 分钟
   - 重点：TTFT、TPOT 等关键指标、延迟和吞吐量的关系

**🔄 复习计划**：
- **Day 3**：快速回顾 QPS 统计核心概念（10 分钟）
- **Day 7**：完整回顾 Week 7 内容（20 分钟）
- **Day 14**：复习 Week 1-7（80 分钟）
- **Day 30**：综合复习（1 小时）

**SGLang 背景**：
- **QPS 监控**：统计每秒处理的请求数
- **延迟统计**：统计 TTFT（Time To First Token）、TPOT（Time Per Output Token）
- **吞吐量统计**：统计系统的整体吞吐量

#### Day 1: QPS 统计（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `03_TTFT_为什么重要.md` 中 TTFT、TPOT 等关键指标
- 理解延迟和吞吐量的关系

- **LeetCode 362: Design Hit Counter** - 30 分钟
  - **SGLang 关联**：直接对应系统的 QPS 监控
  - **核心考点**：时间窗口、队列/数组、滑动窗口
  - **实际应用**：QPS 监控、请求统计、负载监控
  - **难度**：Medium

- **LeetCode 359: Logger Rate Limiter** - 30 分钟
  - **SGLang 关联**：限流器，理解时间窗口限流
  - **核心考点**：哈希表、时间窗口
  - **实际应用**：限流器
  - **难度**：Easy

- **LeetCode 346: Moving Average from Data Stream** - 30 分钟
  - **SGLang 关联**：滑动窗口平均值，理解移动平均
  - **核心考点**：滑动窗口、队列
  - **实际应用**：移动平均
  - **难度**：Easy

#### Day 2: 动态统计（3 道题）
- **LeetCode 53: Maximum Subarray** - 30 分钟
  - **SGLang 关联**：优化问题的经典例子，类似统计峰值负载
  - **核心考点**：动态规划、贪心算法、Kadane 算法
  - **实际应用**：峰值负载统计、最大延迟、资源使用峰值
  - **难度**：Easy

- **LeetCode 918: Maximum Sum Circular Subarray** - 30 分钟
  - **SGLang 关联**：最大子数组扩展，理解环形数组
  - **核心考点**：动态规划、Kadane 算法、环形数组
  - **实际应用**：环形数组
  - **难度**：Medium

- **LeetCode 152: Maximum Product Subarray** - 30 分钟
  - **SGLang 关联**：最大乘积子数组，理解乘积特性
  - **核心考点**：动态规划、数组处理
  - **实际应用**：乘积子数组
  - **难度**：Medium

#### Day 3: 频率统计（3 道题）
- **LeetCode 451: Sort Characters By Frequency** - 30 分钟
  - **SGLang 关联**：频率排序，理解频率统计
  - **核心考点**：哈希表、排序、频率统计
  - **实际应用**：频率排序
  - **难度**：Medium

- **LeetCode 1636: Sort Array by Increasing Frequency** - 30 分钟
  - **SGLang 关联**：频率排序变种，理解排序规则
  - **核心考点**：哈希表、排序、频率统计
  - **实际应用**：频率排序变种
  - **难度**：Easy

- **LeetCode 1331: Rank Transform of an Array** - 30 分钟
  - **SGLang 关联**：数组排名，理解排名计算
  - **核心考点**：哈希表、排序
  - **实际应用**：排名计算
  - **难度**：Easy

#### Day 4-5: 监控扩展（4 道题）
- **LeetCode 42: Trapping Rain Water** - 30 分钟
  - **SGLang 关联**：类似分配 GPU 资源、内存资源
  - **核心考点**：双指针、单调栈、动态规划
  - **实际应用**：资源分配、优化问题
  - **难度**：Hard

- **LeetCode 11: Container With Most Water** - 30 分钟
  - **SGLang 关联**：双指针经典，理解双指针优化
  - **核心考点**：双指针、贪心算法
  - **实际应用**：双指针优化
  - **难度**：Medium

- **LeetCode 15: 3Sum** - 30 分钟
  - **SGLang 关联**：类似优化 batch 组合，找到最优的请求组合
  - **核心考点**：双指针、排序优化、去重技巧
  - **实际应用**：组合优化、去重处理
  - **难度**：Medium

- **LeetCode 16: 3Sum Closest** - 30 分钟
  - **SGLang 关联**：3Sum 变种，理解最近和
  - **核心考点**：双指针、排序
  - **实际应用**：最近和
  - **难度**：Medium

#### Day 6-7: 综合练习与总结
- 复习本周 13 道题
- 理解统计和监控的核心思想
- 思考如何应用到实际系统
- 总结统计和监控的常见模式

---

### Week 8: 系统设计 & 性能优化（综合复习）

**目标**：掌握系统设计中的算法应用，综合复习

**📖 刷题前必读**（Week 8 开始前，15-20 分钟）：
1. **`22_SGLang完整请求流程详解_纠正版.md`**（完整复习）- 10-15 分钟
   - 重点：完整理解 SGLang 请求流程、各组件协调
2. **`23_面试视角_云端推理常见系统与问题.md`** - 5-10 分钟
   - 重点：面试常见问题、系统设计思路

**🔄 复习计划**：
- **Day 3**：快速回顾系统设计核心思想（10 分钟）
- **Day 7**：完整回顾 Week 8 内容（20 分钟）
- **Day 14**：复习所有内容（2 小时）
- **Day 30**：综合复习（2 小时）

#### Day 1: 系统设计基础（3 道题）

**📖 刷题前快速回顾**（5 分钟）：
- 回顾 `22_SGLang完整请求流程详解_纠正版.md` 中完整请求流程
- 理解各个组件的协调关系

- **LeetCode 706: Design HashMap** - 30 分钟
  - **SGLang 关联**：理解哈希表的实现，KV Cache 管理的基础
  - **核心考点**：哈希表实现、冲突解决、数据结构设计
  - **实际应用**：底层数据结构设计、冲突处理、KV Cache 管理
  - **难度**：Easy

- **LeetCode 705: Design HashSet** - 30 分钟
  - **SGLang 关联**：哈希集合实现，理解集合操作
  - **核心考点**：哈希集合、数据结构设计
  - **实际应用**：集合操作
  - **难度**：Easy

- **LeetCode 355: Design Twitter** - 30 分钟
  - **SGLang 关联**：综合的系统设计题目，包含多个组件
  - **核心考点**：系统设计思维、多个数据结构组合、实时数据处理
  - **实际应用**：系统设计、多组件协调、实时数据处理
  - **难度**：Medium

#### Day 2: 系统设计扩展（3 道题）
- **LeetCode 146: LRU Cache** - 30 分钟（复习）
  - **SGLang 关联**：KV Cache 淘汰策略
  - **核心考点**：哈希表 + 双向链表、O(1) 操作
  - **实际应用**：KV Cache 淘汰
  - **难度**：Medium

- **LeetCode 460: LFU Cache** - 30 分钟（复习）
  - **SGLang 关联**：缓存淘汰策略
  - **核心考点**：复杂数据结构设计、频率统计
  - **实际应用**：缓存优化
  - **难度**：Hard

- **LeetCode 588: Design In-Memory File System** - 30 分钟（复习）
  - **SGLang 关联**：树形结构设计
  - **核心考点**：Trie 树、系统设计
  - **实际应用**：树形数据结构
  - **难度**：Hard

#### Day 3: 性能优化（3 道题）
- **LeetCode 1: Two Sum** - 30 分钟（复习）
  - **SGLang 关联**：最经典的哈希表应用，O(n²) → O(n) 优化
  - **核心考点**：哈希表、空间换时间、算法优化
  - **实际应用**：快速查找、优化思想
  - **难度**：Easy

- **LeetCode 167: Two Sum II - Input array is sorted** - 30 分钟
  - **SGLang 关联**：Two Sum 变种，理解双指针优化
  - **核心考点**：双指针、排序数组
  - **实际应用**：双指针优化
  - **难度**：Easy

- **LeetCode 170: Two Sum III - Data structure design** - 30 分钟
  - **SGLang 关联**：Two Sum 系统设计版
  - **核心考点**：哈希表、系统设计
  - **实际应用**：系统设计
  - **难度**：Easy

#### Day 4-5: 综合优化（4 道题）
- **LeetCode 18: 4Sum** - 30 分钟
  - **SGLang 关联**：3Sum 扩展，理解多指针
  - **核心考点**：双指针、排序、去重
  - **实际应用**：多指针优化
  - **难度**：Medium

- **LeetCode 259: 3Sum Smaller** - 30 分钟
  - **SGLang 关联**：3Sum 变种，理解计数问题
  - **核心考点**：双指针、排序
  - **实际应用**：计数问题
  - **难度**：Medium

- **LeetCode 611: Valid Triangle Number** - 30 分钟
  - **SGLang 关联**：双指针应用，理解三角形判断
  - **核心考点**：双指针、排序
  - **实际应用**：三角形判断
  - **难度**：Medium

- **LeetCode 977: Squares of a Sorted Array** - 30 分钟
  - **SGLang 关联**：双指针基础，理解排序数组处理
  - **核心考点**：双指针、排序数组
  - **实际应用**：排序数组处理
  - **难度**：Easy

#### Day 6-7: 综合复习与模拟面试
- 复习所有核心题目（从 Week 1-7 中挑选重点题目）
- 模拟面试：随机抽取题目
- 总结常见模式和技巧
- **SGLang 技术栈总结**：
  - Radix Tree & Prefix Matching
  - Zero-overhead Scheduler
  - KV Cache 管理
  - Continuous Batching
  - Router & 负载均衡
  - 并发与异步
  - 统计与监控

---

## 🎯 核心题目清单（按 SGLang 技术栈优先级）

**总题目数**：104 道题（8 周 × 13 道题/周）  
**时间分配**：每道题 30 分钟  
**每天完成**：4-6 道题（2-3 小时）

### 第一优先级：Radix Tree & Prefix Matching（Week 1，13 道题）

**核心题目**：
1. ✅ **LeetCode 208: Implement Trie** - Radix Tree 基础
2. ✅ **LeetCode 211: Design Add and Search Words** - 模糊匹配
3. ✅ **LeetCode 212: Word Search II** - 多模式匹配
4. ✅ **LeetCode 438: Find All Anagrams** - 模式匹配
5. ✅ **LeetCode 76: Minimum Window Substring** - 文本匹配
6. ✅ **LeetCode 3: Longest Substring Without Repeating Characters** - 滑动窗口基础
7. ✅ **LeetCode 14: Longest Common Prefix** - 最长公共前缀
8. ✅ **LeetCode 720: Longest Word in Dictionary** - Trie 应用
9. ✅ **LeetCode 692: Top K Frequent Words** - 热点统计
10. ✅ **LeetCode 49: Group Anagrams** - 分组思想
11. ✅ **LeetCode 242: Valid Anagram** - 字符串匹配基础
12. ✅ **LeetCode 205: Isomorphic Strings** - 字符串映射
13. ✅ **LeetCode 387: First Unique Character in a String** - 字符串处理基础

### 第二优先级：调度系统（Week 2，13 道题）

**核心题目**：
14. ✅ **LeetCode 621: Task Scheduler** - 调度核心
15. ✅ **LeetCode 253: Meeting Rooms II** - Prefill-Decode 分离
16. ✅ **LeetCode 252: Meeting Rooms** - 区间调度基础
17. ✅ **LeetCode 295: Find Median from Data Stream** - 动态统计
18. ✅ **LeetCode 703: Kth Largest Element in a Stream** - 流式 Top K
19. ✅ **LeetCode 215: Kth Largest Element in an Array** - 快速选择
20. ✅ **LeetCode 767: Reorganize String** - 任务重排
21. ✅ **LeetCode 358: Rearrange String k Distance Apart** - 带距离约束
22. ✅ **LeetCode 630: Course Schedule III** - 带截止时间的调度
23. ✅ **LeetCode 135: Candy** - 资源分配
24. ✅ **LeetCode 455: Assign Cookies** - 贪心分配
25. ✅ **LeetCode 435: Non-overlapping Intervals** - 区间调度优化
26. ✅ **LeetCode 452: Minimum Number of Arrows to Burst Balloons** - 区间覆盖

### 第三优先级：内存池管理（Week 3，13 道题）

**核心题目**：
27. ✅ **LeetCode 146: LRU Cache** - KV Cache 淘汰
28. ✅ **LeetCode 460: LFU Cache** - 缓存策略
29. ✅ **LeetCode 432: All O`one Data Structure** - 频率统计
30. ✅ **LeetCode 380: Insert Delete GetRandom O(1)** - 内存池管理
31. ✅ **LeetCode 381: Insert Delete GetRandom O(1) - Duplicates** - 复杂内存池
32. ✅ **LeetCode 706: Design HashMap** - 哈希表实现
33. ✅ **LeetCode 355: Design Twitter** - 系统设计
34. ✅ **LeetCode 588: Design In-Memory File System** - 树形结构设计
35. ✅ **LeetCode 642: Design Search Autocomplete System** - Prefix 匹配系统
36. ✅ **LeetCode 1: Two Sum** - 哈希表优化
37. ✅ **LeetCode 217: Contains Duplicate** - 去重基础
38. ✅ **LeetCode 219: Contains Duplicate II** - 滑动窗口 + 哈希表
39. ✅ **LeetCode 220: Contains Duplicate III** - 复杂窗口

### 第四优先级：连续批处理（Week 4，13 道题）

**核心题目**：
40. ✅ **LeetCode 239: Sliding Window Maximum** - Chunked Prefill
41. ✅ **LeetCode 480: Sliding Window Median** - 滑动窗口中位数
42. ✅ **LeetCode 424: Longest Repeating Character Replacement** - 窗口优化
43. ✅ **LeetCode 23: Merge k Sorted Lists** - 批量合并
44. ✅ **LeetCode 21: Merge Two Sorted Lists** - 合并基础
45. ✅ **LeetCode 88: Merge Sorted Array** - 数组合并
46. ✅ **LeetCode 347: Top K Frequent Elements** - 热点统计
47. ✅ **LeetCode 692: Top K Frequent Words** - 高频统计
48. ✅ **LeetCode 973: K Closest Points to Origin** - Top K 变种
49. ✅ **LeetCode 56: Merge Intervals** - 区间合并
50. ✅ **LeetCode 57: Insert Interval** - 区间插入
51. ✅ **LeetCode 986: Interval List Intersections** - 区间交集
52. ✅ **LeetCode 1288: Remove Covered Intervals** - 区间覆盖

### 第五优先级：路由与负载均衡（Week 5，13 道题）

**核心题目**：
53. ✅ **LeetCode 49: Group Anagrams** - Cache-aware Routing
54. ✅ **LeetCode 249: Group Shifted Strings** - 分组模式
55. ✅ **LeetCode 609: Find Duplicate File in System** - 分组应用
56. ✅ **LeetCode 560: Subarray Sum Equals K** - 前缀和
57. ✅ **LeetCode 523: Continuous Subarray Sum** - 前缀和变种
58. ✅ **LeetCode 974: Subarray Sums Divisible by K** - 复杂前缀和
59. ✅ **LeetCode 325: Maximum Size Subarray Sum Equals k** - 最大子数组
60. ✅ **LeetCode 713: Subarray Product Less Than K** - 滑动窗口变种
61. ✅ **LeetCode 209: Minimum Size Subarray Sum** - 最小窗口
62. ✅ **LeetCode 30: Substring with Concatenation of All Words** - 复杂匹配
63. ✅ **LeetCode 159: Longest Substring with At Most Two Distinct Characters** - 字符限制窗口
64. ✅ **LeetCode 340: Longest Substring with At Most K Distinct Characters** - K 字符窗口
65. ✅ **LeetCode 395: Longest Substring with At Least K Repeating Characters** - 复杂窗口

### 第六优先级：并发与异步（Week 6，13 道题）

**核心题目**：
66. ✅ **LeetCode 1114: Print in Order** - 线程同步
67. ✅ **LeetCode 1115: Print FooBar Alternately** - 交替执行
68. ✅ **LeetCode 1116: Print Zero Even Odd** - 复杂同步
69. ✅ **LeetCode 1195: Fizz Buzz Multithreaded** - 多线程协作
70. ✅ **LeetCode 1226: The Dining Philosophers** - 死锁避免
71. ✅ **LeetCode 1279: Traffic Light Controlled Intersection** - 资源控制
72. ✅ **LeetCode 1188: Design Bounded Blocking Queue** - 阻塞队列
73. ✅ **LeetCode 1242: Web Crawler Multithreaded** - 并发处理
74. ✅ **LeetCode 1196: How Many Apples Can You Put into the Basket** - 资源分配
75. ✅ **LeetCode 1249: Minimum Remove to Make Valid Parentheses** - 括号匹配
76. ✅ **LeetCode 20: Valid Parentheses** - 括号匹配基础
77. ✅ **LeetCode 32: Longest Valid Parentheses** - 括号匹配扩展
78. ✅ **LeetCode 301: Remove Invalid Parentheses** - 复杂括号匹配

### 第七优先级：统计与监控（Week 7，13 道题）

**核心题目**：
79. ✅ **LeetCode 362: Design Hit Counter** - QPS 统计
80. ✅ **LeetCode 359: Logger Rate Limiter** - 限流器
81. ✅ **LeetCode 346: Moving Average from Data Stream** - 移动平均
82. ✅ **LeetCode 53: Maximum Subarray** - 峰值统计
83. ✅ **LeetCode 918: Maximum Sum Circular Subarray** - 环形数组
84. ✅ **LeetCode 152: Maximum Product Subarray** - 乘积子数组
85. ✅ **LeetCode 451: Sort Characters By Frequency** - 频率排序
86. ✅ **LeetCode 1636: Sort Array by Increasing Frequency** - 频率排序变种
87. ✅ **LeetCode 1331: Rank Transform of an Array** - 排名计算
88. ✅ **LeetCode 42: Trapping Rain Water** - 资源分配
89. ✅ **LeetCode 11: Container With Most Water** - 双指针优化
90. ✅ **LeetCode 15: 3Sum** - 组合优化
91. ✅ **LeetCode 16: 3Sum Closest** - 最近和

### 第八优先级：系统设计 & 性能优化（Week 8，13 道题）

**核心题目**：
92. ✅ **LeetCode 706: Design HashMap** - 底层原理
93. ✅ **LeetCode 705: Design HashSet** - 哈希集合
94. ✅ **LeetCode 355: Design Twitter** - 系统设计
95. ✅ **LeetCode 146: LRU Cache** - 缓存淘汰（复习）
96. ✅ **LeetCode 460: LFU Cache** - 缓存策略（复习）
97. ✅ **LeetCode 588: Design In-Memory File System** - 树形结构（复习）
98. ✅ **LeetCode 1: Two Sum** - 哈希表优化（复习）
99. ✅ **LeetCode 167: Two Sum II** - 双指针优化
100. ✅ **LeetCode 170: Two Sum III** - 系统设计版
101. ✅ **LeetCode 18: 4Sum** - 多指针优化
102. ✅ **LeetCode 259: 3Sum Smaller** - 计数问题
103. ✅ **LeetCode 611: Valid Triangle Number** - 三角形判断
104. ✅ **LeetCode 977: Squares of a Sorted Array** - 排序数组处理

---

## 💡 核心知识点总结（按 SGLang 技术栈）

### 1. Radix Tree & Prefix Matching（SGLang 核心创新）
- **Trie 基础**：Implement Trie (208) - Radix Tree 的基础数据结构
- **模糊匹配**：Design Add and Search Words (211) - Router 的近似匹配
- **模式匹配**：Find All Anagrams (438) - 文本级别的匹配
- **SGLang 应用**：
  - **RadixAttention**：使用 Radix Tree 存储和查询共享前缀
  - **Router**：文本级别的近似前缀匹配，路由请求到合适的 Worker
  - **Scheduler**：token 级别的精确前缀匹配，查找 KV Cache

### 2. 调度系统（Zero-overhead Scheduler）
- **任务调度**：Task Scheduler (621) - Continuous Batching 的核心
- **资源分配**：Meeting Rooms II (253) - Prefill-Decode 分离
- **动态统计**：Find Median from Data Stream (295) - 实时监控
- **SGLang 应用**：
  - **Zero-overhead CPU Scheduler**：CPU 和 GPU 重叠执行
  - **Prefill-Decode Disaggregation**：Prefill 和 Decode 独立调度
  - **Continuous Batching**：动态添加和移除请求

### 3. 内存池管理（KV Cache & Paged Attention）
- **LRU 淘汰**：LRU Cache (146) - KV Cache 淘汰策略
- **内存池**：Insert Delete GetRandom O(1) (380) - Paged Attention
- **LFU 淘汰**：LFU Cache (460) - 基于频率的淘汰
- **SGLang 应用**：
  - **KV Cache 管理**：高效分配和回收 GPU 内存
  - **Paged Attention**：分页管理 KV Cache，类似虚拟内存
  - **内存池**：快速分配和回收，避免内存碎片

### 4. 连续批处理（Continuous Batching）
- **滑动窗口**：Sliding Window Maximum (239) - Chunked Prefill
- **批量合并**：Merge k Sorted Lists (23) - 多 GPU 合并
- **热点统计**：Top K Frequent Elements (347) - Prefix Cache 优化
- **SGLang 应用**：
  - **Continuous Batching**：动态批处理，最大化效率
  - **Chunked Prefill**：分块处理长序列
  - **优先级队列**：根据前缀长度、延迟等指标动态调整

### 5. 路由与负载均衡（Router）
- **分组思想**：Group Anagrams (49) - Cache-aware Routing
- **前缀和**：Subarray Sum Equals K (560) - 统计和查询
- **文本匹配**：Minimum Window Substring (76) - 文本级别匹配
- **SGLang 应用**：
  - **Router**：文本级别的近似前缀匹配
  - **负载均衡**：根据 Worker 负载选择最合适的 Worker
  - **Cache-aware Routing**：将相似请求路由到同一 Worker

### 6. 并发与异步（CPU-GPU 重叠）
- **线程同步**：Print in Order (1114) - 多组件协调
- **多线程协作**：Fizz Buzz Multithreaded (1195) - 多 Worker 协调
- **交替执行**：Print FooBar Alternately (1115) - Prefill-Decode 交替
- **SGLang 应用**：
  - **Zero-overhead CPU Scheduler**：CPU 和 GPU 重叠执行
  - **多线程协调**：Tokenizer、Scheduler、GPU 之间的协调
  - **异步处理**：异步 tokenize、异步调度、异步推理

### 7. 统计与监控
- **QPS 统计**：Design Hit Counter (362) - 请求统计
- **峰值统计**：Maximum Subarray (53) - 峰值负载
- **频率统计**：All O`one Data Structure (432) - 频率统计
- **SGLang 应用**：
  - **QPS 监控**：统计每秒处理的请求数
  - **延迟统计**：统计 TTFT、TPOT 等指标
  - **吞吐量统计**：统计系统的整体吞吐量

### 8. 性能优化
- **哈希表优化**：Two Sum (1), Subarray Sum Equals K (560)
- **双指针**：3Sum (15), Trapping Rain Water (42)
- **滑动窗口**：Sliding Window Maximum (239), Minimum Window Substring (76)

---

## 🎓 面试技巧（SGLang 导向）

### 1. 题目理解（SGLang 技术栈映射）
- **看到 Trie/Prefix Tree 题** → 联想到 **RadixAttention、Radix Tree、Router 的文本级别匹配**
- **看到调度题** → 联想到 **Zero-overhead CPU Scheduler、Prefill-Decode 分离、Continuous Batching**
- **看到缓存题** → 联想到 **KV Cache 管理、Paged Attention、内存池**
- **看到队列题** → 联想到 **Continuous Batching、Chunked Prefill、优先级队列**
- **看到路由题** → 联想到 **Router、文本级别近似匹配、Cache-aware Routing**
- **看到并发题** → 联想到 **CPU-GPU 重叠、多线程协调、异步处理**

### 2. 算法选择（SGLang 应用场景）
- **调度类**：优先考虑堆/优先级队列 → **Continuous Batching、Prefill-Decode 调度**
- **缓存类**：哈希表 + 链表/数组 → **KV Cache 管理、Paged Attention**
- **统计类**：哈希表 + 堆/桶排序 → **QPS 监控、延迟统计、频率统计**
- **路由类**：Trie 树、哈希表分组 → **Router、文本级别匹配、Cache-aware Routing**

### 3. 优化方向（SGLang 性能优化）
- **时间复杂度**：O(n²) → O(n)，用哈希表优化 → **提高推理速度**
- **空间复杂度**：用时间换空间，或者原地算法 → **优化 GPU 内存使用**
- **实际应用**：结合 SGLang 的实际场景 → **RadixAttention、Zero-overhead Scheduler**

### 4. 系统设计思维（SGLang 架构）
- **每个算法都要思考**：如何在 SGLang 系统中应用
- **理解系统组件**：
  - **Router**：文本级别的近似前缀匹配
  - **Scheduler**：Zero-overhead CPU Scheduler、Prefill-Decode 分离
  - **KV Cache**：RadixAttention、Paged Attention
  - **Worker**：Continuous Batching、Chunked Prefill
- **性能优化**：
  - **延迟**：TTFT、TPOT
  - **吞吐量**：QPS、Token/s
  - **资源利用率**：GPU 利用率、内存利用率

---

## 📝 练习建议

### 第一周（Week 1-2）
1. **先刷核心题**：Task Scheduler, LRU Cache, Meeting Rooms II
2. **理解系统背景**：每道题都要思考如何在推理系统中应用
3. **优化复杂度**：不仅要做对，还要优化到最优解

### 第二周（Week 3-4）
4. **扩展题目**：Sliding Window Maximum, Merge k Sorted Lists, Implement Trie
5. **总结模式**：总结常见的数据结构和算法模式
6. **实际应用**：思考这些算法在实际系统中的使用场景

### 第三周（Week 5-6）
7. **并发编程**：理解多线程、同步、状态机
8. **统计监控**：理解 QPS、延迟、吞吐量等指标
9. **模式匹配**：理解 Prefix Matching、路由算法

### 第四周（Week 7-8）
10. **系统设计**：理解系统设计中的算法应用
11. **性能优化**：理解优化技巧和权衡
12. **综合复习**：模拟面试，查漏补缺

---

## 🔗 相关资源（SGLang 核心技术）

### SGLang 核心技术
- **RadixAttention**：使用 Radix Tree 实现前缀缓存，SGLang 的核心创新
- **Zero-overhead CPU Scheduler**：CPU 和 GPU 重叠执行，避免 GPU 空闲
- **Prefill-Decode Disaggregation**：Prefill 和 Decode 分离，提高资源利用率
- **Continuous Batching**：动态批处理，最大化批处理效率
- **Chunked Prefill**：分块预填充，处理长序列
- **Router**：文本级别的近似前缀匹配，路由请求到合适的 Worker
- **KV Cache 管理**：Paged Attention、内存池管理
- **Speculative Decoding**：推测解码，提升推理速度

### 算法模式（SGLang 应用）
- **调度算法**：优先级队列、贪心算法 → **Continuous Batching、Prefill-Decode 调度**
- **缓存算法**：LRU、LFU、Random → **KV Cache 淘汰策略**
- **路由算法**：Trie、Prefix Matching、负载均衡 → **Router、文本级别匹配**
- **并发模式**：多线程、状态机、同步 → **CPU-GPU 重叠、多线程协调**

### 面试准备
- **SGLang 面试**：AI Infra LLM 工程师，重点考察 SGLang 核心技术
- **系统设计**：LLM 推理系统、调度系统、KV Cache 管理
- **算法优化**：时间复杂度、空间复杂度，结合 SGLang 实际场景

---

## 📅 完整复习时间表（基于记忆曲线）

### 记忆曲线复习原理
- **第1天**：初次学习（100% 记忆）
- **第3天**：第一次复习（保留 80%，复习 10-15 分钟）
- **第7天**：第二次复习（保留 60%，复习 20-30 分钟）
- **第14天**：第三次复习（保留 40%，复习 30-60 分钟）
- **第30天**：第四次复习（长期记忆，复习 1-2 小时）

### 详细复习时间表

#### Week 1（Radix Tree & Prefix Matching）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 1 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的 Radix Tree 基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 1 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-2 所有内容 | 30 分钟 | 重点回顾 + 重刷 5-8 道核心题 |
| **Day 30** | 综合复习 Week 1-4 | 1 小时 | 完整回顾 + 重刷 10-15 道核心题 |

#### Week 2（调度系统）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 2 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的调度系统基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 2 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-2 所有内容 | 30 分钟 | 重点回顾 + 重刷 5-8 道核心题 |
| **Day 30** | 综合复习 Week 1-4 | 1 小时 | 完整回顾 + 重刷 10-15 道核心题 |

#### Week 3（内存池管理）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 3 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的 KV Cache 基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 3 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-3 所有内容 | 40 分钟 | 重点回顾 + 重刷 8-10 道核心题 |
| **Day 30** | 综合复习 Week 1-4 | 1 小时 | 完整回顾 + 重刷 10-15 道核心题 |

#### Week 4（连续批处理）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 4 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的批处理基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 4 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-4 所有内容 | 50 分钟 | 重点回顾 + 重刷 10-12 道核心题 |
| **Day 30** | 综合复习 Week 1-4 | 1 小时 | 完整回顾 + 重刷 10-15 道核心题 |

#### Week 5（路由与负载均衡）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 5 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的 Router 基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 5 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-5 所有内容 | 60 分钟 | 重点回顾 + 重刷 12-15 道核心题 |
| **Day 30** | 综合复习所有内容 | 2 小时 | 完整回顾 + 重刷 20-25 道核心题 |

#### Week 6（并发与异步）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 6 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的并发基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 6 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-6 所有内容 | 70 分钟 | 重点回顾 + 重刷 15-18 道核心题 |
| **Day 30** | 综合复习所有内容 | 2 小时 | 完整回顾 + 重刷 20-25 道核心题 |

#### Week 7（统计与监控）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 7 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的统计基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 7 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习 Week 1-7 所有内容 | 80 分钟 | 重点回顾 + 重刷 18-20 道核心题 |
| **Day 30** | 综合复习所有内容 | 2 小时 | 完整回顾 + 重刷 20-25 道核心题 |

#### Week 8（系统设计 & 性能优化）

| 时间点 | 复习内容 | 时间 | 方法 |
|--------|---------|------|------|
| **Day 1** | 初次学习 Week 8 内容 | - | 刷题 + 阅读文档 |
| **Day 3** | 复习 Day 1 的系统设计基础 | 10 分钟 | 快速回顾文档 + 重刷 1-2 道题 |
| **Day 7** | 完整复习 Week 8 所有内容 | 20 分钟 | 完整阅读文档 + 重刷 3-5 道核心题 |
| **Day 14** | 复习所有内容（综合复习） | 2 小时 | 完整回顾所有文档 + 重刷 25-30 道核心题 |
| **Day 30** | 最终综合复习 | 2 小时 | 完整回顾 + 模拟面试 + 重刷 30-40 道核心题 |

### 复习方法建议

**快速回顾（10-15 分钟）**：
1. 重新阅读必读文档的关键部分（5-10 分钟）
2. 快速浏览本周刷过的题目（2-3 分钟）
3. 总结核心算法模式（2-3 分钟）

**完整回顾（20-30 分钟）**：
1. 完整阅读必读文档（10-15 分钟）
2. 重新刷 3-5 道核心题目（10-15 分钟）
3. 总结技术栈的核心思想（5 分钟）

**综合复习（1-2 小时）**：
1. 回顾所有必读文档（30-60 分钟）
2. 重新刷 10-25 道核心题目（30-60 分钟）
3. 总结所有技术栈的关系（10-20 分钟）
4. 模拟面试（可选，20-30 分钟）

---

## 📊 进度跟踪

### Week 1-2（调度与缓存）
- [ ] Task Scheduler (621)
- [ ] LRU Cache (146)
- [ ] Meeting Rooms II (253)
- [ ] Find Median from Data Stream (295)
- [ ] LFU Cache (460)

### Week 3-4（队列与路由）
- [ ] Sliding Window Maximum (239)
- [ ] Merge k Sorted Lists (23)
- [ ] Implement Trie (208)
- [ ] Subarray Sum Equals K (560)
- [ ] Top K Frequent Elements (347)

### Week 5-6（并发与统计）
- [ ] Insert Delete GetRandom O(1) (380)
- [ ] Design Add and Search Words (211)
- [ ] Group Anagrams (49)
- [ ] Design Hit Counter (362)
- [ ] Find All Anagrams (438)

### Week 7-8（系统设计与优化）
- [ ] Design HashMap (706)
- [ ] Design Twitter (355)
- [ ] Maximum Subarray (53)
- [ ] 3Sum (15)
- [ ] Trapping Rain Water (42)

---

**参考**：
- **SGLang 官方文档**：https://sglang.readthedocs.io/
- **SGLang GitHub**：https://github.com/sgl-project/sglang
- **RadixAttention 论文**：SGLang 的核心创新技术
- **Zero-overhead CPU Scheduler**：SGLang 调度系统的核心技术
- **MiniMax AI Infra LLM 面试**：AI Infra LLM 工程师面试准备
- **LeetCode 官方题目和讨论**：算法基础训练
