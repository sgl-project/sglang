# Hash Table 哈希表刷题集合

这个目录包含了经典的哈希表相关 LeetCode 题目，每道题都包含：
- 题目描述和示例
- 核心思路分析
- 多种解法（包括最优解和对比解法）
- 复杂度分析
- 测试用例

## 📚 题目列表

### 基础题（Easy）

1. **01_two_sum.py** - Two Sum (两数之和)
   - 难度: ⭐ Easy
   - 核心: 哈希表优化查找
   - 时间复杂度: O(n)

2. **04_contains_duplicate.py** - Contains Duplicate (存在重复元素)
   - 难度: ⭐ Easy
   - 核心: 集合去重
   - 时间复杂度: O(n)

3. **05_valid_anagram.py** - Valid Anagram (有效的字母异位词)
   - 难度: ⭐ Easy
   - 核心: 字符频次统计
   - 时间复杂度: O(n)

### 中等题（Medium）

4. **02_group_anagrams.py** - Group Anagrams (字母异位词分组)
   - 难度: ⭐⭐ Medium
   - 核心: 哈希表分组
   - 时间复杂度: O(n*k) 或 O(n*k*log(k))

5. **07_design_hashmap.py** - Design HashMap (设计哈希映射)
   - 难度: ⭐⭐ Medium
   - 核心: 哈希表实现（链地址法）
   - 时间复杂度: 平均 O(1)

6. **08_subarray_sum_equals_k.py** - Subarray Sum Equals K (和为 K 的子数组)
   - 难度: ⭐⭐ Medium
   - 核心: 前缀和 + 哈希表
   - 时间复杂度: O(n)

### 困难题（Hard）

7. **03_longest_consecutive_sequence.py** - Longest Consecutive Sequence (最长连续序列)
   - 难度: ⭐⭐⭐ Hard
   - 核心: 哈希集合 + 只从起点扩展
   - 时间复杂度: O(n)

8. **06_first_missing_positive.py** - First Missing Positive (缺失的第一个正数)
   - 难度: ⭐⭐⭐ Hard
   - 核心: 原地哈希
   - 时间复杂度: O(n)，空间复杂度: O(1)

## 🎯 刷题建议

### 优先级排序

1. **必刷（核心基础）**：
   - Two Sum - 哈希表最经典应用
   - Contains Duplicate - 集合基础
   - Valid Anagram - 字符统计

2. **推荐刷（加深理解）**：
   - Group Anagrams - 哈希表分组
   - Subarray Sum Equals K - 前缀和 + 哈希表
   - Design HashMap - 理解哈希表实现

3. **进阶刷（挑战）**：
   - Longest Consecutive Sequence - 优化技巧
   - First Missing Positive - 原地算法

### 核心技巧总结

1. **查找优化**：
   - 用哈希表将 O(n) 查找优化为 O(1)
   - 典型应用：Two Sum

2. **去重和计数**：
   - 使用集合去重
   - 使用字典统计频次
   - 典型应用：Contains Duplicate, Valid Anagram

3. **分组**：
   - 用哈希表将相同特征的元素分组
   - 典型应用：Group Anagrams

4. **前缀和**：
   - 结合哈希表快速查找子数组
   - 典型应用：Subarray Sum Equals K

5. **原地哈希**：
   - 利用数组本身作为哈希表
   - 典型应用：First Missing Positive

## 🚀 运行方式

每道题都可以独立运行：

```bash
# 运行单道题
python 01_two_sum.py

# 或者使用 Python 解释器
python -m hash_table.01_two_sum
```

## 📝 学习笔记

建议在刷题时：
1. 先自己思考，不要直接看答案
2. 理解哈希表的核心思想：O(1) 查找
3. 注意边界情况和特殊输入
4. 比较不同解法的时间和空间复杂度
5. 总结常见模式和技巧

## 🔗 相关资源

- [LeetCode 哈希表专题](https://leetcode.com/tag/hash-table/)
- [Python 集合和字典文档](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

