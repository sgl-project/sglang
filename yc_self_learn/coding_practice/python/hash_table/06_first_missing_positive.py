"""
LeetCode 41: First Missing Positive (缺失的第一个正数)
难度: ⭐⭐⭐ Hard
标签: Array, Hash Table

题目描述:
给你一个未排序的整数数组 nums，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

示例:
输入: nums = [1,2,0]
输出: 3

输入: nums = [3,4,-1,1]
输出: 2

输入: nums = [7,8,9,11,12]
输出: 1

核心思路:
1. 哈希集合法: O(n) 时间，O(n) 空间 - 用集合存储所有正数
2. 原地哈希法: O(n) 时间，O(1) 空间 - 利用数组本身作为哈希表
   - 关键观察: 答案一定在 [1, n+1] 范围内（n 是数组长度）
   - 将数字 i 放到索引 i-1 的位置（如果 i 在 [1, n] 范围内）
   - 遍历数组，找到第一个 nums[i] != i+1 的位置

时间复杂度: O(n)
空间复杂度: O(1) - 原地修改
"""


def first_missing_positive(nums):
    """
    方法1: 哈希集合法 - 简单直观
    
    思路:
    - 将所有正数存入集合
    - 从 1 开始检查，找到第一个不在集合中的正数
    """
    num_set = set(nums)
    
    # 答案一定在 [1, len(nums)+1] 范围内
    for i in range(1, len(nums) + 2):
        if i not in num_set:
            return i
    
    return 1


def first_missing_positive_inplace(nums):
    """
    方法2: 原地哈希法 - 满足 O(1) 空间要求
    
    思路:
    - 关键观察: 答案一定在 [1, n+1] 范围内
    - 将数字 i 放到索引 i-1 的位置（如果 i 在 [1, n] 范围内）
    - 遍历数组，找到第一个 nums[i] != i+1 的位置
    
    步骤:
    1. 将不在 [1, n] 范围内的数字标记为 n+1（或任意大于 n 的数）
    2. 对于每个在 [1, n] 范围内的数字 num，将索引 num-1 位置的数标记为负数
    3. 遍历数组，找到第一个正数，其索引+1 就是答案
    """
    n = len(nums)
    
    # 步骤1: 将负数和大数替换为 n+1
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1
    
    # 步骤2: 对于每个在 [1, n] 范围内的数字，标记对应位置
    for i in range(n):
        num = abs(nums[i])
        if 1 <= num <= n:
            # 将索引 num-1 位置的数标记为负数
            nums[num - 1] = -abs(nums[num - 1])
    
    # 步骤3: 找到第一个正数
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    
    # 如果所有位置都被标记，说明答案是 n+1
    return n + 1


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    nums1 = [1, 2, 0]
    print(f"输入: {nums1}")
    print(f"输出: {first_missing_positive(nums1)}")  # 期望: 3
    print()
    
    # 测试用例 2
    nums2 = [3, 4, -1, 1]
    print(f"输入: {nums2}")
    print(f"输出: {first_missing_positive(nums2)}")  # 期望: 2
    print()
    
    # 测试用例 3
    nums3 = [7, 8, 9, 11, 12]
    print(f"输入: {nums3}")
    print(f"输出: {first_missing_positive(nums3)}")  # 期望: 1
    print()
    
    # 测试用例 4 - 测试原地哈希法
    nums4 = [3, 4, -1, 1]
    nums4_copy = nums4.copy()
    print(f"输入: {nums4_copy}")
    print(f"输出（原地哈希）: {first_missing_positive_inplace(nums4_copy)}")  # 期望: 2
    print()

