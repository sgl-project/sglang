"""
LeetCode 560: Subarray Sum Equals K (和为 K 的子数组)
难度: ⭐⭐ Medium
标签: Array, Hash Table, Prefix Sum

题目描述:
给你一个整数数组 nums 和一个整数 k，请你统计并返回该数组中和为 k 的连续子数组的个数。

示例:
输入: nums = [1,1,1], k = 2
输出: 2
解释: [1,1] 与 [1,1] 为两种不同的情况。

输入: nums = [1,2,3], k = 3
输出: 2

核心思路:
1. 暴力法: O(n²) - 枚举所有子数组，计算和
2. 前缀和 + 哈希表: O(n) - 利用前缀和的性质
   - 前缀和: prefix_sum[i] = sum(nums[0...i])
   - 子数组 [i...j] 的和 = prefix_sum[j] - prefix_sum[i-1]
   - 如果 prefix_sum[j] - prefix_sum[i-1] == k，则 prefix_sum[j] - k == prefix_sum[i-1]
   - 用哈希表记录每个前缀和出现的次数
   - 遍历时，对于当前前缀和 prefix_sum[j]，查找 prefix_sum[j] - k 出现的次数

时间复杂度: O(n)
空间复杂度: O(n)
"""

from collections import defaultdict


def subarray_sum(nums, k):
    """
    前缀和 + 哈希表法
    
    思路:
    - 用哈希表记录每个前缀和出现的次数
    - 遍历数组，计算当前前缀和
    - 查找 prefix_sum - k 在哈希表中的出现次数
    - 将当前前缀和加入哈希表
    """
    count = 0
    prefix_sum = 0
    # 哈希表: {前缀和: 出现次数}
    prefix_map = defaultdict(int)
    prefix_map[0] = 1  # 初始前缀和为 0，出现 1 次
    
    for num in nums:
        prefix_sum += num
        
        # 如果 prefix_sum - k 在哈希表中，说明存在子数组和为 k
        if prefix_sum - k in prefix_map:
            count += prefix_map[prefix_sum - k]
        
        # 将当前前缀和加入哈希表
        prefix_map[prefix_sum] += 1
    
    return count


def subarray_sum_brute_force(nums, k):
    """
    暴力法: 枚举所有子数组（不推荐，仅作对比）
    
    时间复杂度: O(n²)
    空间复杂度: O(1)
    """
    count = 0
    n = len(nums)
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            if current_sum == k:
                count += 1
    
    return count


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    nums1 = [1, 1, 1]
    k1 = 2
    print(f"输入: nums = {nums1}, k = {k1}")
    print(f"输出: {subarray_sum(nums1, k1)}")  # 期望: 2
    print()
    
    # 测试用例 2
    nums2 = [1, 2, 3]
    k2 = 3
    print(f"输入: nums = {nums2}, k = {k2}")
    print(f"输出: {subarray_sum(nums2, k2)}")  # 期望: 2
    print()
    
    # 测试用例 3
    nums3 = [1, -1, 0]
    k3 = 0
    print(f"输入: nums = {nums3}, k = {k3}")
    print(f"输出: {subarray_sum(nums3, k3)}")  # 期望: 3
    print()

