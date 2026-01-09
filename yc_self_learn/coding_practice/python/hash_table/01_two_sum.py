"""
LeetCode 1: Two Sum (两数之和)
难度: ⭐ Easy
标签: Array, Hash Table

题目描述:
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
你可以按任意顺序返回答案。

示例:
输入: nums = [2,7,11,15], target = 9
输出: [0,1]
解释: 因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

核心思路:
1. 暴力法: O(n²) - 两层循环遍历所有组合
2. 哈希表法: O(n) - 一次遍历，用哈希表存储已访问的元素
   - 遍历数组，对于每个元素 nums[i]，检查 target - nums[i] 是否在哈希表中
   - 如果在，返回两个索引
   - 如果不在，将当前元素和索引存入哈希表

时间复杂度: O(n)
空间复杂度: O(n)
"""


def two_sum(nums, target):
    """
    使用哈希表优化，一次遍历解决问题
    
    Args:
        nums: List[int] - 整数数组
        target: int - 目标值
    
    Returns:
        List[int] - 两个数的索引
    """
    # 哈希表: {值: 索引}
    hash_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # 如果补数在哈希表中，说明找到了
        if complement in hash_map:
            return [hash_map[complement], i]
        
        # 将当前元素存入哈希表
        hash_map[num] = i
    
    # 题目保证有解，这里不会执行到
    return []

    hesh_map = {}
    for i, num in enumerate(nums): 
        need = target - num 
        if need in hash_map:
            return [hash_map[need], i]
        hash_map[need] = i
    return []
    

def two_sum_brute_force(nums, target):
    """
    暴力法: 两层循环（不推荐，仅作对比）
    
    时间复杂度: O(n²)
    空间复杂度: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    nums1 = [2, 7, 11, 15]
    target1 = 9
    print(f"输入: nums = {nums1}, target = {target1}")
    print(f"输出: {two_sum(nums1, target1)}")  # 期望: [0, 1]
    print()
    
    # 测试用例 2
    nums2 = [3, 2, 4]
    target2 = 6
    print(f"输入: nums = {nums2}, target = {target2}")
    print(f"输出: {two_sum(nums2, target2)}")  # 期望: [1, 2]
    print()
    
    # 测试用例 3
    nums3 = [3, 3]
    target3 = 6
    print(f"输入: nums = {nums3}, target = {target3}")
    print(f"输出: {two_sum(nums3, target3)}")  # 期望: [0, 1]
    print()

