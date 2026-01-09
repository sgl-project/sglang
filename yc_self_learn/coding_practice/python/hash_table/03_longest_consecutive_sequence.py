"""
LeetCode 128: Longest Consecutive Sequence (最长连续序列)
难度: ⭐⭐⭐ Hard
标签: Array, Hash Table, Union Find

题目描述:
给定一个未排序的整数数组 nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

示例:
输入: nums = [100,4,200,1,3,2]
输出: 4
解释: 最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

核心思路:
1. 排序法: O(n*log(n)) - 排序后遍历（不满足 O(n) 要求）
2. 哈希表法: O(n) - 用集合存储所有数字，只从序列起点开始扩展
   - 将所有数字存入集合
   - 遍历数组，对于每个数字，如果它是序列起点（num-1 不在集合中），则开始扩展
   - 从起点开始，不断检查 num+1 是否在集合中，计算序列长度
   - 更新最长序列长度

时间复杂度: O(n) - 每个数字最多被访问两次
空间复杂度: O(n)
"""


def longest_consecutive(nums):
    """
    使用哈希集合优化，只从序列起点开始扩展
    
    关键点:
    - 只从序列的起点（num-1 不在集合中）开始扩展
    - 这样每个数字最多被访问两次（一次在遍历，一次在扩展）
    """
    if not nums:
        return 0
    
    # 将所有数字存入集合，O(1) 查找
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # 只从序列起点开始扩展（num-1 不在集合中）
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # 向右扩展，计算序列长度
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            # 更新最长序列长度
            max_length = max(max_length, current_length)
    
    return max_length


def longest_consecutive_naive(nums):
    """
    朴素方法: 排序后遍历（不满足 O(n) 要求，仅作对比）
    
    时间复杂度: O(n*log(n))
    空间复杂度: O(1)
    """
    if not nums:
        return 0
    
    nums.sort()
    max_length = 1
    current_length = 1
    
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            current_length += 1
        elif nums[i] != nums[i-1]:  # 跳过重复数字
            current_length = 1
        
        max_length = max(max_length, current_length)
    
    return max_length


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    nums1 = [100, 4, 200, 1, 3, 2]
    print(f"输入: {nums1}")
    print(f"输出: {longest_consecutive(nums1)}")  # 期望: 4
    print()
    
    # 测试用例 2
    nums2 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    print(f"输入: {nums2}")
    print(f"输出: {longest_consecutive(nums2)}")  # 期望: 9
    print()
    
    # 测试用例 3
    nums3 = []
    print(f"输入: {nums3}")
    print(f"输出: {longest_consecutive(nums3)}")  # 期望: 0
    print()

