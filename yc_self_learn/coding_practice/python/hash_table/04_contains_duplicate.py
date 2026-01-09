"""
LeetCode 217: Contains Duplicate (存在重复元素)
难度: ⭐ Easy
标签: Array, Hash Table, Sorting

题目描述:
给你一个整数数组 nums。如果任一值在数组中出现至少两次，返回 true；如果数组中每个元素互不相同，返回 false。

示例:
输入: nums = [1,2,3,1]
输出: true

输入: nums = [1,2,3,4]
输出: false

核心思路:
1. 哈希集合法: O(n) - 遍历数组，用集合记录已访问的元素
2. 排序法: O(n*log(n)) - 排序后检查相邻元素
3. 集合长度法: O(n) - 将数组转为集合，比较长度

时间复杂度: O(n)
空间复杂度: O(n)
"""


def contains_duplicate(nums):
    """
    方法1: 哈希集合 - 遍历时检查是否已存在
    
    思路:
    - 遍历数组，对于每个元素，检查是否已在集合中
    - 如果在，说明有重复，返回 True
    - 如果不在，加入集合
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False


def contains_duplicate_set_length(nums):
    """
    方法2: 集合长度比较 - 最简洁的方法
    
    思路:
    - 将数组转为集合
    - 如果集合长度小于数组长度，说明有重复
    """
    return len(set(nums)) < len(nums)


def contains_duplicate_sort(nums):
    """
    方法3: 排序法 - 排序后检查相邻元素（不推荐，仅作对比）
    
    时间复杂度: O(n*log(n))
    空间复杂度: O(1)
    """
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            return True
    return False


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    nums1 = [1, 2, 3, 1]
    print(f"输入: {nums1}")
    print(f"输出: {contains_duplicate(nums1)}")  # 期望: True
    print()
    
    # 测试用例 2
    nums2 = [1, 2, 3, 4]
    print(f"输入: {nums2}")
    print(f"输出: {contains_duplicate(nums2)}")  # 期望: False
    print()
    
    # 测试用例 3
    nums3 = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
    print(f"输入: {nums3}")
    print(f"输出: {contains_duplicate(nums3)}")  # 期望: True
    print()

