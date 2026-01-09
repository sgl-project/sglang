"""
LeetCode 49: Group Anagrams (字母异位词分组)
难度: ⭐⭐ Medium
标签: Hash Table, String, Sorting

题目描述:
给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。
字母异位词是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

示例:
输入: strs = ["eat","tea","tan","ate","nat","bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

核心思路:
1. 排序法: 将每个字符串排序后作为 key，相同 key 的字符串归为一组
   - 时间复杂度: O(n*k*log(k))，k 是字符串平均长度
2. 计数法: 统计每个字符串的字符频次，用频次作为 key
   - 时间复杂度: O(n*k)，更优

时间复杂度: O(n*k) 或 O(n*k*log(k))
空间复杂度: O(n*k)
"""

from collections import defaultdict


def group_anagrams(strs):
    """
    方法1: 排序法 - 将排序后的字符串作为 key
    
    思路:
    - 对每个字符串排序，排序后的字符串作为 key
    - 相同 key 的字符串就是字母异位词
    """
    groups = defaultdict(list)
    
    for s in strs:
        # 排序后的字符串作为 key
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())


def group_anagrams_count(strs):
    """
    方法2: 计数法 - 统计字符频次作为 key（更优）
    
    思路:
    - 统计每个字符串的字符频次
    - 用频次数组（转为字符串）作为 key
    - 相同频次的字符串就是字母异位词
    """
    groups = defaultdict(list)
    
    for s in strs:
        # 统计字符频次
        count = [0] * 26  # 假设只有小写字母
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # 将频次数组转为元组作为 key（列表不能作为字典 key）
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    strs1 = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"输入: {strs1}")
    result1 = group_anagrams(strs1)
    print(f"输出: {result1}")
    print()
    
    # 测试用例 2
    strs2 = [""]
    print(f"输入: {strs2}")
    result2 = group_anagrams(strs2)
    print(f"输出: {result2}")
    print()
    
    # 测试用例 3
    strs3 = ["a"]
    print(f"输入: {strs3}")
    result3 = group_anagrams(strs3)
    print(f"输出: {result3}")
    print()

