"""
LeetCode 242: Valid Anagram (有效的字母异位词)
难度: ⭐ Easy
标签: Hash Table, String, Sorting

题目描述:
给定两个字符串 s 和 t，编写一个函数来判断 t 是否是 s 的字母异位词。
注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

示例:
输入: s = "anagram", t = "nagaram"
输出: true

输入: s = "rat", t = "car"
输出: false

核心思路:
1. 排序法: O(n*log(n)) - 排序后比较
2. 哈希表计数法: O(n) - 统计字符频次
3. 数组计数法: O(n) - 用数组代替哈希表（如果只有小写字母）

时间复杂度: O(n)
空间复杂度: O(1) - 字符集大小固定
"""

from collections import Counter


def is_anagram(s, t):
    """
    方法1: 哈希表计数 - 使用 Counter
    
    思路:
    - 统计两个字符串的字符频次
    - 比较频次是否相同
    """
    return Counter(s) == Counter(t)


def is_anagram_dict(s, t):
    """
    方法2: 手动实现哈希表计数
    
    思路:
    - 手动统计字符频次
    - 比较两个字典是否相同
    """
    if len(s) != len(t):
        return False
    
    count_s = {}
    count_t = {}
    
    for char in s:
        count_s[char] = count_s.get(char, 0) + 1
    
    for char in t:
        count_t[char] = count_t.get(char, 0) + 1
    
    return count_s == count_t


def is_anagram_array(s, t):
    """
    方法3: 数组计数 - 最优方法（如果只有小写字母）
    
    思路:
    - 使用长度为 26 的数组统计字符频次
    - 遍历 s 时增加计数，遍历 t 时减少计数
    - 最后检查数组是否全为 0
    """
    if len(s) != len(t):
        return False
    
    count = [0] * 26  # 假设只有小写字母
    
    # 统计 s 的字符频次
    for char in s:
        count[ord(char) - ord('a')] += 1
    
    # 减去 t 的字符频次
    for char in t:
        count[ord(char) - ord('a')] -= 1
    
    # 检查是否全为 0
    return all(c == 0 for c in count)


def is_anagram_sort(s, t):
    """
    方法4: 排序法 - 排序后比较（不推荐，仅作对比）
    
    时间复杂度: O(n*log(n))
    空间复杂度: O(1)
    """
    return sorted(s) == sorted(t)


# 测试用例
if __name__ == "__main__":
    # 测试用例 1
    s1, t1 = "anagram", "nagaram"
    print(f"输入: s = '{s1}', t = '{t1}'")
    print(f"输出: {is_anagram(s1, t1)}")  # 期望: True
    print()
    
    # 测试用例 2
    s2, t2 = "rat", "car"
    print(f"输入: s = '{s2}', t = '{t2}'")
    print(f"输出: {is_anagram(s2, t2)}")  # 期望: False
    print()
    
    # 测试用例 3
    s3, t3 = "listen", "silent"
    print(f"输入: s = '{s3}', t = '{t3}'")
    print(f"输出: {is_anagram(s3, t3)}")  # 期望: True
    print()

