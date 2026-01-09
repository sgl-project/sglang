"""
LeetCode 706: Design HashMap (设计哈希映射)
难度: ⭐⭐ Medium
标签: Array, Hash Table, Linked List, Design

题目描述:
不使用任何内建的哈希表库设计一个哈希映射（HashMap）。
实现 MyHashMap 类：
- MyHashMap() 用空映射初始化对象
- void put(int key, int value) 向 HashMap 插入一个键值对 (key, value)
- int get(int key) 返回特定的 key 所映射的 value；如果映射中不包含 key 的映射，返回 -1
- void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应的 value

示例:
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found)
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found)

核心思路:
1. 数组法: 简单直接，但空间浪费大（key 范围很大时）
2. 链地址法: 使用数组 + 链表，处理哈希冲突
3. 开放寻址法: 使用数组，冲突时线性探测

时间复杂度: 平均 O(1)，最坏 O(n)
空间复杂度: O(n)
"""


class MyHashMap:
    """
    方法1: 链地址法 - 使用数组 + 链表处理冲突
    
    设计要点:
    - 使用固定大小的数组作为桶（bucket）
    - 每个桶存储一个链表，处理哈希冲突
    - 哈希函数: key % bucket_size
    """
    
    def __init__(self):
        """初始化哈希映射"""
        self.bucket_size = 1000
        self.buckets = [[] for _ in range(self.bucket_size)]
    
    def _hash(self, key):
        """哈希函数"""
        return key % self.bucket_size
    
    def put(self, key: int, value: int) -> None:
        """插入或更新键值对"""
        bucket = self.buckets[self._hash(key)]
        
        # 查找是否已存在该 key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                # 更新值
                bucket[i] = (key, value)
                return
        
        # 不存在，添加新键值对
        bucket.append((key, value))
    
    def get(self, key: int) -> int:
        """获取值，不存在返回 -1"""
        bucket = self.buckets[self._hash(key)]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return -1
    
    def remove(self, key: int) -> None:
        """删除键值对"""
        bucket = self.buckets[self._hash(key)]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return


class MyHashMapSimple:
    """
    方法2: 简单数组法 - 适用于 key 范围较小的情况
    
    注意: 如果 key 范围很大，会浪费大量空间
    """
    
    def __init__(self):
        """初始化，使用足够大的数组"""
        self.size = 1000001
        self.data = [-1] * self.size
    
    def put(self, key: int, value: int) -> None:
        """插入或更新键值对"""
        self.data[key] = value
    
    def get(self, key: int) -> int:
        """获取值，不存在返回 -1"""
        return self.data[key]
    
    def remove(self, key: int) -> None:
        """删除键值对"""
        self.data[key] = -1


# 测试用例
if __name__ == "__main__":
    print("=== 测试链地址法 ===")
    myHashMap = MyHashMap()
    
    myHashMap.put(1, 1)
    myHashMap.put(2, 2)
    print(f"get(1): {myHashMap.get(1)}")  # 期望: 1
    print(f"get(3): {myHashMap.get(3)}")  # 期望: -1
    
    myHashMap.put(2, 1)  # 更新
    print(f"get(2): {myHashMap.get(2)}")  # 期望: 1
    
    myHashMap.remove(2)
    print(f"get(2): {myHashMap.get(2)}")  # 期望: -1
    print()
    
    print("=== 测试简单数组法 ===")
    myHashMap2 = MyHashMapSimple()
    
    myHashMap2.put(1, 1)
    myHashMap2.put(2, 2)
    print(f"get(1): {myHashMap2.get(1)}")  # 期望: 1
    print(f"get(3): {myHashMap2.get(3)}")  # 期望: -1
    
    myHashMap2.put(2, 1)  # 更新
    print(f"get(2): {myHashMap2.get(2)}")  # 期望: 1
    
    myHashMap2.remove(2)
    print(f"get(2): {myHashMap2.get(2)}")  # 期望: -1
    print()

