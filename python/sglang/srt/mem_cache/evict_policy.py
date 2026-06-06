"""
KV Cache 淘汰策略模块。
=======================

当 GPU 显存中的 KV Cache 池满了，需要决定"踢掉谁"来腾空间。
这个模块定义了多种淘汰策略，每种策略通过给节点打分（get_priority）
来决定淘汰顺序：分数越小的节点越先被踢掉。

核心思想：
  - 所有策略都实现了同一个接口 EvictionStrategy
  - get_priority 返回一个可比较的值（float 或 tuple）
  - 返回值的比较规则：tuple 按字典序比较，即先比第一个元素，相等再比第二个
  - 缓存系统调用 sorted(nodes, key=strategy.get_priority)，取最前面的淘汰

支持的策略一览：
  ┌─────────────────┬──────────────────────────────────┐
  │ 策略             │ 淘汰逻辑                          │
  ├─────────────────┼──────────────────────────────────┤
  │ LRU             │ 最久没被访问的先踢                  │
  │ LFU             │ 被命中次数最少的先踢（平局看 LRU）   │
  │ FIFO            │ 最早创建的先踢                      │
  │ MRU             │ 最近刚访问的先踢（和 LRU 反过来）     │
  │ FILO            │ 最近创建的先踢                      │
  │ Priority        │ 优先级低的先踢（平局看 LRU）          │
  │ SLRU            │ 分两段：冷数据先踢，热数据保护         │
  └─────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    """淘汰策略的抽象基类:主要作用是定义规范（契约），告诉所有继承它的子类：
    你们必须实现 get_priority 这个方法，并且它的输入和输出格式应该符合我的规定。

    所有淘汰策略都要实现 get_priority 方法。
    返回值小的节点会优先被淘汰。
    """

    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        """返回节点的淘汰优先级。值越小 → 越先被淘汰。

        返回值可以是：
          - float: 直接按数值比较（LRU、MRU、FIFO、FILO）
          - tuple:  按字典序比较，如 (segment, timestamp) 先比 segment 再比时间
        """
        pass


class LRUStrategy(EvictionStrategy):
    """最近最少使用（Least Recently Used）。

    直接返回节点的 last_access_time。时间戳越小 → 越久没被访问 → 优先淘汰。
    这是最经典的缓存淘汰策略。
    """

    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    """最少频率使用（Least Frequently Used）。

    返回 (hit_count, last_access_time) 元组。先比命中次数，次数相同的再按 LRU 比。
    这意味着：
      - 被访问 1 次的节点会先于被访问 100 次的被淘汰
      - 相同命中次数时，更久没被访问的先淘汰
    """

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    """先进先出（First In First Out）。

    按节点的创建时间排序，最早创建的最先被淘汰。
    不管之后被访问了多少次，只看创建时间。
    """

    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    """最近最常使用（Most Recently Used）。

    和 LRU 正相反：返回 -last_access_time，取反后最近访问的节点分数最低。
    这意味着刚被用过的反而最先被淘汰。
    适用于某些负载模式：新请求不太可能复用最近用过的缓存。
    """

    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    """先进后出（First In Last Out）。

    和 FIFO 正相反：返回 -creation_time，最近创建的节点分数最低。
    即后创建的反而先淘汰。
    """

    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """按业务优先级淘汰。

    返回 (priority, last_access_time) 元组。
    priority 值小的先淘汰（优先级 1 比优先级 5 先踢）。
    相同优先级内，更久没被访问的先淘汰。

    这允许上游给不同请求的缓存打上不同的重要性标签。
    """

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    """分段 LRU（Segmented LRU）。

    把缓存分为两段：
      段 0（考察期 probationary）: hit_count < threshold → 容易被淘汰
      段 1（保护期 protected）:    hit_count >= threshold → 受保护，不易被淘汰

    比较逻辑（字典序）：
      - (0, timestamp) < (1, timestamp)  → 段 0 的节点总是先于段 1 被淘汰
      - 同一段内，更久没被访问的先淘汰

    默认 threshold=2，意味着一个节点被访问 2 次后就从"考察期"升到"保护期"。
    """

    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
