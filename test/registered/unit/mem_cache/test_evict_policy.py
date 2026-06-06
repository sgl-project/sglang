"""
evict_policy.py 的单元测试。

测试 7 种淘汰策略的正确性：
  LRU, LFU, FIFO, MRU, FILO, Priority, SLRU

每个策略的核心假设是：
  分数越小 → 越先被淘汰

所以测试验证两件事：
  1. get_priority 返回了正确的值（类型和内容）
  2. 多个节点排序后，淘汰顺序符合预期

运行方法（无需 GPU）：
  python test/registered/unit/mem_cache/test_evict_policy.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

# 注册为 CPU 测试，est_time=6 秒，放在 stage-a 阶段执行
register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)


# 在python中，函数名前加下划线通常表示这是一个私有辅助函数，只在当前测试文件内部使用。
# **kwargs：表示接收任意数量的关键字参数。这些参数会被自动打包成一个字典（Dictionary）。
# 例如调用：_make_node(last_access_time=100.0, hit_count=5)
# 此时 kwargs 内部就是：{"last_access_time": 100.0, "hit_count": 5}
def _make_node(**kwargs):
    """快速构造一个模拟 TreeNode，只设置策略需要的属性。

    用 MagicMock 而非真实 TreeNode，避免依赖整个 radix_cache 模块。
    这保证了单元测试的隔离性——即使 radix_cache.py 改了，测试也不受影响。
    """
    node = MagicMock() # 万能替身，任何属性都可以临时创建
    # 初始化evict_Priority.py中策略需要的属性，默认值为0或0.0
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    return node


# ============================================================
# 以下每个 TestCase 测试一种策略
# 模式：构造节点 → 调用 get_priority → 验证返回值或排序结果
# ============================================================


class TestLRUStrategy(unittest.TestCase):
    """最近最少使用：淘汰最久没被访问的节点"""

    def setUp(self):
        self.strategy = LRUStrategy()

    def test_priority_is_last_access_time(self):
        """LRU 的优先级就是节点的最后访问时间"""
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)

    def test_older_access_evicted_first(self):
        """旧访问（时间戳小）应该比新访问（时间戳大）先淘汰"""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    """最少频率使用：先淘汰被命中次数最少的节点"""

    def setUp(self):
        self.strategy = LFUStrategy()

    def test_priority_is_hit_count_and_time(self):
        """LFU 返回 (命中次数, 最后访问时间) 元组"""
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))

    def test_lower_hit_count_evicted_first(self):
        """命中次数少的节点优先淘汰"""
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        """相同命中次数时，更久未访问的优先淘汰（退化为 LRU）"""
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    """先进先出：最早创建的节点优先淘汰"""

    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_priority_is_creation_time(self):
        """FIFO 的优先级就是节点的创建时间"""
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)

    def test_earlier_created_evicted_first(self):
        """先创建的应该先淘汰"""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    """最近最常使用：最近被访问的反而优先淘汰（和 LRU 相反）"""

    def setUp(self):
        self.strategy = MRUStrategy()

    def test_priority_is_negated_access_time(self):
        """MRU 返回 -last_access_time（取反），所以最近访问的分数反而最低"""
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)

    def test_most_recently_used_evicted_first(self):
        """最近被访问的反而最先被淘汰"""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    """先进后出：最近创建的反而优先淘汰（和 FIFO 相反）"""

    def setUp(self):
        self.strategy = FILOStrategy()

    def test_priority_is_negated_creation_time(self):
        """FILO 返回 -creation_time"""
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)

    def test_last_created_evicted_first(self):
        """后创建的反而先淘汰"""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    """按业务优先级淘汰：优先级低的先踢，相同优先级的按 LRU"""

    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_priority_is_tuple(self):
        """返回 (priority, last_access_time) 元组"""
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))

    def test_lower_priority_evicted_first(self):
        """优先级数字小的先淘汰"""
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        """相同优先级时退化为 LRU：更久未访问的先淘汰"""
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    """分段 LRU：热数据（被多次访问）进入保护段，冷数据在考察段优先淘汰"""

    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        """命中次数 < threshold 的节点在 0 段（考察期，优先淘汰）"""
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        """命中次数 >= threshold 的节点在 1 段（保护期）"""
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        """访问过很多次的也在保护段"""
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        """考察期的节点（段 0）一定先于保护期（段 1）被淘汰"""
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        """同一段内，更久未访问的先淘汰"""
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        """自定义 threshold 能正常工作"""
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))

    def test_default_threshold_is_2(self):
        """不传参数时默认 threshold = 2"""
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)


class TestEvictionOrdering(unittest.TestCase):
    """集成式测试：构造多个节点，用 sorted() 排序，验证淘汰顺序"""

    def test_lru_ordering(self):
        """三个不同时间戳的节点，装进数组排序，最早的在前"""
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        """四类节点混合，验证 SLRU 的分段排序正确"""
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # 保护段，旧
            _make_node(hit_count=0, last_access_time=10.0),  # 考察段，新
            _make_node(hit_count=0, last_access_time=2.0),  # 考察段，旧 ← 最先淘汰
            _make_node(hit_count=3, last_access_time=8.0),  # 保护段，新 ← 最后淘汰
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # 考察段 old → 最先淘汰
            (0, 10.0),  # 考察段 new
            (1, 1.0),  # 保护段 old
            (1, 8.0),  # 保护段 new → 最后淘汰
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
