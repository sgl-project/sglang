"""
mem_cache/utils.py 的单元测试。

测试覆盖：
  - get_eviction_strategy       (策略工厂：创建/大小写/未知策略)
  - get_hash_str                (token_ids → SHA256 hex)
  - hash_str_to_int64           (hex → signed int64)
  - split_node_hash_value        (分页 hash 拆分)
  - maybe_init_custom_mem_pool  (自定义内存池初始化：默认与启用路径)
  - compute_node_hash_values    (Radix Tree 节点哈希值计算)

运行方法（无需 GPU）：
  pytest test/registered/unit/mem_cache/test_mem_cache_utils.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_cpu_ci(est_time=9, suite="base-b-test-cpu")

import hashlib
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    get_hash_str,
    hash_str_to_int64,
    maybe_init_custom_mem_pool,
    split_node_hash_value,
)


# ============================================================================
# get_eviction_strategy（策略工厂函数）
# ============================================================================


class TestGetEvictionStrategy(unittest.TestCase):
    """get_eviction_strategy 根据字符串名创建对应的淘汰策略。"""

    def test_lru(self):
        self.assertIsInstance(get_eviction_strategy("lru"), LRUStrategy)

    def test_lfu(self):
        self.assertIsInstance(get_eviction_strategy("lfu"), LFUStrategy)

    def test_fifo(self):
        self.assertIsInstance(get_eviction_strategy("fifo"), FIFOStrategy)

    def test_mru(self):
        self.assertIsInstance(get_eviction_strategy("mru"), MRUStrategy)

    def test_filo(self):
        self.assertIsInstance(get_eviction_strategy("filo"), FILOStrategy)

    def test_priority(self):
        self.assertIsInstance(get_eviction_strategy("priority"), PriorityStrategy)

    def test_slru(self):
        self.assertIsInstance(get_eviction_strategy("slru"), SLRUStrategy)

    def test_case_insensitive(self):
        """大小写不敏感：'LRU' 和 'lru' 应该等价。"""
        self.assertIsInstance(get_eviction_strategy("LRU"), LRUStrategy)
        self.assertIsInstance(get_eviction_strategy("Lru"), LRUStrategy)
        self.assertIsInstance(get_eviction_strategy("FIFO"), FIFOStrategy)

    def test_unknown_policy_raises_valueerror(self):
        """未知策略名抛出 ValueError，消息中包含所有已知策略名。"""
        with self.assertRaises(ValueError) as ctx:
            get_eviction_strategy("nonexistent")
        msg = str(ctx.exception)
        self.assertIn("Unknown eviction policy", msg)
        self.assertIn("lru", msg)
        self.assertIn("lfu", msg)
        self.assertIn("fifo", msg)
        self.assertIn("mru", msg)
        self.assertIn("filo", msg)
        self.assertIn("priority", msg)
        self.assertIn("slru", msg)

    def test_each_call_creates_new_instance(self):
        """每次调用都返回新实例，而非共享单例。"""
        s1 = get_eviction_strategy("lru")
        s2 = get_eviction_strategy("lru")
        self.assertIsNot(s1, s2)


# ============================================================================
# maybe_init_custom_mem_pool（自定义内存池初始化）
# ============================================================================


class TestMaybeInitCustomMemPool(unittest.TestCase):
    """maybe_init_custom_mem_pool 在特定环境变量下初始化自定义内存池。"""

    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    def test_disabled_by_default(self, mock_env_get):
        """默认情况下，未设置环境变量时不启用自定义内存池。"""
        mock_env_get.return_value = None
        enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertFalse(enabled)
        self.assertIsNone(pool)
        self.assertIsNone(pool_type)

    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    @patch("sglang.srt.disaggregation.mooncake.utils.init_mooncake_custom_mem_pool")
    def test_enabled_via_env(self, mock_init, mock_env_get):
        """设置了环境变量时，应当调用 Mooncake 初始化函数并返回其结果。"""
        mock_env_get.return_value = "enabled"
        mock_init.return_value = (True, "mock_pool_instance", "mooncake")

        enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertTrue(enabled)
        self.assertEqual(pool, "mock_pool_instance")
        self.assertEqual(pool_type, "mooncake")
        mock_init.assert_called_once_with("cuda:0")


# ============================================================================
# get_hash_str（token_ids → SHA256 hex string）
# ============================================================================


class TestGetHashStr(unittest.TestCase):
    """get_hash_str 将 token ID 列表转换为确定性 SHA256 哈希字符串。"""

    def test_empty_list(self):
        """空 token 列表的 hash 是固定的（SHA256 of nothing）。"""
        result = get_hash_str([])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
        expected = hashlib.sha256().hexdigest()
        self.assertEqual(result, expected)

    def test_single_token(self):
        """单个 token 产生确定性 hash。"""
        h1 = get_hash_str([42])
        h2 = get_hash_str([42])
        self.assertEqual(h1, h2)

    def test_multiple_tokens(self):
        """多个 token 产生确定性 hash。"""
        h1 = get_hash_str([1, 2, 3])
        h2 = get_hash_str([1, 2, 3])
        self.assertEqual(h1, h2)

    def test_different_sequence_different_hash(self):
        """不同 token 序列产生不同 hash。"""
        h1 = get_hash_str([1, 2, 3])
        h2 = get_hash_str([3, 2, 1])
        self.assertNotEqual(h1, h2)

    def test_different_values_different_hash(self):
        """不同 token 值产生不同 hash。"""
        h1 = get_hash_str([100])
        h2 = get_hash_str([200])
        self.assertNotEqual(h1, h2)

    def test_bigram_tuple_tokens(self):
        """EAGLE bigram 模式：token 是 tuple，两个元素都被 hash。"""
        h1 = get_hash_str([(1, 2), (3, 4)])
        h2 = get_hash_str([(1, 2), (3, 4)])
        self.assertEqual(h1, h2)

    def test_bigram_vs_flat_token_equivalent(self):
        """bigram tuple 和等价的 int 序列产生相同的 hash。"""
        h_flat = get_hash_str([1, 2])
        h_bigram = get_hash_str([(1, 2)])
        self.assertEqual(h_flat, h_bigram)

    def test_bigram_order_matters(self):
        """bigram tuple 的顺序影响 hash。"""
        h1 = get_hash_str([(1, 2)])
        h2 = get_hash_str([(2, 1)])
        self.assertNotEqual(h1, h2)

    def test_prior_hash_chaining(self):
        """prior_hash 相当于在上次的 hash 基础上继续 hash。"""
        first = get_hash_str([1, 2])
        second = get_hash_str([3, 4], prior_hash=first)

        ref = hashlib.sha256()
        ref.update(bytes.fromhex(first))
        ref.update((3).to_bytes(4, byteorder="little", signed=False))
        ref.update((4).to_bytes(4, byteorder="little", signed=False))
        self.assertEqual(second, ref.hexdigest())

    def test_prior_hash_single_step(self):
        """分步哈希链模式与单次展开哈希应该生成不同的结果（设计预期）。"""
        step1 = get_hash_str([1])
        step2 = get_hash_str([2], prior_hash=step1)
        direct = get_hash_str([1, 2])
        self.assertNotEqual(step2, direct)

    def test_returns_64_char_hex(self):
        """返回值是 64 字符的 hex 字符串。"""
        for tokens in [[], [1], [1, 2, 3], [(1, 2)], [1, 2, 3, 4, 5]]:
            result = get_hash_str(tokens)
            self.assertIsInstance(result, str)
            self.assertEqual(len(result), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in result))


# ============================================================================
# hash_str_to_int64（hex string → signed int64）
# ============================================================================


class TestHashStrToInt64(unittest.TestCase):
    """hash_str_to_int64 将 SHA256 hex 字符串的前 16 个字符转为 signed int64。"""

    def test_zero_hash(self):
        """全零 hash → int64 值为 0。"""
        result = hash_str_to_int64("0" * 64)
        self.assertEqual(result, 0)

    def test_small_positive_value(self):
        """小的 hex 值直接映射为正整数。"""
        result = hash_str_to_int64("0000000000000001" + "0" * 48)
        self.assertEqual(result, 1)

    def test_large_positive_value(self):
        """小于 2^63 的值映射为正整数。"""
        result = hash_str_to_int64("7fffffffffffffff" + "0" * 48)
        self.assertEqual(result, 2**63 - 1)

    def test_negative_overflow(self):
        """>= 2^63 的值映射为负整数（signed int64 溢出表现）。"""
        result = hash_str_to_int64("8000000000000000" + "0" * 48)
        self.assertEqual(result, -(2**63))

    def test_max_unsigned_maps_to_minus_one(self):
        """最大 uint64 值映射为 -1。"""
        result = hash_str_to_int64("f" * 16 + "0" * 48)
        self.assertEqual(result, -1)

    def test_only_first_16_chars_matter(self):
        """只取前 16 个 hex 字符。"""
        h1 = hash_str_to_int64("a" * 16 + "0" * 48)
        h2 = hash_str_to_int64("a" * 16 + "f" * 48)
        self.assertEqual(h1, h2)

    def test_roundtrip_with_get_hash_str(self):
        """get_hash_str → hash_str_to_int64 链路测试。"""
        hash_hex = get_hash_str([42, 99])
        int64_val = hash_str_to_int64(hash_hex)
        self.assertIsInstance(int64_val, int)
        self.assertTrue(-(2**63) <= int64_val < 2**63)


# ============================================================================
# compute_node_hash_values（节点 hash 值计算）
# ============================================================================


class TestComputeNodeHashValues(unittest.TestCase):
    """compute_node_hash_values 为 radix tree 节点计算分页 SHA256 hash 列表。"""

    def setUp(self):
        def mock_hash_page(start, end, parent_hash):
            parts = [f"p{start}-{end}"]
            if parent_hash is not None:
                parts.append(parent_hash)
            return "-".join(parts)

        self.mock_hash_page = mock_hash_page

    def _make_node(self, key_len, parent=None, parent_hash_values=None):
        """构造一个用于测试的模拟 TreeNode。"""
        node = MagicMock()
        node.key.__len__.return_value = key_len
        node.key.hash_page = self.mock_hash_page
        node.parent = parent
        if parent is not None:
            parent.hash_value = parent_hash_values
        return node

    def test_single_page_root(self):
        """单页 root 节点：一个 page 覆盖全部 token。"""
        node = self._make_node(key_len=3)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(len(result), 1)
        self.assertIn("p0-3", result[0])

    def test_multiple_pages(self):
        """多个 page 的节点：每个 page 生成一个 hash。"""
        node = self._make_node(key_len=32)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(len(result), 2)

    def test_page_aligned_boundary(self):
        """key 长度刚好是 page_size 的整数倍。"""
        node = self._make_node(key_len=32)
        result = compute_node_hash_values(node, page_size=8)
        self.assertEqual(len(result), 4)

    def test_key_shorter_than_page_size(self):
        """key 长度小于 page_size 时只有 1 页。"""
        node = self._make_node(key_len=5)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(len(result), 1)

    def test_chained_parent_hash(self):
        """有父节点时，第一个 page 使用 parent 的最后一个 hash 作为种子。"""
        parent = MagicMock()
        parent.key.__len__.return_value = 8
        parent.hash_value = ["parent_hash_0", "parent_hash_1"]
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(
            key_len=16, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 2)
        self.assertIn("parent_hash_1", result[0])

    def test_parent_with_empty_key(self):
        """parent key 为空时不继承 hash（len(key) == 0）。"""
        parent = MagicMock()
        parent.key.__len__.return_value = 0
        parent.hash_value = ["some_hash"]
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(
            key_len=8, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 1)
        self.assertNotIn("some_hash", result[0])

    def test_parent_without_hash_value(self):
        """parent 没有 hash_value 属性或为空列表时正常计算。"""
        parent = MagicMock()
        parent.key.__len__.return_value = 8
        parent.hash_value = []
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(key_len=8, parent=parent, parent_hash_values=[])
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 1)


# ============================================================================
# split_node_hash_value（分页 hash 拆分）
# ============================================================================


class TestSplitNodeHashValue(unittest.TestCase):
    """split_node_hash_value 在 radix tree 节点分裂时拆分 hash 列表。"""

    def test_none_input_returns_none_tuple(self):
        """输入 None 时返回 (None, None)。"""
        result = split_node_hash_value(None, 10, 4)
        self.assertEqual(result, (None, None))

    def test_page_size_one_split(self):
        """page_size=1 时，split_len 直接决定拆分位置。"""
        hash_values = [f"hash_{i}" for i in range(4)]
        new_hash, child_hash = split_node_hash_value(hash_values, 2, 1)
        self.assertEqual(new_hash, ["hash_0", "hash_1"])
        self.assertEqual(child_hash, ["hash_2", "hash_3"])

    def test_page_size_one_split_at_zero(self):
        """split_len=0：parent 取 0 个，child 保留全部。"""
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 0, 1)
        self.assertEqual(new_hash, [])
        self.assertEqual(child_hash, ["a", "b", "c"])

    def test_page_size_one_split_at_len(self):
        """split_len=len：parent 取全部，child 为空。"""
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 3, 1)
        self.assertEqual(new_hash, ["a", "b", "c"])
        self.assertEqual(child_hash, [])

    def test_page_size_larger_than_one(self):
        """page_size > 1 时，拆分按页边界对齐。"""
        hash_values = ["p0", "p1", "p2", "p3", "p4", "p5"]
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 2)
        self.assertEqual(new_hash, ["p0", "p1"])
        self.assertEqual(child_hash, ["p2", "p3", "p4", "p5"])

    def test_page_size_split_exact_page_boundary(self):
        """split_len 正好落在页边界时，精确分页。"""
        hash_values = ["a", "b", "c", "d"]
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c", "d"])

    def test_page_size_split_within_page(self):
        """split_len 在页中间时，按向下取整逻辑划分 page。"""
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 6, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c"])

    def test_total_length_preserved(self):
        """拆分后 parent + child 的总长度等于原列表长度。"""
        hash_values = [f"h_{i}" for i in range(10)]
        for split_len in [0, 1, 3, 5, 7, 9, 10]:
            for page_size in [1, 2, 4]:
                new_hash, child_hash = split_node_hash_value(
                    hash_values, split_len, page_size
                )
                self.assertEqual(
                    len(new_hash) + len(child_hash),
                    len(hash_values),
                    f"split_len={split_len}, page_size={page_size}",
                )


if __name__ == "__main__":
    unittest.main()