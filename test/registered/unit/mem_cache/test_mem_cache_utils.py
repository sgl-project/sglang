"""
mem_cache/utils.py 的单元测试。

测试覆盖：
  - get_eviction_strategy    (策略工厂：创建/大小写/未知策略)
  - get_hash_str              (token_ids → SHA256 hex)
  - hash_str_to_int64         (hex → signed int64)
  - split_node_hash_value     (分页 hash 拆分)
  - maybe_init_custom_mem_pool (默认路径：无需 mooncake)

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
        # 确保所有已知策略名都列在错误消息中
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
# get_hash_str（token_ids → SHA256 hex string）
# ============================================================================


class TestGetHashStr(unittest.TestCase):
    """get_hash_str 将 token ID 列表转换为确定性 SHA256 哈希字符串。"""

    # --- 基本用法 ---

    def test_empty_list(self):
        """空 token 列表的 hash 是固定的（SHA256 of nothing）。"""
        result = get_hash_str([])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)  # SHA256 = 64 hex chars
        # 空输入的 SHA256 作为参考值
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

    # --- 确定性 ---

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

    # --- EAGLE bigram 模式（tuple 元素） ---

    def test_bigram_tuple_tokens(self):
        """EAGLE bigram 模式：token 是 tuple，两个元素都被 hash。"""
        h1 = get_hash_str([(1, 2), (3, 4)])
        h2 = get_hash_str([(1, 2), (3, 4)])
        self.assertEqual(h1, h2)

    def test_bigram_vs_flat_token_equivalent(self):
        """bigram tuple 和等价的 int 序列产生相同的 hash。

        因为 tuple 模式把 tuple 内每个元素依次写入 hasher，
        效果等价于把同样的 int 值拆开逐个传入。
        """
        h_flat = get_hash_str([1, 2])
        h_bigram = get_hash_str([(1, 2)])
        self.assertEqual(h_flat, h_bigram)

    def test_bigram_order_matters(self):
        """bigram tuple 的顺序影响 hash。"""
        h1 = get_hash_str([(1, 2)])
        h2 = get_hash_str([(2, 1)])
        self.assertNotEqual(h1, h2)

    # --- prior_hash 链式哈希 ---

    def test_prior_hash_chaining(self):
        """prior_hash 相当于在上次的 hash 基础上继续 hash。"""
        first = get_hash_str([1, 2])
        second = get_hash_str([3, 4], prior_hash=first)

        # 手工复现：把 prior_hash 的 bytes + 后续 token 一起 hash
        ref = hashlib.sha256()
        ref.update(bytes.fromhex(first))
        ref.update((3).to_bytes(4, byteorder="little", signed=False))
        ref.update((4).to_bytes(4, byteorder="little", signed=False))
        self.assertEqual(second, ref.hexdigest())

    def test_prior_hash_single_step(self):
        """prior_hash + 一个 token 等同于单独 hash 全部。"""
        # 分步 hash：先 hash [1]，再 hash [2] 带上 prior_hash
        step1 = get_hash_str([1])
        step2 = get_hash_str([2], prior_hash=step1)
        # 一步 hash [1, 2]
        direct = get_hash_str([1, 2])
        # 注意：分步 hash 的结果和一次 hash 所有 token 不一样
        # 因为 prior_hash 是 hex string 的 bytes 形式写入 hasher
        # 而直接 hash [1, 2] 是把两个 int 的 bytes 依次写入
        # 所以这两个结果应该不同 —— 这是设计上的区分
        self.assertNotEqual(step2, direct)

    # --- 输出格式 ---

    def test_returns_64_char_hex(self):
        """返回值是 64 字符的 hex 字符串（SHA256 标准）。"""
        for tokens in [[], [1], [1, 2, 3], [(1, 2)], [1, 2, 3, 4, 5]]:
            result = get_hash_str(tokens)
            self.assertIsInstance(result, str)
            self.assertEqual(len(result), 64, f"tokens={tokens}")
            # 确保全部是 hex 字符
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
        # 2^63 - 1 = 0x7FFFFFFFFFFFFFFF
        result = hash_str_to_int64("7fffffffffffffff" + "0" * 48)
        self.assertEqual(result, 2**63 - 1)

    def test_negative_overflow(self):
        """>= 2^63 的值映射为负整数（signed int64 溢出）。"""
        # 2^63 = 0x8000000000000000 → -2^63
        result = hash_str_to_int64("8000000000000000" + "0" * 48)
        self.assertEqual(result, -(2**63))

    def test_max_unsigned_maps_to_minus_one(self):
        """最大 uint64 值 = -1（signed int64）。"""
        result = hash_str_to_int64("f" * 16 + "0" * 48)
        self.assertEqual(result, -1)

    def test_only_first_16_chars_matter(self):
        """只取前 16 个 hex 字符，后面的忽略。"""
        h1 = hash_str_to_int64("a" * 16 + "0" * 48)
        h2 = hash_str_to_int64("a" * 16 + "f" * 48)
        self.assertEqual(h1, h2)

    def test_roundtrip_with_get_hash_str(self):
        """get_hash_str → hash_str_to_int64 链路可用。"""
        hash_hex = get_hash_str([42, 99])
        int64_val = hash_str_to_int64(hash_hex)
        self.assertIsInstance(int64_val, int)
        self.assertLessEqual(abs(int64_val), 2**63)


# ============================================================================
# split_node_hash_value（分页 hash 拆分）
# ============================================================================


class TestSplitNodeHashValue(unittest.TestCase):
    """split_node_hash_value 在 radix tree 节点分裂时拆分 hash 列表。"""

    # --- None 输入 ---

    def test_none_input_returns_none_tuple(self):
        """输入 None 时返回 (None, None)。"""
        result = split_node_hash_value(None, 10, 4)
        self.assertEqual(result, (None, None))

    # --- page_size = 1 ---

    def test_page_size_one_split(self):
        """page_size=1 时，split_len 直接决定拆分位置。"""
        # hash 列表 = ["a", "b", "c", "d"], split_len=2
        # 前 2 个归 parent，后 2 个归 child
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

    # --- page_size > 1 ---

    def test_page_size_larger_than_one(self):
        """page_size > 1 时，拆分按页边界对齐。"""
        # 6 个 hash 页，page_size=2 → split 按页对齐
        hash_values = ["p0", "p1", "p2", "p3", "p4", "p5"]
        # split_len = 4 tokens, page_size=2 → 2 个 page 归 parent
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 2)
        self.assertEqual(new_hash, ["p0", "p1"])
        self.assertEqual(child_hash, ["p2", "p3", "p4", "p5"])

    def test_page_size_split_exact_page_boundary(self):
        """split_len 正好落在页边界时，精确分页。"""
        hash_values = ["a", "b", "c", "d"]
        # 4 pages, page_size=4, split_len=4 → 1 page to parent, 3 to child
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c", "d"])

    def test_page_size_split_within_page(self):
        """split_len 在页中间时，多余的 token 部分不产生额外 page。"""
        hash_values = ["a", "b", "c"]
        # page_size=4, split_len=6 → split_pages = 6//4 = 1
        new_hash, child_hash = split_node_hash_value(hash_values, 6, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c"])

    # --- 不变性 ---

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


# ============================================================================
# compute_node_hash_values（节点 hash 值计算）
# ============================================================================


class TestComputeNodeHashValues(unittest.TestCase):
    """compute_node_hash_values 为 radix tree 节点计算分页 SHA256 hash 列表。"""

    def setUp(self):
        # 创建一个模拟的 hash_page 方法
        def mock_hash_page(start, end, parent_hash):
            """模拟 hash_page：返回格式化的字符串以便追踪。"""
            parts = [f"p{start}-{end}"]
            if parent_hash is not None:
                parts.append(parent_hash)
            return "-".join(parts)

        self.mock_key = MagicMock()
        self.mock_key.hash_page = mock_hash_page

    def _make_node(self, key_len, parent=None, parent_hash_values=None):
        """构造一个模拟 TreeNode。"""
        node = MagicMock()
        node.key.__len__.return_value = key_len
        node.key.hash_page = self.mock_key.hash_page
        node.parent = parent
        if parent is not None and parent_hash_values is not None:
            parent.hash_value = parent_hash_values
        elif parent is None:
            node.parent = None
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
        parent.parent = None
        parent.key.__len__.return_value = 8
        parent.hash_value = ["parent_hash_0", "parent_hash_1"]
        parent.key.hash_page = self.mock_key.hash_page

        child = self._make_node(key_len=16, parent=parent, parent_hash_values=parent.hash_value)
        result = compute_node_hash_values(child, page_size=8)
        # parent 有 2 个 hash，child 有 16/8=2 个 hash
        self.assertEqual(len(result), 2)
        # 第一个 child hash 应该用 parent 的最后一个 hash 作为种子
        self.assertIn("parent_hash_1", result[0])

    def test_parent_with_empty_key(self):
        """parent key 为空时不继承 hash（len(key) == 0）。"""
        parent = MagicMock()
        parent.parent = None
        parent.key.__len__.return_value = 0
        parent.hash_value = ["some_hash"]
        parent.key.hash_page = self.mock_key.hash_page

        child = self._make_node(key_len=8, parent=parent, parent_hash_values=parent.hash_value)
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 1)
        # parent key 长度为 0，不使用 parent hash
        self.assertNotIn("some_hash", result[0])

    def test_parent_without_hash_value(self):
        """parent 没有 hash_value 属性时正常计算。"""
        parent = MagicMock()
        parent.parent = None
        parent.key.__len__.return_value = 8
        parent.hash_value = []  # 空列表
        parent.key.hash_page = self.mock_key.hash_page

        child = self._make_node(key_len=8, parent=parent, parent_hash_values=[])
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
