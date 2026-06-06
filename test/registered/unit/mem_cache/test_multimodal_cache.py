"""
Unit tests for multimodal_cache.py — LRU multimodal embedding cache.

Tests cover:
  - MultimodalCache.combine_hashes   (纯函数，合并 hash 列表)
  - _get_tensor_size                 (计算 tensor 字节数)
  - MultiModalStaticCache            (基于 OrderedDict 的 LRU 缓存)
      - set：存入新 key、覆盖已有 key、触发 LRU 淘汰、超限拒绝
      - get / get_single：命中、未命中
      - has：存在性检查（不改变 LRU 顺序）
      - free：删除条目
      - clear：清空缓存
      - available_size：实时追踪条目数
      - LRU 淘汰顺序验证

运行方法（无需 GPU）：
  pytest test/registered/unit/mem_cache/test_multimodal_cache.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
    MultimodalCache,
    _get_tensor_size,
)


def _make_embedding(num_bytes: int) -> EmbeddingResult:
    """创建一个指定字节大小的 EmbeddingResult（float32，CPU tensor）。"""
    numel = max(1, num_bytes // 4)  # float32 = 4 bytes/element
    return EmbeddingResult(embedding=torch.zeros(numel, dtype=torch.float32))


# ============================================================================
# MultimodalCache.combine_hashes（静态方法，纯函数逻辑）
# ============================================================================


class TestCombineHashes(unittest.TestCase):
    """combine_hashes 将多张图片的 hash 合并为一个缓存 key。"""

    def test_empty_list_returns_none(self):
        """空列表应该返回 None（没有内容可以缓存）。"""
        self.assertIsNone(MultimodalCache.combine_hashes([]))

    def test_single_hash(self):
        """单元素的 hash 列表返回 hash((elem,))。"""
        result = MultimodalCache.combine_hashes([42])
        self.assertEqual(result, hash((42,)))

    def test_multiple_hashes(self):
        """多个 hash 合并为一个元组的 hash。"""
        result = MultimodalCache.combine_hashes([1, 2, 3])
        self.assertEqual(result, hash((1, 2, 3)))

    def test_order_is_sensitive(self):
        """hash 顺序敏感：[A, B] 和 [B, A] 产生不同的 key。"""
        h1 = MultimodalCache.combine_hashes([1, 2])
        h2 = MultimodalCache.combine_hashes([2, 1])
        self.assertNotEqual(h1, h2)

    def test_deterministic(self):
        """相同的输入总是返回相同的 hash。"""
        h1 = MultimodalCache.combine_hashes([100, 200, 300])
        h2 = MultimodalCache.combine_hashes([100, 200, 300])
        self.assertEqual(h1, h2)


# ============================================================================
# _get_tensor_size（工具函数）
# ============================================================================


class TestGetTensorSize(unittest.TestCase):
    """_get_tensor_size 返回 tensor 占用的字节数。"""

    def test_float32_tensor(self):
        """float32 类型，每个元素 4 字节。"""
        t = torch.zeros(10, dtype=torch.float32)
        self.assertEqual(_get_tensor_size(t), 40)

    def test_float16_tensor(self):
        """float16 类型，每个元素 2 字节。"""
        t = torch.zeros(100, 100, dtype=torch.float16)
        self.assertEqual(_get_tensor_size(t), 20_000)

    def test_int64_tensor(self):
        """int64 类型，每个元素 8 字节。"""
        t = torch.zeros(5, dtype=torch.int64)
        self.assertEqual(_get_tensor_size(t), 40)

    def test_scalar_tensor(self):
        """标量 tensor（0 维），numel = 1。"""
        t = torch.tensor(3.14, dtype=torch.float32)
        self.assertEqual(_get_tensor_size(t), 4)


# ============================================================================
# MultiModalStaticCache（带 LRU 的缓存实现）
# ============================================================================


class TestMultiModalStaticCacheSet(unittest.TestCase):
    """set() 方法：存入新 embedding。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_set_new_item(self):
        """正常存入一个新 embedding。"""
        result = self.cache.set(1, _make_embedding(200))
        self.assertTrue(result)
        self.assertEqual(self.cache.available_size(), 1)

    def test_set_duplicate_key(self):
        """重复存入同一个 key，条目数不变，只是更新 LRU 顺序。"""
        self.cache.set(1, _make_embedding(200))
        result = self.cache.set(1, _make_embedding(300))
        self.assertTrue(result)
        self.assertEqual(self.cache.available_size(), 1)

    def test_set_triggers_lru_eviction(self):
        """空间不够时淘汰最旧的条目。"""
        cache = MultiModalStaticCache(max_size=300)
        self.cache = cache

        cache.set(1, _make_embedding(200))  # current = 200
        cache.set(2, _make_embedding(200))  # 200 + 200 = 400 > 300 → evict key 1
        self.assertEqual(cache.available_size(), 1)
        self.assertFalse(cache.has(1))  # evicted
        self.assertTrue(cache.has(2))  # survives

    def test_set_evicts_multiple_items_if_needed(self):
        """如果新条目很大，一次淘汰多个旧条目直到空间足够。"""
        cache = MultiModalStaticCache(max_size=500)
        for i in range(4):
            cache.set(i, _make_embedding(100))  # 4 × 100 = 400
        # 此时缓存里有 0,1,2,3，current_size=400
        cache.set(4, _make_embedding(200))  # 需要 600 → 淘汰 0,1（腾出 200）
        self.assertEqual(cache.available_size(), 3)  # 2,3,4 存活
        self.assertFalse(cache.has(0))
        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))
        self.assertTrue(cache.has(3))
        self.assertTrue(cache.has(4))

    def test_set_item_larger_than_max_size(self):
        """单个条目比 max_size 还大，返回 False。"""
        huge = _make_embedding(1500)
        result = self.cache.set(1, huge)
        self.assertFalse(result)

    def test_set_empty_cache_fails_for_oversized_item(self):
        """即使是空缓存，放不下超大的条目也返回 False。"""
        cache = MultiModalStaticCache(max_size=100)
        result = cache.set(1, _make_embedding(200))
        self.assertFalse(result)

    def test_set_raises_on_invalid_type(self):
        """非 EmbeddingResult 类型会触发 AssertionError。"""
        with self.assertRaises(AssertionError):
            self.cache.set(1, "not an embedding")  # type: ignore[arg-type]


class TestMultiModalStaticCacheGet(unittest.TestCase):
    """get() 方法：按 hash 列表查找，内部调用 combine_hashes。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_get_hit(self):
        """通过 combine_hashes → set → get 的完整流程命中。"""
        key = MultimodalCache.combine_hashes([42])
        self.cache.set(key, _make_embedding(200))
        result = self.cache.get([42])
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EmbeddingResult)

    def test_get_miss(self):
        """不存在的 hash 返回 None。"""
        result = self.cache.get([999])
        self.assertIsNone(result)

    def test_get_empty_list(self):
        """空列表导致 combine_hashes 返回 None，get 也返回 None。"""
        result = self.cache.get([])
        self.assertIsNone(result)


class TestMultiModalStaticCacheGetSingle(unittest.TestCase):
    """get_single() 方法：用单个 hash 直接查找。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_get_single_hit(self):
        self.cache.set(42, _make_embedding(200))
        self.assertIsNotNone(self.cache.get_single(42))

    def test_get_single_miss(self):
        self.assertIsNone(self.cache.get_single(999))

    def test_get_single_returns_embedding_result(self):
        self.cache.set(42, _make_embedding(200))
        result = self.cache.get_single(42)
        self.assertIsInstance(result, EmbeddingResult)


class TestMultiModalStaticCacheHas(unittest.TestCase):
    """has() 方法：存在性检查（不影响 LRU 顺序）。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_has_existing(self):
        self.cache.set(1, _make_embedding(200))
        self.assertTrue(self.cache.has(1))

    def test_has_nonexistent(self):
        self.assertFalse(self.cache.has(999))

    def test_has_after_eviction(self):
        """条目被淘汰后 has 返回 False。"""
        cache = MultiModalStaticCache(max_size=200)
        cache.set(1, _make_embedding(150))
        cache.set(2, _make_embedding(150))  # evicts 1
        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))


class TestMultiModalStaticCacheFree(unittest.TestCase):
    """free() 方法：手动删除条目。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)
        self.cache.set(1, _make_embedding(200))
        self.cache.set(2, _make_embedding(200))

    def test_free_existing(self):
        result = self.cache.free(1, None)
        self.assertTrue(result)
        self.assertEqual(self.cache.available_size(), 1)

    def test_free_nonexistent(self):
        result = self.cache.free(999, None)
        self.assertFalse(result)

    def test_free_updates_current_size(self):
        """free 后 current_size 应减去被删除条目的大小。"""
        size_before = self.cache.current_size
        self.cache.free(1, None)
        self.assertLess(self.cache.current_size, size_before)

    def test_free_clears_all_items_one_by_one(self):
        self.cache.free(1, None)
        self.cache.free(2, None)
        self.assertEqual(self.cache.available_size(), 0)


class TestMultiModalStaticCacheClear(unittest.TestCase):
    """clear() 方法：清空全部。"""

    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_clear_empties_cache(self):
        self.cache.set(1, _make_embedding(200))
        self.cache.set(2, _make_embedding(200))
        self.cache.clear()
        self.assertEqual(self.cache.available_size(), 0)
        self.assertIsNone(self.cache.get_single(1))
        self.assertIsNone(self.cache.get_single(2))

    def test_clear_empty_cache(self):
        """清空空缓存不应该报错。"""
        self.cache.clear()  # should not raise
        self.assertEqual(self.cache.available_size(), 0)


class TestMultiModalStaticCacheLRUOrder(unittest.TestCase):
    """LRU 淘汰顺序验证。"""

    def test_lru_eviction_order(self):
        """最久未使用的条目被优先淘汰。"""
        cache = MultiModalStaticCache(max_size=500)
        cache.set(1, _make_embedding(200))  # order: [1]
        cache.set(2, _make_embedding(200))  # order: [1, 2]
        cache.set(3, _make_embedding(200))  # 200+200+200=600 > 500 → evict 1
        self.assertFalse(cache.has(1))  # 1 was oldest → evicted
        self.assertTrue(cache.has(2))
        self.assertTrue(cache.has(3))

    def test_get_moves_item_to_end(self):
        """get_single 访问的条目被移到 LRU 末尾，不会被优先淘汰。"""
        cache = MultiModalStaticCache(max_size=500)
        cache.set(1, _make_embedding(200))  # order: [1]
        cache.set(2, _make_embedding(200))  # order: [1, 2]
        cache.get_single(1)  # access 1 → order: [2, 1]
        cache.set(3, _make_embedding(200))  # 200+200+200=600 > 500 → evict 2
        self.assertTrue(cache.has(1))  # 1 was recently accessed → survives
        self.assertFalse(cache.has(2))  # 2 was LRU → evicted
        self.assertTrue(cache.has(3))

    def test_get_via_combined_hash_updates_lru(self):
        """通过 get()（使用 combine_hashes）访问也更新 LRU 顺序。"""
        cache = MultiModalStaticCache(max_size=500)
        k1 = MultimodalCache.combine_hashes([1])
        k2 = MultimodalCache.combine_hashes([2])
        cache.set(k1, _make_embedding(200))
        cache.set(k2, _make_embedding(200))
        cache.get([1])  # access k1 → move to end
        cache.set(MultimodalCache.combine_hashes([3]), _make_embedding(200))  # evict LRU
        self.assertTrue(cache.has(k1))  # recently accessed → survives
        self.assertFalse(cache.has(k2))  # LRU → evicted

    def test_set_duplicate_moves_to_end(self):
        """重复 set 同一个 key 也更新 LRU 顺序。"""
        cache = MultiModalStaticCache(max_size=300)
        cache.set(1, _make_embedding(150))  # order: [1]
        cache.set(2, _make_embedding(150))  # order: [1, 2]
        cache.set(1, _make_embedding(150))  # duplicate → order: [2, 1]
        cache.set(3, _make_embedding(150))  # 150+150+150=450 > 300 → evict 2
        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))
        self.assertTrue(cache.has(3))


class TestMultiModalStaticCacheAvailableSize(unittest.TestCase):
    """available_size() 实时反映缓存条目数。"""

    def test_empty_cache_returns_zero(self):
        cache = MultiModalStaticCache(max_size=1000)
        self.assertEqual(cache.available_size(), 0)

    def test_after_one_set(self):
        cache = MultiModalStaticCache(max_size=1000)
        cache.set(1, _make_embedding(200))
        self.assertEqual(cache.available_size(), 1)

    def test_after_multiple_sets(self):
        cache = MultiModalStaticCache(max_size=1000)
        cache.set(1, _make_embedding(200))
        cache.set(2, _make_embedding(200))
        self.assertEqual(cache.available_size(), 2)

    def test_after_clear(self):
        cache = MultiModalStaticCache(max_size=1000)
        cache.set(1, _make_embedding(200))
        cache.set(2, _make_embedding(200))
        cache.clear()
        self.assertEqual(cache.available_size(), 0)

    def test_after_eviction(self):
        cache = MultiModalStaticCache(max_size=300)
        cache.set(1, _make_embedding(200))
        cache.set(2, _make_embedding(200))  # evicts 1
        self.assertEqual(cache.available_size(), 1)

    def test_len_returns_same_as_available_size(self):
        cache = MultiModalStaticCache(max_size=1000)
        self.assertEqual(cache.__len__(), cache.available_size())
        cache.set(1, _make_embedding(200))
        self.assertEqual(cache.__len__(), cache.available_size())
        cache.set(2, _make_embedding(200))
        self.assertEqual(cache.__len__(), cache.available_size())


class TestMultiModalStaticCacheEdgeCases(unittest.TestCase):
    """边界情况。"""

    def test_set_zero_size_embedding(self):
        """0 字节的 tensor 也可以正常缓存。"""
        emb = EmbeddingResult(embedding=torch.zeros(0, dtype=torch.float32))
        cache = MultiModalStaticCache(max_size=100)
        result = cache.set(1, emb)
        self.assertTrue(result)
        self.assertEqual(cache.available_size(), 1)

    def test_evict_up_to_empty_cache(self):
        """淘汰到空缓存仍然放不下超大的条目，返回 False。"""
        cache = MultiModalStaticCache(max_size=100)
        cache.set(1, _make_embedding(50))
        # 尝试存入一个 120 字节的条目 → 先淘汰 1（腾出 50），但 120 > 100 仍然放不下
        result = cache.set(2, _make_embedding(120))
        self.assertFalse(result)
        self.assertEqual(cache.available_size(), 0)  # 1 也被淘汰了

    def test_free_with_none_allocator(self):
        """free 的 allocator 参数在实现中没有被使用，传 None 应该正常工作。"""
        cache = MultiModalStaticCache(max_size=1000)
        cache.set(1, _make_embedding(200))
        result = cache.free(1, None)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
