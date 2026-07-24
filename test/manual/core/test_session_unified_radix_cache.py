"""Tests for session references on UnifiedRadixCache."""

from __future__ import annotations

import ast
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    # Import before sglang: its triton stub can break torch's lazy import.
    import torch._inductor.runtime.triton_heuristics  # noqa: F401

    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


REPO_ROOT = Path(__file__).resolve().parents[3]
MEM_CACHE_ROOT = REPO_ROOT / "python/sglang/srt/mem_cache"


def class_bases(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        base.id if isinstance(base, ast.Name) else ast.unparse(base)
        for base in class_node.bases
    }


class TestSessionCacheOwnership(unittest.TestCase):
    def test_only_unified_radix_cache_owns_session_mixin(self):
        ordinary_mixin = MEM_CACHE_ROOT / "session_radix_cache.py"
        radix_cache = MEM_CACHE_ROOT / "radix_cache.py"
        hiradix_cache = MEM_CACHE_ROOT / "hiradix_cache.py"
        evict_policy = MEM_CACHE_ROOT / "evict_policy.py"
        unified_cache = MEM_CACHE_ROOT / "unified_radix_cache.py"

        self.assertFalse(ordinary_mixin.exists())
        ordinary_source = "\n".join(
            path.read_text() for path in (radix_cache, hiradix_cache, evict_policy)
        )
        for removed_symbol in (
            "SessionRadixCacheMixin",
            "SessionAwareEvictionStrategy",
            "session_ref",
            "_session_on_",
            "_session_forget_node",
            "_account_new_evictable_node",
            "_supports_session_radix_cache",
            "enable_session_radix_cache",
        ):
            self.assertNotIn(removed_symbol, ordinary_source)
        self.assertNotIn(
            "SessionRadixCacheMixin", class_bases(radix_cache, "RadixCache")
        )
        self.assertIn(
            "SessionUnifiedRadixCacheMixin",
            class_bases(unified_cache, "UnifiedRadixCache"),
        )

        for component in (
            "full_component.py",
            "swa_component.py",
            "mamba_component.py",
        ):
            self.assertIn(
                "session_ref",
                (MEM_CACHE_ROOT / "unified_cache_components" / component).read_text(),
            )

        scheduler = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
        self.assertIn(
            'getattr(self.tree_cache, "enable_session_radix_cache", False)',
            scheduler.read_text(),
        )


def make_params(enable_session: bool) -> CacheInitParams:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        enable_session_radix_cache=enable_session,
        tree_components=(ComponentType.FULL,),
    )


def insert(cache, token_ids):
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node


def match_len(cache, token_ids) -> int:
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        ).device_indices
    )


def register(cache, token_ids, session_id, generation=None):
    cache.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=generation,
            session=None,
            last_node=cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", token_ids)))
            ).last_device_node,
            origin_input_ids=array("q", token_ids),
            output_ids=array("q"),
            kv_committed_len=len(token_ids),
            extra_key=None,
        )
    )


@unittest.skipIf(torch is None, "PyTorch is required for cache behavior tests")
class TestRadixCacheSessionRemoval(unittest.TestCase):
    def test_plain_radix_cache_does_not_enable_session_references(self):
        cache = RadixCache(make_params(enable_session=True))

        self.assertFalse(hasattr(cache, "enable_session_radix_cache"))
        self.assertFalse(hasattr(cache, "register_session_ref"))
        self.assertFalse(hasattr(cache, "open_radix_session"))


@unittest.skipIf(torch is None, "PyTorch is required for cache behavior tests")
class TestSessionUnifiedRadixCache(unittest.TestCase):
    def setUp(self):
        self.cache = UnifiedRadixCache(make_params(enable_session=True))
        self.full = self.cache.components[ComponentType.FULL]

    def test_register_and_release_update_full_component_reference(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.assertEqual(self.full.session_ref(leaf), 1)

        self.cache.release_radix_session("s1")
        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_reopen_rejects_stale_generation(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        old_generation = self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", old_generation)

        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_eviction_prefers_unreferenced_full_kv(self):
        referenced = insert(self.cache, [1, 2, 3, 4])
        insert(self.cache, [7, 8, 9])
        register(self.cache, [1, 2, 3, 4], "s1")

        self.cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(match_len(self.cache, [7, 8, 9]), 0)
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(self.full.session_ref(referenced), 1)


if __name__ == "__main__":
    unittest.main()
