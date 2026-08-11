"""Tests for session references on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

REPO_ROOT = Path(__file__).resolve().parents[4]
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


class TestSessionCacheOwnership(CustomTestCase):
    def test_only_unified_radix_cache_owns_session_ref_tracker(self):
        ordinary_mixin = MEM_CACHE_ROOT / "session_radix_cache.py"
        radix_cache = MEM_CACHE_ROOT / "radix_cache.py"
        hiradix_cache = MEM_CACHE_ROOT / "hiradix_cache.py"
        evict_policy = MEM_CACHE_ROOT / "evict_policy.py"
        unified_cache = MEM_CACHE_ROOT / "unified_radix_cache.py"
        session_ref_tracker = (
            MEM_CACHE_ROOT / "unified_cache" / "session_ref_tracker.py"
        )

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
        # Session behavior is composed, not mixed in (general-code-style rule).
        self.assertEqual(
            class_bases(unified_cache, "UnifiedRadixCache"), {"BasePrefixCache"}
        )
        self.assertIn("UnifiedSessionRefTracker", session_ref_tracker.read_text())
        self.assertNotIn("SessionUnifiedRadixCacheMixin", unified_cache.read_text())

        for component in (
            "full_component.py",
            "swa_component.py",
            "mamba_component.py",
        ):
            self.assertIn(
                "session_ref",
                (
                    MEM_CACHE_ROOT / "unified_cache" / "components" / component
                ).read_text(),
            )

        registry = MEM_CACHE_ROOT / "registry.py"
        self.assertIn(
            "--enable-session-radix-cache requires UnifiedRadixCache",
            registry.read_text(),
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
    """Insert and return the tail node; the cache boundary hands back a NodeId."""
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    node_id = cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node
    return cache.tree_core.node_by_id(node_id)


def match_len(cache, token_ids) -> int:
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        ).device_indices
    )


def register(cache, token_ids, session_id, generation=None):
    if generation is None:
        generation = cache.ensure_session_generation(session_id)
    cache.session_refs.register_session_ref(
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


class TestRadixCacheSessionRemoval(CustomTestCase):
    def test_plain_radix_cache_does_not_enable_session_references(self):
        cache = RadixCache(make_params(enable_session=True))

        self.assertFalse(hasattr(cache, "enable_session_radix_cache"))
        self.assertFalse(hasattr(cache, "register_session_ref"))
        self.assertFalse(hasattr(cache, "open_radix_session"))


class TestSessionUnifiedRadixCache(CustomTestCase):
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

    def test_protected_eviction_cutoff_stops_before_session_kv(self):
        protected = insert(self.cache, [1, 2, 3, 4])
        insert(self.cache, [7, 8, 9])
        register(self.cache, [1, 2, 3, 4], "protected")

        self.assertEqual(self.cache.evictable_size(), 7)
        self.assertEqual(self.cache.evictable_size_without_session_refs(), 3)

        result = self.cache.evict(
            EvictParams(
                num_tokens=7,
                allow_protected_session_cache=False,
            )
        )

        self.assertEqual(result.num_tokens_evicted, 3)
        self.assertEqual(match_len(self.cache, [7, 8, 9]), 0)
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(self.full.session_ref(protected), 1)

    def test_demote_and_promote_session_cache_priority(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")
        register(self.cache, [1, 2, 3, 4], "s1", generation)

        demoted = self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )
        self.assertEqual(demoted.status, "updated")
        self.assertEqual(demoted.indexed_component_leaves, 1)
        self.assertEqual(self.full.session_ref(leaf), 0)

        unchanged = self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )
        self.assertEqual(unchanged.status, "unchanged")
        self.assertEqual(self.full.session_ref(leaf), 0)

        promoted = self.cache.set_session_cache_priority(
            "s1", protected=True, generation=generation
        )
        self.assertEqual(promoted.status, "updated")
        self.assertEqual(self.full.session_ref(leaf), 1)

    def test_demoted_request_cannot_evict_protected_session_cache(self):
        generation = self.cache.open_radix_session("s1")
        req = SimpleNamespace(
            session_id="s1",
            session_generation=generation,
            session=None,
        )
        self.assertTrue(self.cache.request_can_evict_protected_session_cache(req))

        self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )

        self.assertFalse(self.cache.request_can_evict_protected_session_cache(req))

    def test_stale_session_request_cannot_evict_protected_session_cache(self):
        stale_generation = self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")
        stale_req = SimpleNamespace(
            session_id="s1",
            session_generation=stale_generation,
            session=None,
        )

        self.assertFalse(
            self.cache.request_can_evict_protected_session_cache(stale_req)
        )

    def test_demoted_session_keeps_future_leaves_evictable(self):
        generation = self.cache.open_radix_session("s1")
        self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )
        leaf = insert(self.cache, [1, 2, 3, 4])

        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.assertEqual(self.full.session_ref(leaf), 0)

        self.cache.set_session_cache_priority(
            "s1", protected=True, generation=generation
        )
        self.assertEqual(self.full.session_ref(leaf), 1)

    def test_session_cache_priority_rejects_stale_generation(self):
        generation = self.cache.open_radix_session("s1")

        result = self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation + 1
        )

        self.assertEqual(result.status, "stale_generation")
        self.assertEqual(result.generation, generation)

    def test_demoting_one_session_keeps_shared_prefix_protected(self):
        shared_leaf = insert(self.cache, [1, 2, 3, 4])
        generation_1 = self.cache.open_radix_session("s1")
        generation_2 = self.cache.open_radix_session("s2")
        register(self.cache, [1, 2, 3, 4], "s1", generation_1)
        register(self.cache, [1, 2, 3, 4], "s2", generation_2)
        self.assertEqual(self.full.session_ref(shared_leaf), 2)

        self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation_1
        )

        self.assertEqual(self.full.session_ref(shared_leaf), 1)

    def test_close_and_reopen_restore_default_protection(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")
        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )
        self.cache.release_radix_session("s1")

        reopened_generation = self.cache.open_radix_session("s1")
        register(self.cache, [1, 2, 3, 4], "s1", reopened_generation)

        self.assertEqual(self.full.session_ref(leaf), 1)

    def test_eviction_prefers_demoted_session_over_protected_session(self):
        demoted_leaf = insert(self.cache, [1, 2, 3])
        protected_leaf = insert(self.cache, [7, 8, 9])
        demoted_generation = self.cache.open_radix_session("demoted")
        protected_generation = self.cache.open_radix_session("protected")
        register(self.cache, [1, 2, 3], "demoted", demoted_generation)
        register(self.cache, [7, 8, 9], "protected", protected_generation)
        self.cache.set_session_cache_priority(
            "demoted", protected=False, generation=demoted_generation
        )

        self.cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(match_len(self.cache, [1, 2, 3]), 0)
        self.assertEqual(match_len(self.cache, [7, 8, 9]), 3)
        self.assertEqual(self.full.session_ref(demoted_leaf), 0)
        self.assertEqual(self.full.session_ref(protected_leaf), 1)

    def test_promote_after_demoted_leaf_eviction_is_safe(self):
        insert(self.cache, [1, 2, 3])
        generation = self.cache.open_radix_session("s1")
        register(self.cache, [1, 2, 3], "s1", generation)
        self.cache.set_session_cache_priority(
            "s1", protected=False, generation=generation
        )
        self.cache.evict(EvictParams(num_tokens=3))

        result = self.cache.set_session_cache_priority(
            "s1", protected=True, generation=generation
        )

        self.assertEqual(result.status, "updated")
        self.assertEqual(result.indexed_component_leaves, 0)


if __name__ == "__main__":
    unittest.main()
