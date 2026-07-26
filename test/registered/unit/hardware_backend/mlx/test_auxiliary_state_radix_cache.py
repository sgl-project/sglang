"""Model-free regression tests for the MLX auxiliary-state unified radix cache.

Unlike the component-contract tests in ``test_attention_patching.py`` (which
hand the component a ``SimpleNamespace`` stand-in for the cache), these build a
REAL ``UnifiedRadixCache`` and drive match/insert/evict through it, so the
inherited ``MambaComponent`` code paths actually run against
``MlxAuxiliaryStateComponent`` and ``MlxAuxiliaryStateReqToTokenPool``:

* ``MlxAuxiliaryStateComponent.__init__`` calls ``TreeComponent.__init__``
  directly (``MambaComponent.__init__`` asserts ``HybridReqToTokenPool``), so
  the inherited match/insert helpers must still find
  ``mamba_cache_chunk_size`` / ``mamba_max_states_per_path``.
* Inherited allocator paths (match CoW alloc, eviction frees, retract
  release) address the pool as ``req_to_token_pool.mamba_allocator``.
"""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx
    import torch

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        MlxAuxiliaryStateComponent,
        MlxAuxiliaryStateReqToTokenPool,
    )
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

CHUNK_SIZE = 4


def _set_server_args(mamba_max_states_per_path: int = -1) -> None:
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = CHUNK_SIZE
    server_args.mamba_max_states_per_path = mamba_max_states_per_path
    set_global_server_args_for_scheduler(server_args)


class FakeNativeCache:
    def __init__(self, state=()):
        self.state = state


def _build_cache(auxiliary_state_size: int = 4, kv_size: int = 64):
    req_to_token_pool = MlxAuxiliaryStateReqToTokenPool(
        size=4,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
        auxiliary_state_size=auxiliary_state_size,
    )
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=4,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        component_registry_override={
            ComponentType.MAMBA: MlxAuxiliaryStateComponent,
        },
    )
    cache = UnifiedRadixCache(params=params)
    return cache, allocator, req_to_token_pool


def _insert_with_aux_state(cache, allocator, req_to_token_pool, tokens, state_value):
    """Insert a token chain whose tail donates an auxiliary-state slot,
    mirroring the finished-request path: snapshot the native cache into a
    fresh slot, then hand the slot to the tree."""
    pool = req_to_token_pool.auxiliary_state_pool
    slot = pool.alloc(1)
    assert slot is not None
    native = [FakeNativeCache(mx.array([state_value], dtype=mx.float32))]
    pool.store_cache(slot[0], native, [0])
    kv = allocator.alloc(len(tokens))
    result = cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=kv, mamba_value=slot)
    )
    return result, slot


def _match(cache, tokens, **kwargs):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), **kwargs)
    )


def _cow_req():
    return SimpleNamespace(
        mamba_pool_idx=None,
        mamba_cow_src_index=None,
        mamba_needs_clear=True,
        session=None,
    )


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAuxiliaryStateRealRadixCache(unittest.TestCase):
    def setUp(self):
        _set_server_args()

    def test_component_reads_mamba_config_from_server_args(self):
        cache, _, _ = _build_cache()
        component = cache.components[ComponentType.MAMBA]
        self.assertIsInstance(component, MlxAuxiliaryStateComponent)
        self.assertEqual(component.mamba_cache_chunk_size, CHUNK_SIZE)
        self.assertEqual(component.mamba_max_states_per_path, -1)

    def test_req_pool_exposes_allocator_alias_with_group_hooks(self):
        _, _, req_to_token_pool = _build_cache()
        allocator = req_to_token_pool.mamba_allocator
        self.assertIs(allocator, req_to_token_pool.auxiliary_state_pool)
        # The scheduler's prefill loop brackets allocs with these hooks once
        # it sees a ``mamba_allocator`` attribute.
        allocator.alloc_group_begin(3)
        slot = allocator.alloc(1)
        allocator.alloc_group_end()
        self.assertIsNotNone(slot)
        allocator.free(slot)
        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(allocator.schedulable_available_size(), 4)

    def test_insert_and_match_prefix_survive_inherited_mamba_paths(self):
        # Insert runs _emit_excess_path_states_eviction
        # (mamba_max_states_per_path); match runs
        # finalize_match_result_in_tree_core (mamba_cache_chunk_size). Both
        # crashed with AttributeError before the component mirrored the base
        # MambaComponent fields.
        cache, allocator, req_to_token_pool = _build_cache()
        tokens = list(range(2 * CHUNK_SIZE))
        insert_result, _ = _insert_with_aux_state(
            cache, allocator, req_to_token_pool, tokens, 1.0
        )
        self.assertEqual(insert_result.prefix_len, 0)

        match_result = _match(cache, tokens)
        self.assertEqual(len(match_result.device_indices), len(tokens))
        # The auxiliary boundary coincides with the full-KV hit: no branching.
        self.assertIsNone(match_result.mamba_branching_seqlen)

    def test_evict_frees_auxiliary_slot_through_allocator_alias(self):
        # Tombstoning an interior auxiliary state routes the freed slot
        # through _free_mamba_value -> req_to_token_pool.mamba_allocator,
        # which crashed with AttributeError before the alias existed. The
        # unified cache itself triggers this evict when the pool runs dry
        # (MlxAuxiliaryStateComponent.prepare_for_caching_req).
        cache, allocator, req_to_token_pool = _build_cache()
        pool = req_to_token_pool.auxiliary_state_pool
        chain_a = list(range(CHUNK_SIZE))
        chain_ab = chain_a + list(range(100, 100 + CHUNK_SIZE + 1))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_a, 1.0)
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_ab, 2.0)
        self.assertEqual(pool.available_size(), 2)

        evict_result = cache.evict(EvictParams(num_tokens=0, mamba_num=1))

        self.assertEqual(evict_result.mamba_num_evicted, 1)
        self.assertEqual(pool.available_size(), 3)
        # The interior node (chain_a's tail) lost its auxiliary state while
        # its full KV stayed matchable, so a match of chain_a now reports the
        # aligned branching point past the (empty) auxiliary boundary.
        match_result = _match(cache, chain_a)
        self.assertEqual(match_result.mamba_branching_seqlen, CHUNK_SIZE)

    def test_match_prefix_cow_copies_snapshot_into_request_slot(self):
        # The CoW alloc in MambaComponent.finalize_match_result_in_cache
        # crashed with AttributeError on mamba_allocator; and without the
        # eager copy the request slot would stay empty (the MLX runner has no
        # deferred forward_batch CoW pass), degrading every prefix hit to a
        # full-prompt recompute.
        cache, allocator, req_to_token_pool = _build_cache()
        pool = req_to_token_pool.auxiliary_state_pool
        tokens = list(range(2 * CHUNK_SIZE))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, tokens, 7.0)

        req = _cow_req()
        match_result = _match(cache, tokens, cow_mamba=True, req=req)

        self.assertEqual(len(match_result.device_indices), len(tokens))
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNone(req.mamba_cow_src_index)
        self.assertFalse(req.mamba_needs_clear)
        restored = [FakeNativeCache()]
        self.assertTrue(pool.restore_cache(req.mamba_pool_idx, restored, [0]))
        self.assertEqual(restored[0].state.tolist(), [7.0])

    def test_match_prefix_cow_evicts_when_pool_exhausted(self):
        # Exhausted pool: the CoW path pins the matched node, evicts one
        # auxiliary state, and retries the alloc — crossing both fixed paths
        # (allocator alias + component config fields) in one flow.
        cache, allocator, req_to_token_pool = _build_cache(auxiliary_state_size=2)
        pool = req_to_token_pool.auxiliary_state_pool
        chain_a = list(range(CHUNK_SIZE))
        chain_ab = chain_a + list(range(100, 100 + CHUNK_SIZE))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_a, 1.0)
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_ab, 2.0)
        self.assertEqual(pool.available_size(), 0)

        req = _cow_req()
        match_result = _match(cache, chain_ab, cow_mamba=True, req=req)

        self.assertEqual(len(match_result.device_indices), len(chain_ab))
        self.assertIsNotNone(req.mamba_pool_idx)
        # The pinned tail survived eviction and its snapshot was copied.
        restored = [FakeNativeCache()]
        self.assertTrue(pool.restore_cache(req.mamba_pool_idx, restored, [0]))
        self.assertEqual(restored[0].state.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
