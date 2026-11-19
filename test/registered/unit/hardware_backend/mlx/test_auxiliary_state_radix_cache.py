"""Model-free regression tests for the MLX auxiliary-state unified radix cache.

Unlike the component-contract tests in ``test_attention_patching.py`` (which
hand the component a ``SimpleNamespace`` stand-in for the cache), these build a
REAL ``UnifiedRadixCache`` and drive match/insert/evict/admission through it,
so the inherited ``MambaComponent`` code paths actually run against
``MlxAuxiliaryStateComponent`` and ``MlxAuxiliaryStateReqToTokenPool``:

* The component must carry the base ``MambaComponent`` config fields
  (``mamba_cache_chunk_size`` / ``mamba_checkpoint_grid`` /
  ``mamba_max_states_per_path``), or the inherited match/insert helpers crash
  with ``AttributeError`` (issue #32449, defect 1).
* Admission must reclaim auxiliary states before binding rows, and an
  auxiliary shortfall must not consume the request row (defect 2).
* A CoW prefix hit must copy the matched snapshot into the request slot
  eagerly: the MLX runner has no deferred CoW apply pass (defect 3).

Ported from PR #32450's regression suite, adapted to main's tree_slots /
req.kv / alloc_req_slots(storage, state, reqs, tree_cache) APIs.
"""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="extra-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    import mlx.core as mx
    import torch

    from sglang.srt.hardware_backend.mlx.kv_cache import (
        MlxAuxiliaryStateComponent,
        MlxAuxiliaryStateReqToTokenPool,
    )
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.mem_cache.allocation import alloc_req_slots
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.kv.plan import TREE_STATE
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.test.tree_slots import tree_slots_of

CHUNK_SIZE = 4


def _set_server_args(mamba_max_states_per_path: int = -1) -> None:
    server_args = ServerArgs(
        model_path="dummy",
        page_size=1,
        mamba_max_states_per_path=mamba_max_states_per_path,
    )
    # The dummy HF config cannot supply mamba_cache_chunk_size; preset it.
    server_args._mamba_cache_chunk_size = CHUNK_SIZE
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
    # The runner wires the state storage onto the token allocator so the
    # radix cache's CoW/eviction paths reach the auxiliary pool.
    allocator.state = req_to_token_pool.state
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        tree_slots=tree_slots_of(
            (ComponentType.FULL, ComponentType.MAMBA),
            checkpoint_interval=CHUNK_SIZE,
        ),
        component_registry_override={
            TREE_STATE: MlxAuxiliaryStateComponent,
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
    return SimpleNamespace(kv=ReqKvInfo(), session=None)


def _admission_req():
    return SimpleNamespace(kv=ReqKvInfo(), session=None)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAuxiliaryStateRealRadixCache(unittest.TestCase):
    def setUp(self):
        _set_server_args()

    def test_component_reads_mamba_config_from_server_args(self):
        # Break caught: the component skips MambaComponent.__init__ and the
        # inherited match/insert helpers crash with AttributeError.
        cache, _, _ = _build_cache()
        component = cache.components[ComponentType.MAMBA]
        self.assertIsInstance(component, MlxAuxiliaryStateComponent)
        self.assertEqual(component.mamba_cache_chunk_size, CHUNK_SIZE)
        # lcm(chunk=4, tree page=1) = 4.
        self.assertEqual(component.mamba_checkpoint_grid, CHUNK_SIZE)
        self.assertEqual(component.mamba_max_states_per_path, -1)

    def test_req_pool_exposes_allocator_alias_with_group_hooks(self):
        # Break caught: eviction/CoW address the slot allocator as
        # ``mamba_allocator`` and the scheduler brackets allocs with the
        # group hooks; a missing alias or hook crashes those paths.
        _, _, req_to_token_pool = _build_cache()
        allocator = req_to_token_pool.mamba_allocator
        self.assertIs(allocator, req_to_token_pool.auxiliary_state_pool)
        allocator.alloc_group_begin(3)
        slot = allocator.alloc(1)
        allocator.alloc_group_end()
        self.assertIsNotNone(slot)
        allocator.free(slot)
        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(allocator.schedulable_available_size(), 4)

    def test_insert_and_match_prefix_survive_inherited_mamba_paths(self):
        # Break caught: insert runs _emit_excess_path_states_eviction
        # (mamba_max_states_per_path); match runs
        # finalize_match_result_in_tree_core (mamba_checkpoint_grid). Both
        # crashed with AttributeError before the component mirrored the base
        # MambaComponent fields.
        cache, allocator, req_to_token_pool = _build_cache()
        tokens = list(range(2 * CHUNK_SIZE))
        insert_result, _ = _insert_with_aux_state(
            cache, allocator, req_to_token_pool, tokens, 1.0
        )
        self.assertEqual(insert_result.prefix_len, 0)

        match_result = _match(cache, tokens)
        self.assertEqual(match_result.device_prefix_len, len(tokens))
        # The auxiliary boundary coincides with the full-KV hit: no branching.
        self.assertIsNone(match_result.mamba_branching_seqlen)

    def test_evict_frees_auxiliary_slot_through_allocator_alias(self):
        # Break caught: tombstoning an interior auxiliary state must route
        # the freed slot through the allocator the tree reaches, and the
        # evicted boundary must surface as the aligned branching point.
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
        # Break caught: without the eager copy the request slot stays empty
        # (the MLX runner has no deferred forward_batch CoW pass), degrading
        # every prefix hit to a full-prompt recompute.
        cache, allocator, req_to_token_pool = _build_cache()
        pool = req_to_token_pool.auxiliary_state_pool
        tokens = list(range(2 * CHUNK_SIZE))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, tokens, 7.0)

        req = _cow_req()
        match_result = _match(cache, tokens, cow_mamba=True, req=req)

        self.assertEqual(match_result.device_prefix_len, len(tokens))
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        restored = [FakeNativeCache()]
        self.assertTrue(pool.restore_cache(req.kv.mamba_pool_idx, restored, [0]))
        self.assertEqual(restored[0].state.tolist(), [7.0])

    def test_match_prefix_cow_evicts_when_pool_exhausted(self):
        # Break caught: an exhausted pool must not fail the CoW match; the
        # path pins the matched node, evicts one auxiliary state, retries
        # the alloc, and copies the surviving snapshot.
        cache, allocator, req_to_token_pool = _build_cache(auxiliary_state_size=2)
        pool = req_to_token_pool.auxiliary_state_pool
        chain_a = list(range(CHUNK_SIZE))
        chain_ab = chain_a + list(range(100, 100 + CHUNK_SIZE))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_a, 1.0)
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_ab, 2.0)
        self.assertEqual(pool.available_size(), 0)

        req = _cow_req()
        match_result = _match(cache, chain_ab, cow_mamba=True, req=req)

        self.assertEqual(match_result.device_prefix_len, len(chain_ab))
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        # The pinned tail survived eviction and its snapshot was copied.
        restored = [FakeNativeCache()]
        self.assertTrue(pool.restore_cache(req.kv.mamba_pool_idx, restored, [0]))
        self.assertEqual(restored[0].state.tolist(), [2.0])

    def test_cache_miss_admission_evicts_auxiliary_states(self):
        # Break caught: a no-prefix request reaches alloc_req_slots without
        # the match-time CoW path. A type gate that skips the MLX pool's
        # preflight consumes the request row and then asserts while the
        # auxiliary pool is exhausted, instead of evicting cached states.
        cache, allocator, req_to_token_pool = _build_cache(auxiliary_state_size=2)
        pool = req_to_token_pool.auxiliary_state_pool
        chain_a = list(range(CHUNK_SIZE))
        chain_ab = chain_a + list(range(100, 100 + CHUNK_SIZE))
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_a, 1.0)
        _insert_with_aux_state(cache, allocator, req_to_token_pool, chain_ab, 2.0)
        self.assertEqual(pool.available_size(), 0)
        self.assertEqual(cache.mamba_evictable_size(), 2)

        req = _admission_req()
        req_pool_indices = alloc_req_slots(
            kv_storage_of(req_to_token_pool, allocator),
            allocator.state,
            [req],
            cache,
        )

        # Which free row a fresh alloc gets is not semantically meaningful
        # (see ReqToTokenPool.alloc); only pin that it is one real, valid row
        # bound to the request.
        self.assertEqual(len(req_pool_indices), 1)
        self.assertIn(req_pool_indices[0], range(1, req_to_token_pool.size + 1))
        self.assertEqual(req.kv.req_pool_idx, req_pool_indices[0])
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        # Slot accounting: of the 2 slots, the request holds one and the tree
        # keeps the other (exactly the demanded state was evicted, no leak).
        self.assertEqual(pool.available_size(), 0)
        self.assertEqual(cache.mamba_evictable_size(), 1)

    def test_auxiliary_allocation_failure_does_not_consume_request_row(self):
        # Break caught: a non-atomic alloc binds the request row first and
        # then fails on the auxiliary shortfall, leaking the row. The
        # shortfall must surface before any row is consumed.
        _, _, req_to_token_pool = _build_cache(auxiliary_state_size=1)
        pool = req_to_token_pool.auxiliary_state_pool
        occupied = pool.alloc(1)
        req = _admission_req()

        with self.assertRaisesRegex(
            RuntimeError, "Not enough MLX auxiliary state slots"
        ):
            req_to_token_pool.alloc([req])

        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(req_to_token_pool.available_size(), 4)
        self.assertEqual(pool.available_size(), 0)
        pool.free(occupied)

    def test_row_exhaustion_returns_reserved_auxiliary_slots(self):
        # Break caught: when no request row is left, the auxiliary slots
        # reserved before the row alloc must go back; leaking them silently
        # shrinks the state pool.
        _, _, req_to_token_pool = _build_cache(auxiliary_state_size=8)
        pool = req_to_token_pool.auxiliary_state_pool
        holders = [_admission_req() for _ in range(4)]
        self.assertIsNotNone(req_to_token_pool.alloc(holders))
        self.assertEqual(req_to_token_pool.available_size(), 0)
        self.assertEqual(pool.available_size(), 4)

        req = _admission_req()
        self.assertIsNone(req_to_token_pool.alloc([req]))

        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(pool.available_size(), 4)


if __name__ == "__main__":
    unittest.main()
