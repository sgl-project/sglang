"""CPU unit tests for the NPU DSA cache layer-split pool.

Covers the bookkeeping and pairing-critical logic of
``LayerSplitNPUMLATokenToKVPool`` without devices: shard-range mapping, the
row-space broadcast slicing, and the set-path decision matrix that must stay
symmetric across the CP group (a rank-dependent get-time broadcast decision
deadlocks the paired HCCL broadcast). Also covers ``_dsa_split_row_bound``,
the token-vs-row bound semantics that the sliced prefix broadcast relies on.

Run directly:
    python -m pytest test/registered/unit/npu/mem_cache/test_npu_dsa_layer_split.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsa_cache_layer_split import (
    LayerSplitNPUMLATokenToKVPool,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.attention.dsa.utils import _dsa_split_row_bound
from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

POOL = LayerSplitNPUMLATokenToKVPool


def _make_pool(
    *,
    layer_num=10,
    start_layer=0,
    shard_rank=1,
    shard_size=4,
    kv_pages=4,
    page_size=8,
    kv_dim=6,
    kr_dim=4,
    index_dim=128,
):
    """Build a pool skeleton via __new__ (no device work) with real shard math."""
    pool = POOL.__new__(POOL)
    pool.layer_num = layer_num
    pool.start_layer = start_layer
    pool.layer_shard_rank = shard_rank
    pool.layer_shard_size = shard_size
    pool.layer_shard_enabled = True
    pool.kv_cache_dim = kv_dim
    pool.kr_cache_dim = kr_dim
    pool.index_head_dim = index_dim
    pool.store_dtype = torch.float32
    pool.dtype = torch.float32
    pool.use_combined_kv_buffer = False
    pool.use_fia_nz = False

    owned_start, owned_end = get_layer_shard_range(shard_rank, shard_size, layer_num)
    num_owned = owned_end - owned_start
    kv_shape = (num_owned, kv_pages, page_size, 1, kv_dim)
    kr_shape = (num_owned, kv_pages, page_size, 1, kr_dim)
    pool.k_buffer = torch.zeros(kv_shape)
    pool.v_buffer = torch.zeros(kr_shape)
    pool.remote_k_buffer = torch.zeros((kv_pages, page_size, 1, kv_dim))
    pool.remote_v_buffer = torch.zeros((kv_pages, page_size, 1, kr_dim))
    pool._kv_capacity_tokens = kv_pages * page_size
    pool._kv_used_bound = 0
    pool._idx_used_bound = 0

    pool.index_k_buffer = torch.zeros((2, kv_pages, page_size, 1, index_dim))
    pool.remote_index_k_buffer = torch.zeros((kv_pages, page_size, 1, index_dim))
    pool.remote_index_k_scale_buffer = None
    pool.indexer_layer_id_to_slot = {}
    return pool


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class TestShardBookkeeping(CustomTestCase):
    def test_uneven_split_ownership_and_offsets(self):
        pool = _make_pool(layer_num=10, shard_rank=1, shard_size=4)
        self.assertEqual(pool._owned_local_layer_range(), (3, 6))

        owned = [lid for lid in range(10) if pool._is_layer_owned(lid)]
        self.assertEqual(owned, [3, 4, 5])

        for lid in range(10):
            self.assertEqual(
                pool._get_layer_owner_rank(lid),
                get_layer_owner(lid, 4, 10),
            )

        # Owned layers rebase onto dense owned-only buffer rows.
        self.assertEqual([pool._kv_local_offset(lid) for lid in owned], [0, 1, 2])

    def test_start_layer_offsets_everything(self):
        pool = _make_pool(layer_num=10, start_layer=2, shard_rank=1, shard_size=4)
        # Owned global layers are start_layer + [3, 6) = [5, 8).
        owned = [lid for lid in range(2, 12) if pool._is_layer_owned(lid)]
        self.assertEqual(owned, [5, 6, 7])
        self.assertEqual([pool._kv_local_offset(lid) for lid in owned], [0, 1, 2])

    def test_kv_slice_tokens_clamps_bound(self):
        pool = _make_pool()
        self.assertEqual(pool._kv_slice_tokens(None), pool._kv_capacity_tokens)
        self.assertEqual(pool._kv_slice_tokens(0), pool._kv_capacity_tokens)
        self.assertEqual(pool._kv_slice_tokens(7), 7)
        self.assertEqual(
            pool._kv_slice_tokens(pool._kv_capacity_tokens + 5),
            pool._kv_capacity_tokens,
        )

    def test_get_kv_layer_ids_doubles_owned_layers(self):
        pool = _make_pool(layer_num=10, start_layer=2, shard_rank=1, shard_size=4)
        with patch.object(POOL, "get_state_layer_ids", lambda self: [10]):
            self.assertEqual(pool.get_kv_layer_ids(), [5, 6, 7, 5, 6, 7, 10])

    def test_get_kv_transfer_layer_span_counts_buffer_groups(self):
        pool = _make_pool()
        cases = [
            # (dsa_kv_cache_store_fp8, has_scale, expected_groups)
            (False, False, 3),  # k + v + indexer k
            (True, False, 2),  # packed k/v + indexer k
            (True, True, 3),  # packed k/v + indexer k + fp32 scale
        ]
        for fp8, has_scale, groups in cases:
            pool.dsa_kv_cache_store_fp8 = fp8
            pool.index_k_scale_buffer = object() if has_scale else None
            self.assertEqual(pool.get_kv_transfer_layer_span(6 * groups), 6)

        # Non-multiple entry counts are a registration bug: refuse silently
        # mis-slicing the layer span.
        pool.dsa_kv_cache_store_fp8 = False
        pool.index_k_scale_buffer = None
        with self.assertRaises(AssertionError):
            pool.get_kv_transfer_layer_span(7)


class _FakeParallel:
    def __init__(self, ranks):
        self.attn_cp_group = SimpleNamespace(ranks=ranks, device_group="fake-group")


class TestBroadcastSlicing(CustomTestCase):
    def setUp(self):
        self._parallel_patch = patch(
            "sglang.srt.hardware_backend.npu.dsa_cache_layer_split.get_parallel",
            lambda: _FakeParallel([10, 11, 12, 13]),
        )
        self._parallel_patch.start()
        self.broadcasts = []
        self._bcast_patch = patch.object(
            torch.distributed,
            "broadcast",
            self._record_broadcast,
            create=True,
        )
        self._bcast_patch.start()

    def tearDown(self):
        self._bcast_patch.stop()
        self._parallel_patch.stop()

    def _record_broadcast(self, tensor, *, src, group, async_op):
        self.broadcasts.append((tensor, src, group))
        return _FakeWork()

    def test_owner_broadcast_reads_owned_rows(self):
        pool = _make_pool(layer_num=10, shard_rank=1, shard_size=4)
        for i in range(3):
            pool.k_buffer[i].fill_(float(i + 1))
        # Layer 4 is owned (local 4 -> owned row 1); owner branch must pass the
        # dense owned-row slice, and the src must be the owner's global rank.
        works = pool._broadcast_kv_from_owner(4, used_bound=16)
        self.assertEqual(len(works), 2)
        k_tensor, src, group = self.broadcasts[0]
        self.assertEqual(src, 11)
        self.assertEqual(group, "fake-group")
        expected = pool.k_buffer[1].view(-1, pool.kv_cache_dim)[:16]
        self.assertTrue(torch.equal(k_tensor, expected))
        self.assertTrue(
            torch.equal(
                self.broadcasts[1][0], pool.v_buffer[1].view(-1, pool.kr_cache_dim)[:16]
            )
        )

    def test_non_owner_broadcast_reads_remote_scratch(self):
        pool = _make_pool(layer_num=10, shard_rank=0, shard_size=4)
        # Rank 0 owns locals [0, 3); layer 7 (local 7) belongs to rank 2.
        pool.remote_k_buffer.fill_(9.0)
        pool.remote_v_buffer.fill_(8.0)
        pool._broadcast_kv_from_owner(7, used_bound=0)
        k_tensor, src, _ = self.broadcasts[0]
        self.assertEqual(src, 12)
        self.assertTrue(
            torch.equal(k_tensor, pool.remote_k_buffer.view(-1, pool.kv_cache_dim))
        )
        # bound=0 means full-capacity rows, not zero rows.
        self.assertEqual(k_tensor.shape[0], pool._kv_capacity_tokens)

    def test_non_owner_broadcast_slices_used_bound(self):
        pool = _make_pool(layer_num=10, shard_rank=0, shard_size=4)
        pool._broadcast_kv_from_owner(7, used_bound=13)
        self.assertEqual(self.broadcasts[0][0].shape[0], 13)

    def test_index_broadcast_owner_slot_and_bound(self):
        pool = _make_pool(layer_num=10, shard_rank=1, shard_size=4)
        pool.indexer_layer_id_to_slot = {4: 1}
        pool._idx_used_bound = 10
        cap_tokens = pool.remote_index_k_buffer.numel() // pool.index_head_dim

        work = pool._broadcast_index_from_owner(4)
        self.assertIsNotNone(work)
        tensor, src, _ = self.broadcasts[0]
        self.assertEqual(src, 11)
        expected = pool.index_k_buffer[1].view(-1)[: 10 * pool.index_head_dim]
        self.assertTrue(torch.equal(tensor, expected))

        # full=True ignores the bound and ships the whole scratch.
        self.broadcasts.clear()
        pool._broadcast_index_from_owner(4, full=True)
        self.assertEqual(
            self.broadcasts[0][0].numel(), cap_tokens * pool.index_head_dim
        )

    def test_index_broadcast_non_owner_uses_remote_scratch(self):
        pool = _make_pool(layer_num=10, shard_rank=0, shard_size=4)
        pool.indexer_layer_id_to_slot = {}
        pool._idx_used_bound = 5
        pool._broadcast_index_from_owner(6)
        expected = pool.remote_index_k_buffer.view(-1)[: 5 * pool.index_head_dim]
        self.assertTrue(torch.equal(self.broadcasts[0][0], expected))


class TestKvSetDecisionMatrix(CustomTestCase):
    """The set path decides, per rank, whether the get skips its broadcast.

    Every rank must reach the SAME registered/not-registered outcome for a
    layer, otherwise one rank reads stale scratch while the other skips the
    paired broadcast and both sides deadlock or diverge.
    """

    def setUp(self):
        self.parent_calls = []

        def _record(_pool, layer, loc, k, v):
            self.parent_calls.append(layer.layer_id)

        patcher = patch.object(
            NPUMLATokenToKVPool,
            "set_kv_buffer",
            _record,
            create=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _pool(self, layer_id):
        pool = _make_pool(layer_num=10, shard_rank=1, shard_size=4)
        pool.remote_kv_layer_id = None
        pool.pending_remote_kv_broadcast = False
        pool.pending_remote_kv_layer_id = None
        pool.pending_kv_works = []
        pool.stores = []

        def store(layer, loc_info, cache_k, cache_v):
            pool.stores.append(layer.layer_id)
            pool.remote_kv_layer_id = layer.layer_id

        pool._store_remote_copy = store
        self.assertEqual(pool._is_layer_owned(layer_id), layer_id in (3, 4, 5))
        return pool

    def _layer(self, layer_id):
        return SimpleNamespace(layer_id=layer_id)

    def test_owner_full_gather_registers_patch_write(self):
        pool = self._pool(4)
        pool.set_kv_buffer(
            self._layer(4), None, object(), object(), cp_gathered_full=True
        )
        self.assertEqual(self.parent_calls, [4])
        self.assertEqual(pool.remote_kv_layer_id, 4)

    def test_non_owner_full_gather_stores_remote_copy(self):
        pool = self._pool(7)
        pool.set_kv_buffer(
            self._layer(7), None, object(), object(), cp_gathered_full=True
        )
        self.assertEqual(pool.stores, [7])
        self.assertEqual(self.parent_calls, [])
        self.assertEqual(pool.remote_kv_layer_id, 7)

    def test_chunk_write_with_pending_prefix_broadcast_registers(self):
        pool = self._pool(7)
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        pool.set_kv_buffer(
            self._layer(7), None, object(), object(), cp_gathered_chunk=True
        )
        self.assertEqual(pool.stores, [7])

    def test_chunk_write_with_cached_remote_copy_registers(self):
        # TBO micro-batch B: the pending broadcast was already consumed by
        # micro-batch A, but this rank still holds the remote copy.
        pool = self._pool(7)
        pool.remote_kv_layer_id = 7
        pool.set_kv_buffer(
            self._layer(7), None, object(), object(), cp_gathered_chunk=True
        )
        self.assertEqual(pool.stores, [7])

    def test_chunk_write_without_prefix_falls_back_to_owner_only(self):
        pool = self._pool(7)
        pool.set_kv_buffer(
            self._layer(7), None, object(), object(), cp_gathered_chunk=True
        )
        self.assertEqual(pool.stores, [])
        self.assertEqual(self.parent_calls, [])
        self.assertIsNone(pool.remote_kv_layer_id)

    def test_unsupported_store_config_keeps_owner_only_write(self):
        pool = self._pool(7)
        pool.use_combined_kv_buffer = True
        pool.set_kv_buffer(
            self._layer(7), None, object(), object(), cp_gathered_full=True
        )
        self.assertEqual(pool.stores, [])
        self.assertEqual(self.parent_calls, [])

    def test_pending_broadcast_consumed_even_on_fallback(self):
        # A plain (non-CP-gather) write must still consume the pending
        # broadcast on EVERY rank before the owned-only early return, keeping
        # the follow-up get decision paired.
        pool = self._pool(7)
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        work = _FakeWork()
        pool.pending_kv_works = [work]
        pool.set_kv_buffer(self._layer(7), None, object(), object())
        self.assertFalse(pool.pending_remote_kv_broadcast)
        self.assertIsNone(pool.pending_remote_kv_layer_id)
        self.assertTrue(work.waited)
        self.assertEqual(pool.stores, [])

    def test_pending_for_other_layer_left_untouched(self):
        pool = self._pool(7)
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 3
        pool.set_kv_buffer(self._layer(7), None, object(), object())
        self.assertTrue(pool.pending_remote_kv_broadcast)
        self.assertEqual(pool.pending_remote_kv_layer_id, 3)
        self.assertEqual(pool.stores, [])
        self.assertEqual(self.parent_calls, [])


class TestIndexSetAndPrefetch(CustomTestCase):
    def setUp(self):
        self.parent_index_calls = []
        self.parent_scale_calls = []
        for name, recorder in (
            ("set_index_k_buffer", self.parent_index_calls),
            ("set_index_k_scale_buffer", self.parent_scale_calls),
        ):
            holder = recorder

            def _record(_pool, *args, _holder=holder):
                _holder.append(args)

            patcher = patch.object(
                NPUMLATokenToKVPool,
                name,
                _record,
                create=True,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def _pool(self, has_scale=False):
        pool = _make_pool(layer_num=10, shard_rank=1, shard_size=4)
        pool.remote_index_layer_id = None
        pool.pending_remote_index_broadcast = False
        pool.pending_remote_index_layer_id = None
        pool.pending_index_works = []
        pool.remote_index_k_scale_buffer = object() if has_scale else None
        pool.index_stores = []

        def store(layer_id, loc, index_k):
            pool.index_stores.append(layer_id)
            pool.remote_index_layer_id = layer_id

        pool._store_remote_index_copy = store
        return pool

    def test_non_owner_full_index_gather_stores_copy(self):
        pool = self._pool()
        pool.set_index_k_buffer(7, "loc", "idx", cp_gathered_full=True)
        self.assertEqual(pool.index_stores, [7])
        self.assertEqual(self.parent_index_calls, [])

    def test_quant_indexer_keeps_broadcast_path(self):
        # The fp32 scale rows are not covered by patch-write, so quant
        # configs must NOT register the remote copy.
        pool = self._pool(has_scale=True)
        pool.set_index_k_buffer(7, "loc", "idx", cp_gathered_full=True)
        self.assertEqual(pool.index_stores, [])
        self.assertEqual(self.parent_index_calls, [])

    def test_owner_index_gather_writes_parent(self):
        pool = self._pool()
        pool.set_index_k_buffer(4, "loc", "idx", cp_gathered_full=True)
        self.assertEqual(self.parent_index_calls, [(4, "loc", "idx")])

    def test_index_write_invalidates_cached_remote_copy(self):
        # A plain write invalidates the cached remote copy before the
        # owned-only early return, so the next read re-broadcasts on every
        # rank.
        pool = self._pool()
        pool.remote_index_layer_id = 7
        pool.set_index_k_buffer(7, "loc", "idx")
        self.assertIsNone(pool.remote_index_layer_id)
        self.assertEqual(pool.index_stores, [])

    def test_chunk_write_with_cached_remote_copy_reregisters(self):
        pool = self._pool()
        pool.remote_index_layer_id = 7
        pool.set_index_k_buffer(7, "loc", "idx", cp_gathered_chunk=True)
        self.assertEqual(pool.index_stores, [7])
        self.assertEqual(pool.remote_index_layer_id, 7)

    def test_scale_write_skips_non_owner(self):
        pool = self._pool(has_scale=True)
        pool.set_index_k_scale_buffer(7, "loc", "scale")
        self.assertEqual(self.parent_scale_calls, [])

    def test_scale_write_owner_reaches_parent(self):
        pool = self._pool(has_scale=True)
        pool.set_index_k_scale_buffer(4, "loc", "scale")
        self.assertEqual(self.parent_scale_calls, [(4, "loc", "scale")])

    def test_prefetch_index_skipped_for_quant_indexer(self):
        pool = self._pool(has_scale=True)
        pool._broadcast_index_from_owner = lambda *a, **k: self.fail(
            "quant indexer must keep the get-time broadcast"
        )
        pool.prefetch_index_k_buffer(7)
        self.assertFalse(pool.pending_remote_index_broadcast)

    def test_prefetch_index_dedups_pending_layer(self):
        pool = self._pool()
        pool.pending_remote_index_broadcast = True
        pool.pending_remote_index_layer_id = 7
        pool._broadcast_index_from_owner = lambda *a, **k: self.fail(
            "pending layer must not be re-kicked"
        )
        pool.prefetch_index_k_buffer(7)
        self.assertEqual(pool.pending_remote_index_layer_id, 7)
        self.assertTrue(pool.pending_remote_index_broadcast)

    def test_prefetch_index_supersedes_stale_pending(self):
        pool = self._pool()
        pool.pending_remote_index_broadcast = True
        pool.pending_remote_index_layer_id = 3
        pool.pending_index_works = [_FakeWork()]
        pool._broadcast_index_from_owner = lambda layer_id: _FakeWork()
        pool.prefetch_index_k_buffer(7, used_bound=12)
        self.assertEqual(pool.pending_remote_index_layer_id, 7)
        self.assertEqual(pool._idx_used_bound, 12)
        self.assertTrue(pool.pending_remote_index_broadcast)

    def test_ensure_remote_index_layer_materializes_scratch(self):
        pool = self._pool()
        pool._broadcast_index_from_owner = lambda layer_id, full=False: _FakeWork()
        pool._ensure_remote_index_layer(6)
        self.assertEqual(pool.remote_index_layer_id, 6)


class TestFinalizeAndPrefetchKv(CustomTestCase):
    def _pool(self):
        pool = _make_pool()
        pool.remote_kv_layer_id = None
        pool.pending_remote_kv_broadcast = False
        pool.pending_remote_kv_layer_id = None
        pool.pending_kv_works = []
        pool.kicks = []
        pool._broadcast_kv_from_owner = lambda layer_id, bound=None: (
            pool.kicks.append((layer_id, bound)) or []
        )
        return pool

    def test_finalize_promotes_pending_layer_id(self):
        pool = self._pool()
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        pool.pending_kv_works = [_FakeWork()]
        pool._finalize_pending_kv_broadcast()
        self.assertEqual(pool.remote_kv_layer_id, 7)
        self.assertFalse(pool.pending_remote_kv_broadcast)
        self.assertIsNone(pool.pending_remote_kv_layer_id)
        self.assertEqual(pool.pending_kv_works, [])

    def test_finalize_without_promotion_keeps_remote_id(self):
        pool = self._pool()
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        pool._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        self.assertIsNone(pool.remote_kv_layer_id)
        self.assertFalse(pool.pending_remote_kv_broadcast)

    def test_finalize_no_wait_skips_work_handles(self):
        pool = self._pool()
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        work = _FakeWork()
        pool.pending_kv_works = [work]
        pool._finalize_pending_kv_broadcast(wait_works=False)
        self.assertFalse(work.waited)
        self.assertEqual(pool.pending_kv_works, [])

    def test_finalize_noop_when_nothing_pending(self):
        pool = self._pool()
        pool._finalize_pending_kv_broadcast()
        self.assertIsNone(pool.remote_kv_layer_id)

    def test_prefetch_passes_row_bound_through(self):
        # The row-space bound (pool rows, NOT prefix tokens) must reach the
        # broadcast untouched: this is the token/row semantics contract of the
        # sliced prefix broadcast.
        pool = self._pool()
        pool.prefetch_kv_buffer(7, used_bound=384)
        self.assertEqual(pool.kicks, [(7, 384)])
        self.assertEqual(pool._kv_used_bound, 384)
        self.assertTrue(pool.pending_remote_kv_broadcast)
        self.assertEqual(pool.pending_remote_kv_layer_id, 7)

    def test_prefetch_dedups_cached_and_pending_layer(self):
        pool = self._pool()
        pool.remote_kv_layer_id = 7
        pool.prefetch_kv_buffer(7)
        self.assertEqual(pool.kicks, [])

        pool.remote_kv_layer_id = None
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 7
        pool.prefetch_kv_buffer(7)
        self.assertEqual(pool.kicks, [])

    def test_prefetch_supersedes_stale_pending_broadcast(self):
        pool = self._pool()
        stale = _FakeWork()
        pool.pending_remote_kv_broadcast = True
        pool.pending_remote_kv_layer_id = 3
        pool.pending_kv_works = [stale]
        pool.prefetch_kv_buffer(7, used_bound=5)
        self.assertEqual(pool.kicks, [(7, 5)])
        self.assertEqual(pool.pending_remote_kv_layer_id, 7)
        self.assertEqual(pool.pending_kv_works, [])


class TestDsaSplitRowBound(CustomTestCase):
    def test_bound_is_max_chunk_first_row(self):
        # Three chunks of one batch, each starting at its own pool row (the
        # request slots are not zero-based). The bound must be the max FIRST
        # ROW, never a token count.
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([500, 501, 9100, 9101, 17700]),
            extend_seq_lens_cpu=[2, 2, 1],
        )
        self.assertEqual(_dsa_split_row_bound(fb), 17700)

    def test_bound_with_slot_base_offset(self):
        # Regression shape for the base-offset bug: with a non-zero slot base
        # the chunk first-ROW values differ from the chunk token counts, and
        # only the row value is a valid broadcast bound.
        base = 384
        loc = [base, 10, base + 2, 11, 12]
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor(loc), extend_seq_lens_cpu=[2, 3]
        )
        # Chunk 0 starts at pool row 384, chunk 1 at row 386; a token-space
        # computation would have yielded 3.
        self.assertEqual(_dsa_split_row_bound(fb), 386)

    def test_cached_bound_is_reused(self):
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([5, 6]),
            extend_seq_lens_cpu=[1, 1],
            _dsa_split_row_bound=42,
        )
        self.assertEqual(_dsa_split_row_bound(fb), 42)

    def test_degenerate_batches_return_none(self):
        self.assertIsNone(_dsa_split_row_bound(SimpleNamespace()))
        self.assertIsNone(
            _dsa_split_row_bound(
                SimpleNamespace(
                    out_cache_loc=torch.tensor([1, 2]), extend_seq_lens_cpu=[]
                )
            )
        )
        self.assertIsNone(
            _dsa_split_row_bound(
                SimpleNamespace(
                    out_cache_loc=torch.tensor([1, 2]), extend_seq_lens_cpu=[0, 0]
                )
            )
        )
        self.assertIsNone(
            _dsa_split_row_bound(
                SimpleNamespace(
                    out_cache_loc=torch.tensor([1, 2]), extend_seq_lens_cpu=[4, 4]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
