"""CPU unit tests for the NPU DSV4 cache layer-split pool."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_cache_layer_split import (
    KIND_C4,
    KIND_C128,
    KIND_INDEX,
    KIND_SWA,
    LayerSplitDSV4NPUTokenToKVPool,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout import NPUDSV4KVLayout
from sglang.srt.layers.cp.utils import get_layer_shard_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

POOL = LayerSplitDSV4NPUTokenToKVPool

_MIXIN_MODULE = "sglang.srt.hardware_backend.npu.layer_split_mixin"

# DeepSeek-V4-Flash style layout: 12 stage layers, ratios 0/4/128 interleaved.
RATIOS = [0, 4, 128, 0, 4, 128, 0, 4, 128, 0, 4, 128]


_POOL_MODULE = "sglang.srt.hardware_backend.npu.dsv4.dsv4_cache_layer_split"


def _patch_arch35():
    """CPU test env: torch_npu imports but acl may not; force the bf16 path."""
    return patch(f"{_POOL_MODULE}.is_npu_arch35", lambda: False)


def _make_pool(
    *,
    layer_num=len(RATIOS),
    start_layer=0,
    shard_rank=1,
    shard_size=4,
    swa_pages=8,
    c4_pages=8,
    c128_pages=4,
    kernel_page_size=8,
    c4_page_size=2,
    c128_page_size=1,
    kv_dim=6,
):
    """Build a pool skeleton via __new__ (no device work) with real shard math
    and the DSV4 multi-pool structure."""
    pool = POOL.__new__(POOL)
    pool._init_layer_split(shard_rank, shard_size)
    pool.layer_num = layer_num
    pool.start_layer = start_layer
    pool.compression_ratios = list(RATIOS)
    pool._ls_stage_start = start_layer
    pool._ls_stage_end = start_layer + layer_num
    pool._ls_owned_start, pool._ls_owned_end = get_layer_shard_range(
        shard_rank, shard_size, layer_num
    )
    pool.layer_shard_start = pool._ls_stage_start + pool._ls_owned_start
    pool._live_pages = {}
    pool._broadcast_pages = {}
    pool._remote_index_scale_scratch = None
    pool.layer_transfer_counter = None
    pool.memory_saver_adapter = SimpleNamespace(
        region=lambda *_: contextlib.nullcontext()
    )

    def _sub(num_layers, pages, kp):
        p = SimpleNamespace(
            kv_buffer=[torch.zeros(pages, kp, 1, kv_dim) for _ in range(num_layers)],
            layout=NPUDSV4KVLayout.V4_BF16,
            size=pages * kp,
            page_size=kp,
            global_page_size=kp,
            kernel_page_size=kp,
            store_dtype=torch.bfloat16,
            device="cpu",
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
        )
        p.fused_writes = []

        def set_key_buffer_fused(slot, loc, kv, _p=p):
            _p.fused_writes.append((slot, loc, kv))

        p.set_key_buffer_fused = set_key_buffer_fused
        return p

    owned = range(start_layer + pool._ls_owned_start, start_layer + pool._ls_owned_end)
    n_swa = pool._ls_owned_end - pool._ls_owned_start
    n_c4 = sum(1 for g in owned if RATIOS[g - start_layer] == 4)
    n_c128 = sum(1 for g in owned if RATIOS[g - start_layer] == 128)

    pool.swa_kv_pool = _sub(n_swa, swa_pages, kernel_page_size)
    pool.c4_kv_pool = _sub(n_c4, c4_pages, c4_page_size)
    pool.c128_kv_pool = _sub(n_c128, c128_pages, c128_page_size)
    pool.kv_pools = {4: pool.c4_kv_pool, 128: pool.c128_kv_pool}
    pool.index_pools = {
        4: SimpleNamespace(
            index_k_buffer=[
                torch.zeros(c4_pages, c4_page_size, 1, 16) for _ in range(n_c4)
            ],
            index_scale_buffer=[
                torch.zeros(c4_pages, c4_page_size, 1, 1) for _ in range(n_c4)
            ],
            size=c4_pages * c4_page_size,
            _kernel_page_size=c4_page_size,
            index_head_dim=16,
            has_npu_storage=True,
            device="cpu",
        )
    }

    def get_index_k(slot):
        return pool.index_pools[4].index_k_buffer[slot]

    def get_index_scale(slot):
        return pool.index_pools[4].index_scale_buffer[slot]

    pool.index_pools[4].get_index_k = get_index_k
    pool.index_pools[4].get_index_scale = get_index_scale
    pool.index_pools[4].index_fused_writes = []
    pool.index_pools[4].set_index_fused = lambda slot, loc, kv, _p=pool.index_pools[4]: (
        _p.index_fused_writes.append((slot, loc, kv))
    )
    pool.index_pools[4].index_k_scale_writes = []
    pool.index_pools[4].set_index_k_scale_buffer = (
        lambda slot, loc, kv, scale, _p=pool.index_pools[4]: (
            _p.index_k_scale_writes.append((slot, loc, kv, scale))
        )
    )
    pool._indexer_pool = lambda ratio: pool.index_pools[ratio]

    # layer_mapping: global entries for this stage.
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

    counts = {0: 0, 4: 0, 128: 0}
    pool.layer_mapping = [None] * (start_layer + layer_num)
    for gid in range(start_layer, start_layer + layer_num):
        ratio = RATIOS[gid - start_layer]
        pool.layer_mapping[gid] = DeepSeekV4LayerItem(
            compress_ratio=ratio,
            compress_layer_id=counts[ratio],
            compress_kv_pool=None if ratio == 0 else pool.kv_pools[ratio],
        )
        counts[ratio] += 1

    pool._build_owned_slot_maps()
    with _patch_arch35():
        pool._create_remote_scratches()
    return pool


class TestOwnedSlotMaps(CustomTestCase):
    def test_swa_slots_cover_every_owned_layer(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        # 12 layers / 4 shards -> rank 1 owns locals [3, 6). The SWA pool
        # holds every stage layer, so the remap covers all owned locals.
        self.assertEqual(pool._ls_owned_start, 3)
        self.assertEqual(pool._ls_owned_end, 6)
        self.assertEqual([pool._swa_local_layer_id(g) for g in (3, 4, 5)], [0, 1, 2])
        with self.assertRaises(AssertionError):
            pool._swa_local_layer_id(6)  # local 6 belongs to rank 2

    def test_compress_slots_rebase_ratio_subsets(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        # RATIOS: [0,4,128,0,4,128,...]; c4 layers are stage-locals 1,4,7,10
        # with compress_layer_id 0..3; rank 1 owns locals 3,4,5 = ratios
        # (0,4,128) -> owned c4 layer is compress_layer_id 1 -> slot 0.
        self.assertEqual(pool._ls_compress_slots[4], {1: 0})
        # c128 layers are stage-locals 2,5,8,11; rank 1 owns local 5 ->
        # compress_layer_id 1 -> slot 0.
        self.assertEqual(pool._ls_compress_slots[128], {1: 0})

    def test_first_rank_owns_dense_prefix(self):
        pool = _make_pool(shard_rank=0, shard_size=4)
        # Rank 0 owns locals [0, 3) = ratios (0, 4, 128): one slot each.
        self.assertEqual([pool._swa_local_layer_id(g) for g in (0, 1, 2)], [0, 1, 2])
        self.assertEqual(pool._ls_compress_slots[4], {0: 0})
        self.assertEqual(pool._ls_compress_slots[128], {0: 0})

    def test_sub_pools_shrink_to_owned_counts(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        self.assertEqual(len(pool.swa_kv_pool.kv_buffer), 3)
        self.assertEqual(len(pool.c4_kv_pool.kv_buffer), 1)
        self.assertEqual(len(pool.c128_kv_pool.kv_buffer), 1)
        self.assertEqual(len(pool.index_pools[4].index_k_buffer), 1)

    def test_layer_kinds_mapping(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        self.assertEqual(pool._split_layer_kinds(0), (KIND_SWA,))
        self.assertEqual(pool._split_layer_kinds(1), (KIND_SWA, KIND_C4, KIND_INDEX))
        self.assertEqual(pool._split_layer_kinds(2), (KIND_SWA, KIND_C128))
        # ratio 1/2 layers are rejected at __init__; simulate directly.
        pool.compression_ratios[3] = 2
        with self.assertRaises(ValueError):
            pool._split_layer_kinds(3)


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _FakeParallel:
    def __init__(self, ranks):
        self.attn_cp_group = SimpleNamespace(ranks=ranks, device_group="fake-group")


class TestPageBroadcast(CustomTestCase):
    def setUp(self):
        self._parallel_patch = patch(
            f"{_MIXIN_MODULE}.get_parallel",
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

    def test_owner_gathers_live_pages(self):
        pool = _make_pool(shard_rank=0, shard_size=4)
        pages = torch.tensor([1, 3])
        pool.set_live_pages({KIND_SWA: pages})
        pool.swa_kv_pool.kv_buffer[0][1].fill_(7.0)
        pool.swa_kv_pool.kv_buffer[0][3].fill_(9.0)

        units = pool._split_broadcast_units(KIND_SWA, 0, None)
        ((src, _dest),) = units
        # The owner packs exactly the live pages, in sorted-unique order.
        self.assertTrue(torch.equal(src[0], pool.swa_kv_pool.kv_buffer[0][1]))
        self.assertTrue(torch.equal(src[1], pool.swa_kv_pool.kv_buffer[0][3]))
        # The broadcast root tensor is the owner's packed gather.
        self.broadcasts.clear()
        pool._broadcast_kind(KIND_SWA, 0, None, wait=True)
        ((tensor, src_rank, _),) = self.broadcasts
        self.assertEqual(src_rank, 10)
        self.assertTrue(torch.equal(tensor[0], pool.swa_kv_pool.kv_buffer[0][1]))

    def test_non_owner_receives_and_scatters_into_scratch(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pages = torch.tensor([2, 5])
        pool.set_live_pages({KIND_SWA: pages})
        # Layer 4 (ratio 0) is owned by rank 1; pick a non-owned ratio-0
        # layer owned by rank 2: local 6 -> rank 2.
        dest_payload = torch.full((2, *pool._split_scratch(KIND_SWA).shape[1:]), 5.0)

        def _fake_broadcast(tensor, *, src, group, async_op):
            tensor.copy_(dest_payload)
            return _FakeWork()

        self._bcast_patch.stop()
        self._bcast_patch = patch.object(
            torch.distributed, "broadcast", _fake_broadcast, create=True
        )
        self._bcast_patch.start()

        pool._ensure_remote_kind(KIND_SWA, 6)
        scratch = pool._split_scratch(KIND_SWA)
        self.assertTrue(torch.equal(scratch[2], dest_payload[0]))
        self.assertTrue(torch.equal(scratch[5], dest_payload[1]))
        # Rows outside the live-page set are untouched (zeros).
        self.assertEqual(scratch[0].abs().sum().item(), 0.0)

    def test_index_kind_broadcasts_k_and_scale(self):
        pool = _make_pool(shard_rank=0, shard_size=4)
        pages = torch.tensor([0, 1])
        pool.set_live_pages({KIND_C4: pages})
        # Rank 0 owns local 1 (ratio 4, compress_layer_id 0 -> slot 0).
        pool.index_pools[4].index_k_buffer[0].fill_(3.0)
        pool.index_pools[4].index_scale_buffer[0].fill_(0.5)
        units = pool._split_broadcast_units(KIND_INDEX, 1, None)
        self.assertEqual(len(units), 2)
        (src_k, _), (src_scale, _) = units
        self.assertTrue(torch.equal(src_k[0], pool.index_pools[4].index_k_buffer[0][0]))
        self.assertTrue(
            torch.equal(src_scale[0], pool.index_pools[4].index_scale_buffer[0][0])
        )

    def test_missing_live_pages_raises(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        with self.assertRaises(AssertionError):
            pool._split_broadcast_units(KIND_SWA, 6, None)


class TestSkipWrites(CustomTestCase):
    def setUp(self):
        p1 = patch(f"{_POOL_MODULE}.is_npu_arch35", lambda: False)
        p2 = patch(
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool.is_npu_arch35",
            lambda: False,
        )
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_non_owner_swa_write_is_skipped(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        loc = torch.tensor([0])
        cache = torch.ones(1, 1, 6)
        pool.set_swa_buffer(6, loc, cache)  # local 6 belongs to rank 2
        self.assertEqual(pool.swa_kv_pool.kv_buffer[0].abs().sum().item(), 0.0)

    def test_owned_swa_write_rebases_slot(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        loc = torch.tensor([0])
        cache = torch.ones(1, 1, 6)
        pool.set_swa_buffer(4, loc, cache)  # local 4 -> owned swa slot 1
        self.assertEqual(pool.swa_kv_pool.kv_buffer[1][0, 0, 0, :].sum().item(), 6.0)

    def test_non_owner_compress_write_is_skipped(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        loc = torch.tensor([0])
        kv = torch.ones(1, 6)
        pool.set_compress_buffer(2, loc, kv, None, from_indexer=False)
        self.assertEqual(pool.c128_kv_pool.kv_buffer[0].abs().sum().item(), 0.0)

    def test_owned_compress_write_rebases_slot(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        loc = torch.tensor([0])
        kv = torch.ones(1, 6)
        pool.set_compress_buffer(4, loc, kv, None, from_indexer=False)
        # local 4 (ratio 4, compress_layer_id 1) -> owned c4 slot 0, routed
        # through the (CUDA-style) fused setter on the CPU skeleton.
        slot, written_loc, written_kv = pool.c4_kv_pool.fused_writes[0]
        self.assertEqual(slot, 0)
        self.assertTrue(torch.equal(written_loc, loc))
        self.assertTrue(torch.equal(written_kv, kv))

    def test_owned_indexer_write_rebases_slot(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        loc = torch.tensor([0])
        kv = torch.ones(1, 16)
        pool.set_compress_buffer(4, loc, kv, None, from_indexer=True)
        # local 4 (ratio 4, compress_layer_id 1) -> owned indexer slot 0,
        # routed through the (CUDA-style) fused setter on the CPU skeleton.
        slot, _, _ = pool.index_pools[4].index_fused_writes[0]
        self.assertEqual(slot, 0)


class TestPatchWrite(CustomTestCase):
    """Patch-write set path: the non-owner seeds its scratch with the
    CP-gathered rows locally and the get skips the broadcast entirely."""

    def setUp(self):
        for mod in (
            _POOL_MODULE,
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool",
        ):
            p = patch(f"{mod}.is_npu_arch35", lambda: False)
            p.start()
            self.addCleanup(p.stop)
        self.broadcasts = []

        def _record(tensor, *, src, group, async_op):
            self.broadcasts.append((tensor, src, group))
            return _FakeWork()

        self._bcast_patch = patch.object(
            torch.distributed, "broadcast", _record, create=True
        )
        self._bcast_patch.start()
        self.addCleanup(self._bcast_patch.stop)
        self._parallel_patch = patch(
            f"{_MIXIN_MODULE}.get_parallel",
            lambda: _FakeParallel([10, 11, 12, 13]),
        )
        self._parallel_patch.start()
        self.addCleanup(self._parallel_patch.stop)

    def test_full_gather_seeds_scratch_and_skips_broadcast(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages({KIND_SWA: torch.tensor([1, 3])})
        cache = torch.full((2, 1, 6), 3.0)
        # Layer 6 (ratio 0) is owned by rank 2; this rank only holds shards.
        pool.set_swa_buffer(6, torch.tensor([9, 20]), cache, cp_gathered_full=True)
        scratch = pool._split_scratch(KIND_SWA)
        self.assertTrue(torch.equal(scratch.flatten(0, 1)[9], cache[0]))
        self.assertTrue(torch.equal(scratch.flatten(0, 1)[20], cache[1]))
        self.assertEqual(pool._split_state(KIND_SWA).remote_layer_id, 6)
        # The registered copy makes the read skip the broadcast entirely.
        pool._ensure_remote_kind(KIND_SWA, 6)
        self.assertEqual(len(self.broadcasts), 0)
        self.assertTrue(torch.equal(pool.get_swa_buffer(6).flatten(0, 1)[9], cache[0]))

    def test_full_gather_seeds_indexer_k_and_flat_scale(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages(
            {
                KIND_SWA: torch.tensor([1, 3]),
                KIND_C4: torch.tensor([0]),
                KIND_INDEX: torch.tensor([0]),
            }
        )
        # Layer 7 (ratio 4) belongs to rank 2: this rank seeds its scratch
        # with the CP-gathered rows; the owner writes its pool instead.
        loc = torch.tensor([3, 11])
        kv = torch.full((2, 16), 2.0)
        # The indexer scale rides the wire flat ([T]); the k rows are [T, D].
        scale = torch.full((2,), 0.25)
        pool.set_compress_buffer(
            7, loc, kv, scale, from_indexer=True, cp_gathered_full=True
        )
        # Registered copy: the read skips the broadcast entirely.
        pool._ensure_remote_kind(KIND_INDEX, 7)
        self.assertEqual(len(self.broadcasts), 0)
        k_scratch = pool._split_scratch(KIND_INDEX)
        # kv rides the wire as [T, D]; the scratch rows are [1, D].
        self.assertTrue(
            torch.equal(
                k_scratch.flatten(0, 1)[3], kv[0].reshape(1, -1).to(k_scratch.dtype)
            )
        )
        self.assertTrue(
            torch.equal(
                k_scratch.flatten(0, 1)[11], kv[1].reshape(1, -1).to(k_scratch.dtype)
            )
        )
        scale_scratch = pool._remote_index_scale_scratch
        # scale rides the wire flat ([T]); the scratch rows are [1, 1].
        self.assertTrue(
            torch.equal(
                scale_scratch.flatten(0, 1)[11],
                scale[1].reshape(1, 1).to(scale_scratch.dtype),
            )
        )
        # The owned rank never touches the scratch (its get reads the pool).
        self.assertEqual(pool.index_pools[4].index_k_scale_writes, [])

    def test_prefix_without_kick_falls_back_to_broadcast(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages({KIND_SWA: torch.tensor([2, 5])})
        payload = torch.full((2, *pool._split_scratch(KIND_SWA).shape[1:]), 5.0)

        def _deliver(tensor, *, src, group, async_op):
            self.broadcasts.append((tensor, src, group))
            tensor.copy_(payload)
            return _FakeWork()

        self._bcast_patch.stop()
        self._bcast_patch = patch.object(
            torch.distributed, "broadcast", _deliver, create=True
        )
        self._bcast_patch.start()
        # cp_gathered_full=False and no pending kick: no seed, the read
        # falls back to the synchronous full-live-page broadcast.
        pool.set_swa_buffer(6, torch.tensor([17]), torch.ones(1, 1, 6))
        self.assertIsNone(pool._split_state(KIND_SWA).remote_layer_id)
        pool._ensure_remote_kind(KIND_SWA, 6)
        self.assertEqual(len(self.broadcasts), 1)
        self.assertTrue(torch.equal(pool._split_scratch(KIND_SWA)[2], payload[0]))

    def test_kick_then_patch_write_covers_prefix_and_new(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pages = torch.tensor([2, 5, 7])
        pool.set_live_pages({KIND_SWA: pages})
        payload = torch.full((3, *pool._split_scratch(KIND_SWA).shape[1:]), 5.0)

        def _deliver(tensor, *, src, group, async_op):
            self.broadcasts.append((tensor, src, group))
            tensor.copy_(payload)
            return _FakeWork()

        self._bcast_patch.stop()
        self._bcast_patch = patch.object(
            torch.distributed, "broadcast", _deliver, create=True
        )
        self._bcast_patch.start()
        pool.prefetch_layer(6)  # kick: pending broadcast for the prefix pages
        st = pool._split_state(KIND_SWA)
        self.assertTrue(st.pending_remote_broadcast)
        self.assertEqual(st.pending_remote_layer_id, 6)
        # The layer's set patches the new-token rows on top of the kicked
        # broadcast (loc 57 sits inside page 7 = flatten rows 56..63).
        cache = torch.full((1, 1, 6), 9.0)
        pool.set_swa_buffer(6, torch.tensor([57]), cache, cp_gathered_full=False)
        self.assertFalse(st.pending_remote_broadcast)
        self.assertEqual(st.remote_layer_id, 6)
        scratch = pool._split_scratch(KIND_SWA)
        self.assertTrue(torch.equal(scratch[2], payload[0]))  # prefix: kicked
        self.assertTrue(torch.equal(scratch[5], payload[1]))
        self.assertTrue(
            torch.equal(scratch.flatten(0, 1)[57], cache[0])
        )  # new: patched
        self.assertTrue(
            torch.equal(scratch.flatten(0, 1)[56], payload[2][0])
        )  # prefix row inside page 7: kicked
        # Registered copy: the read skips the broadcast.
        pool._ensure_remote_kind(KIND_SWA, 6)
        self.assertEqual(len(self.broadcasts), 1)

    def test_prefetch_layer_kicks_every_kind_of_the_layer(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages({KIND_SWA: torch.tensor([1]), KIND_C4: torch.tensor([0])})
        pool.prefetch_layer(4)  # ratio-4 layer: swa + c4 + index(k, scale)
        # swa + c4 + index-k + index-scale = 4 broadcast units.
        self.assertEqual(len(self.broadcasts), 4)
        for kind in (KIND_SWA, KIND_C4, KIND_INDEX):
            st = pool._split_state(kind)
            self.assertTrue(st.pending_remote_broadcast)
            self.assertEqual(st.pending_remote_layer_id, 4)

    def test_foreign_pending_survives_registered_read(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages({KIND_SWA: torch.tensor([1, 3])})
        # Kick layer 6 (foreign pending), then register layer 4 via a
        # full-gather seed (owned layer, patch-write).
        pool.prefetch_layer(6)
        st = pool._split_state(KIND_SWA)
        self.assertEqual(st.pending_remote_layer_id, 6)
        pool.set_swa_buffer(
            4, torch.tensor([9]), torch.full((1, 1, 6), 3.0), cp_gathered_full=True
        )
        # The registered read must NOT consume the foreign pending — the
        # early kick relies on this to overlap the whole layer's compute.
        pool._ensure_remote_kind(KIND_SWA, 4)
        self.assertTrue(st.pending_remote_broadcast)
        self.assertEqual(st.pending_remote_layer_id, 6)
        self.assertEqual(len(self.broadcasts), 1)  # only the kick, no re-read
        # The next layer's own set consumes the kick.
        pool.set_swa_buffer(
            6, torch.tensor([17]), torch.full((1, 1, 6), 9.0), cp_gathered_full=False
        )
        self.assertFalse(st.pending_remote_broadcast)
        self.assertEqual(st.remote_layer_id, 6)

    def test_compute_broadcast_pages(self):
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_cache_layer_split import (
            compute_broadcast_pages,
        )

        live = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        # Pages 0-1 fully written (4 slots each), page 3 partial (1 slot):
        # fully-covered pages drop out, partial and untouched pages stay.
        loc = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12])
        subset = compute_broadcast_pages(live, loc, page_size=4)
        self.assertTrue(torch.equal(subset, torch.tensor([2, 3, 4, 5, 6, 7])))
        # No write loc: no page is covered, fall back to None (full live).
        self.assertIsNone(compute_broadcast_pages(live, None, 4))
        # Empty write loc: nothing covered, every live page stays.
        subset = compute_broadcast_pages(live, torch.empty(0, dtype=torch.long), 4)
        self.assertTrue(torch.equal(subset, live))
        # Long radix prefix: pages 0-5 fully written, page 6 partial ->
        # the subset is exactly the untouched prefix pages.
        loc = torch.arange(0, 26)
        subset = compute_broadcast_pages(live, loc, page_size=4)
        self.assertTrue(torch.equal(subset, torch.tensor([6, 7])))

    def test_subset_kick_delivers_only_subset(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        # Page 2 will be fully covered by this forward's writes; pages 5/7
        # (prefix + partial) need the broadcast.
        pool.set_live_pages(
            {KIND_SWA: torch.tensor([2, 5, 7])},
            broadcast_pages_by_kind={KIND_SWA: torch.tensor([5, 7])},
        )
        payload = torch.full((2, *pool._split_scratch(KIND_SWA).shape[1:]), 5.0)

        def _deliver(tensor, *, src, group, async_op):
            self.broadcasts.append((tensor, src, group))
            tensor.copy_(payload)
            return _FakeWork()

        self._bcast_patch.stop()
        self._bcast_patch = patch.object(
            torch.distributed, "broadcast", _deliver, create=True
        )
        self._bcast_patch.start()
        pool.prefetch_layer(6)
        # The kick packed exactly the subset (2 pages), not the live set (3).
        (tensor, _, _) = self.broadcasts[0]
        self.assertEqual(tensor.shape[0], 2)
        st = pool._split_state(KIND_SWA)
        self.assertEqual(st.pending_remote_layer_id, 6)
        # The set fully covers page 2 and patches one row of page 7.
        loc = torch.tensor(list(range(16, 24)) + [57])
        cache = torch.full((9, 1, 6), 9.0)
        pool.set_swa_buffer(6, loc, cache, cp_gathered_full=False)
        self.assertEqual(st.remote_layer_id, 6)
        scratch = pool._split_scratch(KIND_SWA)
        # Fully-covered page 2: written by the set alone (no broadcast rows).
        self.assertTrue(
            torch.equal(scratch.flatten(0, 1)[16:24].reshape(8, 1, 6), cache[:8])
        )
        # Page 5: broadcast-delivered (not touched by the set).
        self.assertTrue(torch.equal(scratch[5], payload[0]))
        # Page 7: broadcast-delivered, one row patched by the set.
        self.assertTrue(torch.equal(scratch.flatten(0, 1)[56], payload[1][0]))
        self.assertTrue(torch.equal(scratch.flatten(0, 1)[57], cache[8]))
        # Registered copy: the read skips the broadcast.
        pool._ensure_remote_kind(KIND_SWA, 6)
        self.assertEqual(len(self.broadcasts), 1)

    def test_maybe_prefetch_gating(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.set_live_pages({KIND_SWA: torch.tensor([1]), KIND_C4: torch.tensor([0])})

        def _fb(is_extend, prefix_lens):
            mode = SimpleNamespace(is_extend=lambda: is_extend)
            fb = SimpleNamespace(forward_mode=mode)
            if prefix_lens is not None:
                fb.extend_prefix_lens_cpu = prefix_lens
            return fb

        # No prefix: the next layer's set seeds its whole scratch.
        pool.maybe_prefetch_next_layer(4, _fb(True, [0, 0]))
        self.assertEqual(len(self.broadcasts), 0)
        # Non-extend mode: no kick.
        pool.maybe_prefetch_next_layer(4, _fb(False, [3, 0]))
        self.assertEqual(len(self.broadcasts), 0)
        # Past the stage end: no kick.
        pool.maybe_prefetch_next_layer(len(RATIOS), _fb(True, [3, 0]))
        self.assertEqual(len(self.broadcasts), 0)
        # Prefix present: the kick fires for the ratio-4 layer (4 units).
        pool.maybe_prefetch_next_layer(4, _fb(True, [3, 0]))
        self.assertEqual(len(self.broadcasts), 4)


class TestPDTransfer(CustomTestCase):
    """PD owned-layer transfer hooks: the pool must report its entries in the
    owned contiguous global range the Ascend conn slices decode buffers by."""

    def test_layer_shard_start_is_global_owned_start(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        self.assertEqual(pool.layer_shard_start, 3)
        pool = _make_pool(shard_rank=0, shard_size=4)
        self.assertEqual(pool.layer_shard_start, 0)

    def test_swa_layer_ids_are_owned_only(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        self.assertEqual(pool.swa_layer_ids(), [3, 4, 5])
        pool = _make_pool(shard_rank=2, shard_size=4)
        self.assertEqual(pool.swa_layer_ids(), [6, 7, 8])

    def test_transfer_span_is_owned_layer_count(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        # Owned globals 3,4,5 = ratios (0,4,128): one ratio-4 layer, so the
        # contiguous buf infos register 3 entries (C4 KV, index K, scale),
        # while the span must cover all 3 owned layers.
        ptrs, _, _ = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 3)
        self.assertEqual(pool.get_kv_transfer_layer_span(len(ptrs)), 3)
        # A wrong entry count breaks the start/end contract: reject it.
        with self.assertRaises(AssertionError):
            pool.get_kv_transfer_layer_span(4)

    def test_transfer_span_without_owned_c4_layers(self):
        # RATIOS [0,4,128,...] has 4 ratio-4 layers; 3 shards own none only if
        # shard_size > 4. Use a 1-layer stage per rank instead: 4 layers
        # (0,4,128,0) split over 2 ranks — rank 1 owns locals [2, 4), no c4.
        pool = _make_pool(
            layer_num=4,
            shard_rank=1,
            shard_size=2,
            c4_pages=8,
            c128_pages=4,
        )
        ptrs, _, _ = pool.get_contiguous_buf_infos()
        # c4_kv_pool still exists but holds no owned layers -> no entries.
        self.assertEqual(len(ptrs), 0)
        self.assertEqual(pool.get_kv_transfer_layer_span(0), 2)

    def test_device_pool_infos_filter_sources_to_owned(self):
        pool = _make_pool(shard_rank=1, shard_size=4)
        pool.sources_by_ratio = {4: [1, 4, 7, 10], 128: [2, 5, 8, 11]}

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        captured = {}

        def _fake_super_infos(self):
            captured["sources"] = {r: list(s) for r, s in self.sources_by_ratio.items()}
            return ()

        with patch.object(
            DeepSeekV4TokenToKVPool,
            "get_device_pool_infos",
            _fake_super_infos,
        ):
            self.assertEqual(pool.get_device_pool_infos(), ())
        # The base saw only owned sources, and the full stage list is restored.
        self.assertEqual(captured["sources"], {4: [4], 128: [5]})
        self.assertEqual(pool.sources_by_ratio[4], [1, 4, 7, 10])
        self.assertEqual(pool.sources_by_ratio[128], [2, 5, 8, 11])

    def test_empty_owned_families_report_empty_buf_infos(self):
        # 12 layers / 12 shards: rank 0 owns the single ratio-0 layer — no
        # c4/c128 buffers at all. Its registration components must come back
        # EMPTY, never missing: the bootstrap refuses state wires that differ
        # across one stage, so a rank that dropped an empty component (the
        # real-machine failure: CP0 and CP15 400'd at registration, leaving
        # the prefill stuck at 14/16 workers) can never pair with its peers.
        pool = _make_pool(shard_rank=0, shard_size=12)
        pool.compress_state_pools = [None] * 12
        pool.indexer_compress_state_pools = [None] * 12
        self.assertEqual(len(pool.swa_kv_pool.kv_buffer), 1)
        self.assertEqual(pool.c128_kv_pool.kv_buffer, [])
        self.assertEqual(pool.c4_kv_pool.kv_buffer, [])
        self.assertEqual(pool.get_c128_kv_buf_infos(), ([], [], []))


if __name__ == "__main__":
    unittest.main()
