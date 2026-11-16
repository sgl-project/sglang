"""--enable-kda-tp PD decode: a request's head-sharded state slot is allocated at admission.

CPU:
- The switch (SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT) defaults on and applies only under KDA TP
  without ping-pong tracking slots.
- Staging lifecycle with a request-row key: assign at prealloc -> rekey onto the state slot at
  admission -> queue_import / take_pending / finish_import import into that slot.
- Exhaustion: staging full blocks prealloc; a stage whose request row was freed outside
  free_mamba_cache is reclaimed; admission with no free state slot leaves the request waiting
  with its stage intact, and it is admitted once a slot frees.
- Leaving during the handoff (abort / failed transfer): free_mamba_cache returns the stage and
  frees no state slot; after admission it frees the state slot as usual.
- The prealloc capacity check counts no state slot for a lazy request.
- Retraction under KDA TP rebootstraps, so a retracted request re-enters prealloc and defers
  its slot again.
"""

import contextlib
import types
import unittest
import unittest.mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEADS, HEAD_DIM, KERNEL, LAYERS = 8, 4, 4, 3


def _par(world, rank=0):
    return types.SimpleNamespace(
        attn_tp_size=1,
        attn_dp_size=world,
        tp_size=world,
        kda_weight_shard_size=None,
        enable_kda_tp=True,
        attn_dp_rank=rank,
    )


def _staging(world, slots):
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
    )
    from sglang.srt.layers.communication import k3_kda_tp

    shape = KimiLinearStateShape.create(
        tp_world_size=world, num_heads=HEADS, head_dim=HEAD_DIM, conv_kernel_size=KERNEL
    )
    params = KimiLinearCacheParams(shape=shape, layers=list(range(LAYERS)))
    pool = types.SimpleNamespace(
        mamba_layer_ids=list(range(LAYERS)),
        state_allocation_region=contextlib.nullcontext,
    )
    with unittest.mock.patch.object(k3_kda_tp, "get_parallel", lambda: _par(world)):
        return k3_kda_tp.PdStateStaging(pool, params, slots, "cpu")


def _req(row, lazy=True, staged=None, slot=None):
    return types.SimpleNamespace(
        kv=_Kv(row, slot), kda_tp_lazy_mamba=lazy, kda_tp_staged_slot=staged
    )


class _Kv:
    def __init__(self, row, slot):
        self.req_pool_idx = row
        self.mamba_pool_idx = None if slot is None else torch.tensor(slot)
        self.mamba_needs_clear = False

    @property
    def holds_mamba(self):
        return self.mamba_pool_idx is not None


def _pool(staging, first_slot=1, n_slots=2, rows=8):
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    pool = types.SimpleNamespace(
        mamba_allocator=MambaSlotAllocator(
            size=n_slots, device="cpu", first_slot=first_slot
        ),
        req_index_to_mamba_index_mapping=torch.zeros(rows, dtype=torch.int32),
        mamba_pool=types.SimpleNamespace(pd_staging=staging),
        free_slots=[],
    )
    from sglang.srt.mem_cache.kv.state_storage import StateStorage

    pool.state = StateStorage.empty("mamba")
    pool.state.pool = pool.mamba_pool
    pool.state.slots = pool.mamba_allocator
    return pool


class TestSwitch(unittest.TestCase):
    def test_default_on_and_gating(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.communication import k3_kda_tp

        self.assertTrue(envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.get())
        with unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 8):
            self.assertTrue(k3_kda_tp.lazy_mamba_slot(False))
            self.assertFalse(
                k3_kda_tp.lazy_mamba_slot(True)
            )  # ping-pong tracking slots
            with envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.override(False):
                self.assertFalse(k3_kda_tp.lazy_mamba_slot(False))
        with unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 1):
            self.assertFalse(k3_kda_tp.lazy_mamba_slot(False))

    def test_lazy_keys_never_collide_with_slots(self):
        from sglang.srt.layers.communication import k3_kda_tp

        self.assertEqual(k3_kda_tp.lazy_stage_key(3), -3)
        self.assertLess(k3_kda_tp.lazy_stage_key(1), 1)


class TestLifecycle(unittest.TestCase):
    def test_prealloc_admit_import(self):
        from sglang.srt.disaggregation.decode import _admit_lazy_mamba_slot
        from sglang.srt.layers.communication import k3_kda_tp

        st = _staging(4, 2)
        pool = _pool(st)
        req = _req(row=3)
        key = k3_kda_tp.lazy_stage_key(3)
        stage = st.assign(key)  # prealloc (_mamba_payload)
        req.kda_tp_staged_slot = key
        self.assertTrue(_admit_lazy_mamba_slot(pool, req))
        slot = int(req.kv.mamba_pool_idx)
        self.assertEqual(slot, 1)
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertFalse(req.kda_tp_lazy_mamba)
        self.assertEqual(int(pool.req_index_to_mamba_index_mapping[3]), slot)
        self.assertEqual(req.kda_tp_staged_slot, slot)
        self.assertEqual(req.kda_tp_state_slot, slot)
        st.queue_import(req.kda_tp_staged_slot)  # _get_new_prebuilt_batch
        pairs = st.take_pending()
        self.assertEqual(pairs, [(stage, slot)])
        st.finish_import(pairs, None)
        st._reclaim()  # CPU: no event, the stage is free at the next check
        self.assertEqual(sorted(st._free), [0, 1])
        self.assertEqual(st._stage_of, {})

    def test_admission_is_a_noop_for_eager_requests(self):
        from sglang.srt.disaggregation.decode import _admit_lazy_mamba_slot

        pool = _pool(_staging(4, 1))
        req = _req(row=2, lazy=False, slot=5)
        self.assertTrue(_admit_lazy_mamba_slot(pool, req))
        self.assertEqual(int(req.kv.mamba_pool_idx), 5)
        self.assertEqual(pool.mamba_allocator.available_size(), 2)

    def test_rekey_guards(self):
        st = _staging(4, 2)
        stage = st.assign(-2)
        st.assign(7)  # left behind by a request whose slot 7 was freed without release
        with self.assertRaisesRegex(RuntimeError, "no staged state"):
            st.rekey(-5, 9)
        st.rekey(-2, 7)
        self.assertEqual(st._stage_of, {7: stage})
        self.assertEqual(len(st._free), 1)
        st.assign(-3)
        st.queue_import(-3)
        with self.assertRaisesRegex(RuntimeError, "queued before admission"):
            st.rekey(-3, 8)


class TestExhaustion(unittest.TestCase):
    def test_staging_full_blocks_prealloc_and_reclaims_freed_rows(self):
        st = _staging(4, 1)
        alloc = types.SimpleNamespace(free_slots=torch.tensor([], dtype=torch.int64))
        st.assign(-4)
        busy_rows = types.SimpleNamespace(free_slots=[1, 2])
        self.assertFalse(st.has_room(alloc, busy_rows))
        self.assertFalse(
            st.has_room(alloc)
        )  # without the request pool nothing is reclaimed
        freed_rows = types.SimpleNamespace(free_slots=[1, 4])
        self.assertTrue(st.has_room(alloc, freed_rows))

    def test_reservations_bound_admissions_before_publish(self):
        # Deferred publish admits several requests before any of them assigns its stage.
        st = _staging(4, 2)
        alloc = types.SimpleNamespace(free_slots=torch.tensor([], dtype=torch.int64))
        rows = types.SimpleNamespace(free_slots=[])
        admitted = 0
        while st.has_room(alloc, rows):
            st.reserve()
            admitted += 1
        self.assertEqual(admitted, 2)
        with self.assertRaises(RuntimeError):
            st.reserve()
        self.assertIsNone(st.bind(9))  # late binding never takes a reserved stage
        st.assign(-1, reserved=True)
        st.unreserve()  # the second request aborted before its publish
        self.assertTrue(st.has_room(alloc, rows))
        st.assign(-2)
        self.assertFalse(st.has_room(alloc, rows))
        with self.assertRaises(RuntimeError):
            st.unreserve()

    def test_no_free_state_slot_keeps_request_waiting(self):
        from sglang.srt.disaggregation.decode import _admit_lazy_mamba_slot
        from sglang.srt.layers.communication import k3_kda_tp

        st = _staging(4, 3)
        pool = _pool(st, n_slots=1)
        a, b = _req(row=1), _req(row=2)
        for r in (a, b):
            r.kda_tp_staged_slot = k3_kda_tp.lazy_stage_key(r.kv.req_pool_idx)
            st.assign(r.kda_tp_staged_slot)
        self.assertTrue(_admit_lazy_mamba_slot(pool, a))
        self.assertFalse(_admit_lazy_mamba_slot(pool, b))
        # b keeps its deferred state and its stage under the row key.
        self.assertTrue(b.kda_tp_lazy_mamba)
        self.assertIsNone(b.kv.mamba_pool_idx)
        self.assertIn(-2, st._stage_of)
        self.assertEqual(int(pool.req_index_to_mamba_index_mapping[2]), 0)
        st.queue_import(a.kda_tp_staged_slot)
        pairs = st.take_pending()
        st.finish_import(pairs, None)  # a imported; it then finishes and frees its slot
        pool.mamba_allocator.free(a.kv.mamba_pool_idx.unsqueeze(0))
        self.assertTrue(_admit_lazy_mamba_slot(pool, b))
        self.assertIn(int(b.kv.mamba_pool_idx), st._stage_of)


class TestLeave(unittest.TestCase):
    def _free(self, pool, req):
        pool.state.release(pool, req)

    def test_leave_during_handoff_frees_stage_only(self):
        from sglang.srt.layers.communication import k3_kda_tp

        st = _staging(4, 1)
        pool = _pool(st)
        pool.enable_mamba_extra_buffer = False
        req = _req(row=5)
        req.kda_tp_staged_slot = k3_kda_tp.lazy_stage_key(5)
        st.assign(req.kda_tp_staged_slot)
        self._free(pool, req)
        self.assertEqual(st._free, [0])
        self.assertEqual(st._stage_of, {})
        self.assertFalse(req.kda_tp_lazy_mamba)
        self.assertEqual(pool.mamba_allocator.available_size(), 2)

    def test_leave_after_admission_frees_state_slot(self):
        from sglang.srt.disaggregation.decode import _admit_lazy_mamba_slot
        from sglang.srt.layers.communication import k3_kda_tp

        st = _staging(4, 1)
        pool = _pool(st)
        pool.enable_mamba_extra_buffer = False
        req = _req(row=5)
        req.kda_tp_staged_slot = k3_kda_tp.lazy_stage_key(5)
        st.assign(req.kda_tp_staged_slot)
        self.assertTrue(_admit_lazy_mamba_slot(pool, req))
        self.assertEqual(pool.mamba_allocator.available_size(), 1)
        self._free(
            pool, req
        )  # left before its import: stage and state slot both return
        self.assertEqual(pool.mamba_allocator.available_size(), 2)
        self.assertEqual(st._free, [0])
        self.assertIsNone(req.kv.mamba_pool_idx)

    def test_prebuilt_completion_frees_stage_by_state_slot(self):
        # Queued for import (staged key cleared), then finished before the import ran.
        st = _staging(4, 1)
        pool = _pool(st)
        pool.enable_mamba_extra_buffer = False
        req = _req(row=2, lazy=False, slot=5)
        req.kda_tp_state_slot = 5
        st.assign(5)
        st.queue_import(5)
        req.kda_tp_staged_slot = None
        self._free(pool, req)
        self.assertEqual(st._free, [0])
        self.assertEqual(st._stage_of, {})
        self.assertEqual(st.take_pending(), [])
        self.assertIsNone(req.kda_tp_state_slot)

    def test_release_without_pd_staging(self):
        # Under --enable-unified-memory the state pool is a StateView, which has no
        # pd_staging attribute at all.
        from sglang.srt.mem_cache.kv.layout.state import StateView

        pool = _pool(None)
        pool.enable_mamba_extra_buffer = False
        view = StateView.__new__(StateView)
        self.assertFalse(hasattr(view, "pd_staging"))
        pool.state.pool = view
        req = _req(row=2, lazy=False, slot=1)
        pool.mamba_allocator.alloc(1)
        self._free(pool, req)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(pool.mamba_allocator.available_size(), 2)


class TestSizingAndCapacity(unittest.TestCase):
    def test_capacity_check_counts_no_slot_for_lazy(self):
        from sglang.srt.mem_cache import allocation
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
        from sglang.srt.mem_cache.kv.state_storage import StateStorage
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        pool = object.__new__(HybridReqToTokenPool)
        pool.state = StateStorage.empty(pool.state_sub_pool)
        pool.mamba_allocator = MambaSlotAllocator(size=1, device="cpu", first_slot=1)
        pool.mamba_allocator.alloc(1)  # no free state slot left
        pool.mamba_initial_tracking_slots = 0
        pool.enable_mamba_extra_buffer = False
        lazy, eager = _req(row=1), _req(row=2, lazy=False)
        for r in (lazy, eager):
            r.kv.mamba_ping_pong_track_buffer = object()
        self.assertTrue(
            allocation.ensure_mamba_capacity(pool, pool.state, [lazy], None)
        )
        self.assertFalse(
            allocation.ensure_mamba_capacity(pool, pool.state, [eager], None)
        )


class TestRetraction(unittest.TestCase):
    def test_retraction_rebootstraps_under_kda_tp(self):
        # A retracted request frees its state slot and re-enters prealloc (rebootstrap), where
        # it defers its slot again; KDA TP never resumes from a rank-local backup.
        from sglang.srt.layers.communication import k3_kda_tp
        from sglang.srt.mem_cache import common

        alloc = types.SimpleNamespace(get_kvcache=lambda: object())
        with unittest.mock.patch.object(k3_kda_tp, "state_shard_width", lambda: 8):
            self.assertTrue(common.retraction_needs_rebootstrap(alloc))


if __name__ == "__main__":
    unittest.main()
