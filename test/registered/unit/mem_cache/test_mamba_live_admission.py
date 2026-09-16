import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_mod
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid: str = "r0"):
    return SimpleNamespace(rid=rid, kv=ReqKvInfo(), finished=lambda: False)


def _pool(*, mamba_size: int, req_rows: int = 1, lazy: bool = False):
    pool = object.__new__(HybridReqToTokenPool)
    ReqToTokenPool.__init__(
        pool,
        size=req_rows,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = lazy
    pool.mamba_allocator = MambaSlotAllocator(size=mamba_size, device="cpu")
    pool.mamba_ckpt_pool = None
    pool.mamba_pool = SimpleNamespace(
        replayssm_spec_write_pos=None,
        replayssm_write_pos=None,
    )
    pool.req_index_to_mamba_index_mapping = torch.zeros(req_rows + 1, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (req_rows + 1, 2), dtype=torch.int64
    )
    return pool


def _scheduler(pool, tree_cache, *, unified: bool = False):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.req_to_token_pool = pool
    scheduler.token_to_kv_pool_allocator = (
        SimpleNamespace(mamba_slot_full_token_cost=lambda: 7)
        if unified
        else SimpleNamespace()
    )
    scheduler.tree_cache = tree_cache
    return scheduler


class _QueuedReq:
    def __init__(self, rid: str):
        self.rid = rid
        self.kv = ReqKvInfo()
        self.initialized = False
        self.beam_group = None
        self.lora_id = None
        self.cache_request_handle = None
        self.return_logprob = False

    def finished(self) -> bool:
        return False

    def init_next_round_input(self, tree_cache) -> None:
        self.initialized = True


def _queued_req(rid: str):
    return _QueuedReq(rid)


class TestHybridReqToTokenPoolMambaAdmission(CustomTestCase):
    def test_allocation_is_atomic_when_mamba_slots_are_short(self):
        pool = _pool(mamba_size=2)
        req = _req()
        free_slots = pool.mamba_allocator.available_size()

        self.assertIsNone(pool.alloc([req]))

        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_exact_fit_installs_main_and_ping_pong_slots(self):
        pool = _pool(mamba_size=3)
        req = _req()

        self.assertEqual(pool.alloc([req]), [1])

        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertEqual(req.kv.req_pool_idx, 1)
        self.assertEqual(req.kv.mamba_pool_idx.item(), 1)
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.tolist(), [2, 3])
        self.assertEqual(
            pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1].tolist(),
            [2, 3],
        )

    def test_request_row_failure_rolls_back_reserved_mamba_slots(self):
        pool = _pool(mamba_size=3)
        self.assertEqual(pool.alloc_rows(1), [1])
        req = _req()
        free_slots = pool.mamba_allocator.available_size()

        self.assertIsNone(pool.alloc([req]))

        self.assertEqual(pool.mamba_allocator.available_size(), free_slots)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)

    def test_admission_counts_only_missing_slots(self):
        pool = _pool(mamba_size=4, lazy=True)
        req = _req()
        self.assertEqual(pool.mamba_admission_slots(req.kv), 2)

        req.kv.mamba_pool_idx = torch.tensor(1)
        self.assertEqual(pool.mamba_admission_slots(req.kv), 1)

        req.kv.mamba_ping_pong_track_buffer = torch.tensor([2, -1])
        self.assertEqual(pool.mamba_admission_slots(req.kv), 0)

    def test_unified_gap_budget_uses_all_missing_mamba_slots(self):
        pool = _pool(mamba_size=3)
        req = _req()
        adder = object.__new__(PrefillAdder)
        adder._mamba_slot_cost = 7
        adder.tree_cache = SimpleNamespace(req_to_token_pool=pool)

        self.assertEqual(adder._mamba_gap_budget_for_req(req), 21)

        req.kv.mamba_pool_idx = torch.tensor(1)
        self.assertEqual(adder._mamba_gap_budget_for_req(req), 14)

        req.kv.mamba_ping_pong_track_buffer = torch.tensor([2, 3])
        self.assertEqual(adder._mamba_gap_budget_for_req(req), 0)

    def test_scheduler_defers_without_mutating_request(self):
        pool = _pool(mamba_size=1, lazy=True)
        req = _req()
        evictions = []
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: None,
            evict_for_alloc=lambda params: evictions.append(params),
        )
        scheduler = _scheduler(pool, tree_cache)

        admitted = Scheduler._ensure_mamba_admission_capacity(scheduler, req, [])

        self.assertFalse(admitted)
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].mamba_num, 1)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)

    def test_scheduler_rechecks_capacity_after_eviction(self):
        pool = _pool(mamba_size=2, lazy=True)
        held = pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(held)
        req = _req()

        def evict(params):
            self.assertEqual(params.mamba_num, 1)
            pool.mamba_allocator.free(held)

        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: None,
            evict_for_alloc=evict,
        )
        scheduler = _scheduler(pool, tree_cache)

        self.assertTrue(Scheduler._ensure_mamba_admission_capacity(scheduler, req, []))
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)

    def test_scheduler_uses_existing_streaming_session_slots(self):
        pool = _pool(mamba_size=1, lazy=True)
        held = pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(held)
        req = _req()
        session_kv = ReqKvInfo(
            req_pool_idx=1,
            kv_allocated_len=1,
            mamba_pool_idx=torch.tensor(1),
            mamba_ping_pong_track_buffer=torch.tensor([2, -1]),
        )
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: session_kv,
            evict_for_alloc=lambda params: self.fail("session reuse must not evict"),
        )
        scheduler = _scheduler(pool, tree_cache)

        self.assertTrue(Scheduler._ensure_mamba_admission_capacity(scheduler, req, []))

    def test_int8_checkpoint_eviction_is_not_used_for_active_slot_shortage(self):
        pool = _pool(mamba_size=1, lazy=True)
        pool.mamba_ckpt_pool = object()
        req = _req()
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: None,
            evict_for_alloc=lambda params: self.fail(
                "int8 checkpoint eviction cannot free active Mamba slots"
            ),
        )
        scheduler = _scheduler(pool, tree_cache)

        self.assertFalse(Scheduler._ensure_mamba_admission_capacity(scheduler, req, []))

    def test_unified_full_cache_can_donate_with_int8_checkpoints(self):
        pool = _pool(mamba_size=2, lazy=True)
        pool.mamba_ckpt_pool = object()
        held = pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(held)
        req = _req()

        def evict(params):
            self.assertEqual(params.mamba_num, 1)
            pool.mamba_allocator.free(held)

        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: None,
            evict_for_alloc=evict,
        )
        scheduler = _scheduler(pool, tree_cache, unified=True)

        self.assertTrue(Scheduler._ensure_mamba_admission_capacity(scheduler, req, []))

    def test_blocked_fresh_request_does_not_hide_session_resume(self):
        pool = _pool(mamba_size=1, lazy=True)
        self.assertIsNotNone(pool.mamba_allocator.alloc(1))
        fresh = _queued_req("fresh")
        resumed = _queued_req("resumed")
        session_kv = ReqKvInfo(
            req_pool_idx=1,
            kv_allocated_len=1,
            mamba_pool_idx=torch.tensor(1),
            mamba_ping_pong_track_buffer=torch.tensor([2, -1]),
        )
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: True,
            get_session_kv=lambda candidate: (
                session_kv if candidate is resumed else None
            ),
            evict_for_alloc=lambda params: None,
            buffer_pipeline=None,
            storage_prefetch_retries=None,
        )
        scheduler = _scheduler(pool, tree_cache)
        scheduler.grammar_manager = MagicMock(
            has_waiting_grammars=MagicMock(return_value=False)
        )
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.server_args = SimpleNamespace(
            enable_flexkv=False, prefill_max_requests=None
        )
        scheduler.is_hybrid_swa = False
        scheduler.enable_priority_preemption = False
        scheduler.chunked_req = None
        scheduler.waiting_queue = [fresh, resumed]
        scheduler.min_free_slots_delayer = None
        scheduler.get_num_allocatable_reqs = MagicMock(return_value=64)
        scheduler.policy = MagicMock()
        scheduler.processed_tokens_counter = 0
        scheduler.chunked_prefill_size = 8192
        scheduler.page_size = 1
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(), prefill_aware_swa=False
            )
        )
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 8192
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 0
        scheduler.max_running_requests = 64
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = scheduler_mod.DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler.truncation_align_size = None
        scheduler.enable_priority_scheduling = False
        scheduler.load_inquirer = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = MagicMock()

        adder = MagicMock()
        adder.can_run_list = []
        adder.preempt_list = []
        adder.new_chunked_req = None

        def add_one_req(candidate, **kwargs):
            adder.can_run_list.append(candidate)
            return AddReqResult.CONTINUE

        adder.add_one_req.side_effect = add_one_req
        new_batch = MagicMock()
        schedule_batch_cls = MagicMock()
        schedule_batch_cls.init_new.return_value = new_batch
        running_batch = MagicMock(batch_is_full=False, reqs=[])
        running_batch.is_empty.return_value = True

        with (
            patch.object(scheduler_mod, "PrefillAdder", MagicMock(return_value=adder)),
            patch.object(scheduler_mod, "ScheduleBatch", schedule_batch_cls),
            patch.object(scheduler_mod, "PrefillStats", MagicMock()),
            patch.object(scheduler_mod, "set_time_batch", MagicMock()),
            patch.object(
                scheduler_mod,
                "get_schedule",
                MagicMock(return_value=SimpleNamespace(prefill_max_requests=None)),
            ),
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        self.assertIs(batch, new_batch)
        self.assertFalse(fresh.initialized)
        self.assertTrue(resumed.initialized)
        self.assertEqual(scheduler.waiting_queue, [fresh])


if __name__ == "__main__":
    unittest.main()
