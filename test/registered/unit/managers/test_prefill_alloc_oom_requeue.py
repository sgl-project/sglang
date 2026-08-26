"""Regression tests for sgl-project/sglang#34676.

"Hybrid Mamba prefill allocation failure kills scheduler instead of
returning request to waiting queue."

PrefillAdder admits requests against an *estimate* of available KV-cache
capacity; the real allocation happens later in
ScheduleBatch.prepare_for_extend() -> alloc_for_extend(). When it misses
(concurrent decode-side eviction, DCP-owned pages, or hybrid Mamba state
made the estimate wrong), two things must hold:

1. alloc_for_extend() must free the req-pool slots it handed out for this
   batch and raise the typed PrefillOOMError, not a bare RuntimeError
   (TestAllocForExtendOOMContract).
2. Scheduler._get_new_batch_prefill_raw() must catch it, return the admitted
   requests to the waiting queue, and report "no batch this round" -- not
   let the exception escape its event loop, where run_scheduler_process()
   treats it as fatal and kills the whole process tree
   (TestSchedulerRecoversFromPrefillOOM).

Decode already has a retraction path for the equivalent situation; these
tests pin down prefill's.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_mod
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import PrefillOOMError, alloc_for_extend
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class _FakeReq:
    """Minimal stand-in for sglang.srt.managers.schedule_batch.Req."""

    def __init__(self, prefix_len: int):
        self.prefix_indices = torch.empty(prefix_len, dtype=torch.int64)
        self.dllm_incomplete_ids = None
        self.req_pool_idx = None
        self.kv = None


class _FakeAllocator:
    """Stands in for the real TokenToKVPoolAllocator.

    Deliberately a plain object (not a MagicMock): MagicMock auto-creates
    any attribute you ask for, so `hasattr(allocator, "c128_attn_allocator")`
    would silently be True and divert alloc_for_extend() down the DSV4
    branch. A plain object keeps hasattr() honest.
    """

    def __init__(self, page_size: int, *, oom_on_extend: bool):
        self.page_size = page_size
        self._oom_on_extend = oom_on_extend

    def available_size(self) -> int:
        # Real capacity looks fine; the shortfall only shows up on the real
        # alloc_extend() call -- mirrors PrefillAdder's admission estimate
        # being wrong in the reported crash.
        return 10**9

    def alloc_extend(self, *args, **kwargs):
        if self._oom_on_extend:
            return None
        return torch.arange(4, dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self, allocator: _FakeAllocator):
        self.token_to_kv_pool_allocator = allocator
        self.page_size = allocator.page_size

    def is_chunk_cache(self) -> bool:
        return False

    def evict(self, params) -> None:
        pass

    def available_and_evictable_str(self) -> str:
        return "fake-tree-cache-state"

    def pretty_print(self) -> None:
        pass


class _FakeReqToTokenPool:
    """Deliberately *not* a HybridReqToTokenPool instance -- keeps
    alloc_req_slots() on the plain (non-mamba) path.

    Mirrors the real ReqToTokenPool alloc()/free() contract closely enough
    to observe alloc_for_extend()'s failure-path cleanup: alloc() stamps
    req.req_pool_idx on each request, free() clears it back to None.
    """

    device = "cpu"

    def __init__(self, num_slots: int = 1, max_context_len: int = 64):
        self.req_to_token = torch.zeros((num_slots, max_context_len), dtype=torch.int64)
        self._next_idx = 0

    def alloc(self, reqs):
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = self._next_idx
                self._next_idx += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        req.req_pool_idx = None

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _FakeBatch:
    def __init__(self, allocator: _FakeAllocator):
        req = _FakeReq(prefix_len=0)
        self.reqs = [req]
        self.prefix_lens = [0]
        self.extend_lens = [4]
        self.extend_num_tokens = 4
        self.device = "cpu"
        self.seq_lens = torch.tensor([4], dtype=torch.int64)
        self.seq_lens_cpu = torch.tensor([4], dtype=torch.int64)
        self.tree_cache = _FakeTreeCache(allocator)
        self.req_to_token_pool = _FakeReqToTokenPool()
        self.token_to_kv_pool_allocator = allocator

    def is_dllm(self) -> bool:
        return False

    def maybe_evict_swa(self) -> None:
        pass


class TestAllocForExtendOOMContract(CustomTestCase):
    """Layer 1: the allocation-side contract of the #34676 fix."""

    def setUp(self):
        # torch_native keeps write_cache_indices() off the triton kernel path,
        # so this CPU-CI test doesn't need a GPU driver.
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", attention_backend="torch_native")
        )

    def test_oom_raises_typed_error_and_frees_req_slot(self):
        # page_size > 1 routes through alloc_paged_token_slots_extend(), the
        # exact function behind the reported "Prefill out of memory" error.
        allocator = _FakeAllocator(page_size=16, oom_on_extend=True)
        batch = _FakeBatch(allocator)
        req = batch.reqs[0]

        with self.assertRaises(PrefillOOMError):
            alloc_for_extend(batch)

        # The req-pool slot handed out earlier in the same call must be freed
        # on failure -- otherwise every miss leaks one slot, a slow-drain
        # crash of its own (same family as sgl-project/sglang#21404/#16067).
        self.assertIsNone(req.req_pool_idx)

    def test_preexisting_req_slot_is_not_freed_on_oom(self):
        # A chunked-prefill continuation arrives already holding a slot; the
        # failure path may only free slots *this* call allocated.
        allocator = _FakeAllocator(page_size=16, oom_on_extend=True)
        batch = _FakeBatch(allocator)
        req = batch.reqs[0]
        req.req_pool_idx = 7  # pre-existing slot, not ours to free

        with self.assertRaises(PrefillOOMError):
            alloc_for_extend(batch)

        self.assertEqual(req.req_pool_idx, 7)

    def test_success_path_unaffected(self):
        # Guard against the fixture silently no-op'ing: the same fakes must
        # complete a real allocation when the allocator does not miss.
        allocator = _FakeAllocator(page_size=16, oom_on_extend=False)
        batch = _FakeBatch(allocator)
        out_cache_loc, _, _ = alloc_for_extend(batch)
        self.assertEqual(out_cache_loc.numel(), 4)
        self.assertIsNotNone(batch.reqs[0].req_pool_idx)


def _scheduler_for_prefill_raw(*, waiting_reqs) -> Scheduler:
    """Scheduler.__new__ with exactly the attributes
    _get_new_batch_prefill_raw() touches before/around prepare_for_extend().
    Follows the pattern of test_scheduler_chunked_req_gate.py."""
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock(has_waiting_grammars=MagicMock(return_value=False))
    s.enable_hierarchical_cache = False
    s.server_args = MagicMock(enable_flexkv=False, prefill_max_requests=None)
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.chunked_req = None
    s.waiting_queue = list(waiting_reqs)
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = MagicMock(return_value=64)
    s.policy = MagicMock()
    s.chunked_prefill_size = 8192
    s.enable_dynamic_chunking = False
    s.page_size = 16
    s.tree_cache = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    s.max_prefill_tokens = 8192
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = 64
    s.dllm_config = None
    s.enable_lora = False
    # Plain namespace: getattr(pool, "mamba_allocator", None) must be None,
    # and a MagicMock would fabricate one.
    s.req_to_token_pool = SimpleNamespace(available_size=lambda: 64)
    s.disaggregation_mode = scheduler_mod.DisaggregationMode.NULL
    s.enable_hicache_storage = False
    s.truncation_align_size = None
    s.enable_priority_scheduling = False
    s.load_inquirer = MagicMock()
    s.model_config = MagicMock()
    s.enable_overlap = False
    s.spec_algorithm = MagicMock()
    s.tp_worker = MagicMock()
    # Requeue must really land in waiting_queue so the test can observe it.
    s._add_request_to_queue = lambda req: s.waiting_queue.append(req)
    return s


class TestSchedulerRecoversFromPrefillOOM(CustomTestCase):
    """Layer 2: the scheduler-side recovery of the #34676 fix.

    Drives the real Scheduler._get_new_batch_prefill_raw() with an adder
    that admits one request and a batch whose prepare_for_extend() raises
    PrefillOOMError. On pre-fix code the exception escapes this method (and
    would kill the scheduler process); post-fix it must be absorbed: the
    admitted request goes back to the waiting queue and no batch is
    scheduled this round.
    """

    def test_prefill_oom_requeues_and_returns_no_batch(self):
        req = MagicMock(name="admitted_req")
        s = _scheduler_for_prefill_raw(waiting_reqs=[req])

        fake_adder = MagicMock()
        fake_adder.can_run_list = [req]
        fake_adder.preempt_list = []
        fake_adder.new_chunked_req = None
        fake_adder.add_one_req = MagicMock(return_value=AddReqResult.CONTINUE)

        fake_batch = MagicMock(name="new_batch")
        fake_batch.prepare_for_extend.side_effect = PrefillOOMError(
            "Prefill out of memory (injected)"
        )
        fake_schedule_batch_cls = MagicMock()
        fake_schedule_batch_cls.init_new.return_value = fake_batch

        running_batch = MagicMock()
        running_batch.batch_is_full = False
        running_batch.reqs = []

        with patch.object(
            scheduler_mod, "PrefillAdder", MagicMock(return_value=fake_adder)
        ), patch.object(scheduler_mod, "ScheduleBatch", fake_schedule_batch_cls):
            # Pre-fix code re-raises PrefillOOMError here and this call blows
            # up -- exactly the process-killing escape from #34676.
            result, returned_running = Scheduler._get_new_batch_prefill_raw(
                s, prefill_delayer_single_pass=None, running_batch=running_batch
            )

        # The batch really was attempted (guards against the test passing
        # because the method bailed out before prepare_for_extend()).
        fake_batch.prepare_for_extend.assert_called_once()
        # No batch this round; the running batch flows back unchanged.
        self.assertIsNone(result)
        self.assertIs(returned_running, running_batch)
        # The admitted request is back in the waiting queue, not dropped.
        self.assertIn(req, s.waiting_queue)
        # And the scheduler stops re-admitting into the same full pool this
        # round instead of hot-looping the failure.
        self.assertTrue(running_batch.batch_is_full)

    def test_prefill_success_still_schedules_batch(self):
        # Symmetric guard against over-catching: when prepare_for_extend()
        # succeeds, the method must still produce the batch.
        req = MagicMock(name="admitted_req")
        s = _scheduler_for_prefill_raw(waiting_reqs=[req])

        fake_adder = MagicMock()
        fake_adder.can_run_list = [req]
        fake_adder.preempt_list = []
        fake_adder.new_chunked_req = None
        fake_adder.add_one_req = MagicMock(return_value=AddReqResult.CONTINUE)

        fake_batch = MagicMock(name="new_batch")
        fake_schedule_batch_cls = MagicMock()
        fake_schedule_batch_cls.init_new.return_value = fake_batch

        running_batch = MagicMock()
        running_batch.batch_is_full = False
        running_batch.reqs = []
        running_batch.is_empty = MagicMock(return_value=True)

        with patch.object(
            scheduler_mod, "PrefillAdder", MagicMock(return_value=fake_adder)
        ), patch.object(
            scheduler_mod, "ScheduleBatch", fake_schedule_batch_cls
        ), patch.object(
            scheduler_mod, "PrefillStats", MagicMock()
        ):
            result, _ = Scheduler._get_new_batch_prefill_raw(
                s, prefill_delayer_single_pass=None, running_batch=running_batch
            )

        fake_batch.prepare_for_extend.assert_called_once()
        self.assertIs(result, fake_batch)
        self.assertNotIn(req, s.waiting_queue)


if __name__ == "__main__":
    unittest.main()
