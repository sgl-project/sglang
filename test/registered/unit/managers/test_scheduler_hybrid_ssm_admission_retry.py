"""With a PrefillAdder stand-in that refuses every request with NO_TOKEN, a
hybrid SSM model with a mamba-aware cache re-offers the waiting request each
round; a model that is neither hybrid SSM nor hybrid SWA offers it once and
keeps batch_is_full set. A retry whose prefix match finds no Mamba slot for the
state copy is refused and stays queued instead of raising, and is admitted once
capacity recovers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    IncLockRefResult,
    MambaCowAllocError,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RefusingAdder:
    """PrefillAdder stand-in: every add_one_req is refused with NO_TOKEN."""

    calls = 0

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.new_chunked_req = None
        self.rem_mamba_slots = None
        self.rem_chunk_tokens = 4096
        self.page_size = 1

    def budget_state(self):
        return AddReqResult.CONTINUE

    def chunk_budget_exhausted(self):
        return False

    def add_chunked_req(self, req):
        return req

    def preempt_to_schedule(self, req):
        return False

    def add_one_req(self, req, **kwargs):
        _RefusingAdder.calls += 1
        return AddReqResult.NO_TOKEN


class _Req(SimpleNamespace):
    # The scheduler keeps admitted requests in a set.
    __eq__ = object.__eq__
    __hash__ = object.__hash__


def _req():
    return _Req(
        rid="r1",
        beam_group=None,
        session=None,
        lora_id=None,
        init_next_round_input=MagicMock(),
        kv=SimpleNamespace(
            holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False
        ),
    )


def _scheduler(*, is_hybrid_ssm: bool) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_lmcache = False
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.is_hybrid_ssm = is_hybrid_ssm
    s.tree_cache = SimpleNamespace(
        supports_mamba=lambda: True,
        buffer_pipeline=None,
        req_to_token_pool=SimpleNamespace(),
    )
    s.waiting_queue = [_req()]
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda *args, **kwargs: 1
    s.policy = MagicMock()
    s.chunked_prefill_size = 4096
    s.enable_dynamic_chunking = False
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.page_size = 1
    s.token_to_kv_pool_allocator = MagicMock()
    s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    s.max_prefill_tokens = 4096
    s.is_mixed_chunk = False
    s.processed_tokens_counter = 0
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 1
    s.max_running_requests = 4
    s.dllm_config = None
    s.enable_lora = False
    s.lora_drainer = None
    s.req_to_token_pool = SimpleNamespace()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.truncation_align_size = None
    return s


class TestHybridSsmAdmissionRetry(CustomTestCase):
    def _rounds(self, s, n):
        running_batch = SimpleNamespace(batch_is_full=False, reqs=[])
        _RefusingAdder.calls = 0
        with (
            patch("sglang.srt.managers.scheduler.PrefillAdder", _RefusingAdder),
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
        ):
            for _ in range(n):
                ret, running_batch = Scheduler._get_new_batch_prefill_raw(
                    s, prefill_delayer_single_pass=None, running_batch=running_batch
                )
                self.assertIsNone(ret)
        return _RefusingAdder.calls, running_batch.batch_is_full

    def test_hybrid_ssm_retries_admission_every_round(self):
        calls, _ = self._rounds(_scheduler(is_hybrid_ssm=True), 3)
        self.assertEqual(calls, 3)

    def test_non_hybrid_model_latches_after_one_refusal(self):
        calls, latched = self._rounds(_scheduler(is_hybrid_ssm=False), 3)
        self.assertEqual(calls, 1)
        self.assertTrue(latched)


class _ScriptedAdder(_RefusingAdder):
    """PrefillAdder stand-in that returns scripted results, admitting the
    request on CONTINUE."""

    script = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preempt_list = []

    def add_one_req(self, req, **kwargs):
        _ScriptedAdder.calls += 1
        res = _ScriptedAdder.script.pop(0)
        if res == AddReqResult.CONTINUE:
            self.can_run_list.append(req)
        return res


class TestMambaCowAllocRefusal(CustomTestCase):
    def test_cow_alloc_failure_raises_recoverable_error(self):
        # The matched node is locked during eviction and nothing else is
        # evictable, so the copy-on-write slot cannot be allocated.
        node = SimpleNamespace(id=3)
        locks = []
        cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(alloc=lambda n: None)
            ),
            inc_lock_ref=lambda n: locks.append(("inc", n)) or IncLockRefResult(),
            dec_lock_ref=lambda n, params: locks.append(("dec", n)),
            evict_for_alloc=lambda params: None,
        )
        component = object.__new__(MambaComponent)
        component.cache = cache
        component.tree_core = SimpleNamespace(
            get_component_device_value=lambda n, ct: torch.tensor([5])
        )
        component.component_type = ComponentType.MAMBA
        kv = SimpleNamespace(
            holds_mamba=False,
            mamba_pool_idx=None,
            mamba_cow_src_index=None,
            mamba_needs_clear=True,
        )
        params = MatchPrefixParams(key=None, cow_mamba=True, req=SimpleNamespace(kv=kv))
        with self.assertRaises(MambaCowAllocError):
            component.finalize_match_result_in_cache(
                params, SimpleNamespace(best_match_node=node)
            )
        self.assertEqual(locks, [("inc", node), ("dec", node)])
        self.assertIsNone(kv.mamba_pool_idx)
        self.assertIsNone(kv.mamba_cow_src_index)

    def test_request_stays_queued_until_mamba_capacity_recovers(self):
        s = _scheduler(is_hybrid_ssm=True)
        req = s.waiting_queue[0]
        req.token_indices_to_pool = None
        # Round 1: the adder refuses. Round 2: the prefix match can no longer
        # allocate a Mamba slot for the state copy. Round 3: capacity is back.
        req.init_next_round_input.side_effect = [
            None,
            MambaCowAllocError("Can not alloc mamba cache"),
            None,
        ]
        _ScriptedAdder.script = [AddReqResult.NO_TOKEN, AddReqResult.CONTINUE]
        _ScriptedAdder.calls = 0
        s.tree_cache.storage_prefetch_retries = None
        s.enable_priority_scheduling = False
        s.load_inquirer = MagicMock()
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.tp_worker.model_runner.prefill_aware_swa = False
        running_batch = SimpleNamespace(batch_is_full=False, reqs=[])
        results = []
        with (
            patch("sglang.srt.managers.scheduler.PrefillAdder", _ScriptedAdder),
            patch("sglang.srt.managers.scheduler.ScheduleBatch") as schedule_batch,
            patch("sglang.srt.managers.scheduler.PrefillStats"),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
        ):
            for _ in range(3):
                ret, running_batch = Scheduler._get_new_batch_prefill_raw(
                    s, prefill_delayer_single_pass=None, running_batch=running_batch
                )
                results.append((ret, list(s.waiting_queue), _ScriptedAdder.calls))

        # The failed state copy is a refusal: no add_one_req, request queued.
        self.assertIsNone(results[1][0])
        self.assertEqual(results[1][1], [req])
        self.assertEqual(results[1][2], 1)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        # Recovered capacity admits the request.
        self.assertIs(results[2][0], schedule_batch.init_new.return_value)
        self.assertEqual(results[2][1], [])
        self.assertEqual(results[2][2], 2)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [req])


if __name__ == "__main__":
    unittest.main()
