"""With a PrefillAdder stand-in that refuses every request with NO_TOKEN, a
hybrid SSM model with a mamba-aware cache re-offers the waiting request each
round; a model that is neither hybrid SSM nor hybrid SWA offers it once and
keeps batch_is_full set."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RefusingAdder:
    """PrefillAdder stand-in: every add_one_req is refused with NO_TOKEN."""

    calls = 0

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.new_chunked_req = None
        self.rem_mamba_slots = None

    def chunk_budget_exhausted(self):
        return False

    def add_chunked_req(self, req):
        return req

    def preempt_to_schedule(self, req):
        return False

    def add_one_req(self, req, **kwargs):
        _RefusingAdder.calls += 1
        return AddReqResult.NO_TOKEN


def _req():
    return SimpleNamespace(
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
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.is_hybrid_ssm = is_hybrid_ssm
    s.tree_cache = SimpleNamespace(
        supports_mamba=lambda: True,
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
        with patch("sglang.srt.managers.scheduler.PrefillAdder", _RefusingAdder), patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_flexkv=False),
        ), patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=None),
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


if __name__ == "__main__":
    unittest.main()
