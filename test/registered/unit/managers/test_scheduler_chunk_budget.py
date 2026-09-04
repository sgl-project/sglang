"""Regression tests for waiting admission after resuming a chunked request."""

import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_policy import (  # noqa: E402
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingPrefillAdder(PrefillAdder):
    instances: ClassVar[list["_RecordingPrefillAdder"]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.instances.append(self)

    def add_one_req(self, req, **kwargs):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


def _make_scheduler(waiting_req, chunked_req, available_tokens):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.grammar_manager = MagicMock()
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_priority_preemption = False
    scheduler.enable_priority_scheduling = False
    scheduler.is_hybrid_swa = False
    scheduler.min_free_slots_delayer = None
    scheduler.waiting_queue = [waiting_req]
    scheduler.chunked_req = chunked_req
    scheduler.get_num_allocatable_reqs = MagicMock(return_value=8)
    scheduler.policy = MagicMock()
    scheduler.chunked_prefill_size = 4
    scheduler.dynamic_chunk_sizer = None
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(), prefill_aware_swa=False
        )
    )
    scheduler.page_size = 1
    scheduler.tree_cache = MagicMock()
    scheduler.tree_cache.supports_mamba.return_value = False
    scheduler.tree_cache.evictable_size.return_value = 0
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.token_to_kv_pool_allocator.available_size.return_value = available_tokens
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    scheduler.max_prefill_tokens = 32
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 8
    scheduler.max_running_requests = 8
    scheduler.dllm_config = None
    scheduler.enable_lora = False
    scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=MagicMock())
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.truncation_align_size = None
    scheduler.model_config = MagicMock()
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = MagicMock()
    scheduler.load_inquirer = MagicMock()
    return scheduler


class TestSchedulerChunkBudget(CustomTestCase):
    def setUp(self):
        _RecordingPrefillAdder.instances.clear()

    def _run_scheduler(self, chunk_tokens, available_tokens=128):
        chunked_req = MagicMock()
        chunked_req.full_untruncated_fill_ids = list(range(chunk_tokens))
        chunked_req.prefix_indices = []
        chunked_req.sampling_params.max_new_tokens = 1
        chunked_req.retracted_stain = False
        chunked_req.kv.holds_mamba = False
        chunked_req.inflight_middle_chunks = 0
        chunked_req.set_extend_range.side_effect = lambda start, end: setattr(
            chunked_req, "extend_range", Range(start, end)
        )
        waiting_req = MagicMock()
        running_batch = MagicMock()
        running_batch.reqs = []
        running_batch.batch_is_full = False
        running_batch.is_empty.return_value = True
        scheduler = _make_scheduler(waiting_req, chunked_req, available_tokens)

        batch = MagicMock()
        batch.return_logprob = False
        batch.input_embeds = None
        with (
            patch(
                "sglang.srt.managers.scheduler.PrefillAdder",
                _RecordingPrefillAdder,
            ),
            patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=batch,
            ),
            patch(
                "sglang.srt.managers.scheduler.PrefillStats.from_adder",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
            patch(
                "sglang.srt.managers.schedule_policy.is_dsa_prefill_cp_in_seq_split",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.schedule_policy.is_prefill_context_parallel_enabled",
                return_value=False,
            ),
        ):
            Scheduler._get_new_batch_prefill_raw(
                scheduler,
                prefill_delayer_single_pass=None,
                running_batch=running_batch,
            )

        return (
            scheduler,
            running_batch,
            waiting_req,
            _RecordingPrefillAdder.instances[-1],
        )

    def test_waiting_queue_not_scanned_after_resumed_chunk_exhausts_budget(self):
        scheduler, _, waiting_req, adder = self._run_scheduler(chunk_tokens=4)

        self.assertEqual(adder.rem_chunk_tokens, 0)
        waiting_req.init_next_round_input.assert_not_called()
        scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
            0
        )
        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(scheduler.waiting_queue, [waiting_req])

    def test_waiting_queue_scanned_when_resumed_chunk_leaves_budget(self):
        scheduler, _, waiting_req, adder = self._run_scheduler(chunk_tokens=2)

        self.assertEqual(adder.rem_chunk_tokens, 2)
        waiting_req.init_next_round_input.assert_called_once()
        scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
            1
        )
        self.assertEqual(len(adder.can_run_list), 2)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_total_token_exhaustion_marks_batch_full_without_scanning(self):
        scheduler, running_batch, waiting_req, adder = self._run_scheduler(
            chunk_tokens=4,
            available_tokens=6,
        )

        self.assertEqual(adder.rem_chunk_tokens, 0)
        self.assertEqual(adder.rem_total_tokens, 0)
        waiting_req.init_next_round_input.assert_not_called()
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(scheduler.waiting_queue, [waiting_req])


if __name__ == "__main__":
    unittest.main()
