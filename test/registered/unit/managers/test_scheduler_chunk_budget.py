"""Regression tests for waiting admission after resuming a chunked request."""

import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import (  # noqa: E402
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.prefill_budget import PrefillBudget  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingPrefillAdder(PrefillAdder):
    instances: ClassVar[list["_RecordingPrefillAdder"]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.instances.append(self)


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
    scheduler.policy.shortest_prefill_chunk_limit.return_value = None
    scheduler.processed_tokens_counter = 0
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
    scheduler.tree_cache.is_tree_cache.return_value = False
    scheduler.tree_cache.buffer_pipeline = None
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.token_to_kv_pool_allocator.page_size = 1
    scheduler.token_to_kv_pool_allocator.available_size.return_value = available_tokens
    scheduler.token_to_kv_pool_allocator.create_prefill_budget.side_effect = (
        lambda tree_cache, num_mixed_decode_tokens=0: PrefillBudget(
            scheduler.token_to_kv_pool_allocator,
            tree_cache,
            num_mixed_decode_tokens=num_mixed_decode_tokens,
        )
    )
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

    def _run_scheduler(
        self,
        chunk_tokens,
        available_tokens=4096,
        *,
        page_size=1,
        chunk_budget=4,
        mixed_decode_tokens=0,
        cache_mode="radix",
        waiting_tokens=(1,),
        max_prefill_tokens=4096,
    ):
        def make_req(rid, tokens):
            req = Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=list(range(tokens)),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = list(range(tokens))
            req.prefix_indices = torch.empty(0, dtype=torch.int64)
            req.last_node = MagicMock()
            req.init_next_round_input = MagicMock()
            return req

        chunked_req = (
            make_req("chunked", chunk_tokens) if chunk_tokens is not None else None
        )
        waiting_reqs = [
            make_req(f"waiting-{i}", n) for i, n in enumerate(waiting_tokens)
        ]
        waiting_req = waiting_reqs[0]
        running_batch = MagicMock()
        running_batch.reqs = [
            make_req(f"decode-{i}", 1) for i in range(mixed_decode_tokens)
        ]
        running_batch.batch_is_full = False
        running_batch.return_logprob = False
        running_batch.is_empty.side_effect = lambda: not running_batch.reqs
        scheduler = _make_scheduler(waiting_req, chunked_req, available_tokens)
        scheduler.waiting_queue = waiting_reqs
        scheduler.page_size = page_size
        scheduler.token_to_kv_pool_allocator.page_size = page_size
        scheduler.chunked_prefill_size = chunk_budget
        scheduler.max_prefill_tokens = max_prefill_tokens
        scheduler.is_mixed_chunk = mixed_decode_tokens > 0
        scheduler.enable_hierarchical_cache = cache_mode == "hierarchical"
        scheduler.enable_unified_cache_external_linker = cache_mode == "external"

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
                "sglang.srt.managers.schedule_policy.get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
                ),
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

        self.assertEqual(adder.rem_chunk_tokens, 1)
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

    def test_mixed_decode_leaves_subpage_budget_after_resumed_chunk(self):
        scheduler, _, waiting, adder = self._run_scheduler(
            64, page_size=64, chunk_budget=128, mixed_decode_tokens=1
        )
        self.assertEqual(adder.rem_chunk_tokens, 63)
        waiting.init_next_round_input.assert_not_called()
        scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
            0
        )
        self.assertEqual([req.rid for req in adder.can_run_list], ["chunked"])
        self.assertEqual(scheduler.waiting_queue, [waiting])

    def test_mixed_decode_leaves_subpage_budget_without_resumed_chunk(self):
        scheduler, _, waiting, adder = self._run_scheduler(
            None, page_size=64, chunk_budget=64, mixed_decode_tokens=1
        )
        self.assertEqual(adder.rem_chunk_tokens, 63)
        waiting.init_next_round_input.assert_not_called()
        scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
            0
        )
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(scheduler.waiting_queue, [waiting])

    def test_mixed_decode_stops_scan_after_first_waiting_prefill(self):
        scheduler, _, first_waiting, adder = self._run_scheduler(
            None,
            page_size=64,
            chunk_budget=128,
            mixed_decode_tokens=1,
            waiting_tokens=(64, 64),
        )
        self.assertEqual(adder.rem_chunk_tokens, 63)
        first_waiting.init_next_round_input.assert_called_once()
        self.assertEqual([req.rid for req in adder.can_run_list], ["waiting-0"])
        self.assertEqual(len(scheduler.waiting_queue), 1)
        scheduler.waiting_queue[0].init_next_round_input.assert_not_called()

    def test_page_boundaries_preserve_resumed_chunk_and_waiting_admission(self):
        for chunk_tokens in (1, 63, 64, 65, 127, 128, 129):
            for mixed in (0, 1):
                with self.subTest(chunk_tokens=chunk_tokens, mixed=mixed):
                    scheduler, _, waiting, adder = self._run_scheduler(
                        chunk_tokens,
                        page_size=64,
                        chunk_budget=128,
                        mixed_decode_tokens=mixed,
                    )
                    resumed_charge = ((min(chunk_tokens, 128 - mixed) + 63) // 64) * 64
                    should_admit = 128 - mixed - resumed_charge >= 64
                    self.assertEqual(
                        waiting.init_next_round_input.call_count, int(should_admit)
                    )
                    self.assertEqual(adder.can_run_list[0].rid, "chunked")
                    self.assertEqual(len(adder.can_run_list), 1 + int(should_admit))
                    scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
                        int(should_admit)
                    )

    def test_exhausted_chunk_preserves_batch_full_semantics_in_cache_modes(self):
        for cache_mode in ("radix", "hierarchical", "external"):
            for available_tokens, full in ((4096, False), (6, True)):
                with self.subTest(
                    cache_mode=cache_mode, available_tokens=available_tokens
                ):
                    scheduler, running, waiting, adder = self._run_scheduler(
                        4, available_tokens=available_tokens, cache_mode=cache_mode
                    )
                    waiting.init_next_round_input.assert_not_called()
                    self.assertEqual(running.batch_is_full, full)
                    self.assertEqual(
                        [req.rid for req in adder.can_run_list], ["chunked"]
                    )
                    scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
                        0
                    )

    def test_subpage_and_total_exhaustion_preserve_batch_full_semantics(self):
        for cache_mode in ("radix", "hierarchical", "external"):
            with self.subTest(cache_mode=cache_mode):
                scheduler, running, waiting, adder = self._run_scheduler(
                    64,
                    available_tokens=131,
                    page_size=64,
                    chunk_budget=128,
                    mixed_decode_tokens=1,
                    cache_mode=cache_mode,
                )
                self.assertEqual(adder.rem_chunk_tokens, 63)
                self.assertEqual(adder.budget_state(), AddReqResult.NO_TOKEN)
                waiting.init_next_round_input.assert_not_called()
                self.assertTrue(running.batch_is_full)
                scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
                    0
                )

    def test_total_exhaustion_alone_preserves_waiting_scan(self):
        for cache_mode in ("radix", "hierarchical", "external"):
            for chunk_tokens, available_tokens in ((None, 0), (2, 4)):
                with self.subTest(cache_mode=cache_mode, chunk_tokens=chunk_tokens):
                    scheduler, running, waiting, adder = self._run_scheduler(
                        chunk_tokens,
                        available_tokens=available_tokens,
                        cache_mode=cache_mode,
                    )
                    self.assertGreater(adder.rem_chunk_tokens, 0)
                    waiting.init_next_round_input.assert_called_once()
                    self.assertEqual(
                        running.batch_is_full,
                        cache_mode == "radix" or chunk_tokens is not None,
                    )
                    scheduler.req_to_token_pool.mamba_allocator.alloc_group_begin.assert_called_once_with(
                        1
                    )

    def test_dllm_budget_does_not_use_autoregressive_chunk_remainder(self):
        _, _, _, adder = self._run_scheduler(4)
        adder.dllm_config = SimpleNamespace()
        adder.rem_dllm_tokens = 64
        self.assertEqual(adder.rem_chunk_tokens, 0)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)
        adder.rem_dllm_tokens = 0
        self.assertEqual(adder.budget_state(), AddReqResult.OTHER)

    def test_input_budget_alone_does_not_block_first_waiting_request(self):
        for chunk_budget in (None, 64):
            with self.subTest(chunk_budget=chunk_budget):
                scheduler, _, waiting, adder = self._run_scheduler(
                    None, chunk_budget=chunk_budget, max_prefill_tokens=0
                )
                waiting.init_next_round_input.assert_called_once()
                self.assertEqual(adder.can_run_list, [waiting])
                self.assertEqual(scheduler.waiting_queue, [])


if __name__ == "__main__":
    unittest.main()
