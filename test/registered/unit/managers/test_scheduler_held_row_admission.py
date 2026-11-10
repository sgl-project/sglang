"""A request that already holds a req_to_token row is not counted twice at admission (#38118, #35537)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from parameterized import parameterized

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NUM_ROWS = 4


class _Req:
    def __init__(self, holds_row):
        self.kv = SimpleNamespace(holds_kv=holds_row)
        self.beam_group = None
        self.token_indices_to_pool = None

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder(PrefillAdder):
    """Admits whatever the gate lets through; the final chunk joins first."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestHeldRowAdmission(CustomTestCase):
    def _scheduler(self, disaggregation_mode, running, final_chunk, waiting):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.waiting_queue.join(waiting)
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.chunked_req = final_chunk
        s.min_free_slots_delayer = None
        s.invariant_checker = Mock()
        s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.request_retirer = None
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = SimpleNamespace(state_slots=None)
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = NUM_ROWS
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = disaggregation_mode
        s.disagg_prefill_bootstrap_queue = None
        s.disagg_prefill_transfer_queue = []
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_unified_cache_external_linker = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        # The final chunk and the running requests hold rows; one row is free.
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: NUM_ROWS - len(running) - 1
        )
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        return s

    def _admitted(self, disaggregation_mode):
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        final_chunk = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        running_batch = ScheduleBatch(reqs=running)
        s = self._scheduler(disaggregation_mode, running, final_chunk, [waiting])

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return (
            built.exception.args[0],
            running_batch.batch_is_full,
            final_chunk,
            waiting,
        )

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_mixed_logprobs(self, prefill_lp, decode_lp):
        prefill, decode = [
            Req(str(i), "", array("q", [i, i + 1]), SamplingParams(), return_logprob=lp)
            for i, lp in enumerate((prefill_lp, decode_lp), 1)
        ]
        prefill.extend_end = decode.extend_end = 2
        s = self._scheduler(DisaggregationMode.NULL, [], decode, [])
        s.model_config = SimpleNamespace(is_encoder_decoder=False, vocab_size=17)
        s.req_to_token_pool.device = "cpu"
        s.is_mixed_chunk = True
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.hisparse_coordinator = None
        s.enable_hierarchical_cache = s.enable_skip_finishing_decode = False
        s.tp_worker.model_runner.prefill_aware_swa = False
        s.enable_priority_scheduling = False
        s.load_inquirer = Mock()
        allocations = [(torch.arange(2), row, row) for row in torch.arange(2).split(1)]
        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(schedule_batch, "alloc_for_extend", side_effect=allocations),
            patch.object(
                schedule_batch, "alloc_for_decode", return_value=torch.tensor([4])
            ),
            patch.object(ScheduleBatch, "check_decode_mem", return_value=True),
            patch("sglang.srt.managers.scheduler.PrefillStats.from_adder"),
        ):
            running, _ = s._get_new_batch_prefill_raw(None, ScheduleBatch([]))
            decode.output_ids.append(4)
            s.chunked_req = prefill
            batch, _ = s._get_new_batch_prefill_raw(None, running)
        self.assertEqual(batch.forward_mode, ForwardMode.MIXED)
        self.assertEqual(batch.reqs, [prefill, decode])
        self.assertEqual(batch.return_logprob, prefill_lp or decode_lp)
        self.assertEqual(batch.extend_lens, [2, 1])
        self.assertEqual(
            batch.extend_logprob_start_lens,
            [2 if decode_lp and not prefill_lp else 0, int(prefill_lp or decode_lp)],
        )
        targets = batch.extend_input_logprob_token_ids
        self.assertEqual(
            None if targets is None else targets.tolist(),
            [2, 0] if prefill_lp else None,
        )

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            with self.subTest(mode=mode):
                can_run_list, batch_is_full, final_chunk, waiting = self._admitted(mode)
                self.assertEqual(can_run_list, [final_chunk, waiting])
                self.assertFalse(batch_is_full)


if __name__ == "__main__":
    unittest.main()
