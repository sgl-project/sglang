"""CPU regressions for overlapped output accounting and terminal KV ownership.

Use the real request, result, batch filtering, planner and capacity checks.
Model execution, sampling metadata, admission and KV release are boundaries;
these tests do not exercise CUDA streams or distributed rank agreement.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult, OutputBudgetReservation
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOutputBudgetReservations(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        self.addCleanup(reset_context)
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.enable_overlap_output_budget = True

    def _req(self, rid="r", max_new_tokens=2, output_ids=(), input_len=1):
        params = SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=True)
        params.normalize(tokenizer=None)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [1] * input_len),
            sampling_params=params,
            eos_token_ids=set(),
            vocab_size=128,
        )
        req.output_ids.extend(output_ids)
        return req

    def _batch(self, reqs, mode=ForwardMode.DECODE, chunked_req=None):
        batch = ScheduleBatch(
            reqs=list(reqs),
            forward_mode=mode,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            chunked_req=chunked_req,
        )
        batch.device = "cpu"
        batch.model_config = SimpleNamespace(is_encoder_decoder=False)
        batch.req_pool_indices = torch.arange(len(reqs), dtype=torch.int64)
        batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
        batch.seq_lens = torch.tensor(
            [len(r.origin_input_ids) + len(r.output_ids) for r in reqs],
            dtype=torch.int64,
        )
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.orig_seq_lens = batch.seq_lens.clone()
        # Sampling metadata is outside output-budget accounting.
        batch.sampling_info = Mock()
        return batch

    def _reserve(self, batch):
        result = GenerationBatchResult()
        result.reserve_output_budget(
            self.scheduler._build_output_budget_reservations(batch)
        )
        return result

    @staticmethod
    def _commit(req, token):
        # Model/result-consumption boundary: materialize one ordinary output.
        req.output_ids.append(token)
        req.update_finish_state()

    def test_two_outputs_stop_compute_before_final_result_is_consumed(self):
        req = self._req()
        prefill = self._reserve(self._batch([req], ForwardMode.EXTEND))
        self.assertEqual(req.num_pending_output_tokens, 1)
        batch = self._batch([req])
        self.assertEqual(self.scheduler._get_output_budget_exhausted_reqs(batch), set())

        decode = self._reserve(batch)
        self.assertEqual(req.num_pending_output_tokens, 2)
        self._commit(req, 10)
        prefill.settle_output_budget()
        self.assertEqual((len(req.output_ids), req.num_pending_output_tokens), (1, 1))
        self.assertFalse(req.finished())

        self.assertEqual(self.scheduler._get_output_budget_exhausted_reqs(batch), {req})

        # The final result owns its original rows even after all compute stops.
        snapshot = batch.copy()
        batch.batch_is_full = True
        self.assertIs(self.scheduler.update_running_batch(batch), batch)
        self.assertTrue(batch.is_empty())
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(snapshot.reqs, [req])
        self.assertEqual(req.num_pending_output_tokens, 1)
        self.assertFalse(req.finished())

        self._commit(req, 11)
        decode.settle_output_budget()
        self.assertEqual(list(req.output_ids), [10, 11])
        self.assertEqual(req.num_pending_output_tokens, 0)
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)

    def test_budget_accumulates_across_multiple_pending_results(self):
        req = self._req(max_new_tokens=3)
        batch = self._batch([req])
        results = []
        for pending_count in (1, 2, 3):
            results.append(self._reserve(batch))
            self.assertEqual(req.num_pending_output_tokens, pending_count)
            self.assertEqual(
                self.scheduler._get_output_budget_exhausted_reqs(batch),
                {req} if pending_count == 3 else set(),
            )
        for token, result in zip((10, 11, 12), results):
            self._commit(req, token)
            result.settle_output_budget()
        self.assertEqual(list(req.output_ids), [10, 11, 12])
        self.assertEqual(req.num_pending_output_tokens, 0)

    def test_middle_chunk_does_not_reserve_an_output(self):
        final = self._req("final", max_new_tokens=1)
        middle = self._req("middle")
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                batch = self._batch([final, middle], mode, chunked_req=middle)
                result = self._reserve(batch)
                self.assertEqual(final.num_pending_output_tokens, 1)
                self.assertEqual(middle.num_pending_output_tokens, 0)
                self.assertEqual(
                    self.scheduler._get_output_budget_exhausted_reqs(batch), {final}
                )
                result.settle_output_budget()

        # DECODE must ignore stale chunk metadata on a reused batch.
        result = self._reserve(self._batch([middle], chunked_req=middle))
        self.assertEqual(middle.num_pending_output_tokens, 1)
        result.settle_output_budget()

    def test_unsupported_rows_and_forward_modes_do_not_reserve(self):
        for attribute, value in (("grammar", object()), ("beam_group", object())):
            with self.subTest(attribute=attribute):
                req = self._req()
                setattr(req, attribute, value)
                result = self._reserve(self._batch([req]))
                self.assertEqual(result.output_budget_reservations, ())
                self.assertEqual(req.num_pending_output_tokens, 0)
        for mode, spec in (
            (ForwardMode.IDLE, SpeculativeAlgorithm.NONE),
            (ForwardMode.TARGET_VERIFY, SpeculativeAlgorithm.EAGLE),
        ):
            with self.subTest(mode=mode):
                batch = self._batch([self._req()], mode)
                batch.spec_algorithm = spec
                self.assertEqual(self._reserve(batch).output_budget_reservations, ())

    def test_disabled_and_empty_batches_do_not_filter(self):
        req = self._req(max_new_tokens=1)
        batch = self._batch([req])
        result = self._reserve(batch)
        self.scheduler.enable_overlap_output_budget = False
        self.assertEqual(self.scheduler._get_output_budget_exhausted_reqs(batch), set())
        self.scheduler.enable_overlap_output_budget = True
        # An empty batch need not carry a speculative algorithm.
        self.assertEqual(
            self.scheduler._get_output_budget_exhausted_reqs(ScheduleBatch(reqs=[])),
            set(),
        )
        self.assertEqual(req.num_pending_output_tokens, 1)
        result.settle_output_budget()

    def test_retraction_invalidates_old_results_without_debiting_new_work(self):
        req = self._req(max_new_tokens=4)
        old_results = [self._reserve(self._batch([req])) for _ in range(2)]
        old_epoch = req.retraction_count
        req.reset_for_retract()
        self.assertEqual(req.retraction_count, old_epoch + 1)
        self.assertEqual(req.num_pending_output_tokens, 0)

        req.is_retracted = False  # Re-admission boundary.
        new_result = self._reserve(self._batch([req], ForwardMode.EXTEND))
        for result in old_results:
            result.settle_output_budget()
            self.assertEqual(result.output_budget_reservations, ())
            self.assertEqual(req.num_pending_output_tokens, 1)
        new_result.settle_output_budget()
        new_result.settle_output_budget()
        self.assertEqual(req.num_pending_output_tokens, 0)

    def test_reservation_validation_is_atomic(self):
        a, b = self._req("a"), self._req("b")
        result = GenerationBatchResult()
        with self.assertRaises(AssertionError):
            result.reserve_output_budget(
                (
                    OutputBudgetReservation(a, a.retraction_count, 1),
                    OutputBudgetReservation(b, b.retraction_count + 1, 1),
                )
            )
        self.assertEqual(
            (a.num_pending_output_tokens, b.num_pending_output_tokens), (0, 0)
        )
        self.assertEqual(result.output_budget_reservations, ())

    def test_double_registration_and_settlement_underflow_are_detected(self):
        a, b = self._req("a"), self._req("b")
        result = self._reserve(self._batch([a, b]))
        with self.assertRaises(AssertionError):
            result.reserve_output_budget(result.output_budget_reservations)
        self.assertEqual(
            (a.num_pending_output_tokens, b.num_pending_output_tokens), (1, 1)
        )
        b.num_pending_output_tokens = 0  # Corrupt the second row's accounting.
        with self.assertRaises(AssertionError):
            result.settle_output_budget()
        self.assertEqual(a.num_pending_output_tokens, 1)
        self.assertEqual(len(result.output_budget_reservations), 2)

    def test_common_result_processing_settles_after_flag_is_disabled(self):
        req = self._req(max_new_tokens=1)
        batch = self._batch([req])
        result = self._reserve(batch)
        scheduler = self.scheduler
        scheduler.enable_overlap_output_budget = False
        scheduler.scheduler_stage_metrics = None
        scheduler.publish_load_snapshot = Mock(return_value=None)
        scheduler.load_publisher = Mock()
        scheduler.load_inquirer = Mock()
        scheduler.tree_cache = Mock()
        scheduler.metrics_reporter = Mock()
        scheduler.enable_fpm = False
        scheduler._record_step_counters = Mock()
        scheduler._maybe_clear_mm_inputs = Mock()
        scheduler.maybe_send_health_check_signal = Mock()

        def consume_result(batch, result):
            self._commit(req, 10)
            # Reservations remain owned until result processing completes.
            self.assertEqual(req.num_pending_output_tokens, 1)

        scheduler.batch_result_processor = SimpleNamespace(
            process_batch_result_decode=consume_result
        )
        scheduler.process_batch_result(batch, result)
        self.assertTrue(req.finished())
        self.assertEqual(list(req.output_ids), [10])
        self.assertEqual(req.num_pending_output_tokens, 0)
        self.assertEqual(result.output_budget_reservations, ())

    def _pressure_fixture(self):
        a = self._req("a", 2, [10], input_len=10)
        b = self._req("b", 8, [10, 11, 12], input_len=4)
        allocator = TokenToKVPoolAllocator(
            size=18, dtype=torch.int64, device="cpu", kvcache=None, need_sort=False
        )
        slots = {a: allocator.alloc(11), b: allocator.alloc(7)}
        a.kv.kv_committed_len = 11
        b.kv.kv_committed_len = 7
        batch = self._batch([a, b])
        batch.batch_is_full = True
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = SimpleNamespace(
            supports_prefix_sharing=lambda: False,
            req_to_token_pool=SimpleNamespace(mamba_allocator=None),
        )

        # Replace device preparation / cache-release boundaries, not the
        # production filtering, capacity gate or retraction decision.
        def prepare_decode():
            batch.out_cache_loc = allocator.alloc(len(batch.reqs))
            self.assertIsNotNone(batch.out_cache_loc)

        def release_req(index, remaining, offload_kv=True):
            req = batch.reqs[index]
            allocator.free(slots.pop(req))
            req.reset_for_retract()
            return True

        batch.prepare_for_decode = prepare_decode
        batch.release_req = release_req
        scheduler = self.scheduler
        scheduler.new_token_ratio_tracker = SimpleNamespace(
            current=1.0, decay_step=Mock()
        )
        scheduler.decode_offload_manager = None
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.tree_cache = batch.tree_cache
        scheduler.metrics_reporter = Mock(enable_metrics=False)
        scheduler.ipc_channels = Mock()
        scheduler.beam_coordinator = Mock()
        scheduler._add_request_to_queue = Mock()
        scheduler.forward_ct = 1
        result = self._reserve(batch)
        return a, b, batch, result, allocator, slots

    def _configure_planner(self):
        scheduler = self.scheduler
        scheduler.scheduler_stage_metrics = None
        scheduler.process_pending_chunked_abort = Mock()
        scheduler._process_hicache_events = Mock()
        scheduler.enable_fpm = False
        scheduler.dllm_config = None
        scheduler.chunked_req = None
        scheduler.enable_hisparse = False
        scheduler.require_mlp_sync = False
        scheduler._should_defer_prefill = lambda: False
        scheduler.get_new_batch_prefill = lambda batch: NextBatchPlan(
            batch_to_run=None, running_batch=batch
        )
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        )
        scheduler.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        )
        scheduler._arm_prefill_decode_interval = Mock()

    @patch("sglang.srt.managers.scheduler.TEST_RETRACT", False)
    def test_pending_terminal_kv_defers_then_retries_survivor(self):
        a, b, batch, result, allocator, slots = self._pressure_fixture()
        snapshot = batch.copy()
        self._configure_planner()
        self.assertEqual(allocator.available_size(), 0)

        plan = self.scheduler.get_next_batch_to_run(batch, last_batch=batch)
        self.assertIsNone(plan.batch_to_run)
        self.assertIs(plan.running_batch, batch)
        self.assertEqual(batch.reqs, [b])
        self.assertEqual(batch.req_pool_indices.tolist(), [1])
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(snapshot.reqs, [a, b])
        self.assertEqual(snapshot.req_pool_indices.tolist(), [0, 1])
        self.assertEqual([r.req for r in result.output_budget_reservations], [a, b])
        self.assertFalse(a.finished())
        self.assertIsNone(b.to_finish)
        self.assertEqual(allocator.available_size(), 0)

        self._commit(a, 11)
        self._commit(b, 13)
        result.settle_output_budget()
        allocator.free(slots.pop(a))  # Terminal result releases its KV.
        retry = self.scheduler.get_next_batch_to_run(
            plan.running_batch, last_batch=None
        )
        self.assertIs(retry.batch_to_run, batch)
        self.assertEqual(batch.reqs, [b])
        self.assertEqual(batch.out_cache_loc.numel(), 1)
        self.assertEqual(allocator.available_size(), 10)
        self.assertEqual(
            (a.num_pending_output_tokens, b.num_pending_output_tokens), (0, 0)
        )
        self.assertIsNone(b.to_finish)
        self.assertFalse(b.is_retracted)

    @patch("sglang.srt.managers.scheduler.TEST_RETRACT", False)
    def test_disabled_budget_keeps_existing_retraction_behavior(self):
        a, b, batch, result, allocator, _ = self._pressure_fixture()
        self.scheduler.enable_overlap_output_budget = False
        self.assertIs(self.scheduler.update_running_batch(batch), batch)
        self.assertEqual(batch.reqs, [b])
        self.assertTrue(a.is_retracted)
        self.assertIsNone(b.to_finish)
        self.assertEqual(allocator.available_size(), 10)
        result.settle_output_budget()
        self.assertEqual(
            (a.num_pending_output_tokens, b.num_pending_output_tokens), (0, 0)
        )


if __name__ == "__main__":
    unittest.main()
