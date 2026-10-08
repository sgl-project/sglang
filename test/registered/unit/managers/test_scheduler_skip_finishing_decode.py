"""CPU regressions for skipping the decode of requests finishing in flight.

Use the real request, batch filtering, planner and capacity checks. Model
execution, sampling metadata, admission and KV release are boundaries; these
tests do not exercise CUDA streams or distributed rank agreement.
"""

import unittest
from array import array
from collections import deque
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
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSkipFinishingDecode(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        self.addCleanup(reset_context)
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.enable_skip_finishing_decode = True
        self.scheduler.reqs_finishing_in_flight = []
        self.scheduler.result_queue = deque()

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

    def _batch(self, reqs, mode=ForwardMode.DECODE):
        batch = ScheduleBatch(
            reqs=list(reqs),
            forward_mode=mode,
            spec_algorithm=SpeculativeAlgorithm.NONE,
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
        # Sampling metadata is outside this feature.
        batch.sampling_info = Mock()
        return batch

    def _queue(self, batch):
        # Overlap boundary: the launched batch waits for result processing.
        self.scheduler.result_queue.append((batch.copy(), GenerationBatchResult()))

    @staticmethod
    def _commit(req, token):
        # Result-consumption boundary: materialize one ordinary output.
        req.output_ids.append(token)
        req.update_finish_state()

    def test_only_queued_final_outputs_are_dropped(self):
        finishing = self._req("finishing", output_ids=[10])
        # Same remaining length, but its last result was already processed.
        not_queued = self._req("not_queued", output_ids=[10])
        unfinished = self._req("unfinished", max_new_tokens=3, output_ids=[10])
        batch = self._batch([finishing, not_queued, unfinished])
        self._queue(self._batch([finishing, unfinished]))

        dropped = self.scheduler._filter_reqs_finishing_in_flight(batch)
        self.assertEqual(dropped, [finishing])
        self.assertEqual(batch.reqs, [not_queued, unfinished])
        self.assertEqual(self.scheduler.reqs_finishing_in_flight, [finishing])
        self.assertFalse(finishing.finished())

        self._commit(finishing, 11)
        self.assertIsInstance(finishing.finished_reason, FINISH_LENGTH)

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
        self._queue(batch)
        self._configure_planner()
        return a, b, batch, allocator, slots

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
    def test_finishing_kv_defers_decode_instead_of_retracting(self):
        a, b, batch, allocator, slots = self._pressure_fixture()
        self.assertEqual(allocator.available_size(), 0)

        plan = self.scheduler.get_next_batch_to_run(batch, last_batch=batch)
        self.assertIsNone(plan.batch_to_run)
        self.assertIs(plan.running_batch, batch)
        self.assertEqual(batch.reqs, [b])
        self.assertFalse(batch.batch_is_full)
        self.assertFalse(b.is_retracted)
        self.assertEqual(allocator.available_size(), 0)

        # The queued result finishes `a` and releases its KV.
        self._commit(a, 11)
        self._commit(b, 13)
        allocator.free(slots.pop(a))
        self.scheduler.result_queue.clear()
        retry = self.scheduler.get_next_batch_to_run(
            plan.running_batch, last_batch=None
        )
        self.assertIs(retry.batch_to_run, batch)
        self.assertEqual(batch.reqs, [b])
        self.assertEqual(batch.out_cache_loc.numel(), 1)
        self.assertFalse(b.is_retracted)

    @patch("sglang.srt.managers.scheduler.TEST_RETRACT", False)
    def test_disabled_keeps_existing_retraction(self):
        a, b, batch, allocator, _ = self._pressure_fixture()
        self.scheduler.enable_skip_finishing_decode = False
        plan = self.scheduler.get_next_batch_to_run(batch, last_batch=batch)
        self.assertIs(plan.batch_to_run, batch)
        self.assertEqual(batch.reqs, [b])
        self.assertTrue(a.is_retracted)
        self.assertEqual(self.scheduler.reqs_finishing_in_flight, [])


if __name__ == "__main__":
    unittest.main()
