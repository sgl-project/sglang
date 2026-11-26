"""The next decode replay can overwrite hidden states before host readback."""

import unittest
from contextlib import nullcontext
from dataclasses import replace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult, allocate_distinct_stream
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")

COPY_DELAY_CYCLES = 500_000_000


class TestOverlapDecodeGraphHiddenStates(CustomTestCase):
    def setUp(self):
        self.schedule_stream = torch.cuda.current_stream()
        self.scheduler = object.__new__(Scheduler)
        self.scheduler.device_module = torch.cuda
        self.scheduler.enable_overlap = True
        self.scheduler.enable_async_d2h_copy = False
        self.scheduler.forward_stream = allocate_distinct_stream(
            torch.cuda, (self.schedule_stream,)
        )
        self.scheduler.copy_stream = allocate_distinct_stream(
            torch.cuda, (self.schedule_stream, self.scheduler.forward_stream)
        )
        self.scheduler.forward_stream_ctx = torch.cuda.stream(
            self.scheduler.forward_stream
        )
        self.scheduler.copy_stream_ctx = torch.cuda.stream(self.scheduler.copy_stream)

    def _check_graph_readback(
        self,
        inline: bool,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        spec_algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.NONE,
    ):
        scheduler = self.scheduler
        with scheduler.forward_stream_ctx:
            hidden = torch.zeros((4, 8), device="cuda")
            last_hidden = torch.zeros_like(hidden)
            next_token_ids = torch.zeros(2, dtype=torch.int64, device="cuda")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=scheduler.forward_stream):
                hidden.add_(1)
                last_hidden.add_(3)
            graph.replay()

        logits = LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden[:2],
            last_hidden_states=last_hidden[:2],
        )
        result = GenerationBatchResult(
            logits_output=logits,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=True,
            copy_done=torch.cuda.Event(),
        )
        batch = ScheduleBatch(
            reqs=[],
            forward_mode=forward_mode,
            spec_algorithm=spec_algorithm,
            return_hidden_states=True,
        )

        if not inline:
            # A clone queued on the caller's stream would see the second replay.
            torch.cuda._sleep(2 * COPY_DELAY_CYCLES)
            schedule_delay_done = self.schedule_stream.record_event()
        with scheduler.copy_stream_ctx:
            torch.cuda._sleep(COPY_DELAY_CYCLES)
            copy_delay_done = scheduler.copy_stream.record_event()

        caller_ctx = (
            scheduler.forward_stream_ctx if inline else nullcontext()
        )
        with caller_ctx:
            scheduler._copy_result_to_cpu(result, batch)
            if inline:
                self.assertEqual(torch.cuda.current_stream(), scheduler.forward_stream)
        with scheduler.forward_stream_ctx:
            graph.replay()

        self.assertEqual(torch.cuda.current_stream(), self.schedule_stream)
        scheduler.forward_stream.synchronize()
        self.assertFalse(copy_delay_done.query(), "copy must trail the second replay")
        if not inline:
            self.assertFalse(
                schedule_delay_done.query(),
                "caller stream must trail the second replay",
            )
        result.copy_done.synchronize()
        self.schedule_stream.synchronize()

        self.assertEqual(logits.hidden_states.device.type, "cpu")
        # The last hidden states stay on device for in-process consumers.
        self.assertEqual(
            (
                logits.hidden_states.unique().tolist(),
                logits.last_hidden_states.unique().tolist(),
            ),
            ([1.0], [3.0]),
        )

    def test_inline_readback(self):
        self._check_graph_readback(inline=True)

    def test_delayed_sample_readback(self):
        self._check_graph_readback(inline=False)

    def test_extend_graph_readback(self):
        self._check_graph_readback(inline=False, forward_mode=ForwardMode.EXTEND)

    def test_speculative_graph_readback(self):
        self._check_graph_readback(
            inline=True, spec_algorithm=SpeculativeAlgorithm.EAGLE
        )

    def test_inapplicable_batches_keep_the_original_tensors(self):
        # D2H preserves CPU tensor identity, so either field exposes an
        # unintended clone even when return_hidden_states is enabled.
        hidden = torch.ones((2, 8))
        last_hidden = torch.empty_like(hidden, device="cuda")
        next_token_ids = torch.zeros(2, dtype=torch.int64)
        decode = ScheduleBatch(
            reqs=[],
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            return_hidden_states=True,
        )
        cases = (
            ("eager", True, False, decode),
            ("no capture", True, True, replace(decode, return_hidden_states=False)),
            ("no overlap", False, True, decode),
        )
        for name, overlap, graph, batch in cases:
            with self.subTest(name=name):
                self.scheduler.enable_overlap = overlap
                logits = LogitsProcessorOutput(
                    next_token_logits=None,
                    hidden_states=hidden,
                    last_hidden_states=last_hidden,
                )
                result = GenerationBatchResult(
                    logits_output=logits,
                    next_token_ids=next_token_ids,
                    can_run_cuda_graph=graph,
                    copy_done=torch.cuda.Event(),
                )
                self.scheduler._copy_result_to_cpu(result, batch)
                result.copy_done.synchronize()
                self.assertIs(logits.hidden_states, hidden)
                self.assertIs(logits.last_hidden_states, last_hidden)


if __name__ == "__main__":
    unittest.main()
