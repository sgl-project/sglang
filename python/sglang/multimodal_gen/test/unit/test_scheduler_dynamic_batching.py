# SPDX-License-Identifier: Apache-2.0

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


class TestSchedulerDynamicBatching(unittest.TestCase):
    def test_return_frames_mismatch_prevents_request_merge(self) -> None:
        scheduler = Scheduler.__new__(Scheduler)
        requests = [
            Req(
                sampling_params=SamplingParams(
                    prompt=f"prompt-{index}",
                    seed=index,
                    return_frames=return_frames,
                )
            )
            for index, return_frames in enumerate((False, True))
        ]

        self.assertFalse(scheduler._can_dynamic_batch(*requests))
        self.assertEqual(
            scheduler._get_dynamic_batch_reject_reason(*requests),
            "sampling_params.return_frames",
        )
        self.assertIsNone(scheduler._try_merge_generation_reqs(requests))

    def test_non_primary_tp_rank_skips_output_splitting(self) -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.receiver = None
        scheduler.worker = SimpleNamespace(
            is_sleeping=lambda: False,
            execute_forward=Mock(return_value=OutputBatch()),
        )
        scheduler._batching_max_size = 2
        scheduler._batching_delay_s = 0.001
        merged_request = object()
        scheduler._try_merge_generation_reqs = Mock(return_value=merged_request)
        scheduler._split_batched_output = Mock(
            side_effect=AssertionError("non-primary rank must not split media")
        )
        requests = [
            SimpleNamespace(
                is_warmup=False,
                trace_ctx=SimpleNamespace(rebuild_thread_context=lambda: None),
            )
            for _ in range(2)
        ]

        with patch(
            "sglang.multimodal_gen.runtime.managers.scheduler.trace_slice",
            return_value=nullcontext(),
        ):
            outputs = scheduler._handle_generation(requests)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(isinstance(output, OutputBatch) for output in outputs))
        scheduler.worker.execute_forward.assert_called_once_with([merged_request])
        scheduler._split_batched_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
