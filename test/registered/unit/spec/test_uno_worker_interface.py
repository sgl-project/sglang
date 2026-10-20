"""Scheduler-call compatibility for the single-stage UNO worker."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.uno_worker_v2 import UnoWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnoWorkerInterface(unittest.TestCase):
    def setUp(self):
        self.worker = UnoWorkerV2.__new__(UnoWorkerV2)
        self.worker._validate_batch = Mock()
        self.worker._forward_prefill = Mock()
        self.worker._forward_decode = Mock()

    def test_single_stage_scheduler_keyword_routes_prefill_and_decode(self):
        for mode, handler in (
            (ForwardMode.EXTEND, self.worker._forward_prefill),
            (ForwardMode.DECODE, self.worker._forward_decode),
        ):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode)
                publish = Mock()
                result = self.worker.forward_batch_generation(
                    batch,
                    on_publish=publish,
                    grammar_barrier=None,
                    pp_proxy_tensors=None,
                )
                self.worker._validate_batch.assert_called_with(batch)
                handler.assert_called_once_with(batch, publish)
                self.assertIs(result, handler.return_value)

    def test_existing_call_without_pipeline_keyword_still_works(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        result = self.worker.forward_batch_generation(batch)
        self.worker._forward_decode.assert_called_once_with(batch, None)
        self.assertIs(result, self.worker._forward_decode.return_value)

    def test_real_pipeline_input_is_rejected_before_forward(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with self.assertRaisesRegex(NotImplementedError, "PP=1"):
            self.worker.forward_batch_generation(batch, pp_proxy_tensors=object())
        self.worker._validate_batch.assert_not_called()
        self.worker._forward_prefill.assert_not_called()
        self.worker._forward_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
