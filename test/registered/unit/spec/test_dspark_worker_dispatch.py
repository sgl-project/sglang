"""DSpark accepts the non-overlap scheduler's PP=None call (#39262)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkWorkerDispatch(CustomTestCase):
    def setUp(self):
        self.worker = object.__new__(DSparkWorkerV2)
        self.prefill_result = object()
        self.decode_result = object()
        self.worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        self.worker._observers = SimpleNamespace(note_prefill_step=Mock())
        self.worker._forward_prefill = Mock(return_value=self.prefill_result)
        self.worker._forward_decode = Mock(return_value=self.decode_result)

    def make_batch(self, *, extend=False, mixed=False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            is_extend_in_batch=mixed,
        )

    def test_non_overlap_accepts_empty_pp_input(self):
        for extend, mixed in ((True, False), (False, True), (False, False)):
            with self.subTest(extend=extend, mixed=mixed):
                batch = self.make_batch(extend=extend, mixed=mixed)
                result = self.worker.forward_batch_generation(
                    batch, pp_proxy_tensors=None
                )
                self.assertIs(
                    result,
                    self.prefill_result if extend or mixed else self.decode_result,
                )

    def test_existing_callbacks_are_preserved(self):
        batch = self.make_batch()
        publish, barrier = Mock(), Mock()
        result = self.worker.forward_batch_generation(
            batch, on_publish=publish, grammar_barrier=barrier
        )
        self.assertIs(result, self.decode_result)
        self.worker._forward_decode.assert_called_once_with(batch, publish, barrier)

    def test_nonempty_pp_input_is_rejected_before_forward(self):
        for extend in (False, True):
            with self.subTest(extend=extend):
                with self.assertRaisesRegex(
                    NotImplementedError, "pipeline parallelism"
                ):
                    self.worker.forward_batch_generation(
                        self.make_batch(extend=extend), pp_proxy_tensors=object()
                    )
        self.worker._forward_prefill.assert_not_called()
        self.worker._forward_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
