"""Regression tests for the shared EAGLE verify-input builder.

This test covers only the idle path which returns before any tree-mask or
kernel work, allowing the idle-rank capture-mode invariant to be tested on CPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBuildEagleVerifyInput(CustomTestCase):
    def test_idle_verify_capture_mode_matches_active_ranks(self):
        # An idle DP rank must request the same capture mode as the active ranks
        # in the same round, or can_run_graph() diverges (graph on one rank,
        # eager on another) and the target-verify NCCL deadlock.
        for algorithm, is_standalone, expected in (
            ("eagle", False, CaptureHiddenMode.FULL),
            ("standalone", True, CaptureHiddenMode.NULL),
        ):
            with self.subTest(algorithm=algorithm):
                target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        spec_algorithm=SimpleNamespace(
                            is_standalone=MagicMock(return_value=is_standalone)
                        )
                    )
                )

                verify_input = build_eagle_verify_input(
                    SimpleNamespace(forward_mode=ForwardMode.IDLE),
                    # draft input and tree tensors are unused on the idle path.
                    None,
                    None,
                    None,
                    None,
                    None,
                    target_worker=target_worker,
                    topk=1,
                    num_steps=3,
                    num_draft_tokens=4,
                    tree_mask_mode=None,
                    device="cpu",
                )

                self.assertEqual(verify_input.capture_hidden_mode, expected)


if __name__ == "__main__":
    unittest.main()
