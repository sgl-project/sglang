"""TBO must not split a verify batch whose spec input it cannot split.

split_spec_info only knows EagleVerifyInput. With a spec-v2 algorithm such as
DSpark the verify input (DFlashVerifyInput) has no retrieve_index, so splitting
it crashes, both on the eager path (TboForwardBatchPreparer.prepare) and while
capturing decode CUDA graphs (TboCudaGraphRunnerPlugin.capture_one_batch_size,
which calls prepare_raw directly). Idle DP ranks join such a step as IDLE with
the same kind of verify input, and must skip the split too.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboCudaGraphRunnerPlugin,
    TboForwardBatchPreparer,
    _verify_input_blocks_split,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VerifyInput:
    """A verify input without EAGLE's retrieve_* fields (like DFlashVerifyInput)."""

    draft_token_num = 4

    def is_verify_input(self):
        return True


class _EagleVerifyInput(_VerifyInput):
    retrieve_index = None


class _DraftInput:
    def is_verify_input(self):
        return False


def _batch(forward_mode, spec_info):
    return SimpleNamespace(
        forward_mode=forward_mode, spec_info=spec_info, tbo_split_seq_index=1
    )


class TestTboUnsplittableVerify(CustomTestCase):
    def test_blocks_only_unsplittable_verify_inputs(self):
        for mode, spec_info, expected in (
            (ForwardMode.TARGET_VERIFY, _VerifyInput(), True),
            (ForwardMode.IDLE, _VerifyInput(), True),
            (ForwardMode.TARGET_VERIFY, _EagleVerifyInput(), False),
            # DSpark idle batches carry a draft input between verify steps; they
            # must keep joining an EXTEND step's TBO.
            (ForwardMode.IDLE, _DraftInput(), False),
            (ForwardMode.IDLE, None, False),
            (ForwardMode.EXTEND, None, False),
        ):
            with self.subTest(mode=mode, spec_info=type(spec_info).__name__):
                self.assertEqual(
                    _verify_input_blocks_split(_batch(mode, spec_info)), expected
                )

    def test_graph_capture_skips_the_split(self):
        with get_context().override_server_args():
            with get_device().override(device="meta"):
                plugin = TboCudaGraphRunnerPlugin()
        batch = _batch(ForwardMode.TARGET_VERIFY, _VerifyInput())
        with (
            patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True),
            patch.object(
                TboForwardBatchPreparer, "prepare_raw", side_effect=AssertionError
            ),
        ):
            plugin.capture_one_batch_size(batch, num_tokens=64 * 4)


if __name__ == "__main__":
    unittest.main()
