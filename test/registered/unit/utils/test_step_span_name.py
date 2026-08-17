"""
Unit tests for build_step_span_name (profiler trace span naming).

Usage:
    python test_step_span_name.py
    python -m unittest test_step_span_name.py -v
"""

import types
import unittest

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.profile_utils import build_step_span_name
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def _fake_forward_batch(mode: ForwardMode, bs: int, extend_num_tokens=None):
    # build_step_span_name only reads these three attributes, so a lightweight
    # stub avoids constructing a real (GPU/tensor-backed) ForwardBatch.
    return types.SimpleNamespace(
        forward_mode=mode,
        batch_size=bs,
        extend_num_tokens=extend_num_tokens,
    )


class TestBuildStepSpanName(CustomTestCase):
    def test_target_forward_has_no_prefix(self):
        fb = _fake_forward_batch(ForwardMode.DECODE, bs=32)
        self.assertEqual(build_step_span_name(fb), "step[DECODE bs=32]")
        self.assertEqual(build_step_span_name(fb, is_draft=False), "step[DECODE bs=32]")

    def test_draft_forward_gets_draft_prefix(self):
        fb = _fake_forward_batch(ForwardMode.DECODE, bs=32)
        self.assertEqual(
            build_step_span_name(fb, is_draft=True), "draft step[DECODE bs=32]"
        )

    def test_draft_and_target_spans_are_distinct(self):
        # The bug this guards against: a draft proposer reuses a target
        # ForwardMode, so without the prefix the two spans collide.
        fb = _fake_forward_batch(ForwardMode.TARGET_VERIFY, bs=8)
        target = build_step_span_name(fb, is_draft=False)
        draft = build_step_span_name(fb, is_draft=True)
        self.assertNotEqual(target, draft)
        self.assertEqual(target, "step[TARGET_VERIFY bs=8]")
        self.assertEqual(draft, "draft step[TARGET_VERIFY bs=8]")

    def test_extend_includes_token_count(self):
        fb = _fake_forward_batch(ForwardMode.EXTEND, bs=4, extend_num_tokens=100)
        self.assertEqual(build_step_span_name(fb), "step[EXTEND bs=4 toks=100]")
        self.assertEqual(
            build_step_span_name(fb, is_draft=True),
            "draft step[EXTEND bs=4 toks=100]",
        )

    def test_extend_none_token_count_defaults_to_zero(self):
        fb = _fake_forward_batch(ForwardMode.EXTEND, bs=4, extend_num_tokens=None)
        self.assertEqual(build_step_span_name(fb), "step[EXTEND bs=4 toks=0]")


if __name__ == "__main__":
    unittest.main()
