from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability.profile_utils import build_step_span_name


class _FakeNvtx(types.ModuleType):
    """Stand-in for the ``nvtx`` package that records every range opened."""

    def __init__(self):
        super().__init__("nvtx")
        self.ranges = []

    @contextmanager
    def annotate(self, message, color=None):
        self.ranges.append(message)
        yield


def _reload_nvtx_utils(fake_nvtx):
    """Re-import nvtx_utils so its import-time gates pick up the current env."""
    with patch.dict(sys.modules, {"nvtx": fake_nvtx}):
        import sglang.srt.utils.nvtx_utils as nvtx_utils

        return importlib.reload(nvtx_utils)


def _forward_batch(mode, bs, extend_tokens=None):
    return SimpleNamespace(
        forward_mode=mode, batch_size=bs, extend_num_tokens=extend_tokens
    )


class TestNvtxForwardSpan(unittest.TestCase):
    def tearDown(self):
        # Leave the module in its default (all gates off) state for other tests.
        import sglang.srt.utils.nvtx_utils as nvtx_utils

        importlib.reload(nvtx_utils)

    def test_forward_gate_emits_step_span_as_nvtx_range(self):
        fake = _FakeNvtx()
        with envs.SGLANG_ENABLE_NVTX_FORWARD.override(True):
            m = _reload_nvtx_utils(fake)
            self.assertTrue(m.NVTX_FORWARD_ENABLED)
            self.assertFalse(m.NVTX_SCHEDULER_ENABLED)
            name = build_step_span_name(
                _forward_batch(ForwardMode.EXTEND, 8, 4096), detailed_annotations=False
            )
            with m.forward_nvtx_range(name):
                pass
        self.assertEqual(fake.ranges, ["step[EXTEND bs=8 toks=4096]"])

    def test_forward_gate_off_opens_no_nvtx_range(self):
        fake = _FakeNvtx()
        with envs.SGLANG_ENABLE_NVTX_SCHEDULER.override(True):
            # Scheduler gate on, forward gate off: the nvtx package is loaded but
            # the forward span must stay torch-profiler only.
            m = _reload_nvtx_utils(fake)
            self.assertTrue(m.NVTX_SCHEDULER_ENABLED)
            self.assertFalse(m.NVTX_FORWARD_ENABLED)
            with m.forward_nvtx_range("step[DECODE bs=48]"):
                pass
            m.scheduler_nvtx_method("scheduler.run_batch")(lambda: None)()
        self.assertEqual(fake.ranges, ["scheduler.run_batch"])

    def test_step_span_names_carry_mode_and_batch(self):
        cases = [
            (_forward_batch(ForwardMode.DECODE, 48), False, "step[DECODE bs=48]"),
            (
                _forward_batch(ForwardMode.EXTEND, 2, 1024),
                False,
                "step[EXTEND bs=2 toks=1024]",
            ),
            (
                _forward_batch(ForwardMode.TARGET_VERIFY, 48),
                False,
                "step[VERIFY bs=48]",
            ),
            (_forward_batch(ForwardMode.TARGET_VERIFY, 1), True, "step[DRAFT bs=1]"),
        ]
        for fb, is_draft, expected in cases:
            self.assertEqual(
                build_step_span_name(
                    fb, detailed_annotations=False, is_draft_worker=is_draft
                ),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
