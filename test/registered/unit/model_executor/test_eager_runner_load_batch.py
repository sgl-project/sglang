"""Unit tests for ``EagerRunner.load_batch`` — registry staging vs no-copy wrap.

Covers the three input paths:
  * default: the live batch is staged into the eager registry buffers.
  * ``model_capture_mode``: staging is skipped so no copy nodes are baked
    into a captured graph (spec-draft loops route per-step forwards through
    the eager path during capture); the wrap must alias the live tensors.
  * env overrides: ``SGLANG_EAGER_INPUT_NO_COPY`` forces the wrap anywhere,
    ``SGLANG_EAGER_INPUT_NO_COPY_IN_CAPTURE=0`` restores staging in capture.

Imports pull ``sgl_kernel`` transitively, hence CUDA CI instead of CPU.
"""

import dataclasses
import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@dataclasses.dataclass
class _MiniForwardBatch:
    """Minimal FB stand-in: dataclass so ``dataclasses.replace`` works."""

    batch_size: int = 1
    input_ids: Optional[torch.Tensor] = None
    input_embeds: Optional[torch.Tensor] = None


def _make_runner() -> Tuple[EagerRunner, MagicMock]:
    runner = object.__new__(EagerRunner)
    registry = MagicMock()
    registry.extract_buffer.return_value = "staged-batch"
    runner._eager_registry = registry
    return runner, registry


class TestEagerRunnerLoadBatch(CustomTestCase):
    def setUp(self):
        self.fb = _MiniForwardBatch(input_ids=torch.zeros(2, dtype=torch.long))

    def test_default_stages_into_registry(self):
        runner, registry = _make_runner()
        out = runner.load_batch(self.fb)
        registry.fill_from.assert_called_once()
        self.assertEqual(out, "staged-batch")

    def test_capture_mode_skips_staging(self):
        runner, registry = _make_runner()
        with model_capture_mode():
            out = runner.load_batch(self.fb)
        registry.fill_from.assert_not_called()
        # A fresh wrapper object, but aliasing the live tensors.
        self.assertIsNot(out, self.fb)
        self.assertIs(out.input_ids, self.fb.input_ids)

    def test_capture_mode_exit_restores_staging(self):
        runner, registry = _make_runner()
        with model_capture_mode():
            pass
        out = runner.load_batch(self.fb)
        registry.fill_from.assert_called_once()
        self.assertEqual(out, "staged-batch")

    def test_capture_kill_switch_restores_staging(self):
        runner, registry = _make_runner()
        with envs.SGLANG_EAGER_INPUT_NO_COPY_IN_CAPTURE.override(False):
            with model_capture_mode():
                out = runner.load_batch(self.fb)
        registry.fill_from.assert_called_once()
        self.assertEqual(out, "staged-batch")

    def test_global_no_copy_flag_skips_staging(self):
        runner, registry = _make_runner()
        with envs.SGLANG_EAGER_INPUT_NO_COPY.override(True):
            out = runner.load_batch(self.fb)
        registry.fill_from.assert_not_called()
        self.assertIs(out.input_ids, self.fb.input_ids)


if __name__ == "__main__":
    unittest.main()
