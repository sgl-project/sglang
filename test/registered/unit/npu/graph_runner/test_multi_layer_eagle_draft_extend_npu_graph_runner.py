"""
Unit tests for sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Mock the entire sglang package tree.
# ---------------------------------------------------------------------------
for _pkg in (
    "sglang",
    "sglang.test",
    "sglang.test.ci",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.speculative",
    "sglang.srt.model_executor",
    "sglang.srt.hardware_backend",
    "sglang.srt.hardware_backend.npu",
    "sglang.srt.hardware_backend.npu.graph_runner",
):
    sys.modules.setdefault(_pkg, MagicMock())

_ci_mod = types.ModuleType("sglang.test.ci.ci_register")
_ci_mod.register_npu_ci = lambda *a, **kw: None
sys.modules.setdefault("sglang.test.ci.ci_register", _ci_mod)

from sglang.test.ci.ci_register import register_npu_ci  # noqa: E402

register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")

# ---------------------------------------------------------------------------
# Stub cuda_graph_config.
# ---------------------------------------------------------------------------
_cg_mod = types.ModuleType("sglang.srt.model_executor.cuda_graph_config")


def cuda_graph_fully_disabled():
    return False


_cg_mod.cuda_graph_fully_disabled = cuda_graph_fully_disabled
sys.modules["sglang.srt.model_executor.cuda_graph_config"] = _cg_mod

# ---------------------------------------------------------------------------
# Stub parent classes.
# ---------------------------------------------------------------------------
_parent_mod = types.ModuleType(
    "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner"
)


class _FakeParent1:
    """Stand-in for MultiLayerEagleDraftExtendCudaGraphRunner."""

    def __init__(self, eagle_worker=None, step=None):
        self.eagle_worker = eagle_worker
        self.step = step


class _FakeParent2:
    """Stand-in for MultiLayerEagleMultiStepDraftExtendCudaGraphRunner."""

    def __init__(self, eagle_worker=None):
        self.eagle_worker = eagle_worker


_parent_mod.MultiLayerEagleDraftExtendCudaGraphRunner = _FakeParent1
_parent_mod.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner = _FakeParent2
sys.modules[
    "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner"
] = _parent_mod

# ---------------------------------------------------------------------------
# Load the target module directly from file.
# ---------------------------------------------------------------------------
_TARGET_FILE = (
    Path(__file__).resolve().parents[5]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "graph_runner"
    / "multi_layer_eagle_draft_extend_npu_graph_runner.py"
)

_spec = importlib.util.spec_from_file_location(
    "sglang.srt.hardware_backend.npu.graph_runner"
    ".multi_layer_eagle_draft_extend_npu_graph_runner",
    str(_TARGET_FILE),
)
_target_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _target_mod
_spec.loader.exec_module(_target_mod)

MultiLayerEagleDraftExtendNpuGraphRunner = (
    _target_mod.MultiLayerEagleDraftExtendNpuGraphRunner
)
MultiLayerEagleMultiStepDraftExtendNpuGraphRunner = (
    _target_mod.MultiLayerEagleMultiStepDraftExtendNpuGraphRunner
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(bs=4, raw_bs=2):
    r = object.__new__(MultiLayerEagleDraftExtendNpuGraphRunner)
    r.buffers = MagicMock()
    r.buffers.seq_lens_cpu = torch.tensor([10, 20, 30, 40])
    r.backend = MagicMock()
    r.bs = bs
    r.raw_bs = raw_bs
    return r


def _make_multi_step_runner():
    r = object.__new__(MultiLayerEagleMultiStepDraftExtendNpuGraphRunner)
    r.eagle_worker = MagicMock(name="eagle_worker")
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiLayerEagleDraftExtendNpuGraphRunner(unittest.TestCase):
    def test_replay_graph(self):
        """seq_lens from buffers (not forward_batch), attr_name=actual_seq_kvlen."""
        r = _make_runner(bs=4, raw_bs=2)
        r._replay_graph("key", MagicMock())
        r.backend.replay_with_input_update.assert_called_once_with(
            "key",
            seq_lens=[10, 20, 0, 0],
            attr_name="actual_seq_kvlen",
            attr_type=[],
        )

    def test_replay_graph_no_padding(self):
        """bs == raw_bs -> no zero padding."""
        r = _make_runner(bs=2, raw_bs=2)
        r._replay_graph("key", MagicMock())
        kwargs = r.backend.replay_with_input_update.call_args.kwargs
        self.assertEqual(kwargs["seq_lens"], [10, 20])


class TestMultiLayerEagleMultiStepDraftExtendNpuGraphRunner(unittest.TestCase):
    def test_create_runner(self):
        """_create_runner(step) returns MultiLayerEagleDraftExtendNpuGraphRunner
        constructed with self.eagle_worker and the given step."""
        r = _make_multi_step_runner()
        result = r._create_runner(step=3)
        self.assertIsInstance(result, MultiLayerEagleDraftExtendNpuGraphRunner)
        self.assertIs(result.eagle_worker, r.eagle_worker)
        self.assertEqual(result.step, 3)

    def test_cuda_graph_disabled_delegates(self):
        """_cuda_graph_disabled() delegates to cuda_graph_fully_disabled()."""
        r = _make_multi_step_runner()
        with patch.object(_target_mod, "cuda_graph_fully_disabled", return_value=True):
            self.assertTrue(r._cuda_graph_disabled())
        with patch.object(_target_mod, "cuda_graph_fully_disabled", return_value=False):
            self.assertFalse(r._cuda_graph_disabled())


if __name__ == "__main__":
    unittest.main()
