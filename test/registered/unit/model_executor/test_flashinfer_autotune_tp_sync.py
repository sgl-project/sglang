"""CPU-only tests for TP-synchronized flashinfer autotune tactic selection.

Verifies ``flashinfer_autotune_context`` calls flashinfer's
``set_autotune_process_group`` (flashinfer #3186 / #3187) with the TP group's
gloo ``cpu_group`` on entry and resets it to ``None`` on exit, so every rank
picks the same kernel tactic. Divergent tactics size the symmetric-memory
scratch buffers differently and deadlock ``ncclCommWindowRegister`` during CUDA
graph capture under ``--enable-symm-mem``. No GPU and no real flashinfer needed.
"""

import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.runner.flashinfer_autotune"


def _make_model_runner(tp_size):
    """Fake ModelRunner with only the fields the context manager touches."""
    mr = MagicMock()
    mr.device = "cuda"
    mr.ps.tp_size = tp_size
    # Identity sentinel we assert set_autotune_process_group is called with.
    mr.tp_group.cpu_group = object()
    return mr


@contextlib.contextmanager
def _patched_flashinfer(*, has_set_group):
    """Inject a fake ``flashinfer.autotuner`` and stub torch cuda/stream calls.

    ``has_set_group=False`` omits ``set_autotune_process_group`` so the context
    manager's guarded import falls back to ``None`` (pre-0.6.15 flashinfer).
    """
    autotuner = types.ModuleType("flashinfer.autotuner")
    autotuner.autotune = lambda *a, **k: contextlib.nullcontext()
    set_group = MagicMock(name="set_autotune_process_group")
    if has_set_group:
        autotuner.set_autotune_process_group = set_group

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.autotuner = autotuner

    # ``.stream(...)`` must return a real context manager (not a bare MagicMock,
    # whose ``__exit__`` returns a truthy value and would swallow exceptions).
    device_module = MagicMock()
    device_module.stream.side_effect = lambda *a, **k: contextlib.nullcontext()

    with (
        patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.autotuner": autotuner},
        ),
        # The cache-path builder touches real flashinfer/cuda/fs; replace it.
        patch(
            f"{_MODULE}.flashinfer_autotune_cache_path",
            return_value=Path("/tmp/fake_autotune.json"),
        ),
        patch.object(torch.cuda, "current_stream", MagicMock()),
        patch.object(torch, "get_device_module", MagicMock(return_value=device_module)),
        envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
    ):
        yield set_group


class TestFlashinferAutotuneTPSync(CustomTestCase):
    @staticmethod
    def _ctx():
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            flashinfer_autotune_context,
        )

        return flashinfer_autotune_context

    def test_tp_gt_1_sets_on_entry_and_resets_on_exit(self):
        ctx = self._ctx()
        mr = _make_model_runner(tp_size=8)
        with _patched_flashinfer(has_set_group=True) as set_group:
            with ctx(mr, skip_logits=False):
                # entry: called once with the gloo cpu_group sentinel
                set_group.assert_called_once_with(mr.tp_group.cpu_group)
            # exit: reset to None
            self.assertEqual(
                set_group.call_args_list,
                [call(mr.tp_group.cpu_group), call(None)],
            )

    def test_tp_eq_1_never_syncs(self):
        ctx = self._ctx()
        mr = _make_model_runner(tp_size=1)
        with _patched_flashinfer(has_set_group=True) as set_group:
            with ctx(mr, skip_logits=False):
                pass
            set_group.assert_not_called()

    def test_reset_runs_when_body_raises(self):
        ctx = self._ctx()
        mr = _make_model_runner(tp_size=4)
        with _patched_flashinfer(has_set_group=True) as set_group:
            with self.assertRaises(RuntimeError):
                with ctx(mr, skip_logits=False):
                    raise RuntimeError("boom")
            # finally still reset the group to None
            self.assertEqual(
                set_group.call_args_list,
                [call(mr.tp_group.cpu_group), call(None)],
            )

    def test_missing_symbol_is_noop(self):
        ctx = self._ctx()
        mr = _make_model_runner(tp_size=8)
        ran = []
        with _patched_flashinfer(has_set_group=False) as set_group:
            with ctx(mr, skip_logits=False):
                ran.append(True)
        # body executed and the (absent) symbol was never invoked
        self.assertEqual(ran, [True])
        set_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
