"""Capture-state guard for the fused AR + RMSNorm + per-token FP8 quant wrapper.

Regression coverage for the P0 in
``GroupCoordinator.fused_allreduce_rmsnorm_quant_per_token``
(``srt/distributed/parallel_state.py``). Under AITER's TC-piecewise CUDA-graph
replay state -- the global ``_IS_CAPTURING`` flag is set, the current stream is
NOT capturing, and we are inside a piecewise graph -- the ``custom_fused_ar_rms_
quant`` kernel returns dummy zero outputs. Without a guard that non-``None`` tuple
is consumed downstream as real ``(fp8, residual, per-token scale)`` activations,
silently corrupting the forward. The sibling non-quant ``fused_allreduce_rmsnorm``
already guards this state; the per-token quant wrapper must too.

The wrapper only reads ``self.ca_comm``, so the guard logic is driven directly
with a fake communicator (no process group / GPU needed) and the three capture
predicates patched. Each case pins one leg of the three-way AND so a predicate
that degrades to always-true/false is caught:

  * full capture state  -> returns None, kernel NOT invoked (the P0),
  * not in piecewise gr. -> kernel invoked (guard must not disable the fold),
  * stream IS capturing  -> kernel invoked (real capture uses the fused kernel),
  * _IS_CAPTURING unset   -> kernel invoked (normal decode/prefill).

Pure control-flow logic, no server/engine -- runs on CPU CI.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PIECEWISE = "sglang.srt.distributed.parallel_state.is_in_tc_piecewise_cuda_graph"
_STREAM_CAPTURING = "torch.cuda.is_current_stream_capturing"

# Sentinel the fake kernel returns; distinct from the guard's None so the two
# outcomes ("fell through to kernel" vs "guard returned None") are never confused.
_KERNEL_RESULT = ("fp8", "residual", "scale")


class _FakeCaComm:
    """Minimal stand-in for the aiter custom-all-reduce communicator: exposes a
    dummy-zero ``custom_fused_ar_rms_quant`` and records whether it was called."""

    def __init__(self, is_capturing):
        self.disabled = False
        self._IS_CAPTURING = is_capturing
        self.called = False

    def custom_fused_ar_rms_quant(
        self, input_, residual_inp_, weight_, eps, use_1stage
    ):
        self.called = True
        return _KERNEL_RESULT


def _call(ca_comm):
    # The wrapper only touches self.ca_comm; call it unbound with a fake self.
    fake_self = types.SimpleNamespace(ca_comm=ca_comm, world_size=2)
    x = torch.zeros(4, 16)
    with patch(
        "sglang.srt.distributed.parallel_state.is_hip", return_value=True
    ), patch(
        "sglang.srt.distributed.parallel_state.is_gfx95_supported",
        return_value=True,
    ):
        return GroupCoordinator.fused_allreduce_rmsnorm_quant_per_token(
            fake_self, x, x.clone(), torch.ones(16), 1e-6
        )


class TestFusedARQuantCaptureGuard(CustomTestCase):
    @patch(_PIECEWISE, return_value=True)
    @patch(_STREAM_CAPTURING, return_value=False)
    def test_piecewise_replay_returns_none_without_kernel(self, *_):
        # The P0 state: _IS_CAPTURING set, stream not capturing, in piecewise
        # graph. Must return None (fall back), NOT the dummy-zero kernel tuple.
        ca_comm = _FakeCaComm(is_capturing=True)
        self.assertIsNone(_call(ca_comm))
        self.assertFalse(ca_comm.called)

    @patch(_PIECEWISE, return_value=False)
    @patch(_STREAM_CAPTURING, return_value=False)
    def test_outside_piecewise_graph_uses_kernel(self, *_):
        # Not inside a piecewise graph -> the guard must NOT fire, else the fold
        # is silently disabled in normal operation.
        ca_comm = _FakeCaComm(is_capturing=True)
        self.assertEqual(_call(ca_comm), _KERNEL_RESULT)
        self.assertTrue(ca_comm.called)

    @patch(_PIECEWISE, return_value=True)
    @patch(_STREAM_CAPTURING, return_value=True)
    def test_stream_capturing_uses_kernel(self, *_):
        # Real capture (current stream capturing) legitimately uses the fused
        # kernel; the guard targets only the replay-on-non-capture-stream state.
        ca_comm = _FakeCaComm(is_capturing=True)
        self.assertEqual(_call(ca_comm), _KERNEL_RESULT)
        self.assertTrue(ca_comm.called)

    @patch(_PIECEWISE, return_value=True)
    @patch(_STREAM_CAPTURING, return_value=False)
    def test_not_capturing_uses_kernel(self, *_):
        # _IS_CAPTURING unset (normal decode/prefill) -> kernel path.
        ca_comm = _FakeCaComm(is_capturing=False)
        self.assertEqual(_call(ca_comm), _KERNEL_RESULT)
        self.assertTrue(ca_comm.called)


if __name__ == "__main__":
    unittest.main()
