"""Unit tests for Qwen2MoeSparseMoeBlock._forward_deepep CUDA/DeepEP overlap.

Exercises the ``self._enable_deepep_cuda_shared_overlap`` init-static gate and
the fork/join scheduling contract added to ``_forward_deepep``:

- When the gate is True and the NPU ``enable_dual_stream`` branch is not taken,
  ``shared_expert`` forks onto ``self.alt_stream`` and joins via a CUDA event
  before the final ``add_``.
- When the gate is False the code path is bit-identical to the serial baseline
  (no ``wait_stream``, no event record/wait), and the numerical result
  ``experts + shared`` is unchanged.

Pure Python — no GPU, no model weights, no server. Mocks ``torch.cuda`` stream
APIs and ``torch.Tensor.record_stream`` so the scheduling contract is exercised
on any host.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.test.test_utils import CustomTestCase


class _StubTopKOutput:
    """Identity placeholder for StandardTopKOutput; not inspected downstream."""


def _build_block(*, gate_enabled: bool):
    """Bare stub exposing exactly the attributes _forward_deepep reads."""
    stub = SimpleNamespace()
    stub._enable_deepep_cuda_shared_overlap = gate_enabled
    stub.alt_stream = mock.MagicMock(name="alt_stream")
    stub.alt_stream.record_event = mock.MagicMock(return_value=mock.sentinel.event)
    stub.shared_expert = object()  # non-None marker; not called directly
    stub.is_nextn = True  # skip ExpertLocationDispatchInfo.init_new branch
    stub.layer_id = 0

    router_logits = torch.zeros(1, 8)
    stub.gate = mock.MagicMock(return_value=(router_logits, None))
    stub.topk = mock.MagicMock(return_value=_StubTopKOutput())
    stub.topk.empty_topk_output = mock.MagicMock(return_value=_StubTopKOutput())

    shared_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    stub._forward_shared_experts = mock.MagicMock(return_value=shared_tensor)

    experts_tensor = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    stub.experts = mock.MagicMock(return_value=experts_tensor)

    return stub


def _call(stub, hidden_states):
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_cuda_graph=lambda: False),
        num_token_non_padded=None,
    )
    return Qwen2MoeSparseMoeBlock._forward_deepep(stub, hidden_states, forward_batch)


class TestQwen2MoeDeepepCudaOverlap(CustomTestCase):
    def _run(self, *, gate_enabled):
        stub = _build_block(gate_enabled=gate_enabled)
        hidden = torch.ones(1, 4)

        current_stream = mock.MagicMock(name="current_stream")
        alt_ctx = mock.MagicMock()
        alt_ctx.__enter__ = mock.MagicMock(return_value=None)
        alt_ctx.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch(
            "sglang.srt.models.qwen2_moe.torch.cuda.current_stream",
            return_value=current_stream,
        ), mock.patch(
            "sglang.srt.models.qwen2_moe.torch.cuda.stream",
            return_value=alt_ctx,
        ), mock.patch.object(
            torch.Tensor, "record_stream", new=lambda self, s: None
        ):
            result = _call(stub, hidden)

        return result, stub, current_stream

    def test_gate_true_forks_and_joins_via_event(self):
        result, stub, current_stream = self._run(gate_enabled=True)

        stub.alt_stream.wait_stream.assert_called_once_with(current_stream)
        stub.alt_stream.record_event.assert_called_once_with()
        current_stream.wait_event.assert_called_once_with(mock.sentinel.event)

        self.assertTrue(
            torch.equal(result, torch.tensor([[11.0, 22.0, 33.0, 44.0]])),
            f"unexpected result: {result}",
        )

    def test_gate_false_stays_serial(self):
        result, stub, current_stream = self._run(gate_enabled=False)

        stub.alt_stream.wait_stream.assert_not_called()
        stub.alt_stream.record_event.assert_not_called()
        current_stream.wait_event.assert_not_called()

        self.assertTrue(
            torch.equal(result, torch.tensor([[11.0, 22.0, 33.0, 44.0]])),
            f"unexpected result: {result}",
        )


if __name__ == "__main__":
    unittest.main()
