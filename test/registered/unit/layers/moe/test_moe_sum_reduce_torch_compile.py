"""Regression test for MoE CUDA graph capture with the torch.compile sum-reduce.

Bug: serving an MoE model with CUDA graphs enabled aborted during capture with
``Capture cuda graph failed: RuntimeError when making fake tensor call ...
sum(): functions with out=... arguments don't support automatic
differentiation, but one of the arguments requires grad``.

Mechanism: ``moe_sum_reduce_torch_compile`` reduces the per-expert outputs into
a pre-allocated ``out`` buffer via ``torch.sum(..., out=out)`` plus an in-place
``mul_``, and is chosen for small batches -- exactly the batch sizes captured
for CUDA graphs. Capture traces the helper with ``torch.compile`` while autograd
is enabled, so a grad-tracked ``out`` makes the ``out=`` and in-place ops
illegal and capture aborts. Only the non-AITER Triton MoE path reaches this
helper (e.g. RDNA / gfx1151); CDNA and CUDA route through AITER ``moe_sum``.

This case invokes the helper with a grad-tracked ``out`` while autograd is
enabled: it raises on the pre-fix code and must pass, with the correct
reduction, after.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    moe_sum_reduce_torch_compile,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestMoeSumReduceTorchCompile(CustomTestCase):
    def test_grad_tracked_out_does_not_break_capture(self):
        device = "cuda"
        torch.manual_seed(0)
        num_tokens, topk, hidden = 8, 4, 16
        routed_scaling_factor = 1.5

        x = torch.randn(num_tokens, topk, hidden, device=device)
        # A capture buffer that carries autograd metadata, as it does during
        # CUDA graph capture (which does not run under no_grad).
        out = torch.zeros(num_tokens, hidden, device=device, requires_grad=True)

        with torch.enable_grad():
            moe_sum_reduce_torch_compile(x, out, routed_scaling_factor)

        expected = x.sum(dim=1) * routed_scaling_factor
        torch.testing.assert_close(out.detach(), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
