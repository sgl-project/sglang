"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=13, stage="extra-a", runner_config="1-gpu-small")

_SKIP = {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _eligible(op):
    return [
        b for b in op.available_backends() if b not in _SKIP and op.backend_eligible(b)
    ]


@pytest.mark.parametrize("rows", [0, 1, 64, 1024])
def test_beam_attention_gate(rows):
    from sglang.kernels.ops.elementwise import _BEAM_ATTENTION_GATE as op

    generator = torch.Generator().manual_seed(5)
    bias = torch.randn(12, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    attention = torch.randn(rows, 12 * 128, generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )
    gate = (
        torch.randn(rows, 12 * 128, generator=generator).to(
            device="cuda", dtype=torch.bfloat16
        )
        * 2
    )
    expected = op.forward_native(attention, gate, bias, 128)
    for backend in _eligible(op):
        torch.testing.assert_close(
            op(attention, gate, bias, 128, backend=backend), expected, atol=0, rtol=0
        )


def test_beam_fp16_gate_rounding_boundaries():
    from sglang.kernels.ops.elementwise import _BEAM_ATTENTION_GATE as op

    gate = torch.tensor(
        [[0.0068359375, -0.00634765625]], device="cuda", dtype=torch.float16
    )
    attention = torch.ones_like(gate)
    bias = torch.zeros(1, device="cuda", dtype=torch.float16)
    expected = op.forward_native(attention, gate, bias, 2)
    for backend in _eligible(op):
        torch.testing.assert_close(
            op(attention, gate, bias, 2, backend=backend), expected, atol=0, rtol=0
        )


def test_beam_custom_op_contracts():
    from sglang.kernels.ops.elementwise.beam_fused import fused_attention_gate

    # Sliced rows cover real/fake output strides.
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)[::2]
    bias = torch.zeros(40, device="cuda", dtype=x.dtype)
    torch.library.opcheck(fused_attention_gate, (x, x.clone(), bias, 128))


@pytest.mark.parametrize("forced_backend", [False, True])
def test_beam_compiled_rounding(forced_backend):
    """Outer compilation must preserve sigmoid rounding."""
    from torch._inductor import config as inductor_config

    from sglang.kernels.ops.elementwise import BeamAttentionGateOp

    op = BeamAttentionGateOp()
    gate = torch.tensor(
        [[0.0068359375, -0.00634765625]], device="cuda", dtype=torch.float16
    )
    args = (
        torch.ones_like(gate),
        gate,
        torch.zeros(1, device="cuda", dtype=gate.dtype),
        2,
    )
    expected = op.forward_native(*args)
    op(*args)  # Warm backend imports before full-graph capture.
    op.enter_torch_compile(1)
    try:
        with inductor_config.patch(compile_threads=1):
            actual = (
                op(*args, backend=KernelBackend.TORCH_COMPILE)
                if forced_backend
                else torch.compile(op, fullgraph=True)(*args)
            )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        op.leave_torch_compile()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
