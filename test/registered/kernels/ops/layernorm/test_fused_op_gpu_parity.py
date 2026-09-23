"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import math

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=41, stage="extra-a", runner_config="1-gpu-small")

# torch_compile is native under the hood; skip it here (compile time dominates)
# -- it is exercised in the CPU lane.
_SKIP = {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE}
_TOL = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _eligible(op):
    return [
        b for b in op.available_backends() if b not in _SKIP and op.backend_eligible(b)
    ]


def _close(got, ref, dtype, msg):
    torch.testing.assert_close(got, ref, **_TOL[dtype], msg=msg)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4096), (128, 4096), (7, 2048)])
def test_rmsnorm(dtype, shape):
    from sglang.kernels.ops.layernorm import _RMSNORM

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    ref = _RMSNORM.forward_native(x, w, 1e-6)
    for b in _eligible(_RMSNORM):
        _close(
            _RMSNORM.forward(x, w, 1e-6, backend=b), ref, dtype, f"rmsnorm {b.value}"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4096), (128, 4096)])
def test_fused_add_rmsnorm(dtype, shape):
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    torch.manual_seed(0)
    x0 = torch.randn(shape, dtype=dtype, device="cuda")
    r0 = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    x_ref, r_ref = x0.clone(), r0.clone()
    _FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
    for b in _eligible(_FUSED_ADD_RMSNORM):
        x, r = x0.clone(), r0.clone()
        _FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=b)
        _close(x, x_ref, dtype, f"fused_add {b.value} (normed)")
        _close(r, r_ref, dtype, f"fused_add {b.value} (residual)")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_rmsnorm(dtype):
    from sglang.kernels.ops.layernorm import _GEMMA_RMSNORM

    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=dtype, device="cuda")
    w = torch.randn(2048, dtype=dtype, device="cuda")
    ref = _GEMMA_RMSNORM.forward_native(x, w, 1e-6)
    for b in _eligible(_GEMMA_RMSNORM):
        _close(
            _GEMMA_RMSNORM.forward(x, w, 1e-6, backend=b),
            ref,
            dtype,
            f"gemma {b.value}",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_fused_add_rmsnorm(dtype):
    from sglang.kernels.ops.layernorm import _GEMMA_FUSED_ADD_RMSNORM

    torch.manual_seed(0)
    x0 = torch.randn(64, 2048, dtype=dtype, device="cuda")
    r0 = torch.randn(64, 2048, dtype=dtype, device="cuda")
    w = torch.randn(2048, dtype=dtype, device="cuda")
    x_ref, r_ref = x0.clone(), r0.clone()
    _GEMMA_FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
    for b in _eligible(_GEMMA_FUSED_ADD_RMSNORM):
        x, r = x0.clone(), r0.clone()
        _GEMMA_FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=b)
        _close(x, x_ref, dtype, f"gemma_fused_add {b.value} (normed)")
        _close(r, r_ref, dtype, f"gemma_fused_add {b.value} (residual)")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 64, 1024])
@pytest.mark.parametrize("combine_in_fp32", [False, True])
def test_beam_sandwich_residual_rmsnorm(dtype, rows, combine_in_fp32):
    from sglang.kernels.ops.layernorm import _BEAM_SANDWICH_RESIDUAL_RMSNORM as op
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.beam import depth_scaled_residual

    torch.manual_seed(0)
    post = torch.empty(5120, device="cuda", dtype=dtype).uniform_(0.5, 1.5)
    nxt = torch.empty_like(post).uniform_(0.5, 1.5)
    branch = (
        torch.randn(rows, 5120, generator=torch.Generator().manual_seed(1)).to(
            device="cuda", dtype=dtype
        )
        * 3
    )
    residual = (
        torch.randn(rows, 5120, generator=torch.Generator().manual_seed(2)).to(
            device="cuda", dtype=dtype
        )
        * 3
    )
    args = (branch, residual, post, nxt, 1e-6, (0.25, 0.968), combine_in_fp32)
    expected_out, expected_combined = op.forward_native(*args)
    for backend in _eligible(op):
        out, combined = op(*args, backend=backend)
        _close(combined, expected_combined, dtype, f"beam residual {backend.value}")
        _close(out, expected_out, dtype, f"beam norm {backend.value}")
        if dtype == torch.bfloat16:
            # Retain the fixed-sample bound against the serving RMSNorm backend.
            # Its reduction differs from the pure-torch reference above.
            post_norm = RMSNorm(5120, eps=1e-6).cuda().to(dtype)
            next_norm = RMSNorm(5120, eps=1e-6).cuda().to(dtype)
            post_norm.weight.data.copy_(post)
            next_norm.weight.data.copy_(nxt)
            serving_combined = depth_scaled_residual(
                post_norm(branch), residual, (0.25, 0.968), combine_in_fp32
            )
            assert (combined != serving_combined).float().mean().item() < 2e-3
            torch.testing.assert_close(
                combined, serving_combined, atol=1e-6, rtol=2**-7
            )
            torch.testing.assert_close(
                out, next_norm(serving_combined), atol=1e-6, rtol=2**-6
            )


@pytest.mark.parametrize("norms", [False, True])
def test_beam_residual_cancellation(norms):
    from sglang.kernels.ops.layernorm import _BEAM_SANDWICH_RESIDUAL_RMSNORM as op

    branch = torch.ones(1, 5120, device="cuda", dtype=torch.bfloat16)
    residual = -torch.ones_like(branch)
    post = torch.full((5120,), 5, device="cuda", dtype=branch.dtype) if norms else None
    nxt = torch.ones_like(post) if norms else None
    scales = (1 / math.sqrt(26), 5 / math.sqrt(26)) if norms else (0.5, 0.5)
    args = (branch, residual, post, nxt, 1e-6, scales, norms)
    expected = op.forward_native(*args)
    assert torch.count_nonzero(expected[1]).item() == 0
    for backend in _eligible(op):
        actual = op(*args, backend=backend)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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


@pytest.mark.parametrize("with_norms", [False, True])
def test_beam_custom_op_contracts(with_norms):
    from sglang.kernels.ops.elementwise.beam_fused import fused_attention_gate
    from sglang.kernels.ops.layernorm.beam_fused import _fused_sandwich_residual_rmsnorm

    # Sliced rows cover real/fake output strides as well as optional norm outputs.
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)[::2]
    weight = torch.ones(5120, device="cuda", dtype=x.dtype) if with_norms else None
    torch.library.opcheck(
        _fused_sandwich_residual_rmsnorm,
        (
            x,
            x.clone(),
            weight,
            weight,
            1e-6,
            0.25,
            0.968,
            True,
            torch.empty_like(x) if with_norms else None,
            torch.empty_like(x),
        ),
    )
    bias = torch.zeros(40, device="cuda", dtype=x.dtype)
    torch.library.opcheck(fused_attention_gate, (x, x.clone(), bias, 128))


@pytest.mark.parametrize("op_name", ["norm", "gate"])
@pytest.mark.parametrize("forced_backend", [False, True])
def test_beam_compiled_rounding(op_name, forced_backend):
    """Outer compilation must preserve residual cancellation and sigmoid rounding."""
    from torch._inductor import config as inductor_config

    from sglang.kernels.ops.elementwise import BeamAttentionGateOp
    from sglang.kernels.ops.layernorm import BeamSandwichResidualRMSNormOp

    if op_name == "norm":
        op = BeamSandwichResidualRMSNormOp()
        x = torch.ones(1, 5120, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(5120, device="cuda", dtype=x.dtype)
        args = (
            x,
            -x,
            weight * 5,
            weight,
            1e-6,
            (1 / math.sqrt(26), 5 / math.sqrt(26)),
            True,
        )
    else:
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
