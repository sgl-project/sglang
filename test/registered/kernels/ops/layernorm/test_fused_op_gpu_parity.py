"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")

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


@pytest.mark.parametrize(
    "op_attr", ["_SILU_AND_MUL", "_GELU_AND_MUL", "_GELU_TANH_AND_MUL"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8192), (128, 8192)])
def test_gated_activation(op_attr, dtype, shape):
    import sglang.kernels.ops.activation as act

    torch.manual_seed(0)
    op = getattr(act, op_attr)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = op.forward_native(x)
    for b in _eligible(op):
        _close(op.forward(x, backend=b), ref, dtype, f"{op.op} {b.value}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
