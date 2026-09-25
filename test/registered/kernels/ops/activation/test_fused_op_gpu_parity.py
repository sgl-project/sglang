"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="extra-a", runner_config="1-gpu-small")

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
