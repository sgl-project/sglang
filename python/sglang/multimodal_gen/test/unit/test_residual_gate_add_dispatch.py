# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the PTX-only diffusion residual fast path."""

import pytest
import torch

from sglang.kernels.kda_kernels import residual_gate_add_jit as residual_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _inputs(dtype, layout):
    torch.manual_seed(13)
    residual = torch.randn(1, 16, 32, device="cuda", dtype=dtype)
    if layout == "transposed_row":
        residual = residual.transpose(1, 2).contiguous().transpose(1, 2)
    update = torch.randn(residual.shape, device="cuda", dtype=dtype)
    shape = {
        "full": residual.shape,
        "row": (1, 1, 32),
        "token": (1, 16, 1),
        "transposed_row": (1, 1, 32),
    }[layout]
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    return residual, update, gate


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["full", "row", "token", "transposed_row"])
def test_hip_dispatch_never_attempts_ptx(monkeypatch, dtype, layout):
    residual, update, gate = _inputs(dtype, layout)
    expected = residual + update * gate
    monkeypatch.setattr(torch.version, "hip", "test-hip")

    def unsupported(*args):
        pytest.fail("HIP must not attempt the PTX-only kernel")

    monkeypatch.setattr(residual_ops, "_residual_gate_add_custom_op", unsupported)
    assert not residual_ops.can_use_residual_gate_add_cuda(residual, update, gate)
    torch.testing.assert_close(
        residual_ops.residual_gate_add(residual, update, gate), expected, atol=0, rtol=0
    )
    with pytest.raises(RuntimeError, match="unsupported input"):
        residual_ops.residual_gate_add_cuda(residual, update, gate)


@pytest.mark.skipif(torch.version.hip is not None, reason="PTX kernel requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["full", "row", "token", "transposed_row"])
def test_cuda_fast_path_remains_exact(dtype, layout):
    residual, update, gate = _inputs(dtype, layout)
    assert residual_ops.can_use_residual_gate_add_cuda(residual, update, gate)
    torch.testing.assert_close(
        residual_ops.residual_gate_add_cuda(residual, update, gate),
        residual + update * gate,
        atol=0,
        rtol=0,
    )
