import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.residual_gate_add import (
    can_use_residual_gate_add_cuda,
    residual_gate_add,
    residual_gate_add_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


CASES = [
    ((1, 1024, 4096), (1, 1, 4096)),
    ((1, 512, 4096), (1, 512, 4096)),
    ((1, 17, 65), (1, 1, 65)),
    ((1, 17, 65), (1, 17, 65)),
    # FLUX.1 / FLUX.2-klein 1024^2 shapes (D=3072): dual-stream image/text
    # and single-stream/joint concat; gates are [1, 1, D] modulation rows.
    ((1, 4096, 3072), (1, 1, 3072)),
    ((1, 512, 3072), (1, 1, 3072)),
    ((1, 4608, 3072), (1, 1, 3072)),
    # FLUX.2-dev (D=6144) joint sequence.
    ((1, 4608, 6144), (1, 1, 6144)),
    # ERNIE-4.5-VL 1024^2 image tokens plus text tokens.
    ((1, 4216, 4096), (1, 1, 4096)),
]


def _tol(dtype: torch.dtype) -> float:
    return 1e-5 if dtype == torch.float32 else 5e-2


def _assert_matches_torch(out: torch.Tensor, ref: torch.Tensor) -> None:
    if ref.dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=_tol(ref.dtype), rtol=_tol(ref.dtype))
    else:
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("residual_shape,gate_shape", CASES)
def test_residual_gate_add_matches_torch(residual_shape, gate_shape):
    residual = torch.randn(residual_shape, device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn(gate_shape, device="cuda", dtype=torch.bfloat16)

    out = residual_gate_add_cuda(residual, update, gate)
    ref = residual + update * gate
    _assert_matches_torch(out, ref)
    assert torch.equal(residual_gate_add(residual, update, gate), ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_shape", [(1, 1, 64), (1, 9, 64)])
def test_residual_gate_add_dtypes(dtype, gate_shape):
    residual = torch.randn((1, 9, 64), device="cuda", dtype=dtype)
    update = torch.randn_like(residual)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)

    out = residual_gate_add_cuda(residual, update, gate)
    ref = residual + update * gate
    _assert_matches_torch(out, ref)


def test_can_use_residual_gate_add_cuda_rejects_unsupported_inputs():
    residual = torch.randn((1, 8, 64), device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn((1, 1, 64), device="cuda", dtype=torch.bfloat16)

    assert can_use_residual_gate_add_cuda(residual, update, gate)
    assert not can_use_residual_gate_add_cuda(residual.cpu(), update, gate)
    assert not can_use_residual_gate_add_cuda(residual, update.float(), gate)
    assert not can_use_residual_gate_add_cuda(residual, update[:, ::2], gate)
    assert not can_use_residual_gate_add_cuda(residual, update, gate[:, :, ::2])
    empty_residual = residual[:, :0]
    empty_update = update[:, :0]
    assert not can_use_residual_gate_add_cuda(empty_residual, empty_update, gate)
    assert torch.equal(
        residual_gate_add(empty_residual, empty_update, gate),
        empty_residual + empty_update * gate,
    )

    # Only [1, ..., 1, D] row-broadcast gates are supported; a batched
    # [B>1, 1, D] gate is not row-broadcast here and must fall back.
    batched_residual = torch.randn((2, 8, 64), device="cuda", dtype=torch.bfloat16)
    batched_update = torch.randn_like(batched_residual)
    batched_gate = torch.randn((2, 1, 64), device="cuda", dtype=torch.bfloat16)
    assert not can_use_residual_gate_add_cuda(
        batched_residual, batched_update, batched_gate
    )
    assert torch.equal(
        residual_gate_add(batched_residual, batched_update, batched_gate),
        batched_residual + batched_update * batched_gate,
    )


def test_residual_gate_add_custom_op_torch_compile_fullgraph():
    residual = torch.randn((1, 32, 128), device="cuda", dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    gate = torch.randn((1, 1, 128), device="cuda", dtype=torch.bfloat16)

    def fn(residual, update, gate):
        return residual_gate_add(residual, update, gate)

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(residual, update, gate)
    ref = residual + update * gate
    _assert_matches_torch(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
