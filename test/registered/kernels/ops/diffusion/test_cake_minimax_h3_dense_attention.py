"""Cake MiniMax-H3 dense BF16 attention through sglang.kernels.

Checks: the registry resolves the explicit FlashInfer backend; the facade
result is bitwise identical to calling FlashInfer directly; and the output
matches an exact-softmax FP32 reference over the BF16 operands (query
pre-scaled by ``bf16(1/sqrt(128))``, softmax scale 1.0) within BF16 tolerance.
Skips when FlashInfer lacks the module or the GPU is outside cc 9.0 / 10.x /
12.x (the JIT builds for those majors; SM120 is the performance target).
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_attention as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_dense_attention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OP = "diffusion.minimax_h3_dense_attention"
WIDTH, HEADS, HEAD_DIM = 7168, 56, 128


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_attention:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(cake.FI_DENSE_MODULE, cake.FI_DENSE_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks cake_minimax_h3_dense_attention")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.DENSE_ARCHS:
        pytest.skip(
            f"Cake MiniMax-H3 dense attention builds for 9.x/10.x/12.x, got {cc}"
        )


def _inputs(tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(tokens, WIDTH, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )


def _reference(q, k, v):
    tokens = q.shape[0]
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        scale = torch.tensor(
            cake.DENSE_QUERY_SCALE_BF16, dtype=torch.bfloat16, device=q.device
        )
        qh = (q * scale).view(tokens, HEADS, HEAD_DIM).transpose(0, 1).float()
        kh = k.view(tokens, HEADS, HEAD_DIM).transpose(0, 1).float()
        vh = v.view(tokens, HEADS, HEAD_DIM).transpose(0, 1).float()
        out = torch.softmax(qh @ kh.transpose(1, 2), dim=-1) @ vh
        return out.transpose(0, 1).reshape(tokens, WIDTH)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("tokens", [65, 257])
def test_matches_flashinfer_and_reference(tokens):
    _skip_unless_supported()
    device = torch.device("cuda")
    q, k, v = _inputs(tokens, seed=tokens, device=device)
    out = torch.empty_like(q)
    assert cake.supports_minimax_h3_dense_attention(q, k, v, out)
    result = cake_minimax_h3_dense_attention(q, k, v, out=out)
    assert result is out
    from flashinfer.diffusion_ops.cake_minimax_h3_dense_attention import (
        minimax_h3_dense_attention as fi_direct,
    )

    direct = fi_direct(q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = _reference(q, k, v)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


def test_admission_rejects_out_of_contract():
    _skip_unless_supported()
    device = torch.device("cuda")
    q, k, v = _inputs(8, seed=0, device=device)
    assert cake.supports_minimax_h3_dense_attention(q, k, v)
    assert not cake.supports_minimax_h3_dense_attention(q[:, :128], k, v)
    assert not cake.supports_minimax_h3_dense_attention(q.float(), k, v)
    assert not cake.supports_minimax_h3_dense_attention(q, k, v, out=q[:4])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
