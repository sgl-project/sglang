"""Cake MiniMax-H3 packed-varlen attention (SM100a / SM103a) through sglang.kernels.

Checks, for the BF16 and NVFP4 one-shot entries and their prepared runners:
the registry resolves the explicit FlashInfer backend; the facade result is
bitwise identical to calling FlashInfer directly (and the prepared runner is
bitwise identical to the one-shot form); and the output matches a per-segment
FP32 torch reference within BF16 tolerance (1e-2) for the BF16 route and
within the FP4 block-scaled tolerance (atol 1.0 / rtol 0.1) for the NVFP4
routes. The BF16 route also consumes the engine's strided token-major views
(column chunks of the fused ``[T, 3 * H * 128]`` QKV projection and kind
slices of the ``[T, H, 3, 128]`` pre-attention pack) in place, bitwise equal
to the contiguous result. Skips when FlashInfer lacks the module, the GPU is
not sm_100a / sm_103a, or the generated program is not registered for the arch.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_attention as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_minimax_h3_varlen_attention,
    cake_minimax_h3_varlen_nvfp4_attention,
    cake_prepare_minimax_h3_varlen_attention,
    cake_prepare_minimax_h3_varlen_nvfp4_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 128
HEADS = 7
CU = [0, 133, 133, 300, 364]  # includes an empty segment and unaligned lengths


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.minimax_h3_varlen_attention",
        "diffusion.minimax_h3_varlen_nvfp4_attention",
        "diffusion.prepare_minimax_h3_varlen_attention",
        "diffusion.prepare_minimax_h3_varlen_nvfp4_attention",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_attention:"
    )


def _skip_unless_supported(variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake.FI_VARLEN_MODULE, cake.FI_VARLEN_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks experimental.minimax_h3_varlen_attention"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake.VARLEN_ARCHS:
        pytest.skip(
            f"Cake MiniMax-H3 varlen attention needs sm_100a/103a, device is {cc}"
        )
    from flashinfer.experimental.minimax_h3_varlen_attention import cake_backend

    if not cake_backend.generated_program_available(torch.device("cuda"), variant):
        pytest.skip(f"FlashInfer does not register the {variant} program for {cc}")


def _inputs(cu, heads, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (cu[-1], heads, HEAD_DIM)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g)
    return q, k, v, torch.tensor(cu, dtype=torch.int32, device=device)


def _reference(q, k, v, cu, scale):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        for a, b in zip(cu, cu[1:]):
            if b <= a:
                continue
            qs, ks, vs = (t[a:b].float().transpose(0, 1) for t in (q, k, v))
            logits = torch.matmul(qs, ks.transpose(1, 2)) * scale
            out[a:b] = torch.matmul(torch.softmax(logits, dim=-1), vs).transpose(0, 1)
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_bf16_matches_flashinfer_and_reference():
    _skip_unless_supported("bf16")
    device = torch.device("cuda")
    q, k, v, cu_seqlens = _inputs(CU, HEADS, 6090, device)
    assert cake.supports_minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    out = cake_minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    from flashinfer.prefill import minimax_h3_varlen_attention as fi_direct

    direct = fi_direct(q, k, v, cu_seqlens, cu_seqlens_host=CU)
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.equal(out, direct)
    expected = _reference(q, k, v, CU, 1.0 / math.sqrt(HEAD_DIM))
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)

    # Prepared runner: allocation-free launch, bitwise equal to the one-shot form.
    runner = cake_prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, cu_seqlens_host=CU
    )
    prepared = runner.launch()
    torch.cuda.synchronize()
    assert torch.equal(prepared, out)
    q.mul_(-1.0)
    runner.launch()
    torch.cuda.synchronize()
    again = cake_minimax_h3_varlen_attention(q, k, v, cu_seqlens, cu_seqlens_host=CU)
    torch.cuda.synchronize()
    assert torch.equal(runner.out, again)


def _fused_views(q, k, v):
    """Column chunks of the fused [T, 3*H*D] projection (stock q/k/v views)."""
    t, h, d = q.shape
    fused = torch.cat([x.reshape(t, h * d) for x in (q, k, v)], dim=-1)
    views = [x.view(t, h, d) for x in fused.split(h * d, dim=-1)]
    assert all(not x.is_contiguous() and x.stride(0) == 3 * h * d for x in views)
    return views


def _pack_views(q, k, v):
    """Kind slices of the destination-major pre-attention pack [T, H, 3, D]."""
    pack = torch.stack((q, k, v), dim=2)
    views = [pack[:, :, kind, :] for kind in range(3)]
    assert all(not x.is_contiguous() and x.stride(1) == 3 * HEAD_DIM for x in views)
    return views


@pytest.mark.parametrize("views", [_fused_views, _pack_views])
def test_bf16_strided_views_match_contiguous(views):
    _skip_unless_supported("bf16")
    device = torch.device("cuda")
    q, k, v, cu_seqlens = _inputs(CU, HEADS, 6091, device)
    dense = cake_minimax_h3_varlen_attention(q, k, v, cu_seqlens, cu_seqlens_host=CU)
    sq, sk, sv = views(q, k, v)
    assert torch.equal(sq, q) and torch.equal(sk, k) and torch.equal(sv, v)
    assert cake.supports_minimax_h3_varlen_attention(sq, sk, sv, cu_seqlens)
    out = cake_minimax_h3_varlen_attention(sq, sk, sv, cu_seqlens, cu_seqlens_host=CU)
    torch.cuda.synchronize()
    assert out.is_contiguous() and out.shape == q.shape
    assert torch.equal(out, dense)
    # The NVFP4 route quantizes contiguous operands only.
    assert not cake.supports_minimax_h3_varlen_nvfp4_attention(sq, sk, sv, cu_seqlens)


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
def test_nvfp4_matches_flashinfer_and_reference(pv_mode):
    _skip_unless_supported("nvfp4_fp4pv" if pv_mode == "fp4" else "nvfp4_fp8pv")
    device = torch.device("cuda")
    q, k, v, cu_seqlens = _inputs(CU, HEADS, 6091, device)
    assert cake.supports_minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode
    )
    out = cake_minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode=pv_mode)
    from flashinfer.prefill import minimax_h3_varlen_nvfp4_attention as fi_direct

    direct = fi_direct(q, k, v, cu_seqlens, pv_mode=pv_mode, cu_seqlens_host=CU)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = _reference(q, k, v, CU, 1.0 / math.sqrt(HEAD_DIM))
    torch.testing.assert_close(out.float(), expected, atol=1.0, rtol=0.1)

    runner = cake_prepare_minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode, cu_seqlens_host=CU
    )
    prepared = runner.launch()
    torch.cuda.synchronize()
    assert torch.equal(prepared, out)


def test_admission_rejects_out_of_contract():
    _skip_unless_supported("bf16")
    device = torch.device("cuda")
    q, k, v, cu_seqlens = _inputs([0, 16], HEADS, 1, device)
    assert cake.supports_minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    assert not cake.supports_minimax_h3_varlen_attention(q, k, v, cu_seqlens.long())
    assert not cake.supports_minimax_h3_varlen_attention(q, k[:, :3], v, cu_seqlens)
    assert not cake.supports_minimax_h3_varlen_attention(q.float(), k, v, cu_seqlens)
    # Strided views need a unit last stride and 16-byte head / token strides.
    wide = torch.zeros((16, HEADS, 2 * HEAD_DIM), dtype=torch.bfloat16, device=device)
    assert not cake.supports_minimax_h3_varlen_attention(
        wide[:, :, ::2], k, v, cu_seqlens
    )
    padded = torch.zeros((16, HEADS, HEAD_DIM + 4), dtype=torch.bfloat16, device=device)
    assert not cake.supports_minimax_h3_varlen_attention(
        q, padded[:, :, :HEAD_DIM], v, cu_seqlens
    )
    assert not cake.supports_minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode="int8"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
