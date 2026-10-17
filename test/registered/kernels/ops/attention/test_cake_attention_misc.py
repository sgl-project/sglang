"""Cake Kimi-K3 AttnRes, MiniMax-H3 varlen attention and NVFP4 attention through
sglang.kernels.

Registry resolution for every ``attention_misc`` adapter; bitwise parity of
the facade with direct FlashInfer calls; references: MiniMax-H3 BF16 vs FP32
at 1e-2, NVFP4 variants vs FP32 / SDPA at atol 1.0 rtol 0.1, AttnRes vs
FlashInfer's FP32 ``reference_kimi_k3_attn_res`` at the tolerance FlashInfer
measured for the fused BF16 kernel (atol 8e-2 / rtol 3e-2). GPU tests skip
when FlashInfer lacks the module, the cell has no generated program, or the
device is outside sm_100a / sm_103a (NVFP4 attention: sm_103a only).
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_misc as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_kimi_k3_attn_res,
    cake_minimax_h3_varlen_attention,
    cake_minimax_h3_varlen_nvfp4_attention,
    cake_prepare_kimi_k3_attn_res,
    cake_prepare_minimax_h3_varlen_attention,
    cake_prepare_nvfp4_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "kimi_k3_attn_res",
    "prepare_kimi_k3_attn_res",
    "minimax_h3_varlen_attention",
    "prepare_minimax_h3_varlen_attention",
    "minimax_h3_varlen_nvfp4_attention",
    "prepare_minimax_h3_varlen_nvfp4_attention",
    "prepare_nvfp4_attention",
)


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_misc:")
    assert spec.load() is not None


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake kernel is built for {archs}, device is {cc}")


# --------------------------------------------------------------------------
# Kimi-K3 AttnRes
# --------------------------------------------------------------------------


def _attn_res_inputs(M, K, device, *, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    H = cake.ATTN_RES_HIDDEN
    bf16 = dict(device=device, dtype=torch.bfloat16)
    return dict(
        prefix=torch.empty((M, H), **bf16).uniform_(-1.0, 1.0, generator=g),
        delta=torch.empty((M, H), **bf16).uniform_(-0.015625, 0.015625, generator=g),
        blocks=torch.empty((M, cake.ATTN_RES_MAX_BLOCKS, H), **bf16).uniform_(
            -1.0, 1.0, generator=g
        ),
        norm_weight=torch.empty(H, **bf16).normal_(1.0, 0.05, generator=g),
        qk_weight=torch.empty(H, **bf16).normal_(0.0, 0.02, generator=g),
        output_norm_weight=torch.empty(H, **bf16).normal_(1.0, 0.05, generator=g),
        out=torch.full((M, H), float("nan"), **bf16),
    )


def _clone(inputs):
    return {k: v.clone() for k, v in inputs.items()}


@pytest.mark.parametrize("M,K", [(16, 8), (1, 4)])
def test_kimi_k3_attn_res_matches_flashinfer_and_reference(M, K):
    _skip_unless(cake.ARCHS, cake.FI_ATTN_RES_MODULE, cake.FI_ATTN_RES_BACKEND_MODULE)
    from flashinfer.experimental.cake_kimi_k3_attn_res import reference_kimi_k3_attn_res
    from flashinfer.kimi_k3_attn_res import kimi_k3_attn_res

    device = torch.device("cuda")
    inputs = _attn_res_inputs(M, K, device, seed=40)
    if not cake.supports_kimi_k3_attn_res(
        inputs["prefix"], inputs["delta"], inputs["blocks"], inputs["out"], num_blocks=K
    ):
        pytest.skip(f"no generated AttnRes program for M={M}, K={K}")
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(**expected, num_blocks=K)
    direct = _clone(inputs)
    kimi_k3_attn_res(**direct, num_blocks=K, backend="cake")
    prepared_inputs = _clone(inputs)

    result = cake_kimi_k3_attn_res(**inputs, num_blocks=K)
    runner = cake_prepare_kimi_k3_attn_res(**prepared_inputs, num_blocks=K)
    assert runner.launch() is prepared_inputs["out"]
    torch.cuda.synchronize()
    assert result is inputs["out"]
    for name in ("out", "prefix", "blocks"):
        assert torch.equal(inputs[name], direct[name]), name
        assert torch.equal(prepared_inputs[name], direct[name]), name
    assert torch.isfinite(inputs["out"].float()).all()
    torch.testing.assert_close(
        inputs["out"].float(), expected["out"].float(), atol=8e-2, rtol=3e-2
    )
    assert torch.equal(inputs["prefix"], expected["prefix"])
    assert torch.equal(inputs["blocks"], expected["blocks"])


# --------------------------------------------------------------------------
# MiniMax-H3 packed-varlen attention
# --------------------------------------------------------------------------


def _minimax_reference(q, k, v, cu, scale):
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for a, b in zip(cu, cu[1:]):
        if b <= a:
            continue
        for h in range(q.shape[1]):
            logits = (q[a:b, h].float() @ k[a:b, h].float().T) * scale
            out[a:b, h] = torch.softmax(logits, dim=-1) @ v[a:b, h].float()
    return out


def _minimax_inputs(cu, heads, device, *, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (cu[-1], heads, cake.MINIMAX_HEAD_DIM)
    q, k, v = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
        for _ in range(3)
    )
    return q, k, v, torch.tensor(cu, dtype=torch.int32, device=device)


def test_minimax_h3_varlen_bf16_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_PREFILL_MODULE, cake.FI_MINIMAX_BACKEND_MODULE)
    from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
        generated_program_available,
    )
    from flashinfer.prefill import minimax_h3_varlen_attention

    device = torch.device("cuda")
    if not generated_program_available(device, "bf16"):
        pytest.skip("no generated MiniMax-H3 bf16 program registered")
    cu, heads = [0, 128, 129, 400], 7
    q, k, v, cu_seqlens = _minimax_inputs(cu, heads, device, seed=41)
    assert cake.supports_minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    out = cake_minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    out_fi = minimax_h3_varlen_attention(q, k, v, cu_seqlens, backend="cake")
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    scale = 1.0 / math.sqrt(cake.MINIMAX_HEAD_DIM)
    ref = _minimax_reference(q, k, v, cu, scale)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)

    out_p = torch.full_like(out, float("nan"))
    runner = cake_prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, out=out_p, cu_seqlens_host=cu
    )
    assert runner.launch() is out_p
    torch.cuda.synchronize()
    torch.testing.assert_close(out_p.float(), ref, atol=1e-2, rtol=1e-2)
    q.copy_(torch.randn(q.shape, device=device, dtype=torch.bfloat16))
    before = torch.cuda.memory_allocated()
    runner.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    torch.testing.assert_close(
        out_p.float(), _minimax_reference(q, k, v, cu, scale), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
def test_minimax_h3_varlen_nvfp4_matches_flashinfer_and_reference(pv_mode):
    _skip_unless(cake.ARCHS, cake.FI_PREFILL_MODULE, cake.FI_MINIMAX_BACKEND_MODULE)
    from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
        NVFP4_VARIANT,
        generated_program_available,
    )
    from flashinfer.prefill import minimax_h3_varlen_nvfp4_attention

    device = torch.device("cuda")
    if not generated_program_available(device, NVFP4_VARIANT[pv_mode]):
        pytest.skip(f"no generated MiniMax-H3 {NVFP4_VARIANT[pv_mode]} program")
    cu, heads = [0, 256, 300], 7
    q, k, v, cu_seqlens = _minimax_inputs(cu, heads, device, seed=42)
    assert cake.supports_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode
    )
    out = cake_minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode=pv_mode)
    out_fi = minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode, backend="cake"
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    ref = _minimax_reference(q, k, v, cu, 1.0 / math.sqrt(cake.MINIMAX_HEAD_DIM))
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=0.1)


# --------------------------------------------------------------------------
# NVFP4 dense attention (SM103)
# --------------------------------------------------------------------------


def test_nvfp4_attention_prepare_launch_matches_sdpa():
    _skip_unless(
        cake.NVFP4_ATTENTION_ARCHS,
        cake.FI_PREFILL_MODULE,
        cake.FI_NVFP4_ATTENTION_BACKEND_MODULE,
    )
    from flashinfer.prefill import prepare_nvfp4_attention

    device = torch.device("cuda")
    torch.manual_seed(43)
    q = torch.randn((1, 2, 512, 128), dtype=torch.bfloat16, device=device)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    assert cake.supports_nvfp4_attention(q, k, v, out)
    runner = cake_prepare_nvfp4_attention(q, k, v, out)
    assert runner() is out
    out_fi = torch.empty_like(q)
    prepare_nvfp4_attention(q, k, v, out_fi, backend="cake")()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1.0, rtol=0.1)
    snapshot = out.clone()
    out.zero_()
    runner()
    torch.cuda.synchronize()
    assert torch.equal(out, snapshot)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
