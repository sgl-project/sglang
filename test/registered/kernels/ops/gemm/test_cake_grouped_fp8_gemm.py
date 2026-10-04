"""Cake contiguous grouped FP8 GEMM (plain + fused SwiGLU/quant) through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that the
facade's prepared runner is bitwise identical to FlashInfer's own, and that the
result matches a pure-torch reference within FP8 tolerance; the block-scaled
contract (packed UE8M0 int32 scales, native ``-1`` padding rows, ``alignment``,
``launch(a=..., ...)`` rebinding) is covered when the installed FlashInfer
provides it. Skips (with the reason) when FlashInfer lacks the Cake modules /
generated programs or the GPU is not SM100a.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import gemm_grouped_fp8 as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.gemm.cake import (
    cake_prepare_group_gemm_fp8_nt_groupwise_contiguous,
    cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous",
    "gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
)
ATOL = RTOL = 0.1  # FP8 operands


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.gemm_grouped_fp8:")


def _skip_unless_supported(fused: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    modules = (cake.FI_MODULE, cake.FI_JIT_MODULE)
    if fused:
        modules += (cake.FI_SILU_MODULE, cake.FI_SILU_JIT_MODULE)
    if not flashinfer_module_available(*modules):
        pytest.skip("installed FlashInfer lacks the Cake grouped FP8 GEMM modules")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake grouped FP8 GEMM is built for sm_100a only, device is {cc}")


def _make_inputs(group_counts, n, k, *, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    groups, m = len(group_counts), sum(group_counts)
    a = torch.randn((m, k), generator=g, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n, k), generator=g, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.pow(
        2.0, torch.randint(-8, 1, (m, k // 128), generator=g, device=device).float()
    )
    b_scale = torch.pow(
        2.0,
        torch.randint(
            -8, 1, (groups, n // 128, k // 128), generator=g, device=device
        ).float(),
    )
    counts = torch.tensor(group_counts, dtype=torch.int64, device=device)
    m_indices = torch.repeat_interleave(
        torch.arange(groups, dtype=torch.int32, device=device), counts
    )
    return a, b, a_scale.contiguous(), b_scale.contiguous(), m_indices.contiguous()


def _reference_gemm(a, b, a_scale, b_scale, m_indices):
    """K128 partial dots, post-dot scaling, ordered FP32 accumulation (FP32 out)."""
    m, k = a.shape
    groups, n, _ = b.shape
    a32, b32 = a.float(), b.float()
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for g in range(groups):
        rows = m_indices == g
        if not bool(rows.any()):
            continue
        acc = torch.zeros((int(rows.sum()), n), dtype=torch.float32, device=a.device)
        for q in range(k // 128):
            partial = (
                a32[rows, q * 128 : (q + 1) * 128]
                @ b32[g, :, q * 128 : (q + 1) * 128].T
            )
            scale = a_scale[rows, q].reshape(-1, 1) * b_scale[
                g, :, q
            ].repeat_interleave(128).reshape(1, n)
            acc = acc + partial * scale
        out[rows] = acc
    return out


@pytest.mark.parametrize(
    "group_counts,n,k",
    [
        pytest.param([128, 256, 128, 256], 256, 128, id="k128"),
        pytest.param([128, 1], 384, 1024, id="deepk_c2_128_1"),
        pytest.param([0, 128, 1], 256, 384, id="k384_leading_empty"),
    ],
)
def test_plain_matches_flashinfer_and_reference(group_counts, n, k):
    _skip_unless_supported(fused=False)
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n, k, seed=1234, device=device
    )
    if not cake.supports_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    ):
        pytest.skip("FlashInfer registers no generated program for this device")
    runner = cake_prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    assert isinstance(
        runner, cake.get_prepared_group_gemm_fp8_nt_groupwise_contiguous_class()
    )
    out = runner.launch()
    from flashinfer.gemm.cake_grouped_fp8_gemm import (
        prepare_group_gemm_fp8_nt_groupwise_contiguous as fi_prepare,
    )

    out_fi = fi_prepare(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    ).launch()
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (sum(group_counts), n)
    assert torch.equal(out, out_fi)
    ref = _reference_gemm(a, b, a_scale, b_scale, m_indices)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    # Second launch picks up new contents of the bound tensors.
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    out2 = runner.launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out2.float(),
        _reference_gemm(a, b, a_scale, b_scale, m_indices),
        atol=ATOL,
        rtol=RTOL,
    )


def _pack_ue8m0_mn_major(exponents: torch.Tensor) -> torch.Tensor:
    """uint8 ``(rows, kg)`` exponents -> int32 ``(rows, ceil(kg/4))`` with the
    DeepGEMM dispatcher's MN-major ``(1, rows)`` strides (byte 0 = lowest block)."""
    rows, kg = exponents.shape
    cols = -(-kg // 4)
    padded = torch.zeros((rows, 4 * cols), dtype=torch.uint8, device=exponents.device)
    padded[:, :kg] = exponents
    return padded.view(torch.int32).t().contiguous().t()


def _pack_weight_ue8m0(block_exponents: torch.Tensor) -> torch.Tensor:
    """uint8 ``(G, N/128, kg)`` -> row-repeated int32 ``(G, N, ceil(kg/4))`` with
    the ``transform_scale_ue8m0`` strides ``(N*cols, 1, N)``."""
    groups, n_blocks, kg = block_exponents.shape
    n = n_blocks * 128
    rows = block_exponents.repeat_interleave(128, dim=1).reshape(groups * n, kg)
    cols = -(-kg // 4)
    padded = torch.zeros((groups * n, 4 * cols), dtype=torch.uint8, device=rows.device)
    padded[:, :kg] = rows
    packed = padded.view(torch.int32).view(groups, n, cols)
    return packed.permute(0, 2, 1).contiguous().permute(0, 2, 1)


def _make_block_scaled_inputs(group_counts, n, k, alignment, *, seed, device):
    """Compact layout: every expert's run padded to ``alignment`` rows, padding
    rows ``-1`` in ``m_indices``; scales as packed UE8M0 plus their FP32 values."""
    g = torch.Generator(device=device).manual_seed(seed)
    groups = len(group_counts)
    padded = [-(-c // alignment) * alignment for c in group_counts]
    m = sum(padded)
    a = torch.randn((m, k), generator=g, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n, k), generator=g, device=device).to(torch.float8_e4m3fn)
    exp_a = torch.randint(119, 128, (m, k // 128), generator=g, device=device).to(
        torch.uint8
    )
    exp_b = torch.randint(
        119, 128, (groups, n // 128, k // 128), generator=g, device=device
    ).to(torch.uint8)
    m_indices = torch.full((m,), -1, dtype=torch.int32, device=device)
    start = 0
    for expert, (count, rows) in enumerate(zip(group_counts, padded)):
        m_indices[start : start + count] = expert
        start += rows
    return SimpleNamespace(
        a=a,
        b=b,
        a_scale=_pack_ue8m0_mn_major(exp_a),
        b_scale=_pack_weight_ue8m0(exp_b),
        a_scale_f32=torch.exp2(exp_a.float() - 127.0),
        b_scale_f32=torch.exp2(exp_b.float() - 127.0),
        m_indices=m_indices,
    )


@pytest.mark.parametrize(
    "group_counts,n,k,alignment",
    [
        pytest.param([100, 120, 0, 128], 256, 512, 128, id="ue8m0_a128_padding"),
        pytest.param([1, 300, 64], 384, 1024, 128, id="ue8m0_a128_deepk"),
        pytest.param([30, 64, 1], 256, 1024, 32, id="ue8m0_a32_multirun"),
    ],
)
def test_block_scaled_ue8m0_skips_padding_and_rebinds(group_counts, n, k, alignment):
    _skip_unless_supported(fused=False)
    if not cake.block_scaled_contract_available():
        pytest.skip(
            "installed FlashInfer lacks the block-scaled contiguous grouped FP8 "
            "GEMM contract (alignment keyword)"
        )
    device = torch.device("cuda")
    x = _make_block_scaled_inputs(group_counts, n, k, alignment, seed=99, device=device)
    assert x.a_scale.dtype == torch.int32 and x.a_scale.stride() == (1, x.a.shape[0])
    out = torch.full(
        x.a.shape[:1] + (n,), float("nan"), dtype=torch.bfloat16, device=device
    )
    if not cake.supports_group_gemm_fp8_nt_groupwise_contiguous(
        x.a, x.b, x.a_scale, x.b_scale, x.m_indices, out, alignment=alignment
    ):
        pytest.skip("FlashInfer registers no generated program for this device")
    runner = cake_prepare_group_gemm_fp8_nt_groupwise_contiguous(
        x.a,
        x.b,
        x.a_scale,
        x.b_scale,
        x.m_indices,
        out,
        validate_indices=True,
        alignment=alignment,
    )
    runner.launch()
    torch.cuda.synchronize()
    valid = x.m_indices >= 0
    assert torch.isnan(out[~valid].float()).all(), "padding rows must stay untouched"
    ref = _reference_gemm(x.a, x.b, x.a_scale_f32, x.b_scale_f32, x.m_indices)
    assert torch.isfinite(out[valid].float()).all()
    torch.testing.assert_close(out[valid].float(), ref[valid], atol=ATOL, rtol=RTOL)
    # Rebind the per-token operands (same geometry) on a later launch.
    a2 = torch.randn(x.a.shape, device=device).to(torch.float8_e4m3fn)
    out2 = torch.full_like(out, float("nan"))
    runner.launch(a=a2, a_scale=x.a_scale, m_indices=x.m_indices, out=out2)
    torch.cuda.synchronize()
    assert torch.isnan(out2[~valid].float()).all()
    ref2 = _reference_gemm(a2, x.b, x.a_scale_f32, x.b_scale_f32, x.m_indices)
    torch.testing.assert_close(out2[valid].float(), ref2[valid], atol=ATOL, rtol=RTOL)
    # The first result was not disturbed by the rebinding.
    torch.testing.assert_close(out[valid].float(), ref[valid], atol=ATOL, rtol=RTOL)


def _dequantize(q, s, group=128):
    m, h = q.shape
    return (
        q.float().reshape(m, h // group, group) * s.reshape(m, h // group, 1)
    ).reshape(m, h)


@pytest.mark.parametrize(
    "group_counts,n2,k",
    [
        pytest.param([128], 256, 512, id="one_block_min_shape"),
        pytest.param([128, 128], 256, 512, id="two_experts_one_block_each"),
        pytest.param([256, 0, 64], 512, 1024, id="empty_and_partial_tail"),
    ],
)
def test_fused_silu_quant_matches_flashinfer_and_reference(group_counts, n2, k):
    _skip_unless_supported(fused=True)
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=4321, device=device
    )
    if not cake.supports_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    ):
        pytest.skip("FlashInfer registers no generated fused program for this device")
    runner = cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    assert isinstance(
        runner,
        cake.get_prepared_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_class(),
    )
    out_q, out_s = runner.launch()
    from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant as fi_prepare,
    )

    out_q_fi, out_s_fi = fi_prepare(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    ).launch()
    torch.cuda.synchronize()
    m, h = sum(group_counts), n2 // 2
    assert out_q.dtype == torch.float8_e4m3fn and tuple(out_q.shape) == (m, h)
    assert out_s.dtype == torch.float32 and tuple(out_s.shape) == (m, h // 128)
    assert torch.equal(out_q.view(torch.uint8), out_q_fi.view(torch.uint8))
    assert torch.equal(out_s, out_s_fi)
    y = _reference_gemm(a, b, a_scale, b_scale, m_indices).to(torch.bfloat16).float()
    gate, up = y[:, :h], y[:, h:]
    act = gate * torch.sigmoid(gate) * up
    assert torch.isfinite(out_s).all()
    torch.testing.assert_close(_dequantize(out_q, out_s), act, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
