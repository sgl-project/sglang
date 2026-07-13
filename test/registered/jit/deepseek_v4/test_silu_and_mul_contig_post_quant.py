from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)

GROUP_SIZE = 128


def _reference(x: torch.Tensor, swiglu_limit: float | None, ue8m0: bool):
    n = x.shape[1]
    gate, up = x[:, : n // 2].clone(), x[:, n // 2 :].clone()
    if swiglu_limit is not None:
        gate.clamp_(max=swiglu_limit)
        up.clamp_(min=-swiglu_limit, max=swiglu_limit)
    act = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    return sglang_per_token_group_quant_fp8(
        act,
        GROUP_SIZE,
        column_major_scales=ue8m0,
        scale_tma_aligned=ue8m0,
        scale_ue8m0=ue8m0,
    )


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("swiglu_limit", [None, 7.0])
@pytest.mark.parametrize("m", [128, 512])
def test_matches_unfused_reference(m: int, swiglu_limit: float | None, ue8m0: bool):
    torch.manual_seed(5)
    n = 2048
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 3

    out = torch.empty(m, n // 2, device="cuda", dtype=torch.float8_e4m3fn)
    out_scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=(m, n // 2),
        device=x.device,
        group_size=GROUP_SIZE,
        column_major_scales=ue8m0,
        scale_tma_aligned=ue8m0,
        scale_ue8m0=ue8m0,
    )
    silu_and_mul_contig_post_quant(
        input=x,
        output=out,
        output_scale=out_scale,
        quant_group_size=GROUP_SIZE,
        scale_ue8m0=ue8m0,
        transposed=ue8m0,
        swiglu_limit=swiglu_limit,
        swizzle=False,
    )

    ref_q, ref_s = _reference(x, swiglu_limit, ue8m0)
    # fp8 values may differ where the rounding of the group scale differs;
    # require near-total byte agreement rather than exact.
    byte_match = (
        (out.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    )
    assert byte_match > 0.95, f"fp8 byte match {byte_match:.4f}"
    if ue8m0:
        # Packed UE8M0 exponents are directly comparable.
        scale_match = (
            (
                out_scale.contiguous().view(torch.uint8)
                == ref_s.contiguous().view(torch.uint8)
            )
            .float()
            .mean()
            .item()
        )
        assert scale_match > 0.95, f"scale byte match {scale_match:.4f}"
    else:
        # fp32 scale storage conventions differ between the two kernels, and
        # clamp-boundary values may round differently; require the fused
        # kernel to be as accurate as the unfused path against the float
        # reference rather than bitwise-equal to it.
        def dequant(q, s):
            return q.float() * s.float().repeat_interleave(GROUP_SIZE, dim=1)

        gate, up = x[:, : n // 2].clone(), x[:, n // 2 :].clone()
        if swiglu_limit is not None:
            gate.clamp_(max=swiglu_limit)
            up.clamp_(min=-swiglu_limit, max=swiglu_limit)
        exact = torch.nn.functional.silu(gate.float()) * up.float()
        fused_err = (dequant(out, out_scale) - exact).abs().max().item()
        unfused_err = (dequant(ref_q, ref_s) - exact).abs().max().item()
        assert (
            fused_err < unfused_err * 1.5 + 1e-3
        ), f"fused dequant err {fused_err:.4f} vs unfused {unfused_err:.4f}"


def test_swizzled_layout_differs_from_standard():
    """The swizzled output layout is NOT what the SM120 DeepGEMM contiguous
    grouped GEMM consumes; feeding it there produced garbage (GSM8K 1.00 ->
    0.40). Pin the standard layout as the contract."""
    torch.manual_seed(5)
    m, n = 512, 2048
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 3

    outs = {}
    for swizzle in (False, True):
        out = torch.empty(m, n // 2, device="cuda", dtype=torch.float8_e4m3fn)
        out_scale = create_per_token_group_quant_fp8_output_scale(
            x_shape=(m, n // 2),
            device=x.device,
            group_size=GROUP_SIZE,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        silu_and_mul_contig_post_quant(
            input=x,
            output=out,
            output_scale=out_scale,
            quant_group_size=GROUP_SIZE,
            scale_ue8m0=True,
            transposed=True,
            swiglu_limit=7.0,
            swizzle=swizzle,
        )
        outs[swizzle] = out

    ref_q, _ = _reference(x, 7.0, True)
    std_match = (
        (outs[False].view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    )
    swz_match = (
        (outs[True].view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    )
    assert std_match > 0.95, f"standard layout must match reference: {std_match:.4f}"
    assert swz_match < 0.5, (
        "swizzled layout unexpectedly matches the standard reference; "
        f"({swz_match:.4f}) — if the kernel changed, revisit use_swizzle in "
        "DeepGemmRunnerCore"
    )
