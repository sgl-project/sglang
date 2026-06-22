import pytest
import torch
from sgl_kernel import sgl_per_token_group_quant_8bit  # AOT v2 reference op

from sglang.jit_kernel.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.srt.layers.quantization import fp8_kernel as fp8_kernel_module
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="base-b-kernel-unit-1-gpu-large")

G = 128


def _alloc(x_shape, scale_ue8m0):
    """Pre-allocated (zeroed) output_q + output_s for a given input/output shape.
    Zeroing makes the unwritten (padding / aligned) regions compare equal."""
    x_q = torch.zeros(x_shape, device="cuda", dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x_shape,
        device="cuda",
        group_size=G,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    x_s.zero_()
    return x_q, x_s


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 333])
@pytest.mark.parametrize("hidden", [128, 2048, 4096])
@pytest.mark.parametrize("fuse_silu_and_mul", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
def test_v2_jit_matches_aot(dtype, num_tokens, hidden, fuse_silu_and_mul, scale_ue8m0):
    """JIT v2 must be bit-exact with the AOT v2 across vanilla/silu and float/ue8m0
    scales (NaiveScheduler)."""
    torch.manual_seed(
        hidden + num_tokens + int(fuse_silu_and_mul) + 7 * int(scale_ue8m0)
    )
    in_hidden = hidden * (2 if fuse_silu_and_mul else 1)
    x = torch.randn(num_tokens, in_hidden, device="cuda", dtype=dtype)
    out_shape = (num_tokens, hidden)

    q_ref, s_ref = _alloc(out_shape, scale_ue8m0)
    sgl_per_token_group_quant_8bit(
        x,
        q_ref,
        s_ref,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0,
        fuse_silu_and_mul,
        None,
        enable_v2=True,
    )

    x_q, x_s = _alloc(out_shape, scale_ue8m0)
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
    )
    torch.cuda.synchronize()

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "fp8 codes differ"
    assert torch.equal(x_s, s_ref), "scales differ"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden", [128, 512, 4096, 7168])
def test_sglang_per_token_group_quant_fp8_row_major_ue8m0(dtype, num_tokens, hidden):
    """Row-major scale_ue8m0=True quantizes WITH the rounded (power-of-2) scale.
    Verify: (1) scales are exact powers of 2, (2) dequant ≈ original within FP8 tolerance.
    """
    torch.manual_seed(num_tokens * 1000 + hidden)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)

    x_q, x_s = sglang_per_token_group_quant_fp8(x, G, scale_ue8m0=True)
    torch.cuda.synchronize()

    # Scales must be exact powers of 2
    log2_s = torch.log2(x_s.abs())
    assert torch.equal(log2_s, log2_s.round()), "scales are not power-of-2"

    # Dequant should approximate original within FP8 precision
    x_deq = x_q.float().view(num_tokens, -1, G) * x_s.unsqueeze(-1)
    x_deq = x_deq.view(num_tokens, hidden)
    rel_err = (x.float() - x_deq).abs() / (x.float().abs() + 1e-6)
    assert (
        rel_err.mean() < 0.05
    ), f"mean relative dequant error too large: {rel_err.mean():.4f}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [2, 17])
@pytest.mark.parametrize("num_groups", [8])
@pytest.mark.parametrize("hidden", [256, 4096])
def test_sglang_per_token_group_quant_fp8_ue8m0_outer_major_scale_layout(
    dtype, num_tokens, num_groups, hidden, monkeypatch
):
    torch.manual_seed(num_tokens * 1000 + num_groups * 100 + hidden)
    x = torch.randn(num_tokens, num_groups, hidden, device="cuda", dtype=dtype)

    jit_v2_call_count = 0
    original_jit_v2 = fp8_kernel_module.sgl_per_token_group_quant_8bit_jit_v2

    def wrapped_jit_v2(*args, **kwargs):
        nonlocal jit_v2_call_count
        jit_v2_call_count += 1
        return original_jit_v2(*args, **kwargs)

    def fail_aot(*args, **kwargs):
        raise AssertionError("scale_outer_major must not call the AOT quant path")

    monkeypatch.setattr(
        fp8_kernel_module, "sgl_per_token_group_quant_8bit_jit_v2", wrapped_jit_v2
    )
    monkeypatch.setattr(fp8_kernel_module, "sgl_per_token_group_quant_8bit", fail_aot)

    x_q, x_s = fp8_kernel_module.sglang_per_token_group_quant_fp8(
        x.contiguous(),
        G,
        scale_ue8m0=True,
        scale_outer_major=True,
    )
    assert jit_v2_call_count == 1

    q_ref, s_ref = fp8_kernel_module.sglang_per_token_group_quant_fp8(
        x.reshape(num_tokens * num_groups, hidden).contiguous(),
        G,
        scale_ue8m0=True,
    )
    assert jit_v2_call_count == 2
    torch.cuda.synchronize()

    num_scale_groups = hidden // G
    q_ref = q_ref.view_as(x_q)
    s_ref = (
        s_ref.view(num_tokens, num_groups, num_scale_groups)
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "fp8 codes differ"
    assert torch.equal(x_s, s_ref), "scales differ"
    assert x_s.stride() == s_ref.stride()
    assert x_s[:, 0, :].is_contiguous()


# Masked (EP-MoE) path: the v2 op only has a masked scheduler for the
# column-major + ue8m0 + fused-silu+mul + masked combination. Input is 3D
# [num_experts, tokens_padded, hidden*2]; only tokens < masked_m[e] are processed
# (padding left untouched → zeros in both). Compare JIT vs AOT bit-exact.
@pytest.mark.parametrize("num_experts", [2, 5])
@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("tokens_pad", [128, 384])
def test_v2_jit_masked_matches_aot(num_experts, hidden, tokens_pad):
    torch.manual_seed(num_experts * 1000 + hidden + tokens_pad)
    x = torch.randn(
        num_experts, tokens_pad, hidden * 2, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.randint(
        0, tokens_pad + 1, (num_experts,), device="cuda", dtype=torch.int32
    )
    out_shape = (num_experts, tokens_pad, hidden)

    q_ref, s_ref = _alloc(out_shape, scale_ue8m0=True)
    sgl_per_token_group_quant_8bit(
        x,
        q_ref,
        s_ref,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        True,
        True,
        masked_m,
        enable_v2=True,
    )

    x_q, x_s = _alloc(out_shape, scale_ue8m0=True)
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_m=masked_m,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        x_q.view(torch.int8), q_ref.view(torch.int8)
    ), "masked fp8 differ"
    assert torch.equal(x_s, s_ref), "masked scales differ"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
