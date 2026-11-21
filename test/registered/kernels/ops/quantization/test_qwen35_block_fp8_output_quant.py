"""CUDA correctness and CUDA Graph tests for fused Qwen3.5 blockwise FP8 output quantization."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("z_is_3d", [False, True])
@pytest.mark.parametrize(
    "scale_ue8m0,scale_transposed",
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 4, 8, 32, 64])
def test_layernorm_gated_block_fp8_quant(
    z_is_3d: bool,
    scale_ue8m0: bool,
    scale_transposed: bool,
    num_tokens: int,
):
    nv, hd = 32, 128
    torch.manual_seed(num_tokens * 17 + int(z_is_3d) * 3 + int(scale_ue8m0))
    w = torch.randn(hd, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(num_tokens * nv, hd, device="cuda", dtype=torch.bfloat16)
    if z_is_3d:
        backing = torch.randn(
            num_tokens, nv * 2, hd, device="cuda", dtype=torch.bfloat16
        )
        z = backing[:, :nv, :]
    else:
        z = torch.randn(num_tokens * nv, hd, device="cuda", dtype=torch.bfloat16)

    (q, s), mean, rstd = _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z,
        is_rms_norm=True,
        activation="swish",
        quant_heads=nv,
        block_fp8_quant=True,
        scale_ue8m0=scale_ue8m0,
        scale_transposed=scale_transposed,
    )
    assert mean is None and rstd is None
    assert q.shape == (num_tokens, nv * hd)
    assert q.dtype == torch.float8_e4m3fn

    if num_tokens == 0:
        return

    ref_y, _, _ = _layer_norm_fwd(
        x, w, None, 1e-6, z=z, is_rms_norm=True, activation="swish"
    )
    ref_q, ref_s = sglang_per_token_group_quant_fp8(
        ref_y.view(num_tokens, nv * hd),
        128,
        column_major_scales=scale_transposed,
        scale_tma_aligned=scale_transposed,
        scale_ue8m0=scale_ue8m0,
    )
    assert s.shape == ref_s.shape
    assert s.stride() == ref_s.stride()
    assert s.dtype == ref_s.dtype
    assert torch.equal(s, ref_s)
    byte_match = (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    assert byte_match >= 0.9999, f"Expected >=99.99% byte match, got {byte_match:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("gate_is_3d", [False, True])
@pytest.mark.parametrize(
    "scale_ue8m0,scale_transposed",
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 4, 8, 32, 64])
def test_fused_sigmoid_mul_block_fp8_quant(
    gate_is_3d: bool,
    scale_ue8m0: bool,
    scale_transposed: bool,
    num_tokens: int,
):
    nh, hd = 16, 256
    hidden_dim = nh * hd
    torch.manual_seed(num_tokens * 31 + int(gate_is_3d) * 5 + int(scale_ue8m0))
    attn_out = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    if gate_is_3d:
        backing = torch.randn(
            num_tokens, nh, hd * 2, device="cuda", dtype=torch.bfloat16
        )
        gate = backing[:, :, hd:]
    else:
        gate = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)

    q, s = fused_sigmoid_mul(
        attn_out,
        gate,
        block_fp8_quant=True,
        scale_ue8m0=scale_ue8m0,
        scale_transposed=scale_transposed,
    )
    assert q.shape == (num_tokens, hidden_dim)
    assert q.dtype == torch.float8_e4m3fn

    if num_tokens == 0:
        return

    ref_y = fused_sigmoid_mul(attn_out, gate, inplace=False)
    ref_q, ref_s = sglang_per_token_group_quant_fp8(
        ref_y,
        128,
        column_major_scales=scale_transposed,
        scale_tma_aligned=scale_transposed,
        scale_ue8m0=scale_ue8m0,
    )
    assert s.shape == ref_s.shape
    assert s.stride() == ref_s.stride()
    assert s.dtype == ref_s.dtype
    assert torch.equal(s, ref_s)
    byte_match = (q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    assert byte_match >= 0.9999, f"Expected >=99.99% byte match, got {byte_match:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_block_fp8_output_quant_cuda_graph():
    torch.manual_seed(123)
    T, nv, hd = 8, 32, 128
    nh, attn_hd = 16, 256
    w = torch.randn(hd, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(T * nv, hd, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(T * nv, hd, device="cuda", dtype=torch.bfloat16)
    attn_out = torch.randn(T, nh * attn_hd, device="cuda", dtype=torch.bfloat16)
    gate_backing = torch.randn(T, nh, attn_hd * 2, device="cuda", dtype=torch.bfloat16)
    gate = gate_backing[:, :, attn_hd:]

    # Warmup
    _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z,
        is_rms_norm=True,
        activation="swish",
        quant_heads=nv,
        block_fp8_quant=True,
        scale_ue8m0=False,
        scale_transposed=True,
    )
    fused_sigmoid_mul(
        attn_out,
        gate,
        block_fp8_quant=True,
        scale_ue8m0=False,
        scale_transposed=True,
    )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        (gdn_q, gdn_s), _, _ = _layer_norm_fwd(
            x,
            w,
            None,
            1e-6,
            z=z,
            is_rms_norm=True,
            activation="swish",
            quant_heads=nv,
            block_fp8_quant=True,
            scale_ue8m0=False,
            scale_transposed=True,
        )
        attn_q, attn_s = fused_sigmoid_mul(
            attn_out,
            gate,
            block_fp8_quant=True,
            scale_ue8m0=False,
            scale_transposed=True,
        )

    # Overwrite inputs in-place and replay graph.
    x.normal_()
    z.normal_()
    attn_out.normal_()
    gate_backing.normal_()
    g.replay()
    torch.cuda.synchronize()

    (exp_gdn_q, exp_gdn_s), _, _ = _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z,
        is_rms_norm=True,
        activation="swish",
        quant_heads=nv,
        block_fp8_quant=True,
        scale_ue8m0=False,
        scale_transposed=True,
    )
    exp_attn_q, exp_attn_s = fused_sigmoid_mul(
        attn_out,
        gate,
        block_fp8_quant=True,
        scale_ue8m0=False,
        scale_transposed=True,
    )
    assert torch.equal(gdn_q.view(torch.uint8), exp_gdn_q.view(torch.uint8))
    assert torch.equal(gdn_s, exp_gdn_s)
    assert torch.equal(attn_q.view(torch.uint8), exp_attn_q.view(torch.uint8))
    assert torch.equal(attn_s, exp_attn_s)
