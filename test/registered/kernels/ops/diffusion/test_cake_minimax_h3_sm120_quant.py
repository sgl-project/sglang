"""Cake MiniMax-H3 SM120 (GB202) quantized kernels through sglang.kernels.

Covers the SM120-only FlashInfer entries: FP8 / NVFP4 pre-attention and their
weight quantizers, FP8 FC1 + SwiGLU and its weight preparations, FP8 / NVFP4
gated-residual out-projection and their weight quantizers, and the FP8 /
NVFP4 packed-varlen attention. For each: the registry resolves the explicit
FlashInfer backend; the facade result is bitwise identical to calling
FlashInfer directly; and the output matches a torch reference at the
precision's tolerance (BF16 1e-2 where the GEMM consumes the kernel's own
quantized stage-1 activation; FP8 0.1; FP4 block-scaled atol 1.0 / rtol 0.1).
GPU tests skip unless the device is compute capability 12.0 (12.x for the
varlen attention, whose JIT targets major version 12) and FlashInfer ships
the modules.
"""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_attention as attn
from sglang.kernels.cake_kernels import diffusion_minimax_h3_sm120 as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_minimax_h3_fc1_swiglu_fp8,
    cake_minimax_h3_fp8_out_proj,
    cake_minimax_h3_fp8_pre_attention,
    cake_minimax_h3_nvfp4_out_proj,
    cake_minimax_h3_nvfp4_pre_attention,
    cake_minimax_h3_sm120_varlen_attention_fp8,
    cake_minimax_h3_sm120_varlen_attention_nvfp4,
    cake_prepare_minimax_h3_fc1_weight_fp8,
    cake_prepare_minimax_h3_fc1_weight_nvfp4_sm120,
    cake_quantize_minimax_h3_o_weight_fp8,
    cake_quantize_minimax_h3_o_weight_nvfp4,
    cake_quantize_minimax_h3_qkv_weight_fp8,
    cake_quantize_minimax_h3_qkv_weight_nvfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HIDDEN, NUM_HEADS, HEAD_DIM, QKV_WIDTH, ROPE_DIM = 5376, 56, 128, 21504, 96
FFN, FC1_ROWS, ATTN_DIM, ADALN_ROWS, EPS = 14336, 28672, 7168, 9, 1.0e-5
ROWS = 129
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

SM120_OPS = [
    "diffusion.minimax_h3_fp8_pre_attention",
    "diffusion.minimax_h3_nvfp4_pre_attention",
    "diffusion.quantize_minimax_h3_qkv_weight_fp8",
    "diffusion.quantize_minimax_h3_qkv_weight_nvfp4",
    "diffusion.minimax_h3_fc1_swiglu_fp8",
    "diffusion.prepare_minimax_h3_fc1_weight_fp8",
    "diffusion.prepare_minimax_h3_fc1_weight_nvfp4_sm120",
    "diffusion.minimax_h3_fp8_out_proj",
    "diffusion.minimax_h3_nvfp4_out_proj",
    "diffusion.quantize_minimax_h3_o_weight_fp8",
    "diffusion.quantize_minimax_h3_o_weight_nvfp4",
]


@pytest.mark.parametrize("op", SM120_OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_sm120:"
    )


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.minimax_h3_sm120_varlen_attention_fp8",
        "diffusion.minimax_h3_sm120_varlen_attention_nvfp4",
    ],
)
def test_registry_resolves_varlen_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_attention:"
    )


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {modules}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake MiniMax-H3 SM120 kernels need {archs}, device is {cc}")


def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    return prev


# --- shared fixtures / references ----------------------------------------------


def make_norm_model(device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    return {
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "qkv_weight": normal((QKV_WIDTH, HIDDEN), 0.01),
        "fc1_weight": normal((FC1_ROWS, HIDDEN), 0.002),
        "o_weight": normal((HIDDEN, ATTN_DIM), 0.005),
        "gate": uniform((ADALN_ROWS, HIDDEN), -1.0, 1.0),
    }


def make_x(rows, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device).normal_(
        0.0, 0.5, generator=g
    )
    r = torch.arange(rows, device=device)
    idx = torch.div(r * ADALN_ROWS, rows, rounding_mode="floor").clamp_max(8)
    return x, idx.to(torch.int32)


def rope_cache(rows, device):
    pos = torch.arange(rows, dtype=torch.float32, device=device)
    axes = (
        torch.div(pos, 4096, rounding_mode="floor"),
        torch.div(pos, 64, rounding_mode="floor").remainder(64),
        pos.remainder(64),
    )
    inv_freq = 10000.0 ** (-torch.arange(16, dtype=torch.float32, device=device) / 16)
    ang = torch.cat([a[:, None] * inv_freq[None, :] for a in axes], dim=-1)
    return torch.cat((ang.cos(), ang.sin()), dim=-1).to(torch.bfloat16).contiguous()


def modulated(x, model, idx):
    norm = F.rms_norm(x, (HIDDEN,), model["x_norm_weight"], eps=EPS).to(torch.bfloat16)
    index = idx.long()
    return torch.addcmul(
        model["adaln_shift"].index_select(0, index),
        norm,
        (model["adaln_scale"].index_select(0, index) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)


def partial_rope(x, rope_cos_sin):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :48], rope_cos_sin[:, :48]), -1).float()[:, None]
    sin = torch.cat((rope_cos_sin[:, 48:], rope_cos_sin[:, 48:]), -1).float()[:, None]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    return torch.cat(((rotary * cos + rotated_half * sin).to(torch.bfloat16), tail), -1)


def qkv_from_operands(a_deq, w_deq, model, rope):
    prev = _no_tf32()
    try:
        y = (a_deq.float() @ w_deq.float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    grouped = y.view(-1, NUM_HEADS, 3, HEAD_DIM)
    q = F.rms_norm(grouped[:, :, 0], (HEAD_DIM,), model["q_norm_weight"], eps=EPS)
    k = F.rms_norm(grouped[:, :, 1], (HEAD_DIM,), model["k_norm_weight"], eps=EPS)
    return (
        partial_rope(q.to(torch.bfloat16), rope).contiguous(),
        partial_rope(k.to(torch.bfloat16), rope).contiguous(),
        grouped[:, :, 2].contiguous(),
    )


def nvfp4_dequant(packed_u8, sf_u8, global_scale):
    rows = packed_u8.shape[0]
    grid = torch.tensor(E2M1, dtype=torch.float32, device=packed_u8.device)
    codes = torch.empty(
        (rows, 2 * packed_u8.shape[1]), dtype=torch.uint8, device=packed_u8.device
    )
    codes[:, 0::2] = packed_u8 & 0xF
    codes[:, 1::2] = packed_u8 >> 4
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    sf = sf_u8.reshape(rows, -1).view(torch.float8_e4m3fn).float()
    return (vals.reshape(rows, -1, 16) * sf[..., None]).reshape(rows, -1) / global_scale


def unswizzle(sf_flat, rows, cols):
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import _unswizzle_sf_128x4

    return _unswizzle_sf_128x4(sf_flat.reshape(-1).view(torch.uint8), rows, cols)


def assert_bf16(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


# --- pre-attention ----------------------------------------------------------------


def test_fp8_pre_attention_matches_flashinfer_and_reference():
    _skip_unless(
        cake.ARCHS, cake.FI_PRE_ATTENTION_MODULE, cake.FI_PRE_ATTENTION_JIT_MODULE
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        minimax_h3_fp8_pre_attention as fi_direct,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        quantize_minimax_h3_qkv_weight_fp8 as fi_quantize,
    )

    device = torch.device("cuda")
    model = make_norm_model(device, 4532)
    x, idx = make_x(ROWS, device, 1)
    rope = rope_cache(ROWS, device)
    assert cake.supports_quantize_minimax_h3_qkv_weight(model["qkv_weight"])
    w_q, w_scale = cake_quantize_minimax_h3_qkv_weight_fp8(model["qkv_weight"])
    w_q_fi, w_scale_fi = fi_quantize(model["qkv_weight"])
    assert torch.equal(w_q.view(torch.uint8), w_q_fi.view(torch.uint8))
    assert torch.equal(w_scale, w_scale_fi)

    args = (
        x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx,
        w_q, w_scale, model["q_norm_weight"], model["k_norm_weight"], rope,
    )  # fmt: skip
    act_q = torch.empty((ROWS, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    act_scale = torch.empty((ROWS,), dtype=torch.float32, device=device)
    outs = {
        n: torch.empty((ROWS, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
        for n in ("q", "k", "v")
    }
    assert cake.supports_minimax_h3_fp8_pre_attention(
        *args, act_q=act_q, act_scale=act_scale, **outs
    )
    result = cake_minimax_h3_fp8_pre_attention(
        *args, act_q=act_q, act_scale=act_scale, **outs
    )
    assert result.q is outs["q"] and result.k is outs["k"] and result.v is outs["v"]
    direct = fi_direct(*args)
    torch.cuda.synchronize()
    for name in ("q", "k", "v"):
        assert torch.equal(getattr(result, name), getattr(direct, name))

    a_ref = modulated(x, model, idx)
    a_deq = act_q.float() * act_scale[:, None]
    torch.testing.assert_close(a_deq, a_ref.float(), atol=0.1, rtol=0.1)
    q_exp, k_exp, v_exp = qkv_from_operands(
        a_deq, w_q.float() * w_scale[:, None], model, rope
    )
    assert_bf16(result.q, q_exp)
    assert_bf16(result.k, k_exp)
    assert_bf16(result.v, v_exp)


def test_nvfp4_pre_attention_matches_flashinfer_and_reference():
    _skip_unless(
        cake.ARCHS, cake.FI_PRE_ATTENTION_MODULE, cake.FI_PRE_ATTENTION_JIT_MODULE
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        minimax_h3_nvfp4_pre_attention as fi_direct,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention import (
        quantize_minimax_h3_qkv_weight_nvfp4 as fi_quantize,
    )

    device = torch.device("cuda")
    model = make_norm_model(device, 4533)
    x, idx = make_x(ROWS, device, 2)
    rope = rope_cache(ROWS, device)
    w_q, w_sf, w_gs = cake_quantize_minimax_h3_qkv_weight_nvfp4(model["qkv_weight"])
    w_q_fi, w_sf_fi, w_gs_fi = fi_quantize(model["qkv_weight"])
    assert torch.equal(w_q, w_q_fi) and torch.equal(w_sf, w_sf_fi)
    assert torch.equal(w_gs, w_gs_fi)

    a_ref = modulated(x, model, idx)
    act_gs = ((448.0 * 6.0) / a_ref.float().abs().amax()).reshape(1).float().to(device)
    alpha = 1.0 / float(act_gs.item() * w_gs.float().item())
    args = (
        x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx,
        w_q, w_sf, w_gs, act_gs, model["q_norm_weight"], model["k_norm_weight"], rope,
    )  # fmt: skip
    act_q = torch.empty((ROWS, HIDDEN // 2), dtype=torch.uint8, device=device)
    act_sf = torch.empty((ROWS, HIDDEN // 16), dtype=torch.uint8, device=device)
    assert cake.supports_minimax_h3_nvfp4_pre_attention(
        *args, alpha=alpha, act_q=act_q, act_sf=act_sf
    )
    result = cake_minimax_h3_nvfp4_pre_attention(
        *args, alpha=alpha, act_q=act_q, act_sf=act_sf
    )
    direct = fi_direct(*args, alpha=alpha)
    torch.cuda.synchronize()
    for name in ("q", "k", "v"):
        assert torch.equal(getattr(result, name), getattr(direct, name))

    a_deq = nvfp4_dequant(act_q, act_sf, act_gs)
    torch.testing.assert_close(a_deq, a_ref.float(), atol=1.0, rtol=0.1)
    w_deq = nvfp4_dequant(
        w_q, unswizzle(w_sf, QKV_WIDTH, HIDDEN // 16), w_gs.to(device)
    )
    q_exp, k_exp, v_exp = qkv_from_operands(a_deq, w_deq, model, rope)
    assert_bf16(result.q, q_exp)
    assert_bf16(result.k, k_exp)
    assert_bf16(result.v, v_exp)


# --- FC1 + SwiGLU -----------------------------------------------------------------


def test_fc1_swiglu_fp8_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_FC1_MODULE, cake.FI_FC1_JIT_MODULE)
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        minimax_h3_fc1_swiglu_fp8 as fi_direct,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_fp8 as fi_prepare,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
        prepare_minimax_h3_fc1_weight_nvfp4_sm120 as fi_prepare_nvfp4,
    )

    device = torch.device("cuda")
    model = make_norm_model(device, 4611)
    x, idx = make_x(ROWS, device, 3)
    assert cake.supports_prepare_minimax_h3_fc1_weight_sm120(model["fc1_weight"])
    w_q, w_scale = cake_prepare_minimax_h3_fc1_weight_fp8(model["fc1_weight"])
    w_q_fi, w_scale_fi = fi_prepare(model["fc1_weight"])
    assert torch.equal(w_q.view(torch.uint8), w_q_fi.view(torch.uint8))
    assert torch.equal(w_scale, w_scale_fi)
    w_gs = ((448.0 * 6.0) / model["fc1_weight"].float().abs().amax()).reshape(1)
    n_q, n_sf = cake_prepare_minimax_h3_fc1_weight_nvfp4_sm120(
        model["fc1_weight"], w_gs
    )
    n_q_fi, n_sf_fi = fi_prepare_nvfp4(model["fc1_weight"], w_gs)
    assert torch.equal(n_q, n_q_fi) and torch.equal(n_sf, n_sf_fi)
    assert n_sf.numel() == FC1_ROWS * (HIDDEN // 16)

    args = (x, model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    out = torch.empty((ROWS, FFN), dtype=torch.bfloat16, device=device)
    ws_q = torch.empty((ROWS, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    ws_scale = torch.empty((ROWS,), dtype=torch.float32, device=device)
    assert cake.supports_minimax_h3_fc1_swiglu_fp8(
        *args, w_q, w_scale, out=out, workspace_q=ws_q, workspace_scale=ws_scale
    )
    result = cake_minimax_h3_fc1_swiglu_fp8(
        *args, w_q, w_scale, out=out, workspace_q=ws_q, workspace_scale=ws_scale
    )
    assert result is out
    direct = fi_direct(*args, w_q, w_scale)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)

    a = modulated(x, model, idx)
    prev = _no_tf32()
    try:
        h = (a.float() @ model["fc1_weight"].float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    gate, up = h.chunk(2, dim=-1)
    expected = (F.silu(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.1)


# --- out-projection ---------------------------------------------------------------


def _out_proj_inputs(device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    attn_out = torch.empty(
        (ROWS, ATTN_DIM), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.5, generator=g)
    residual = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device).normal_(
        0.0, 0.5, generator=g
    )
    r = torch.arange(ROWS, device=device)
    idx = torch.div(r * ADALN_ROWS, ROWS, rounding_mode="floor").clamp_max(8)
    idx[50] = -1
    idx[60] = ADALN_ROWS
    return attn_out, idx.to(torch.int32), residual


def _out_proj_reference(attn_out, model, idx, residual):
    prev = _no_tf32()
    try:
        o = (attn_out.float() @ model["o_weight"].float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    index = idx.long()
    valid = (index >= 0) & (index < ADALN_ROWS)
    g = model["gate"].index_select(0, index.clamp(0, ADALN_ROWS - 1))
    g = torch.where(valid[:, None], g, torch.zeros_like(g))
    return (residual.float() + (g.float() * o.float()).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )


def test_fp8_out_proj_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_OUT_PROJ_MODULE, cake.FI_OUT_PROJ_JIT_MODULE)
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        minimax_h3_fp8_out_proj as fi_direct,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        quantize_minimax_h3_o_weight_fp8 as fi_quantize,
    )

    device = torch.device("cuda")
    model = make_norm_model(device, 4616)
    attn_out, idx, residual = _out_proj_inputs(device, 4)
    assert cake.supports_quantize_minimax_h3_o_weight(model["o_weight"])
    w_q, w_scale = cake_quantize_minimax_h3_o_weight_fp8(model["o_weight"])
    w_q_fi, w_scale_fi = fi_quantize(model["o_weight"])
    assert torch.equal(w_q.view(torch.uint8), w_q_fi.view(torch.uint8))
    assert torch.equal(w_scale, w_scale_fi)

    out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    act_q = torch.empty((ROWS, ATTN_DIM), dtype=torch.float8_e4m3fn, device=device)
    act_scale = torch.empty((ROWS,), dtype=torch.float32, device=device)
    assert cake.supports_minimax_h3_fp8_out_proj(
        attn_out, w_q, w_scale, model["gate"], idx, residual,
        out=out, act_q=act_q, act_scale=act_scale,
    )  # fmt: skip
    result = cake_minimax_h3_fp8_out_proj(
        attn_out, w_q, w_scale, model["gate"], idx, residual,
        out=out, act_q=act_q, act_scale=act_scale,
    )  # fmt: skip
    assert result is out
    direct = fi_direct(attn_out, w_q, w_scale, model["gate"], idx, residual)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = _out_proj_reference(attn_out, model, idx, residual)
    torch.testing.assert_close(out.float(), expected.float(), atol=0.1, rtol=0.1)
    assert torch.equal(out[50], residual[50]) and torch.equal(out[60], residual[60])


def test_nvfp4_out_proj_matches_flashinfer_and_reference():
    _skip_unless(cake.ARCHS, cake.FI_OUT_PROJ_MODULE, cake.FI_OUT_PROJ_JIT_MODULE)
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        minimax_h3_nvfp4_out_proj as fi_direct,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj import (
        quantize_minimax_h3_o_weight_nvfp4 as fi_quantize,
    )

    device = torch.device("cuda")
    model = make_norm_model(device, 4617)
    attn_out, idx, residual = _out_proj_inputs(device, 5)
    w_q, w_sf, w_gs = cake_quantize_minimax_h3_o_weight_nvfp4(model["o_weight"])
    w_q_fi, w_sf_fi, w_gs_fi = fi_quantize(model["o_weight"])
    assert torch.equal(w_q, w_q_fi) and torch.equal(w_sf, w_sf_fi)
    assert torch.equal(w_gs, w_gs_fi)
    act_gs = (
        ((448.0 * 6.0) / attn_out.float().abs().amax()).reshape(1).float().to(device)
    )

    out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
    act_q = torch.empty((ROWS, ATTN_DIM // 2), dtype=torch.uint8, device=device)
    act_sf = torch.empty((ROWS, ATTN_DIM // 16), dtype=torch.uint8, device=device)
    assert cake.supports_minimax_h3_nvfp4_out_proj(
        attn_out, w_q, w_sf, w_gs, act_gs, model["gate"], idx, residual,
        out=out, act_q=act_q, act_sf=act_sf,
    )  # fmt: skip
    result = cake_minimax_h3_nvfp4_out_proj(
        attn_out, w_q, w_sf, w_gs, act_gs, model["gate"], idx, residual,
        out=out, act_q=act_q, act_sf=act_sf,
    )  # fmt: skip
    assert result is out
    direct = fi_direct(attn_out, w_q, w_sf, w_gs, act_gs, model["gate"], idx, residual)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    expected = _out_proj_reference(attn_out, model, idx, residual)
    torch.testing.assert_close(out.float(), expected.float(), atol=1.0, rtol=0.1)
    assert torch.equal(out[50], residual[50])


# --- packed-varlen attention ------------------------------------------------------

CU = [0, 133, 300, 364]
HEADS = 7


def _attention_inputs(device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (
        torch.randn(CU[-1], HEADS, HEAD_DIM, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(3)
    )
    return q, k, v, torch.tensor(CU, dtype=torch.int32, device=device)


def _attention_reference(q, k, v):
    prev = _no_tf32()
    try:
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        for a, b in zip(CU, CU[1:]):
            if b <= a:
                continue
            qs, ks, vs = (t[a:b].float().transpose(0, 1) for t in (q, k, v))
            probs = torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) * scale, dim=-1)
            out[a:b] = torch.matmul(probs, vs).transpose(0, 1)
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_varlen_attention_fp8_matches_flashinfer_and_reference():
    _skip_unless(
        attn.SM120_ARCHS, attn.FI_SM120_FP8_MODULE, attn.FI_SM120_FP8_JIT_MODULE
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
        minimax_h3_sm120_varlen_attention_fp8 as fi_direct,
    )

    device = torch.device("cuda")
    q, k, v, cu = _attention_inputs(device, 7)
    out = torch.empty_like(q)
    assert attn.supports_minimax_h3_sm120_varlen_attention_fp8(q, k, v, cu, out)
    result = cake_minimax_h3_sm120_varlen_attention_fp8(
        q, k, v, cu, out, cu_seqlens_host=CU
    )
    assert result is out
    direct = fi_direct(q, k, v, cu)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    torch.testing.assert_close(
        out.float(), _attention_reference(q, k, v), atol=0.1, rtol=0.1
    )


def test_varlen_attention_nvfp4_matches_flashinfer_and_reference():
    _skip_unless(
        attn.SM120_ARCHS,
        attn.FI_SM120_NVFP4_MODULE,
        attn.FI_SM120_NVFP4_JIT_MODULE,
        attn.FI_SM120_FP8_MODULE,
    )
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
        minimax_h3_sm120_varlen_attention_nvfp4 as fi_direct,
    )

    device = torch.device("cuda")
    q, k, v, cu = _attention_inputs(device, 8)
    assert attn.supports_minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu)
    out = cake_minimax_h3_sm120_varlen_attention_nvfp4(q, k, v, cu, cu_seqlens_host=CU)
    direct = fi_direct(q, k, v, cu)
    torch.cuda.synchronize()
    assert torch.equal(out, direct)
    torch.testing.assert_close(
        out.float(), _attention_reference(q, k, v), atol=1.0, rtol=0.1
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
