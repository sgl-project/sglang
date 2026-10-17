"""Cake MiniMax-H3 MXFP8 / NVFP4 prepared pre-attention (SM100a / SM103a).

Checks, for ``prepare_minimax_h3_mxfp8_pre_attention`` and
``prepare_minimax_h3_nvfp4_pre_attention``: the registry resolves the explicit
FlashInfer backend; the facade's prepared runner is bitwise identical to the
one obtained directly from FlashInfer; and every stage agrees with a torch
reference at the tolerance of its precision:

* the in-kernel activation quantization dequantizes back to the BF16
  RMSNorm + indexed-AdaLN reference within FP8 (MXFP8) / FP4 block-scaled
  (NVFP4) tolerance;
* the QKV projection equals the FP32 GEMM of the dequantized operands within
  BF16 tolerance (MXFP8: the caller-owned ``qkv_bf16``; NVFP4: the post-RoPE
  ``debug_q_bf16`` / ``debug_k_bf16`` after the per-head RMSNorm + RoPE);
* the destination-major packed ``(out_q, out_sf)`` dequantizes to the packed
  BF16 Q/K/V within FP8 / FP4 tolerance.

The MXFP8 chain is routed by an exact ``(M, P)`` table
(``flashinfer.jit.cake_minimax_h3_mxfp8.MINIMAX_H3_MXFP8_SHAPES``, FlashInfer
``e4f94f9484``); this test uses the admitted pair ``M=128, P=8``. Both chains
take their TMA tensor maps by value: FlashInfer rejects any non-``None``
``*_descriptor_workspace`` and the adapters do not expose those keywords.
Skips when FlashInfer lacks the modules or the GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import diffusion_minimax_h3_pre_attention as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.diffusion.cake import (
    cake_prepare_minimax_h3_mxfp8_pre_attention,
    cake_prepare_minimax_h3_nvfp4_pre_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN, NUM_HEADS, HEAD_DIM, KINDS = 5376, 56, 128, 3
QKV_WIDTH, ROPE_DIM, ADALN_ROWS, EPS = 21504, 96, 9, 1.0e-5
M, P = 128, 8  # admitted MXFP8 route (128, 8)
ROWS_PER_DEST = M * (NUM_HEADS // P) * KINDS  # 2688, a multiple of 128
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.mark.parametrize(
    "op",
    [
        "diffusion.prepare_minimax_h3_mxfp8_pre_attention",
        "diffusion.prepare_minimax_h3_nvfp4_pre_attention",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.diffusion_minimax_h3_pre_attention:"
    )


def test_mxfp8_route_table_has_the_test_pair():
    assert (M, P) in cake.MXFP8_ROUTES and len(cake.MXFP8_ROUTES) == 44


def _skip_unless_supported(*modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {modules}")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake MiniMax-H3 pre-attention needs sm_100a/103a, device is {cc}")


# --- shared torch reference pieces ---------------------------------------------


def make_rope_cache(m, device):
    rows = torch.arange(m, dtype=torch.float32, device=device)
    axes = (
        torch.div(rows, 4096, rounding_mode="floor"),
        torch.div(rows, 64, rounding_mode="floor").remainder(64),
        rows.remainder(64),
    )
    inv_freq = 10000.0 ** (-torch.arange(16, dtype=torch.float32, device=device) / 16)
    phase = torch.cat([a[:, None] * inv_freq[None, :] for a in axes], dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16).contiguous()


def apply_rope(x, rope_cos_sin):
    rotary = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos = torch.cat((rope_cos_sin[:, :48], rope_cos_sin[:, :48]), -1).float()[:, None]
    sin = torch.cat((rope_cos_sin[:, 48:], rope_cos_sin[:, 48:]), -1).float()[:, None]
    rotated_half = torch.cat((-rotary[..., 48:], rotary[..., :48]), dim=-1)
    return torch.cat(((rotary * cos + rotated_half * sin).to(torch.bfloat16), tail), -1)


def make_case(device, seed):
    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    rows = torch.arange(M, device=device)
    return {
        "x": normal((M, HIDDEN), 0.5),
        "x_norm_weight": uniform((HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((ADALN_ROWS, HIDDEN), -0.05, 0.05),
        "adaln_index": torch.div(rows * ADALN_ROWS, M, rounding_mode="floor")
        .clamp_max(8)
        .to(torch.int32),
        "qkv_weight": normal((QKV_WIDTH, HIDDEN), 0.01),
        "q_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "k_norm_weight": uniform((HEAD_DIM,), 0.9, 1.1),
        "rope_cos_sin": make_rope_cache(M, device),
    }


def modulated(case):
    norm = F.rms_norm(case["x"], (HIDDEN,), case["x_norm_weight"], eps=EPS)
    index = case["adaln_index"].long()
    return torch.addcmul(
        case["adaln_shift"].index_select(0, index),
        norm.to(torch.bfloat16),
        (case["adaln_scale"].index_select(0, index) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)


def fp32_gemm(a, w):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return (a.float() @ w.float().t()).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def qkv_epilogue(qkv_bf16, case):
    """Per-head Q/K RMSNorm + partial RoPE on a BF16 ``[M, 21504]`` projection."""
    grouped = qkv_bf16.view(M, NUM_HEADS, KINDS, HEAD_DIM)
    q = F.rms_norm(grouped[:, :, 0], (HEAD_DIM,), case["q_norm_weight"], eps=EPS)
    k = F.rms_norm(grouped[:, :, 1], (HEAD_DIM,), case["k_norm_weight"], eps=EPS)
    q = apply_rope(q.to(torch.bfloat16), case["rope_cos_sin"])
    k = apply_rope(k.to(torch.bfloat16), case["rope_cos_sin"])
    return q.contiguous(), k.contiguous(), grouped[:, :, 2].contiguous()


def pack(q, k, v):
    fused = torch.stack((q, k, v), dim=2)
    return (
        fused.view(M, P, NUM_HEADS // P, KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def unswizzle(sf_flat, rows, cols):
    from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import _unswizzle_sf_128x4

    return _unswizzle_sf_128x4(sf_flat.reshape(-1).view(torch.uint8), rows, cols)


def mxfp8_dequant(q_e4m3, sf_u8):
    """E4M3 ``[R, K]`` + UE8M0 ``[R, K/32]`` -> FP32."""
    rows, cols = q_e4m3.shape
    scale = torch.exp2(sf_u8.float() - 127.0)
    return (q_e4m3.float().reshape(rows, cols // 32, 32) * scale[:, :, None]).reshape(
        rows, cols
    )


def nvfp4_dequant(packed_u8, sf_u8, global_scale):
    """Packed E2M1 ``[R, K/2]`` + UE4M3 ``[R, K/16]`` -> FP32 ``codes * sf / global_scale``."""
    rows = packed_u8.shape[0]
    grid = torch.tensor(E2M1, dtype=torch.float32, device=packed_u8.device)
    codes = torch.empty(
        (rows, 2 * packed_u8.shape[1]), dtype=torch.uint8, device=packed_u8.device
    )
    codes[:, 0::2] = packed_u8 & 0xF
    codes[:, 1::2] = packed_u8 >> 4
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    sf = sf_u8.view(torch.float8_e4m3fn).float()
    return (vals.reshape(rows, -1, 16) * sf[..., None]).reshape(rows, -1) / global_scale


def global_scale_of(t):
    return ((448.0 * 6.0) / t.float().abs().amax()).reshape(1).float()


# --- MXFP8 ------------------------------------------------------------------------


def test_mxfp8_prepared_matches_flashinfer_and_reference():
    _skip_unless_supported(cake.FI_MXFP8_MODULE, cake.FI_MXFP8_JIT_MODULE)
    from flashinfer import mxfp8_quantize
    from flashinfer.diffusion_ops.cake_minimax_h3_mxfp8 import (
        prepare_minimax_h3_mxfp8_pre_attention as fi_prepare,
    )
    from flashinfer.gemm import gemm_base
    from flashinfer.jit.cake_minimax_h3_mxfp8 import MINIMAX_H3_MXFP8_SHAPES

    assert cake.MXFP8_ROUTES == frozenset(MINIMAX_H3_MXFP8_SHAPES)

    device = torch.device("cuda")
    case = make_case(device, seed=128 + P)
    w_q, w_sf = mxfp8_quantize(case["qkv_weight"], is_sf_swizzled_layout=True)
    w_q = w_q.view(torch.float8_e4m3fn).reshape(QKV_WIDTH, HIDDEN).contiguous()
    w_sf = w_sf.reshape(-1).view(torch.uint8).contiguous()

    def buffers():
        return dict(
            out_q=torch.empty(
                (P, M, NUM_HEADS // P, KINDS, HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            out_sf=torch.empty(
                (P, cake.mxfp8_out_sf_numel(M, P)), dtype=torch.uint8, device=device
            ),
            activation_q=torch.empty(
                (M, HIDDEN), dtype=torch.float8_e4m3fn, device=device
            ),
            activation_sf=torch.empty(
                (cake.mxfp8_activation_sf_numel(M),), dtype=torch.uint8, device=device
            ),
            qkv_bf16=torch.empty((M, QKV_WIDTH), dtype=torch.bfloat16, device=device),
            gemm_workspace=torch.empty(
                (int(gemm_base.DEFAULT_WORKSPACE_SIZE),),
                dtype=torch.uint8,
                device=device,
            ),
        )

    inputs = dict(
        x=case["x"],
        x_norm_weight=case["x_norm_weight"],
        adaln_scale=case["adaln_scale"],
        adaln_shift=case["adaln_shift"],
        adaln_index=case["adaln_index"],
        qkv_weight_q=w_q,
        qkv_weight_sf=w_sf,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
        rope_cos_sin=case["rope_cos_sin"],
        P=P,
    )
    ours = buffers()
    assert cake.supports_prepare_minimax_h3_mxfp8_pre_attention(**inputs, **ours)
    runner = cake_prepare_minimax_h3_mxfp8_pre_attention(**inputs, **ours)
    out_q, out_sf = runner()
    torch.cuda.synchronize()
    assert out_q is ours["out_q"] and out_sf is ours["out_sf"]

    theirs = buffers()
    fi_q, fi_sf = fi_prepare(**inputs, **theirs)()
    torch.cuda.synchronize()
    assert torch.equal(out_q.view(torch.uint8), fi_q.view(torch.uint8))
    assert torch.equal(out_sf, fi_sf)
    assert torch.equal(ours["qkv_bf16"], theirs["qkv_bf16"])

    # Stage 1: the quantized activation dequantizes to the modulated reference.
    a_ref = modulated(case)
    a_sf = unswizzle(ours["activation_sf"], M, HIDDEN // 32)
    a_deq = mxfp8_dequant(ours["activation_q"], a_sf)
    torch.testing.assert_close(a_deq, a_ref.float(), atol=0.1, rtol=0.1)

    # Stage 2: the BF16 projection is the FP32 GEMM of the dequantized operands.
    w_deq = mxfp8_dequant(w_q, unswizzle(w_sf, QKV_WIDTH, HIDDEN // 32))
    qkv_ref = fp32_gemm(a_deq, w_deq)
    torch.testing.assert_close(
        ours["qkv_bf16"].float(), qkv_ref.float(), atol=1e-2, rtol=1e-2
    )

    # Stage 3: the packed MXFP8 output dequantizes to the packed BF16 Q/K/V.
    expected = pack(*qkv_epilogue(ours["qkv_bf16"], case))
    for dest in range(P):
        sf = unswizzle(out_sf[dest], ROWS_PER_DEST, HEAD_DIM // 32)
        got = mxfp8_dequant(out_q[dest].reshape(ROWS_PER_DEST, HEAD_DIM), sf)
        torch.testing.assert_close(
            got,
            expected[dest].reshape(ROWS_PER_DEST, HEAD_DIM).float(),
            atol=0.1,
            rtol=0.1,
        )

    # Out-of-route M is refused by admission (no FlashInfer call needed).
    short = {
        k: (v[:64] if k in ("x", "adaln_index", "rope_cos_sin") else v)
        for k, v in inputs.items()
    }
    assert (64, P) not in cake.MXFP8_ROUTES
    assert not cake.supports_prepare_minimax_h3_mxfp8_pre_attention(**short, **ours)

    # The stages take their tensor maps by value: FlashInfer rejects a
    # descriptor workspace, and the adapter does not expose the keyword.
    with pytest.raises(ValueError, match="descriptor_workspace"):
        fi_prepare(
            **inputs,
            **theirs,
            norm_descriptor_workspace=torch.empty(
                640, dtype=torch.uint8, device=device
            ),
        )


# --- NVFP4 ------------------------------------------------------------------------


def test_nvfp4_prepared_matches_flashinfer_and_reference():
    _skip_unless_supported(cake.FI_NVFP4_MODULE, cake.FI_NVFP4_JIT_MODULE)
    from flashinfer.diffusion_ops.cake_minimax_h3_nvfp4 import (
        prepare_minimax_h3_nvfp4_pre_attention as fi_prepare,
    )
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    device = torch.device("cuda")
    case = make_case(device, seed=4128 + P)
    a_ref = modulated(case)
    qkv_bf16_ref = fp32_gemm(a_ref, case["qkv_weight"])
    q_ref, k_ref, v_ref = qkv_epilogue(qkv_bf16_ref, case)

    w_gs = global_scale_of(case["qkv_weight"]).to(device)
    x_gs = global_scale_of(a_ref).to(device)
    out_gs = global_scale_of(torch.stack((q_ref, k_ref, v_ref))).to(device)
    w_q, w_sf = nvfp4_quantize(
        case["qkv_weight"], w_gs, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_q = w_q.view(torch.uint8).reshape(QKV_WIDTH, HIDDEN // 2).contiguous()
    w_sf = w_sf.view(torch.uint8).reshape(-1).contiguous()
    assert w_sf.numel() == QKV_WIDTH * (HIDDEN // 16)

    def buffers():
        return dict(
            out_q=torch.empty(
                (P, M, NUM_HEADS // P, KINDS, HEAD_DIM // 2),
                dtype=torch.uint8,
                device=device,
            ),
            out_sf=torch.empty(
                (P, cake.nvfp4_out_sf_numel(M, P)), dtype=torch.uint8, device=device
            ),
            activation_q=torch.empty(
                (M, HIDDEN // 2), dtype=torch.uint8, device=device
            ),
            activation_sf=torch.empty(
                (cake.nvfp4_activation_sf_numel(M),), dtype=torch.uint8, device=device
            ),
            debug_q_bf16=torch.empty(
                (M, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
            ),
            debug_k_bf16=torch.empty(
                (M, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
            ),
            debug_adaln_bf16=torch.empty(
                (M, HIDDEN), dtype=torch.bfloat16, device=device
            ),
        )

    inputs = dict(
        x=case["x"],
        x_norm_weight=case["x_norm_weight"],
        adaln_scale=case["adaln_scale"],
        adaln_shift=case["adaln_shift"],
        adaln_index=case["adaln_index"],
        x_global_scale=x_gs,
        qkv_weight_q=w_q,
        qkv_weight_sf=w_sf,
        w_global_scale=w_gs,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
        rope_cos_sin=case["rope_cos_sin"],
        out_global_scale=out_gs,
        P=P,
    )
    ours = buffers()
    admission = {k: v for k, v in ours.items() if not k.startswith("debug_")}
    assert cake.supports_prepare_minimax_h3_nvfp4_pre_attention(**inputs, **admission)
    runner = cake_prepare_minimax_h3_nvfp4_pre_attention(**inputs, **ours)
    out_q, out_sf = runner()
    torch.cuda.synchronize()
    assert out_q is ours["out_q"] and out_sf is ours["out_sf"]

    theirs = buffers()
    fi_q, fi_sf = fi_prepare(**inputs, **theirs)()
    torch.cuda.synchronize()
    assert torch.equal(out_q, fi_q) and torch.equal(out_sf, fi_sf)
    assert torch.equal(ours["debug_q_bf16"], theirs["debug_q_bf16"])
    assert torch.equal(ours["debug_k_bf16"], theirs["debug_k_bf16"])

    # Stage 1: exact AdaLN intermediate and its NVFP4 quantization.
    torch.testing.assert_close(
        ours["debug_adaln_bf16"].float(), a_ref.float(), atol=1e-2, rtol=1e-2
    )
    a_sf = unswizzle(ours["activation_sf"], M, HIDDEN // 16)
    a_deq = nvfp4_dequant(ours["activation_q"], a_sf, x_gs)
    torch.testing.assert_close(a_deq, a_ref.float(), atol=1.0, rtol=0.1)

    # Stage 2: fused GEMM + QK-norm + RoPE on the dequantized NVFP4 operands.
    w_deq = nvfp4_dequant(w_q, unswizzle(w_sf, QKV_WIDTH, HIDDEN // 16), w_gs)
    q_exp, k_exp, v_exp = qkv_epilogue(fp32_gemm(a_deq, w_deq), case)
    torch.testing.assert_close(
        ours["debug_q_bf16"].float(), q_exp.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        ours["debug_k_bf16"].float(), k_exp.float(), atol=1e-2, rtol=1e-2
    )

    # Stage 3: the packed NVFP4 send buffer dequantizes to the packed BF16 Q/K/V.
    expected = pack(ours["debug_q_bf16"], ours["debug_k_bf16"], v_exp)
    for dest in range(P):
        sf = unswizzle(out_sf[dest], ROWS_PER_DEST, HEAD_DIM // 16)
        got = nvfp4_dequant(
            out_q[dest].reshape(ROWS_PER_DEST, HEAD_DIM // 2), sf, out_gs
        )
        torch.testing.assert_close(
            got,
            expected[dest].reshape(ROWS_PER_DEST, HEAD_DIM).float(),
            atol=1.0,
            rtol=0.1,
        )

    # Admission refuses an undersized activation scale buffer.
    bad = dict(
        admission,
        activation_sf=torch.empty(
            cake.nvfp4_activation_sf_numel(M) - 1, dtype=torch.uint8, device=device
        ),
    )
    assert not cake.supports_prepare_minimax_h3_nvfp4_pre_attention(**inputs, **bad)

    # The stages take their tensor maps by value: FlashInfer rejects a
    # descriptor workspace, and the adapter does not expose the keyword.
    with pytest.raises(ValueError, match="descriptor_workspace"):
        fi_prepare(
            **inputs,
            **theirs,
            gemm_descriptor_workspace=torch.empty(
                640, dtype=torch.uint8, device=device
            ),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
