"""Numerical correctness tests for cutlass_mxfp4a8_moe_mm (SM90 only).

Each test runs the grouped GEMM on exact fp8/mxfp4 inputs and compares against
a fp32 golden computed by dequantizing those same inputs, so quantization
error is excluded: any mismatch is a kernel, nibble-packing, or scale-layout
bug. The multi-expert cases pin the activation block-scale layout contracts
that eager and CUDA-graph capture depend on (per-expert even token-count
padding for 16B TMA alignment, (E, 2) stride descriptors).
"""

import pytest
import torch
from sgl_kernel import cutlass_mxfp4a8_moe_mm
from utils import is_hopper

from sglang.srt.layers.mxfp4a8_utils import build_grouped_act_block_scale

CHUNK = 32  # MXFP4 block size along K
# E2M1 magnitude table indexed by the 3-bit magnitude index (exp<<1 | mant),
# matching mxfp4_numeric_conversion.hpp: idx 0..7 -> {0,.5,1,1.5,2,3,4,6}.
E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def make_e2m1_weights(num_experts, n, k, device):
    """nibble codes in [0,16) with bit3=sign, bits2-0=magnitude index, plus the
    decoded signed float values, both [E, N, K]."""
    idx = torch.randint(0, 8, (num_experts, n, k), device=device)
    sign = torch.randint(0, 2, (num_experts, n, k), device=device)
    mag = E2M1_MAG.to(device)[idx]
    values = torch.where(sign.bool(), -mag, mag).to(torch.float32)
    codes = (sign << 3) | idx
    return codes.to(torch.int8), values


def pack_nibbles_natural(codes):
    """[E, N, K] int8 codes -> packed [E, N, K//2] int8.

    The kernel's DirectConvert mainloop reuses the int4a8 path, so it expects
    natural order (low nibble = even K index), NOT the ORDER_MAP interleave
    that the production int4fp8 repack applies.
    """
    codes = codes.to(torch.int8)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return ((high << 4) | (low & 0x0F)).to(torch.int8)


def interleave_scales(scales):
    """[E, N, K//CHUNK] -> [E, K//(CHUNK*4), N*4].

    4-wide because TMA loads each group scale as one packed
    Array<bf16, TileK(128)/GroupSize(32)> = 64-bit element.
    """
    s0, s1, s2 = scales.shape
    alignment = 4 if s2 % 4 == 0 else 1
    si = scales.reshape(s0, s1, s2 // alignment, alignment)
    si = si.permute(0, 2, 1, 3)
    return si.reshape(s0, s2 // alignment, s1 * alignment).contiguous()


def interleave_act_scale_padded(scale):
    """[M, K//CHUNK] -> (flat [K//(CHUNK*4) * M_pad*4], M_pad).

    The token dim is padded to an EVEN count first: the packed scale element
    is Array<bf16,4> = 8 bytes, so the TMA scale_k gmem stride M_pad*8 must
    stay 16B-aligned. Odd per-expert token counts are the norm in production;
    the padded rows are never read (the TMA box N-shape stays at the real M).
    """
    m, sk = scale.shape
    m_pad = (m + 1) & ~1
    if m_pad != m:
        pad = torch.zeros(m_pad - m, sk, dtype=scale.dtype, device=scale.device)
        scale = torch.cat([scale, pad], dim=0)
    alignment = 4 if sk % 4 == 0 else 1
    # token unit-stride, scale_k stride = M_pad
    si = scale.reshape(m_pad, sk // alignment, alignment).permute(1, 0, 2)
    return si.reshape(-1), m_pad


def quant_act_mxfp8(x):
    """Per-token + per-block(=CHUNK) fp8 e4m3 quant.

    Returns x_fp8 [M, K] float8_e4m3fn and scale [M, K//CHUNK] bf16 (amax/448).
    """
    m, k = x.shape
    xf = x.reshape(m, k // CHUNK, CHUNK).float()
    amax = xf.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / 448.0
    xq = (xf / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return xq.reshape(m, k).to(torch.float8_e4m3fn), scale.to(torch.bfloat16)


def ref_grouped_gemm_mxfp8_act(a_fp8, a_scale, w_values, w_scale, num_experts, sel):
    """Golden: dequant both weight (block=32) and activation (per-token,
    block=32) in fp32, then matmul."""
    c_ref = torch.zeros(
        (a_fp8.shape[0], w_values.shape[1]), dtype=torch.bfloat16, device=a_fp8.device
    )
    a_deq = a_fp8.float() * a_scale.float().repeat_interleave(CHUNK, dim=1)
    for e in range(num_experts):
        tok = torch.where(sel == e)[0]
        if len(tok) == 0:
            continue
        w = w_values[e] * w_scale[e].repeat_interleave(CHUNK, dim=1).float()
        c_ref[tok] = torch.matmul(a_deq[tok], w.t()).to(torch.bfloat16)
    return c_ref


def run_mxfp4a8_moe_mm(counts, k, n, device, capture_safe=False, seed=0):
    """Grouped GEMM with per-expert token counts `counts` on the full mxfp8
    activation path; returns (kernel output, fp32-dequant golden)."""
    torch.manual_seed(seed)
    num_experts = len(counts)
    m_total = int(sum(counts))

    a = torch.randn(m_total, k, dtype=torch.bfloat16, device=device)
    a_fp8, a_blk_scale = quant_act_mxfp8(a)

    codes, w_values = make_e2m1_weights(num_experts, n, k, device)
    # E8M0 weight scale: exact powers of two in [2^-4, 2^2).
    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = 2.0 ** exps.to(torch.float32)
    b_packed = pack_nibbles_natural(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    starts = [0]
    for c in counts:
        starts.append(starts[-1] + int(c))
    sel = torch.zeros(m_total, dtype=torch.long, device=device)
    for e in range(num_experts):
        sel[starts[e] : starts[e + 1]] = e
    expert_offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    problem_sizes = torch.tensor(
        [[n, int(c), k] for c in counts], dtype=torch.int32, device=device
    )
    a_strides = torch.full((num_experts, 3), k, dtype=torch.int64, device=device)
    c_strides = torch.full((num_experts, 3), n, dtype=torch.int64, device=device)

    if capture_safe:
        # Production graph-safe fixed-stride layout, built by the same helper
        # the model code uses during CUDA-graph capture.
        as_packed, as_strides = build_grouped_act_block_scale(
            a_blk_scale, expert_offsets, block_size=CHUNK, capture_safe=True
        )
    else:
        # Eager compact layout: per-expert-concatenated interleave, each
        # expert's token block padded to an even count (TMA alignment).
        blocks, m_pads = zip(
            *(
                interleave_act_scale_padded(a_blk_scale[starts[e] : starts[e + 1]])
                for e in range(num_experts)
            )
        )
        as_packed = torch.cat(blocks).contiguous()
        # StrideScale = Stride<Int<1>, int64, int64>: exactly TWO int64 per
        # expert (the leading dim is compile-time); the kernel indexes dAS[e]
        # by sizeof(StrideScale), so an (E, 3) buffer mis-strides every
        # expert > 0.
        as_strides = torch.tensor(
            [[m_pad] * 2 for m_pad in m_pads], dtype=torch.int64, device=device
        )

    # Activation scale is applied inside the mainloop, so the epilogue alpha is 1.
    a_scale_one = torch.ones(1, dtype=torch.float32, device=device)
    c = torch.empty(m_total, n, dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c,
        a_fp8,
        b_packed,
        a_scale_one,
        b_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        a_strides,  # b_strides
        c_strides,
        c_strides,  # s_strides
        CHUNK,
        1,  # topk: each token row belongs to exactly one expert
        as_packed,
        as_strides,
        CHUNK,
    )
    c_ref = ref_grouped_gemm_mxfp8_act(
        a_fp8, a_blk_scale, w_values, w_scale, num_experts, sel
    )
    return c, c_ref


@pytest.mark.skipif(
    not is_hopper(),
    reason="cutlass_mxfp4a8_moe_mm is only supported on sm90",
)
@pytest.mark.parametrize(
    "m,k,n", [(4, 256, 512), (8, 512, 1024), (16, 1024, 2048), (128, 512, 1024)]
)
def test_mxfp4a8_moe_mm_single_expert(m, k, n):
    c, c_ref = run_mxfp4a8_moe_mm([m], k, n, "cuda")
    torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)


@pytest.mark.skipif(
    not is_hopper(),
    reason="cutlass_mxfp4a8_moe_mm is only supported on sm90",
)
@pytest.mark.parametrize(
    "counts,k,n",
    [
        ([4, 4, 4, 4], 512, 1024),
        # Odd per-expert counts exercise the even-padding path; uneven counts
        # exercise the per-expert stride advance of the act-scale pointer.
        ([3, 5, 4, 8], 512, 1024),
        ([16, 8, 32, 8], 1024, 2048),
    ],
)
@pytest.mark.parametrize("capture_safe", [False, True])
def test_mxfp4a8_moe_mm_multi_expert(counts, k, n, capture_safe):
    c, c_ref = run_mxfp4a8_moe_mm(counts, k, n, "cuda", capture_safe=capture_safe)
    torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)
