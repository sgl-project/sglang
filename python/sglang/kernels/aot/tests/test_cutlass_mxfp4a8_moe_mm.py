"""Single-GEMM numerical comparison for cutlass_mxfp4a8_moe_mm.

Goal: locate the correctness bug in the MXFP4A8 grouped GEMM by comparing the
kernel output against a bf16-dequant golden reference, and empirically decide
which nibble packing order the kernel expects (NATURAL vs order_map
[0,2,4,6,1,3,5,7]).

The mxfp4a8 kernel reuses the int4a8 mainloop DirectConvert path, so the byte
packing convention SHOULD be identical to the int4a8 test
(``pack_int4_values_to_int8`` = natural order).  We verify that directly here.

Run on remote (SM90 required):
    cd /sgl-workspace/sglang-bytedance/python/sglang/kernels/aot
    PYTHONPATH=... python tests/test_cutlass_mxfp4a8_moe_mm.py
"""

import sys

import torch
from sgl_kernel import cutlass_mxfp4a8_moe_mm

from sglang.srt.layers.mxfp4a8_utils import build_grouped_act_block_scale

# E2M1 magnitude table indexed by the 3-bit magnitude index (exp<<1 | mant).
# Matches mxfp4_numeric_conversion.hpp: idx 0..7 -> {0,.5,1,1.5,2,3,4,6}.
E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

CHUNK = 32  # MXFP4 block size in K
ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def _per_tensor_quant_fp8(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    assert x.is_contiguous(), "`x` is not contiguous"
    amax = x.float().abs().amax().clamp(min=1e-12)
    x_s = (amax / 448.0).reshape(1).to(torch.float32)
    x_q = (x.float() / x_s).clamp(-448.0, 448.0).to(dtype)
    return x_q, x_s


def make_e2m1_weights(num_experts, n, k, device):
    """Return (nibble_codes[int8], values[float32]) both shape [E,N,K].

    nibble_codes in [0,16): bit3=sign, bits2-0=magnitude index.
    values = decoded signed float on the E2M1 grid.
    """
    idx = torch.randint(0, 8, (num_experts, n, k), device=device)  # magnitude idx
    sign = torch.randint(0, 2, (num_experts, n, k), device=device)  # 0/1
    mag = E2M1_MAG.to(device)[idx]
    values = torch.where(sign.bool(), -mag, mag).to(torch.float32)
    codes = (sign << 3) | idx  # int8 nibble code
    return codes.to(torch.int8), values


def pack_nibbles_natural(codes: torch.Tensor) -> torch.Tensor:
    """codes [E,N,K] int8 -> packed [E,N,K//2] int8.

    Natural order (same as int4a8 test pack_int4_values_to_int8):
    low nibble = even K index (2i), high nibble = odd K index (2i+1).
    """
    codes = codes.to(torch.int8)
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return ((high << 4) | (low & 0x0F)).to(torch.int8)


def pack_nibbles_ordermap(codes: torch.Tensor) -> torch.Tensor:
    """Apply order_map [0,2,4,6,1,3,5,7] within each group of 8 along K,
    then pack natural. Mirrors the production int4fp8 reorder packing."""
    e, n, k = codes.shape
    assert k % 8 == 0
    reordered = codes.reshape(e, n, k // 8, 8)[..., ORDER_MAP].reshape(e, n, k)
    return pack_nibbles_natural(reordered)


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    """[E, N, K//CHUNK] -> [E, K//(CHUNK*4), N*4] (4-wide interleave).

    4-wide matches the mxfp4 kernel PackedScalesNum = TileK(128)/GroupSize(32)."""
    s0, s1, s2 = scales.shape
    alignment = 4 if s2 % 4 == 0 else 1
    si = scales.reshape(s0, s1, s2 // alignment, alignment)
    si = si.permute(0, 2, 1, 3)
    si = si.reshape(s0, s2 // alignment, s1 * alignment)
    return si.contiguous()


def ref_grouped_gemm(a_q, a_scale, w_values, w_scale, num_experts, sel):
    """Golden: dequant w (block=32) then matmul in fp32.

    w_values [E,N,K] fp32 (E2M1 grid), w_scale [E,N,K//CHUNK] fp32.
    """
    dtype = torch.bfloat16
    m = a_q.shape[0]
    n = w_values.shape[1]
    c_ref = torch.zeros((m, n), dtype=dtype, device=a_q.device)
    for i in range(num_experts):
        tok = torch.where(sel == i)[0]
        if len(tok) == 0:
            continue
        a = a_q[tok].to(torch.float32)
        scale_rep = w_scale[i].repeat_interleave(CHUNK, dim=1).to(torch.float32)
        w = w_values[i] * scale_rep  # [N,K]
        c = torch.matmul(a, w.t()) * a_scale
        c_ref[tok] = c.to(dtype)
    return c_ref


def run_case(pack_fn, label, num_experts, m, k, n, device, seed=0):
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(m, k, dtype=dtype, device=device)
    codes, w_values = make_e2m1_weights(num_experts, n, k, device)

    # E8M0 power-of-2 scale, block=32 along K -> [E,N,K//CHUNK]
    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = 2.0 ** exps.to(torch.float32)  # exact powers of two

    b_packed = pack_fn(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    a_q, a_scale = _per_tensor_quant_fp8(a)

    c = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c,
        a_q,
        b_packed,
        a_scale,
        b_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        CHUNK,
        8,
    )
    c = c.to(dtype)

    sel = torch.zeros((m,), dtype=torch.long, device=device)
    c_ref = ref_grouped_gemm(a_q, a_scale.item(), w_values, w_scale, num_experts, sel)

    max_abs = torch.max(torch.abs(c.float() - c_ref.float())).item()
    mean_abs = torch.mean(torch.abs(c.float() - c_ref.float())).item()
    ref_scale = c_ref.float().abs().mean().item() + 1e-9
    print(
        f"[{label:9s}] m={m} k={k} n={n}  max_abs={max_abs:.4f} "
        f"mean_abs={mean_abs:.4f} rel_mean={mean_abs/ref_scale:.4f}"
    )
    return mean_abs / ref_scale


def interleave_act_scale(scale: torch.Tensor) -> torch.Tensor:
    """[M, K//CHUNK] -> [K//(CHUNK*4), M*4] (4-wide interleave over K-blocks),
    mirroring the weight-scale interleave but tiled over tokens (M) instead of
    weight channels (N). This is the physical layout the kernel's activation
    block-scale TMA expects: token unit-stride, scale_k stride = M.
    4-wide matches PackedScalesNum = TileK(128)/GroupSize(32)."""
    m, sk = scale.shape
    alignment = 4 if sk % 4 == 0 else 1
    si = scale.reshape(m, sk // alignment, alignment)  # [M, sk/A, A]
    si = si.permute(1, 0, 2)  # [sk/A, M, A]
    si = si.reshape(sk // alignment, m * alignment)  # [sk/A, M*A]
    return si.contiguous()


def interleave_act_scale_padded(scale: torch.Tensor):
    """Same as interleave_act_scale but pads the token (M) dim up to an EVEN
    count first. The packed act-scale element is Array<bf16,4> = 8 bytes, so the
    TMA scale_k gmem stride = M*8 bytes must be a multiple of 16 -> M must be
    even. Odd per-expert token counts (real in production: routed token counts
    are unpadded) would otherwise trip a 16-byte TMA alignment error and read
    garbage. Returns (interleaved_flat[sk/4 * M_pad*4], M_pad). The padded rows
    are never consumed (the TMA box N-shape stays at the real M)."""
    m, sk = scale.shape
    m_pad = (m + 1) & ~1  # round up to even
    if m_pad != m:
        pad = torch.zeros(m_pad - m, sk, dtype=scale.dtype, device=scale.device)
        scale = torch.cat([scale, pad], dim=0)
    return interleave_act_scale(scale).reshape(-1), m_pad


def quant_act_mxfp8(x: torch.Tensor):
    """Per-token + per-block (block=CHUNK) fp8 e4m3 activation quant.

    Returns x_fp8 [M,K] float8_e4m3fn and scale [M, K//CHUNK] bf16 (amax/448)."""
    m, k = x.shape
    nblk = k // CHUNK
    xf = x.reshape(m, nblk, CHUNK).float()
    amax = xf.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / 448.0
    xq = (xf / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    x_fp8 = xq.reshape(m, k).to(torch.float8_e4m3fn)
    return x_fp8, scale.to(torch.bfloat16)


def ref_grouped_gemm_mxfp8_act(a_fp8, a_scale, w_values, w_scale, num_experts, sel):
    """Golden for mxfp8 activation: dequant BOTH weight (block=32) and activation
    (per-token + block=32) in fp32, then matmul.

    a_fp8 [M,K] fp8, a_scale [M,K//CHUNK] bf16, w_values [E,N,K] fp32,
    w_scale [E,N,K//CHUNK] fp32.
    """
    dtype = torch.bfloat16
    m = a_fp8.shape[0]
    n = w_values.shape[1]
    c_ref = torch.zeros((m, n), dtype=dtype, device=a_fp8.device)
    a_scale_rep = a_scale.float().repeat_interleave(CHUNK, dim=1)  # [M,K]
    a_deq = a_fp8.float() * a_scale_rep  # [M,K] real activation
    for i in range(num_experts):
        tok = torch.where(sel == i)[0]
        if len(tok) == 0:
            continue
        a = a_deq[tok]
        scale_rep = w_scale[i].repeat_interleave(CHUNK, dim=1).float()
        w = w_values[i] * scale_rep  # [N,K]
        c = torch.matmul(a, w.t())
        c_ref[tok] = c.to(dtype)
    return c_ref


def run_case_mxfp8_act(pack_fn, label, num_experts, m, k, n, device, seed=0):
    """Exercise the FULL mxfp8 activation path (per-token + per-block) on the
    kernel. The activation scale rides the 4th TMA; epilogue alpha must be 1.0."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(m, k, dtype=dtype, device=device)
    codes, w_values = make_e2m1_weights(num_experts, n, k, device)

    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = 2.0 ** exps.to(torch.float32)

    b_packed = pack_fn(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    # per-token + per-block activation quant
    a_fp8, a_blk_scale = quant_act_mxfp8(a)  # [M,K] fp8, [M,K//CHUNK] bf16
    as_packed = interleave_act_scale(a_blk_scale).contiguous()  # [K//128, M*4]

    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides
    # activation-scale stride: StrideScale = Stride<Int<1>, int64, int64>, so the
    # per-expert descriptor stride buffer holds exactly TWO int64 (16 bytes); the
    # kernel indexes dAS[e] by sizeof(StrideScale)=2. Layout MUST be (E, 2) =
    # [scale_k_stride=M, L_stride=M], NOT (E, 3) (that mis-strides every expert>0).
    as_strides = torch.full((num_experts, 2), m, device=device, dtype=torch.int64)

    # alpha = 1.0 (activation scale now applied inside the mainloop, not epilogue)
    a_scale_one = torch.ones(1, dtype=torch.float32, device=device)

    c = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c,
        a_fp8,
        b_packed,
        a_scale_one,
        b_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        CHUNK,
        8,
        as_packed,
        as_strides,
        CHUNK,
    )
    c = c.to(dtype)

    sel = torch.zeros((m,), dtype=torch.long, device=device)
    c_ref = ref_grouped_gemm_mxfp8_act(
        a_fp8, a_blk_scale, w_values, w_scale, num_experts, sel
    )

    max_abs = torch.max(torch.abs(c.float() - c_ref.float())).item()
    mean_abs = torch.mean(torch.abs(c.float() - c_ref.float())).item()
    ref_scale = c_ref.float().abs().mean().item() + 1e-9
    print(
        f"[{label:12s}] m={m} k={k} n={n}  max_abs={max_abs:.4f} "
        f"mean_abs={mean_abs:.4f} rel_mean={mean_abs/ref_scale:.4f}"
    )
    return mean_abs / ref_scale


def run_case_act_identity(label, num_experts, m, k, n, device, seed=0):
    """Diagnostic: activation values already lie on the fp8 grid and the
    activation block-scale is all ones. The mainloop's activation-scale path
    then MUST behave as identity, so the kernel output should equal a golden
    that dequants only the weight. Isolates scale-application correctness from
    the scale-value TMA layout."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    a = torch.randn(m, k, dtype=dtype, device=device)
    a_fp8 = a.to(torch.float8_e4m3fn)  # already on fp8 grid
    a_blk_scale = torch.ones(m, k // CHUNK, dtype=torch.bfloat16, device=device)

    codes, w_values = make_e2m1_weights(num_experts, n, k, device)
    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = 2.0 ** exps.to(torch.float32)
    b_packed = pack_nibbles_natural(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    as_packed = interleave_act_scale(a_blk_scale).contiguous()

    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides
    as_strides = torch.full((num_experts, 2), m, device=device, dtype=torch.int64)
    a_scale_one = torch.ones(1, dtype=torch.float32, device=device)

    c = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c,
        a_fp8,
        b_packed,
        a_scale_one,
        b_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        CHUNK,
        8,
        as_packed,
        as_strides,
        CHUNK,
    )
    c = c.to(dtype)

    sel = torch.zeros((m,), dtype=torch.long, device=device)
    c_ref = ref_grouped_gemm_mxfp8_act(
        a_fp8, a_blk_scale, w_values, w_scale, num_experts, sel
    )

    max_abs = torch.max(torch.abs(c.float() - c_ref.float())).item()
    mean_abs = torch.mean(torch.abs(c.float() - c_ref.float())).item()
    ref_scale = c_ref.float().abs().mean().item() + 1e-9
    print(
        f"[{label:12s}] m={m} k={k} n={n}  max_abs={max_abs:.4f} "
        f"mean_abs={mean_abs:.4f} rel_mean={mean_abs/ref_scale:.4f}"
    )
    return mean_abs / ref_scale


def run_case_mxfp8_act_multi(label, counts, k, n, device, seed=0, capture_safe=False):
    """Multi-expert grouped GEMM with the FULL mxfp8 activation path and UNEVEN
    per-expert token counts. This is the case the single-expert tests never
    exercised: the get_group_starts pointer advances by
    ``expert_offset * (k/group)`` per expert, which is only self-consistent if
    the global activation-scale buffer is the per-expert-concatenated interleave
    (each expert -> [scale_k/4, M_e*4], concatenated), with as_strides[e]=M_e.
    Tokens are laid out grouped-by-expert (as the MoE reorder produces)."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    num_experts = len(counts)
    m_total = int(sum(counts))
    sk = k // CHUNK

    a = torch.randn(m_total, k, dtype=dtype, device=device)
    a_fp8, a_blk_scale = quant_act_mxfp8(a)  # [M,K] fp8, [M,K//CHUNK] bf16

    codes, w_values = make_e2m1_weights(num_experts, n, k, device)
    exps = torch.randint(-4, 3, (num_experts, n, k // CHUNK), device=device)
    w_scale = 2.0 ** exps.to(torch.float32)
    b_packed = pack_nibbles_natural(codes).view(num_experts, n, k // 2).contiguous()
    b_scale = interleave_scales(w_scale.to(torch.bfloat16)).contiguous()

    # per-expert token boundaries (grouped layout) + expert selection vector
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + int(c))
    sel = torch.zeros((m_total,), dtype=torch.long, device=device)
    for e in range(num_experts):
        sel[starts[e] : starts[e + 1]] = e

    expert_offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    problem_sizes = torch.tensor(
        [[n, int(counts[e]), k] for e in range(num_experts)],
        dtype=torch.int32,
        device=device,
    )
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides
    if capture_safe:
        as_packed, as_strides = build_grouped_act_block_scale(
            a_blk_scale, expert_offsets, block_size=CHUNK, capture_safe=True
        )
    else:
        # per-expert-concatenated interleaved activation scale, each expert's token
        # block PADDED up to an even count (16-byte TMA alignment on scale_k stride).
        as_blocks = []
        m_pads = []
        for e in range(num_experts):
            se = a_blk_scale[starts[e] : starts[e + 1]]  # [M_e, sk]
            flat, m_pad = interleave_act_scale_padded(se)  # [sk/4 * M_pad*4], M_pad
            as_blocks.append(flat)
            m_pads.append(m_pad)
        as_packed = torch.cat(as_blocks).contiguous()
        # activation-scale stride per expert = PADDED token count M_pad. StrideScale
        # holds exactly TWO int64 (Int<1> leading dim is compile-time), so this MUST
        # be (E,2). get_group_starts uses column 0 (M_pad) to advance the per-expert
        # act-scale pointer by the exclusive cumsum of padded strides.
        as_strides = torch.tensor(
            [[int(m_pads[e])] * 2 for e in range(num_experts)],
            dtype=torch.int64,
            device=device,
        )
    a_scale_one = torch.ones(1, dtype=torch.float32, device=device)

    c = torch.empty((m_total, n), dtype=torch.bfloat16, device=device)
    cutlass_mxfp4a8_moe_mm(
        c,
        a_fp8,
        b_packed,
        a_scale_one,
        b_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        CHUNK,
        1,
        as_packed,
        as_strides,
        CHUNK,
    )
    c = c.to(dtype)

    c_ref = ref_grouped_gemm_mxfp8_act(
        a_fp8, a_blk_scale, w_values, w_scale, num_experts, sel
    )

    max_abs = torch.max(torch.abs(c.float() - c_ref.float())).item()
    mean_abs = torch.mean(torch.abs(c.float() - c_ref.float())).item()
    ref_scale = c_ref.float().abs().mean().item() + 1e-9
    print(
        f"[{label:12s}] E={num_experts} counts={list(counts)} k={k} n={n}  "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.4f} rel_mean={mean_abs/ref_scale:.4f}"
    )
    # per-expert breakdown to localize which group is wrong
    for e in range(num_experts):
        tok = torch.where(sel == e)[0]
        ce = c.float()[tok]
        re = c_ref.float()[tok]
        rs = re.abs().mean().item() + 1e-9
        print(
            f"    expert {e}: tok[{starts[e]}:{starts[e+1]}] "
            f"rel_mean={ (ce-re).abs().mean().item()/rs :.4f} "
            f"max_abs={ (ce-re).abs().max().item() :.2f}"
        )
    return mean_abs / ref_scale


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    device = "cuda"
    print("=== DIAGNOSTIC: activation scale = ones (should be identity) ===")
    for m, k, n in [(4, 256, 512), (8, 512, 1024), (128, 512, 1024)]:
        run_case_act_identity("act_ones", 1, m, k, n, device)
    print("=== mxfp8 per-token + per-block activation (FULL path, single expert) ===")
    for m, k, n in [(4, 256, 512), (8, 512, 1024), (16, 1024, 2048), (128, 512, 1024)]:
        run_case_mxfp8_act(pack_nibbles_natural, "mxfp8_act", 1, m, k, n, device)
    print("=== mxfp8 activation MULTI-EXPERT (uneven token counts) ===")
    run_case_mxfp8_act_multi("mxfp8_multi", [4, 4, 4, 4], 512, 1024, device)
    run_case_mxfp8_act_multi("mxfp8_multi", [3, 5, 4, 8], 512, 1024, device)
    run_case_mxfp8_act_multi("mxfp8_multi", [16, 8, 32, 8], 1024, 2048, device)
    print("=== mxfp8 activation MULTI-EXPERT (graph-safe fixed-stride layout) ===")
    run_case_mxfp8_act_multi(
        "mxfp8_graph", [3, 5, 4, 8], 512, 1024, device, capture_safe=True
    )
    run_case_mxfp8_act_multi(
        "mxfp8_graph", [16, 8, 32, 8], 1024, 2048, device, capture_safe=True
    )
    print("=== mxfp8 activation MULTI-EXPERT (UNIFORM counts: isolate N-value bug) ===")
    for c in ([2, 2], [3, 3], [4, 4], [8, 8], [16, 16], [32, 32]):
        run_case_mxfp8_act_multi("mxfp8_unif", c, 512, 1024, device)


if __name__ == "__main__":
    main()
