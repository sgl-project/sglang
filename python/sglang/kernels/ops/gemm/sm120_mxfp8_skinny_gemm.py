"""Split-K MXFP8 GEMM for decode-sized M on SM120.

FlashInfer's SM120 MXFP8 CUTLASS GEMM has no split-K, so decode-sized GEMMs
with small N run on few CTAs. This kernel can split K across CTAs and reduces
the partial sums deterministically (the last CTA of each N tile adds them in
split order). BN and SPLIT depend only on (N, K), so a row's result
does not depend on the batch size.

The MMA is the native block-scaled one (tl.dot_scaled, E4M3 operands with UE8M0
scales per 32-wide K block, FP32 accumulate).

The weight keeps its block-FP8 checkpoint layout: E4M3 [N, K] with FP32
power-of-two block scales [ceil(N / 32), K / 32] (32x32 blocks). The activation
is either BF16/FP16/FP32, quantized in-kernel bit-identically to FlashInfer's
mxfp8_quantize (CuTe-DSL backend), or an already quantized E4M3 tensor with
UE8M0 scales in FlashInfer's 128x4 swizzled layout.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mxfp8_epilogue import ue8m0_scale

# Linear layers dispatch rows up to this count to the kernel.
MAX_M = 16


@triton.jit
def _mxfp8_skinny_gemm_kernel(
    a_ptr,
    a_sf_ptr,
    w_ptr,
    w_s_ptr,
    out_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_am,
    stride_om,
    sf_k_tiles,
    stride_wsn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    FUSE_QUANT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = pid_k * K_PER_SPLIT
    w_rows = w_ptr + offs_n[:, None] * K
    w_s_rows = w_s_ptr + (offs_n // 32) * stride_wsn
    a_rows = a_ptr + offs_m[:, None] * stride_am
    # FlashInfer 128x4 swizzled UE8M0 layout: row part of the offset.
    sf_row = (
        (offs_m // 128) * (sf_k_tiles * 512)
        + (offs_m % 32) * 16
        + ((offs_m % 128) // 32) * 4
    )
    KB: tl.constexpr = BK // 32
    for kk in range(k0, k0 + K_PER_SPLIT, BK):
        offs_k = kk + tl.arange(0, BK)
        offs_kb = kk // 32 + tl.arange(0, KB)
        w = tl.load(w_rows + offs_k[None, :], mask=n_mask[:, None], other=0.0)
        # FP32 power-of-two block scale -> its UE8M0 exponent byte.
        w_bits = tl.load(
            w_s_rows[:, None] + offs_kb[None, :], mask=n_mask[:, None], other=1.0
        ).to(tl.int32, bitcast=True)
        w_e = ((w_bits >> 23) & 0xFF).to(tl.uint8)
        if FUSE_QUANT:
            x = tl.load(a_rows + offs_k[None, :], mask=m_mask[:, None], other=0.0).to(
                tl.float32
            )
            x3 = tl.reshape(x, (BM, KB, 32))
            sf, inv = ue8m0_scale(tl.max(tl.abs(x3), axis=2))
            q = tl.minimum(tl.maximum(x3 * inv[:, :, None], -448.0), 448.0)
            a = tl.reshape(q.to(tl.float8e4nv), (BM, BK))
            a_e = sf.to(tl.uint8)
        else:
            a = tl.load(a_rows + offs_k[None, :], mask=m_mask[:, None], other=0.0)
            kb = offs_kb[None, :]
            a_e = tl.load(
                a_sf_ptr + sf_row[:, None] + (kb // 4) * 512 + (kb % 4),
                mask=m_mask[:, None],
                other=0,
            )
        acc = tl.dot_scaled(a, a_e, "e4m3", tl.trans(w), w_e, "e4m3", acc)

    out_offs = offs_m[:, None] * stride_om + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT == 1:
        tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)
    else:
        ws_offs = offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptr + pid_k * M * N + ws_offs, acc, mask=out_mask)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT - 1:
            # Last CTA of this N tile: add the partials in split order.
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(SPLIT):
                total += tl.load(
                    ws_ptr + s * M * N + ws_offs,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(
                out_ptr + out_offs, total.to(out_ptr.dtype.element_ty), mask=out_mask
            )
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


# (N, K) -> (BN, SPLIT, BK), measured on RTX PRO 6000 Blackwell (188 SMs) with
# cold weights for M = 6..16. Only these shapes are dispatched from linear layers.
# The DeepSeek-V4 shared expert stays on CUTLASS: it overlaps the routed MoE on
# a side stream, where a many-CTA kernel slowed decode.
_TUNED = {
    (1792, 5120): (32, 8, 64),  # DeepSeek-V4.1-Flash wqkv_a
    (8192, 1280): (64, 1, 128),  # DeepSeek-V4.1-Flash TP4 wq_b
    (5120, 2048): (32, 1, 256),  # DeepSeek-V4.1-Flash TP4 wo_b
    (25600, 6144): (64, 1, 128),  # DeepSeek-V4.1-Flash Engram wkv
    (1280, 5120): (32, 8, 128),  # DeepSeek-V4.1-Flash DSpark draft wq_a
    (512, 5120): (32, 10, 128),  # DeepSeek-V4.1-Flash DSpark draft wkv
    (5120, 15360): (64, 4, 128),  # DeepSeek-V4.1-Flash DSpark draft main_proj
}


def tuned_config(n: int, k: int) -> Optional[Tuple[int, int, int]]:
    """(BN, SPLIT, BK) for a tuned (N, K), else None."""
    return _TUNED.get((n, k))


def mxfp8_skinny_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    weight_block_scale: torch.Tensor,
    counters: Optional[torch.Tensor] = None,
    a_sf: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """out[M, N] = mxfp8(a) @ mxfp8(weight).T for a tuned (N, K) and M <= 128.

    a: BF16/FP16/FP32 [M, K] (quantized in-kernel) or E4M3 [M, K] with `a_sf`
    (UE8M0, FlashInfer 128x4 swizzled). weight: E4M3 [N, K].
    weight_block_scale: FP32 power-of-two scales [ceil(N / 32), K / 32].
    counters: only for shapes that split K; a contiguous int32 tensor of at
    least ceil(N / BN) zeros on the device of `a`, used by one stream at a
    time; the kernel leaves them zero.
    """
    m, k = a.shape
    n = weight.shape[0]
    assert 0 < m <= 128 and weight.shape[1] == k
    assert weight_block_scale.dtype == torch.float32
    assert weight_block_scale.shape == (triton.cdiv(n, 32), k // 32)
    assert weight_block_scale.stride(1) == 1
    fuse_quant = a_sf is None
    if fuse_quant:
        assert a.dtype in (torch.bfloat16, torch.float16, torch.float32)
    else:
        assert a.dtype == torch.float8_e4m3fn and a_sf.dtype == torch.uint8
        assert a_sf.is_contiguous()
    assert a.stride(1) == 1 and weight.is_contiguous()
    bn, split, bk = _TUNED[(n, k)]
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    tiles_n = triton.cdiv(n, bn)
    if split > 1 and (
        counters is None
        or counters.dtype != torch.int32
        or counters.device != a.device
        or not counters.is_contiguous()
        or counters.numel() < tiles_n
    ):
        raise ValueError(
            f"(N, K) = ({n}, {k}) splits K, so counters must be a contiguous int32 "
            f"tensor of at least {tiles_n} zeros on {a.device}"
        )
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    ws = (
        torch.empty((split, m, n), dtype=torch.float32, device=a.device)
        if split > 1
        else out
    )
    _mxfp8_skinny_gemm_kernel[(tiles_n, split)](
        a,
        a_sf if a_sf is not None else a,
        weight,
        weight_block_scale,
        out,
        ws,
        counters if split > 1 else out,
        m,
        n,
        k,
        a.stride(0),
        out.stride(0),
        triton.cdiv(k // 32, 4),
        weight_block_scale.stride(0),
        BM=bm,
        BN=bn,
        BK=bk,
        SPLIT=split,
        K_PER_SPLIT=k // split,
        FUSE_QUANT=fuse_quant,
        num_warps=4,
        num_stages=3,
    )
    return out
