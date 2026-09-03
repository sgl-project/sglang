"""Batch-1 int2 MoE GEMV on the N-contiguous int32-word layout, through a pointer table.

Layout per expert (shared with the tiled prefill kernel's word-load int2 branch):
    qweight  int32 [K/16, N]    word kw holds values 16kw..16kw+15 at bits 2*(k%16), N contiguous
    scales   f16|bf16 [K/128, N]  symmetric, zero point 2  ->  w = (q - 2) * s

Lanes run along N, so every warp-load is one contiguous 128-B line of int32 words. That is the
property measured at 320 GB/s from device memory and 51 GB/s (PCIe line rate) from pinned host
memory (micro-benchmark on real layer-5 tensors); the byte-row variant of the same idea
reached only 157 / 14 GB/s and was rejected.

Pointer tables: int64 [E] of per-expert base addresses for qweight and scales, on the device.
An entry may point into device memory or into pinned host memory; the kernel does not care.
The kernel indexes with the ORIGINAL expert ids. No gather, no staging, no renumbering.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_gemv_int2_tab(
    a_ptr,
    stride_am,  # activations bf16 [M, K]; row r uses a[r // TOP_K]
    wtab_ptr,
    stab_ptr,  # int64 [E]: base address of qweight / scales per expert
    topk_ids_ptr,
    topk_w_ptr,  # [R] original ids, [R] routed weights (fp32)
    c_ptr,
    stride_cm,  # out bf16 [R, N]
    N,
    K,
    stride_ww,
    stride_sg,  # row strides: qweight words per K/16 row, scales per group row
    TOP_K: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    SCALE_BF16: tl.constexpr,
):
    tl.static_assert(BLOCK_K % GROUP == 0)
    tl.static_assert(BLOCK_K % 16 == 0)
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    e = tl.load(topk_ids_ptr + r).to(tl.int64)
    wbase = tl.load(wtab_ptr + e).to(tl.pointer_type(tl.int32))
    if SCALE_BF16:
        sbase = tl.load(stab_ptr + e).to(tl.pointer_type(tl.bfloat16))
    else:
        sbase = tl.load(stab_ptr + e).to(tl.pointer_type(tl.float16))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    NW: tl.constexpr = BLOCK_K // 16
    NG: tl.constexpr = BLOCK_K // GROUP
    offs_w = tl.arange(0, NW)
    a_base = a_ptr + (r // TOP_K) * stride_am

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w = tl.load(
            wbase + (k0 // 16 + offs_w[:, None]) * stride_ww + offs_n[None, :],
            mask=n_mask[None, :],
            other=0,
        )  # [NW, BN] int32
        part = tl.zeros([NW, BLOCK_N], dtype=tl.float32)
        for i in tl.static_range(16):
            x_i = tl.load(a_base + k0 + offs_w * 16 + i).to(
                tl.float32
            )  # [NW], warp-uniform
            part += ((w >> (2 * i)) & 3).to(tl.float32) * x_i[:, None]
        xs = tl.sum(
            tl.load(a_base + k0 + tl.arange(0, BLOCK_K))
            .to(tl.float32)
            .reshape([NG, GROUP]),
            axis=1,
        )  # [NG] zero-point term
        part_g = tl.sum(
            tl.reshape(part, [NG, GROUP // 16, BLOCK_N]), axis=1
        )  # [NG, BN]
        s = tl.load(
            sbase
            + (k0 // GROUP + tl.arange(0, NG)[:, None]) * stride_sg
            + offs_n[None, :],
            mask=n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum((part_g - 2.0 * xs[:, None]) * s, axis=0)
    if MUL_ROUTED_WEIGHT:
        acc = acc * tl.load(topk_w_ptr + r)
    tl.store(c_ptr + r * stride_cm + offs_n, acc.to(tl.bfloat16), mask=n_mask)


def moe_gemv_int2_tab(
    a,
    wtab,
    stab,
    topk_ids,
    topk_w,
    N,
    K,
    stride_ww,
    stride_sg,
    top_k,
    mul_routed_weight,
    block_n=64,
    block_k=128,
    num_warps=4,
    scale_bf16=True,
):
    """a [M, K] bf16; topk_ids/topk_w flat [R]; returns bf16 [R, N].
    scale_bf16: True on the server (params_dtype), False for raw checkpoint f16 scales.
    """
    assert K % block_k == 0 and block_k % 128 == 0, (K, block_k)  # BK=256 breaks K=640
    R = topk_ids.numel()
    c = torch.empty((R, N), dtype=torch.bfloat16, device=a.device)
    _moe_gemv_int2_tab[(triton.cdiv(N, block_n), R)](
        a,
        a.stride(0),
        wtab,
        stab,
        topk_ids,
        topk_w,
        c,
        c.stride(0),
        N,
        K,
        stride_ww,
        stride_sg,
        TOP_K=top_k,
        GROUP=128,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        SCALE_BF16=scale_bf16,
        num_warps=num_warps,
    )
    return c


def make_tables(qweight, scales):
    """qweight int32 [E, K/16, N], scales f16/bf16 [E, K/128, N] (device or pinned host, contiguous)
    -> int64 [E] address tables on the CUDA device, plus the row strides in elements."""
    assert qweight.dtype == torch.int32 and scales.dtype in (
        torch.float16,
        torch.bfloat16,
    ), (qweight.dtype, scales.dtype)
    assert qweight.is_contiguous() and scales.is_contiguous()
    E = qweight.shape[0]
    wt = (
        torch.arange(E, dtype=torch.int64) * (qweight.stride(0) * 4)
        + qweight.data_ptr()
    )
    st = torch.arange(E, dtype=torch.int64) * (scales.stride(0) * 2) + scales.data_ptr()
    return wt.cuda(), st.cuda(), qweight.stride(1), scales.stride(1)


def to_word_ncontig(qweight_u8_nk):
    """Loader layout uint8 [E, N, K/4] (what MoeWNA16 holds today) -> int32 [E, K/16, N].

    Byte kb = k//4 of row n becomes byte (kb % 4) of word kb // 4 at column n, little-endian:
    value k lands at bits 8*((k//4)%4) + 2*(k%4) = 2*(k%16). Bit-exact re-arrangement.
    """
    E, N, KB = qweight_u8_nk.shape
    t = qweight_u8_nk.transpose(1, 2)  # [E, K/4, N]
    t = t.reshape(E, KB // 4, 4, N).permute(0, 1, 3, 2)  # [E, K/16, N, 4]
    # view(int32) folds the trailing 4 bytes into one word: [E, K/16, N, 1] -> drop the 1
    return t.contiguous().view(torch.int32).view(E, KB // 4, N)
