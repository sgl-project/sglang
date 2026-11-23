# SPDX-License-Identifier: Apache-2.0
"""SM90 BF16 QSA prefill with compact quad metadata and page64 KV.

Each request may have its own query length and prefix. The token-slot table
must map each logical page to one contiguous physical 64-token page. Selections
use the canonical QSA full-quads-plus-tail contract.
"""

import tilelang
import tilelang.language as T
import torch

from . import PREFILL_HEAD_PROFILES
from .paged_block_mask import build_block_masks
from .paged_quad_metadata import compact_quads


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _make_prefill_kernel(q_heads, kv_heads):
    block_n = 64
    threads = 128
    batch = T.symbolic("batch")
    rows = T.symbolic("rows")
    cache_rows = T.symbolic("cache_rows")
    q_tiles = T.symbolic("q_tiles")
    max_union = T.symbolic("max_union")
    heads_per_kv = q_heads // kv_heads
    dtype = T.bfloat16
    max_tile_rows = (64 + 2 * heads_per_kv - 2) // heads_per_kv
    mask_dtype = T.int32 if max_tile_rows <= 16 else T.int64

    @T.macro
    def gather(src, dst, slots, b, group, kv, count, tile):
        tx = T.get_thread_binding()
        d = tx % 32 * 8
        for part in T.unroll(block_n // (threads // 32 * 4)):
            j = part * (threads // 32) + tx // 32
            i = tile * (block_n // 4) + j
            address = T.if_then_else(
                i < count, slots[b, group, T.min(i, max_union - 1)], 0
            )
            for lane in T.unroll(4):
                T.ptx_cp_async(
                    T.access_ptr(dst[j * 4 + lane, d], "w", 8),
                    T.access_ptr(src[address + lane, kv, d], "r", 8),
                    8,
                    i < count,
                )
        T.ptx_commit_group()

    @T.prim_func
    def main(
        Q: T.Tensor([rows, q_heads, 256], dtype),
        K: T.Tensor([cache_rows, kv_heads, 256], dtype),
        V: T.Tensor([cache_rows, kv_heads, 256], dtype),
        CU: T.Tensor([batch + 1], T.int32),
        TAILS: T.Tensor([rows], T.int8),
        TILE_MASKS: T.Tensor([batch, q_tiles, max_union], mask_dtype),
        SLOTS: T.Tensor([batch, q_tiles, max_union], T.int32),
        COUNTS: T.Tensor([batch, q_tiles], T.int32),
        O: T.Tensor([rows, q_heads, 256], dtype),
    ):
        # Flatten Q x head so the M64 WGMMA tile has no per-query head padding.
        with T.Kernel(q_tiles, batch * kv_heads, threads=threads) as (tile_id, bkv):
            b = bkv // kv_heads
            kv = bkv % kv_heads
            start = CU[b]
            end = CU[b + 1]
            length = end - start
            group = q_tiles - 1 - tile_id
            qs = T.alloc_shared([64, 256], dtype)
            ks = T.alloc_shared([block_n, 256], dtype)
            vs = T.alloc_shared([block_n, 256], dtype)
            # PV consumes P from registers and V from shared memory.
            p_regs = T.alloc_fragment([64, block_n], dtype)
            os = T.alloc_shared([64, 256], dtype)
            acc = T.alloc_fragment([64, block_n], T.float32)
            out = T.alloc_fragment([64, 256], T.float32)
            prev = T.alloc_fragment([64], T.float32)
            mx = T.alloc_fragment([64], T.float32)
            alpha = T.alloc_fragment([64], T.float32)
            total = T.alloc_fragment([64], T.float32)
            sm = T.alloc_fragment([64], T.float32)
            mask_bit = T.alloc_fragment([64, 4], T.int32)
            if group * 64 < length * heads_per_kv:
                first_row = group * 64 // heads_per_kv
                for h, lane in T.Parallel(64, 4):
                    row = (group * 64 + h) // heads_per_kv
                    tail = T.if_then_else(
                        row < length, TAILS[T.min(start + row, rows - 1)], 0
                    )
                    mask_bit[h, lane] = (row - first_row) * 2 + T.cast(
                        lane >= tail, T.int32
                    )
                for h, d in T.Parallel(64, 256):
                    row = (group * 64 + h) // heads_per_kv
                    head = (group * 64 + h) % heads_per_kv
                    qs[h, d] = T.if_then_else(
                        group * 64 + h < length * heads_per_kv,
                        Q[T.min(start + row, rows - 1), kv * heads_per_kv + head, d],
                        0,
                    )
                T.fill(out, 0)
                T.fill(total, 0)
                T.fill(mx, -(2**30))
                count = COUNTS[b, group]
                tiles = T.ceildiv(count, block_n // 4)
                gather(K, ks, SLOTS, b, group, kv, count, 0)
                gather(V, vs, SLOTS, b, group, kv, count, 0)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.fill(acc, 0)
                T.gemm(qs, ks, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if tiles > 1:
                    gather(K, ks, SLOTS, b, group, kv, count, 1)
                for h, n in T.Parallel(64, block_n):
                    qi = n // 4
                    row = (group * 64 + h) // heads_per_kv
                    bit = mask_bit[h, n % 4]
                    mask = T.if_then_else(
                        qi < count, TILE_MASKS[b, group, T.min(qi, max_union - 1)], 0
                    )
                    valid = (qi < count) & (row < length) & (((mask >> bit) & 1) != 0)
                    acc[h, n] = T.if_then_else(valid, acc[h, n], -T.infinity(T.float32))
                T.copy(mx, prev)
                T.reduce_max(acc, mx, dim=1, clear=False)
                for h in T.Parallel(64):
                    mx[h] = T.max(mx[h], prev[h])
                    alpha[h] = T.exp2((prev[h] - mx[h]) * 0.09016844005)
                for h, n in T.Parallel(64, block_n):
                    acc[h, n] = T.exp2((acc[h, n] - mx[h]) * 0.09016844005)
                T.reduce_sum(acc, sm, dim=1)
                T.copy(acc, p_regs)
                for h in T.Parallel(64):
                    total[h] = total[h] * alpha[h] + sm[h]
                # QK(t) is the older committed WGMMA group; PV(t-1) is the newer group.
                for tile in T.serial(1, tiles):
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.fill(acc, 0)
                    T.wgmma_gemm(
                        qs, ks, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    T.wgmma_gemm(p_regs, vs, out, policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    # The WGMMA reads have completed. Refill the same shared
                    # buffers while this warpgroup applies the online softmax.
                    gather(V, vs, SLOTS, b, group, kv, count, tile)
                    if tile + 1 < tiles:
                        gather(K, ks, SLOTS, b, group, kv, count, tile + 1)
                    for h, n in T.Parallel(64, block_n):
                        qi = tile * (block_n // 4) + n // 4
                        row = (group * 64 + h) // heads_per_kv
                        bit = mask_bit[h, n % 4]
                        mask = T.if_then_else(
                            qi < count,
                            TILE_MASKS[b, group, T.min(qi, max_union - 1)],
                            0,
                        )
                        valid = (
                            (qi < count) & (row < length) & (((mask >> bit) & 1) != 0)
                        )
                        acc[h, n] = T.if_then_else(
                            valid, acc[h, n], -T.infinity(T.float32)
                        )
                    T.copy(mx, prev)
                    T.reduce_max(acc, mx, dim=1, clear=False)
                    for h in T.Parallel(64):
                        mx[h] = T.max(mx[h], prev[h])
                        alpha[h] = T.exp2((prev[h] - mx[h]) * 0.09016844005)
                    for h, n in T.Parallel(64, block_n):
                        acc[h, n] = T.exp2((acc[h, n] - mx[h]) * 0.09016844005)
                    T.reduce_sum(acc, sm, dim=1)
                    for h, d in T.Parallel(64, 256):
                        out[h, d] *= alpha[h]
                    T.copy(acc, p_regs)
                    for h in T.Parallel(64):
                        total[h] = total[h] * alpha[h] + sm[h]
                    T.ptx_wait_group(0)
                    T.sync_threads()
                # Drain the final PV before normalization and output stores.
                T.wgmma_gemm(p_regs, vs, out, policy=T.GemmWarpPolicy.FullRow)
                T.wait_wgmma(0)
                for h, d in T.Parallel(64, 256):
                    out[h, d] /= T.max(total[h], 1e-20)
                T.copy(out, os)
                for h, d in T.Parallel(64, 256):
                    if group * 64 + h < length * heads_per_kv:
                        O[
                            start + (group * 64 + h) // heads_per_kv,
                            kv * heads_per_kv + (group * 64 + h) % heads_per_kv,
                            d,
                        ] = os[h, d]

    return main


def sparse_prefill_blocks(
    q,
    k,
    v,
    blocks,
    positions,
    seq_lens,
    cu_q,
    token_slots,
    max_q,
):
    """Run packed, unpadded query rows with `cu_q[-1] == q.shape[0]`."""
    if q.ndim != 3 or q.dtype != torch.bfloat16 or q.shape[-1] != 256:
        raise ValueError("Q must be BF16 [rows, heads, 256]")
    if k.ndim != 3 or v.shape != k.shape or k.shape[-1] != 256:
        raise ValueError("K/V must be [cache_rows, kv_heads, 256]")
    if (q.shape[1], k.shape[1]) not in PREFILL_HEAD_PROFILES:
        raise ValueError("unsupported QSA head profile")
    if k.dtype != q.dtype or v.dtype != q.dtype or k.shape[0] % 64:
        raise ValueError("K/V must be BF16 with a page64-aligned pool")
    if blocks.shape != (q.shape[0], 512) or blocks.dtype != torch.int32:
        raise ValueError("blocks must be int32 [rows, 512]")
    if positions.shape != (q.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be int32/int64 [rows]")
    if cu_q.ndim != 1 or cu_q.dtype != torch.int32 or cu_q.numel() < 2:
        raise ValueError("cu_q must be int32 [batch + 1]")
    if seq_lens.shape != (cu_q.numel() - 1,) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("seq_lens must be int32/int64 [batch]")
    if (
        token_slots.ndim != 2
        or token_slots.dtype != torch.int32
        or token_slots.shape[0] != cu_q.numel() - 1
        or token_slots.shape[1] < 1
    ):
        raise ValueError("invalid token-slot table")
    if not all(
        x.is_cuda and x.device == q.device and x.is_contiguous()
        for x in (q, k, v, blocks, positions, seq_lens, cu_q, token_slots)
    ):
        raise ValueError("contiguous CUDA tensors on one device required")
    if q.shape[0] == 0:
        return torch.empty_like(q)
    if max_q <= 0:
        raise ValueError("max_q must cover at least one query row")
    masks, tails = build_block_masks(
        blocks, positions, seq_lens, cu_q, token_slots, max_q
    )
    slots, tile_masks, quad_counts = compact_quads(
        masks, token_slots, cu_q, seq_lens, max_q, q.shape[1] // k.shape[1], k.shape[0]
    )
    del masks
    out = torch.empty_like(q)
    _make_prefill_kernel(q.shape[1], k.shape[1])(
        q, k, v, cu_q, tails, tile_masks, slots, quad_counts, out
    )
    return out
