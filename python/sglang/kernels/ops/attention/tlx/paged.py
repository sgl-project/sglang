# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paged D64/D128 GQA through uTLX on gfx1250.

Query rows pack the heads sharing one KV head. TDM loads cache pages
directly, avoiding a contiguous copy of the cached prefix. Lengths, page IDs
and query offsets remain device values so graph replay can change requests.
"""

import functools
import math

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx


# Batch-dependent sizes stay runtime values: a serving batch changes them every step,
# and each specialization would compile a new TDM kernel.
@triton.jit(do_not_specialize=["stride_pb", "total_q"])
def _paged_gqa(
    Q,
    K,
    V,
    O,
    LSE,
    Q_PTR,
    K_LEN,
    PAGES,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_pb,
    POOL_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SPLITS: tl.constexpr,
    total_q,
    Q_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    query_tile = tl.program_id(2) // SPLITS
    split = tl.program_id(2) % SPLITS
    q_begin = tl.load(Q_PTR + seq).to(tl.int64)
    q_len = tl.load(Q_PTR + seq + 1).to(tl.int64) - q_begin
    kv_len = tl.load(K_LEN + seq).to(tl.int64)
    BLOCK_Q: tl.constexpr = BLOCK_M // GROUP
    first_q = query_tile * BLOCK_Q
    if first_q < q_len:
        rows = tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        q_pos = first_q + rows // GROUP
        head = kv_head * GROUP + rows % GROUP
        q = tl.load(
            Q
            + (q_begin + q_pos[:, None]) * stride_qt
            + head[:, None].to(tl.int64) * stride_qh
            + dims[None, :],
            mask=q_pos[:, None] < q_len,
            other=0.0,
        )
        q = q.to(COMPUTE_DTYPE)
        q_scale = tl.load(Q_SCALE) if HAS_Q_SCALE else 1.0
        q_absolute = kv_len - q_len + q_pos
        k_desc = tlx.make_tensor_descriptor(
            None,
            K + kv_head.to(tl.int64) * stride_kh,
            shape=[POOL_TOKENS, HEAD_DIM],
            strides=[stride_kt, 1],
            block_shape=[PAGE_SIZE, HEAD_DIM],
        )
        v_desc = tlx.make_tensor_descriptor(
            None,
            V + kv_head.to(tl.int64) * stride_vh,
            shape=[POOL_TOKENS, HEAD_DIM],
            strides=[stride_vt, 1],
            block_shape=[PAGE_SIZE, HEAD_DIM],
        )
        # Explicit physical LDS layout is required by this uTLX TDM revision.
        PAD: tl.constexpr = 16 if K.dtype.element_ty.is_fp8() else 8
        layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(HEAD_DIM, PAD)], [BLOCK_N, HEAD_DIM], [1, 0]
        )
        if BLOCK_N == PAGE_SIZE:
            k_buf = tlx.local_alloc(
                (BLOCK_N, HEAD_DIM), K.dtype.element_ty, 2, layout=layout
            )
            v_buf = tlx.local_alloc(
                (BLOCK_N, HEAD_DIM), V.dtype.element_ty, 2, layout=layout
            )
            k_pages = k_buf
            v_pages = v_buf
        else:
            # Indexed page views carry physical LDS bases. Avoid local_slice:
            # this Triton TDM lowering drops logical memdesc_subslice offsets.
            # The tile view reads the same padded allocation without a copy.
            whole_layout: tl.constexpr = (
                tlx.padded_shared_layout_encoding.with_identity_for(
                    [(HEAD_DIM, PAD)], [2 * BLOCK_N, HEAD_DIM], [1, 0]
                )
            )
            page_layout: tl.constexpr = (
                tlx.padded_shared_layout_encoding.with_identity_for(
                    [(HEAD_DIM, PAD)], [PAGE_SIZE, HEAD_DIM], [1, 0]
                )
            )
            k_whole = tlx.local_view(
                tlx.local_alloc(
                    (2 * BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1, layout=whole_layout
                ),
                0,
            )
            v_whole = tlx.local_view(
                tlx.local_alloc(
                    (2 * BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1, layout=whole_layout
                ),
                0,
            )
            k_pages = tlx.local_reinterpret(
                k_whole,
                K.dtype.element_ty,
                [2 * BLOCK_N // PAGE_SIZE, PAGE_SIZE, HEAD_DIM],
                layout=page_layout,
            )
            v_pages = tlx.local_reinterpret(
                v_whole,
                V.dtype.element_ty,
                [2 * BLOCK_N // PAGE_SIZE, PAGE_SIZE, HEAD_DIM],
                layout=page_layout,
            )
            k_buf = tlx.local_reinterpret(
                k_whole, K.dtype.element_ty, [2, BLOCK_N, HEAD_DIM], layout=layout
            )
            v_buf = tlx.local_reinterpret(
                v_whole, V.dtype.element_ty, [2, BLOCK_N, HEAD_DIM], layout=layout
            )
        k_scale = tl.load(K_SCALE) if HAS_K_SCALE else 1.0
        v_scale = tl.load(V_SCALE) if HAS_V_SCALE else 1.0
        start = tl.full((), 0, tl.int64)
        end = kv_len
        if WINDOW_LEFT >= 0:
            start = tl.maximum(kv_len - q_len + first_q - WINDOW_LEFT, 0)
            start = start // BLOCK_N * BLOCK_N
        if CAUSAL:
            end = tl.minimum(end, kv_len - q_len + first_q + BLOCK_Q)
        if SPLITS > 1:
            blocks = tl.cdiv(tl.maximum(end - start, 0), BLOCK_N)
            split_blocks = tl.cdiv(blocks, SPLITS)
            start = start + split * split_blocks * BLOCK_N
            end = tl.minimum(end, start + split_blocks * BLOCK_N)
        maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        # FP8 values are exactly representable in BF16. Apply scalar descales
        # to the FP32 dot/output instead of rounding every dequantized input.
        scale = SM_SCALE * 1.4426950408889634 * q_scale * k_scale
        if start < end:
            for offset in tl.static_range(0, BLOCK_N, PAGE_SIZE):
                # TDM coordinates are 32-bit; unsupported_reason bounds the pool.
                first_page = tl.load(
                    PAGES
                    + seq.to(tl.int64) * stride_pb
                    + (start + offset) // PAGE_SIZE,
                    mask=start + offset < end,
                    other=0,
                ).to(tl.int32)
                tlx.tdm_async_load(
                    k_desc,
                    tlx.local_view(k_pages, offset // PAGE_SIZE),
                    [first_page * PAGE_SIZE, 0],
                )
                tlx.tdm_async_load(
                    v_desc,
                    tlx.local_view(v_pages, offset // PAGE_SIZE),
                    [first_page * PAGE_SIZE, 0],
                )
        iteration = 0
        for key_start in range(start, end, BLOCK_N):
            key_pos = key_start + tl.arange(0, BLOCK_N)
            tlx.tdm_async_wait(0)
            keys = tlx.local_load(tlx.local_view(k_buf, iteration % 2))
            values = tlx.local_load(tlx.local_view(v_buf, iteration % 2))
            # Every wave must read both LDS tiles before another can reuse them.
            tlx.workgroup_barrier()
            next_start = key_start + BLOCK_N
            if next_start < end:
                for offset in tl.static_range(0, BLOCK_N, PAGE_SIZE):
                    next_page = tl.load(
                        PAGES
                        + seq.to(tl.int64) * stride_pb
                        + (next_start + offset) // PAGE_SIZE,
                        mask=next_start + offset < end,
                        other=0,
                    ).to(tl.int32)
                    slot = (iteration + 1) % 2 * (
                        BLOCK_N // PAGE_SIZE
                    ) + offset // PAGE_SIZE
                    tlx.tdm_async_load(
                        k_desc,
                        tlx.local_view(k_pages, slot),
                        [next_page * PAGE_SIZE, 0],
                    )
                    tlx.tdm_async_load(
                        v_desc,
                        tlx.local_view(v_pages, slot),
                        [next_page * PAGE_SIZE, 0],
                    )
            keys = keys.to(q.dtype)
            values = values.to(q.dtype)
            if next_start >= end:
                # Unused cache tail elements may contain NaNs, which zero
                # softmax probabilities cannot suppress in the PV dot product.
                values = tl.where(key_pos[:, None] < end, values, 0.0)
            scores = tl.dot(q, tl.trans(keys)) * scale
            valid = (key_pos[None, :] < kv_len) & (q_pos[:, None] < q_len)
            if CAUSAL:
                valid &= key_pos[None, :] <= q_absolute[:, None]
            if WINDOW_LEFT >= 0:
                valid &= key_pos[None, :] >= q_absolute[:, None] - WINDOW_LEFT
            scores = tl.where(valid, scores, float("-inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            # Fully masked rows (including Q longer than K) have zero output.
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            p = tl.exp2(scores - safe_max[:, None])
            alpha = tl.exp2(maximum - safe_max)
            denominator = denominator * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(q.dtype), values, acc)
            maximum = next_max
            iteration += 1
        result = acc * v_scale / tl.where(denominator > 0, denominator, 1.0)[:, None]
        if SPLITS > 1:
            O += split.to(tl.int64) * total_q * Q_HEADS * HEAD_DIM
            lse = tl.where(
                denominator > 0, maximum + tl.log2(denominator), float("-inf")
            )
            tl.store(
                LSE
                + split.to(tl.int64) * total_q * Q_HEADS
                + (q_begin + q_pos) * Q_HEADS
                + head,
                lse,
                mask=q_pos < q_len,
            )
        tl.store(
            O
            + (q_begin + q_pos[:, None]) * stride_ot
            + head[:, None].to(tl.int64) * stride_oh
            + dims[None, :],
            result.to(O.dtype.element_ty),
            mask=q_pos[:, None] < q_len,
        )


@triton.jit(do_not_specialize=["total_rows"])
def _merge_splits(
    PARTIAL,
    LSE,
    OUT,
    total_rows,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, HEAD_DIM)
    lse = tl.load(LSE + splits.to(tl.int64) * total_rows + row)
    largest = tl.max(lse, 0)
    largest = tl.where(largest == float("-inf"), 0.0, largest)
    weight = tl.exp2(lse - largest)
    total = tl.sum(weight, 0)
    partial = tl.load(
        PARTIAL
        + (splits[:, None].to(tl.int64) * total_rows + row) * HEAD_DIM
        + dims[None, :]
    )
    value = tl.sum(partial * weight[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(
        OUT
        + (row // Q_HEADS).to(tl.int64) * stride_ot
        + (row % Q_HEADS) * stride_oh
        + dims,
        value.to(OUT.dtype.element_ty),
    )


def unsupported_reason(
    *,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    sinks=None,
):
    """Metadata-only capability check; never copies request data to the host."""
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape or out.shape != q.shape:
        return "expected Q/O [tokens, heads, dim] and K/V [pages, page, heads, dim]"
    dim = q.shape[-1]
    if dim not in (64, 128) or k.shape[-1] != dim:
        return "head dimension must be 64 or 128"
    if min(k.shape) < 1 or q.shape[1] < 1:
        return "empty head or cache dimensions"
    if q.shape[1] % k.shape[2] or q.shape[1] // k.shape[2] not in (1, 2, 4, 8, 16):
        return "GQA ratio must be 1, 2, 4, 8 or 16"
    if out.dtype not in (torch.bfloat16, torch.float16) or q.dtype not in (
        out.dtype,
        torch.float8_e4m3fn,
    ):
        return "Q must match BF16/FP16 output or use FP8 e4m3fn"
    if k.dtype != v.dtype or k.dtype not in (out.dtype, torch.float8_e4m3fn):
        return "KV must match Q or use FP8 e4m3fn"
    if k.shape[1] not in (16, 64, 128):
        return "page size must be 16, 64 or 128"
    if any(t.stride(-1) != 1 or min(t.stride()) <= 0 for t in (q, k, v, out)):
        return "requires positive strides and contiguous head dimensions"
    for tensor in (k, v):
        if tensor.stride(0) != tensor.shape[1] * tensor.stride(1):
            return "cache pages must have a regular token stride"
        if tensor.data_ptr() % 16 or any(
            s * tensor.element_size() % 16 for s in tensor.stride()[:-1]
        ):
            return "TDM cache bases and row strides must be 16-byte aligned"
    if k.shape[0] * k.shape[1] >= 2**31:
        return "TDM row indices must fit signed int32"
    if softcap or sinks is not None:
        return "softcap and attention sinks are unsupported"
    if len(window_size) != 2 or window_size[0] < -1 or window_size[1] not in (-1, 0):
        return "unsupported attention window"
    if not causal and window_size[1] == 0:
        return "a right window of zero requires causal attention"
    if (
        not isinstance(softmax_scale, (float, int))
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        return "softmax scale must be positive and finite"
    if (
        not isinstance(max_seqlen_q, int)
        or max_seqlen_q < 0
        or not isinstance(max_seqlen_k, int)
        or max_seqlen_k < 0
    ):
        return "sequence upper bounds must be nonnegative host integers"
    if (
        block_table.ndim != 2
        or block_table.stride(1) != 1
        or seqused_k.ndim != 1
        or cu_seqlens_q.ndim != 1
    ):
        return "unsupported paged metadata layout"
    batch = seqused_k.numel()
    if block_table.shape[0] < batch or cu_seqlens_q.numel() < batch + 1:
        return "metadata does not cover the batch"
    if cu_seqlens_q.stride(0) != 1 or seqused_k.stride(0) != 1:
        return "request lengths and pointers must be contiguous"
    tensors = [q, k, v, out, cu_seqlens_q, seqused_k, block_table]
    for tensor in (cu_seqlens_q, seqused_k, block_table):
        if tensor.dtype not in (torch.int32, torch.int64):
            return "paged metadata must use integer tensors"
    for scale in (q_descale, k_descale, v_descale):
        if scale is not None:
            if (
                not isinstance(scale, torch.Tensor)
                or scale.numel() != 1
                or scale.dtype != torch.float32
            ):
                return "descales must be scalar FP32 device tensors"
            tensors.append(scale)
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        return "all tensors must reside on one GPU"
    if torch.version.hip is None or not _device_properties(
        q.device.index
    ).gcnArchName.startswith("gfx1250"):
        return "requires gfx1250"
    return None


@functools.lru_cache(maxsize=None)
def _device_properties(index):
    return torch.cuda.get_device_properties(index)


def _compute_units(device):
    return _device_properties(device.index).multi_processor_count


def _launch_config(q, k, batch, max_seqlen_q, max_seqlen_k, window_left):
    group = q.shape[1] // k.shape[2]
    units = _compute_units(q.device)
    # Short queries retain the page-sized tile so split-KV stays lightweight.
    block_n = k.shape[1]
    long_d64_gqa8 = max_seqlen_q >= 1024 and q.shape[-1] == 64 and group == 8
    if max_seqlen_q == 1 and q.shape[-1] == 64:
        block_m, num_warps = 16, 1
    elif max_seqlen_q <= 8:
        block_m, num_warps = 32, 2
    elif (
        long_d64_gqa8
        and window_left < 0
        and batch * k.shape[2] * triton.cdiv(max_seqlen_q, 256 // group) >= 2 * units
    ):
        # Every tile streams the whole prefix, so taller tiles cut KV traffic. They
        # need eight warps (four spill registers) and enough tiles to fill the GPU.
        block_m, block_n, num_warps = 256, max(block_n, 64), 8
    else:
        block_m, num_warps = (64 if max_seqlen_q >= 1024 else 128), 4
        if long_d64_gqa8:
            # Two-page tiles amortize TDM waits for long D64 GQA prefill.
            block_n = max(block_n, 32)
    programs = batch * k.shape[2] * triton.cdiv(max_seqlen_q, block_m // group)
    active_k = (
        min(max_seqlen_k, window_left + max_seqlen_q + 1)
        if window_left >= 0
        else max_seqlen_k
    )
    # Short queries need more independent KV segments to fill a DPX partition.
    # Scale with the reported CU count rather than assuming a fixed partition.
    target_programs = units * (4 if max_seqlen_q <= 8 else 2)
    min_segment = 128 if max_seqlen_q <= 8 else 256
    splits = 1
    if active_k >= 512 and programs < target_programs:
        splits = min(
            256,
            triton.next_power_of_2(triton.cdiv(target_programs, programs)),
            triton.next_power_of_2(max(1, active_k // min_segment)),
        )
    return block_m, block_n, splits, num_warps


def unified_attention(
    *,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    sinks=None,
):
    """AITER-compatible paged interface, with bottom-right causal alignment.

    The caller owns cache writes and metadata validity: Q indptr is monotonic,
    KV lengths fit the page table, and every referenced page belongs to the
    supplied cache. ``window_size[0]`` counts preceding keys, excluding self.
    No metadata values are read on the host, including during graph capture.
    """
    arguments = locals()
    reason = unsupported_reason(**arguments)
    if reason is not None:
        raise ValueError(reason)
    if q.shape[0] == 0 or max_seqlen_q == 0 or seqused_k.numel() == 0:
        return out
    group = q.shape[1] // k.shape[2]
    block_m, block_n, splits, num_warps = _launch_config(
        q, k, seqused_k.numel(), max_seqlen_q, max_seqlen_k, window_size[0]
    )
    query_tiles = triton.cdiv(max_seqlen_q, block_m // group)
    grid = (seqused_k.numel(), k.shape[2], query_tiles * splits)
    partial = (
        torch.empty((splits, *out.shape), device=q.device, dtype=torch.float32)
        if splits > 1
        else out
    )
    lse = (
        torch.empty(
            (splits, q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32
        )
        if splits > 1
        else None
    )
    # Dot operands retain the output precision even though split accumulators
    # are stored in FP32. Pass a BF16/FP16 type independently of that storage.
    _paged_gqa[grid](
        q,
        k,
        v,
        partial,
        lse,
        cu_seqlens_q,
        seqused_k,
        block_table,
        q_descale,
        k_descale,
        v_descale,
        q.stride(0),
        q.stride(1),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        partial.stride(-3),
        partial.stride(-2),
        block_table.stride(0),
        k.shape[0] * k.shape[1],
        k.shape[1],
        q.shape[-1],
        group,
        softmax_scale,
        causal,
        window_size[0],
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16,
        splits,
        q.shape[0],
        q.shape[1],
        block_m,
        block_n,
        num_warps=num_warps,
    )
    if splits > 1:
        _merge_splits[(q.shape[0] * q.shape[1],)](
            partial,
            lse,
            out,
            q.shape[0] * q.shape[1],
            q.shape[-1],
            splits,
            q.shape[1],
            out.stride(0),
            out.stride(1),
            num_warps=4,
        )
    return out
