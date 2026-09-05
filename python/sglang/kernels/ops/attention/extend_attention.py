# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides
from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.kernels.ops.attention.score_mod import unpack_aux_tensors
from sglang.srt.environ import envs
from sglang.srt.utils import (
    is_cuda,
    is_gfx95_supported,
    is_gfx1250_supported,
    is_hip,
)

_is_cuda = is_cuda()
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

_is_hip = is_hip()
_is_gfx95 = _is_hip and is_gfx95_supported()
_is_gfx1250 = _is_hip and is_gfx1250_supported()

try:
    _triton_version_parts = tuple(
        int(part) for part in triton.__version__.split(".")[:2]
    )
except (AttributeError, ValueError):
    _triton_version_parts = (0, 0)
_is_triton_ge_37 = _triton_version_parts >= (3, 7)


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware
    if _is_hip:
        if _is_gfx95 and _is_triton_ge_37 and Lq == 576 and Lv == 512:
            # Triton 3.7's N64 codegen reaches 512 VGPRs and spills 472 bytes
            # of scratch on gfx950. N32 keeps BLOCK_M/launch work unchanged,
            # uses <=433 VGPRs without scratch, and restores the isolated
            # late-prefill kernel from ~12.57 ms to ~5.24 ms.
            BLOCK_M, BLOCK_N = (64, 32)
            num_warps = 4
        elif _is_gfx95 and 128 < Lq <= 256:
            # gfx950 (CDNA4), 128 < head_dim <= 256: a larger query tile halves KV bytes
            # streamed per call (each workgroup reads the whole prefix); 8 warps
            # hide the loads. Measured on MI350X head_dim 256: -36% kernel time,
            # 28% -> 44% MFU, numerically equivalent (BLOCK_N reduction order
            # unchanged). Other AMD archs / head dims keep the default below.
            BLOCK_M, BLOCK_N = (128, 64)
            num_warps = 8
        else:
            BLOCK_M, BLOCK_N = (64, 64)
            num_warps = 4
    else:
        if _is_cuda and CUDA_CAPABILITY[0] == 12:
            # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        elif _is_cuda and CUDA_CAPABILITY[0] == 10:
            # Blackwell data-center architecture (GB200, B200, sm_100a)
            # sm_100a has different register constraints from Hopper; Hopper block sizes
            # cause PTX register exhaustion (>255 regs) for large head dims (Lq=512).
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (16, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 9:
            # Hopper architecture (H100, etc.)
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 64)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
            # Ampere architecture (A100, etc.)
            # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
            if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (64, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 32)
            else:
                if Lq <= 128:
                    BLOCK_M, BLOCK_N = (128, 128)
                elif Lq <= 256:
                    BLOCK_M, BLOCK_N = (64, 64)
                else:
                    BLOCK_M, BLOCK_N = (32, 64)
        else:
            # Older architectures
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


def _get_num_stages_for_extend_attention(
    Lq: int, Lv: int, block_n: int | None = None
) -> int:
    if _is_gfx95 and Lq == 192 and Lv == 128:
        return 2
    if (
        _is_gfx95
        and Lq == 576
        and Lv == 512
        and (block_n == 32 or (block_n is None and _is_triton_ge_37))
    ):
        return 2
    return 1


def _compact_extend_q_tiles_per_head(
    *,
    batch_size: int,
    max_len_extend: int,
    total_extend_tokens: int,
    block_m: int,
    extend_seq_lens_cpu=None,
) -> int | None:
    """Return compact query tiles per head when it reduces launch work.

    The legacy extend grid is rectangular -- ``batch_size * cdiv(max_len_extend,
    BLOCK_M)`` -- so in a ragged mixed-prefill batch every short row pays tile
    work sized by the longest row. This computes the *compact* tile count
    (``sum_i cdiv(extend_len_i, BLOCK_M)``), i.e. work proportional to the real
    per-request lengths. That is the same ragged-aware launch the flash-attn
    varlen kernels (used by the aiter backend via ``flash_attn_varlen_func`` /
    ``mha_batch_prefill_func``) already get from their cu_seqlens scheduler --
    this closes that triton-vs-flash-attn gap rather than inventing a new
    technique. Returns ``None`` (keep the legacy grid) when compacting would not
    reduce launch work, e.g. a uniform batch.
    """
    if batch_size <= 1 or max_len_extend <= 0:
        return None

    legacy_tiles = batch_size * triton.cdiv(max_len_extend, block_m)
    if legacy_tiles <= 0:
        return None

    if extend_seq_lens_cpu is not None:
        if isinstance(extend_seq_lens_cpu, torch.Tensor):
            extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
        if len(extend_seq_lens_cpu) < batch_size:
            return None
        compact_tiles = sum(
            triton.cdiv(max(0, int(extend_seq_lens_cpu[i])), block_m)
            for i in range(batch_size)
        )
    else:
        if total_extend_tokens == batch_size * max_len_extend:
            return None
        compact_tiles = (total_extend_tokens + batch_size * (block_m - 1)) // block_m

    if compact_tiles <= 0 or compact_tiles >= legacy_tiles:
        return None
    return int(compact_tiles)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _copy_unified_indices_kernel(
    # Input buffers
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    unified_kv_indptr,
    # Output buffer
    unified_kv_indices,
    # Size
    bs,
):
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.
    """
    pid = tl.program_id(0)

    if pid >= bs:
        return

    # Load sequence info
    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    # Copy indices in vectorized chunks
    BLOCK_SIZE: tl.constexpr = 128

    # Process prefix indices
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len

        src_idx = prefix_start + offs
        dst_idx = unified_start + offs

        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)

    # Process extend indices
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len

        src_idx = extend_start + offs
        dst_idx = unified_start + prefix_len + offs

        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)


def build_unified_kv_indices(
    prefix_kv_indptr: torch.Tensor,
    prefix_kv_indices: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_kv_indices: torch.Tensor,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build unified KV indices efficiently:
    - Use PyTorch's optimized cumsum (NVIDIA CUB) for indptr
    - Use Triton kernel for parallel index copying

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    device = prefix_kv_indptr.device

    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    # Create unified_kv_indptr avoiding direct assignment (for CUDA graph compatibility)
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    unified_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(unified_lens, dim=0),
        ]
    )

    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)

    unified_kv_indices = torch.empty(max_unified_len, dtype=torch.int64, device=device)

    # Launch Triton kernel for parallel index copying
    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        unified_kv_indptr,
        unified_kv_indices,
        bs,
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    LSE_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    compact_batch_size,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    SKIP_PREFIX: tl.constexpr,
    SKIP_EXTEND: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_COMPACT_TILE_GRID: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_FP8_PREFIX: tl.constexpr,
    USE_FP8_EXTEND: tl.constexpr,
    FP8_MAX: tl.constexpr,
    IS_GFX1250: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    IDENTITY_KV_INDICES: tl.constexpr = False,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
):
    if USE_COMPACT_TILE_GRID:
        output_tile = tl.program_id(0)
        cur_head = tl.program_id(1)

        cur_seq = tl.full((), 0, tl.int64)
        cum_tiles = tl.full((), 0, tl.int64)
        found = tl.full((), 0, tl.int32)
        while (cur_seq < compact_batch_size) & (found == 0):
            seq_q_start = tl.load(qo_indptr + cur_seq)
            seq_q_end = tl.load(qo_indptr + cur_seq + 1)
            seq_q_len = seq_q_end - seq_q_start
            seq_tiles = (seq_q_len + BLOCK_M - 1) // BLOCK_M
            next_cum_tiles = cum_tiles + seq_tiles
            if next_cum_tiles > output_tile:
                found = 1
            else:
                cum_tiles = next_cum_tiles
                cur_seq = cur_seq + 1

        if found == 0:
            return

        cur_block_m = output_tile - cum_tiles
    else:
        cur_seq = tl.program_id(0)
        cur_head = tl.program_id(1)
        cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    # Grid axis 2 spans the batch-max extend length; all stores are masked by mask_m.
    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # For SWA, we should only load the mask in the sliding window
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)
    # The FP8 prefix sweep can use a wider tile than the current-token sweep.
    offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    prefix_end = 0 if SKIP_PREFIX else cur_seq_len_prefix
    for start_n in range(0, prefix_end, BLOCK_N_PREFIX):
        start_n = tl.multiple_of(start_n, BLOCK_N_PREFIX)
        mask_n = (start_n + offs_n_prefix) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n_prefix[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            # q_id = prefix_len + cur_m, kv_id = cur_n
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            if IDENTITY_KV_INDICES:
                offs_kv_loc = cur_seq_kv_start_idx + start_n + offs_n_prefix
            else:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
                    mask=mask_n,
                    other=0,
                )

            # Page-aware KV address math. At PAGE_SIZE==1
            # (legacy / non-shared / shared-at-ps=1), Triton specializes
            # the else-branch away — byte-identical SASS to today.
            if PAGE_SIZE == 1:
                # load k in transposed way
                offs_buf_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            else:
                page_id = offs_kv_loc // PAGE_SIZE
                tok_in_p = offs_kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )
            # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
            # dim K>=128 (K=64 ok). This prefix read fires when a radix-cache prefix is
            # reused (prefill reads the cached fp8 KV), and the MLA nope dot has K=512,
            # so we must upcast the fp8 K to q's dtype and dot in bf16 rather than
            # downcasting q to fp8. No-op for a bf16 cache. (Do NOT revert to q.to(fp8).)
            # On all other platforms keep the original q.to(k.dtype) downcast.
            # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
            if IS_GFX1250:
                qk = tl.dot(q, k.to(q.dtype))
            else:
                qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_kpe = (
                        offs_kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                else:
                    offs_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                if IS_GFX1250:
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                else:
                    qk += tl.dot(qpe.to(kpe.dtype), kpe)
            if USE_EXP2:
                qk *= sm_scale * k_scale * LOG2E
            else:
                qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    start_n + offs_n_prefix[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            if USE_EXP2:
                re_scale = tl.exp2(e_max - n_e_max)
                p = tl.exp2(qk - n_e_max[:, None])
            else:
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            else:
                offs_buf_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            if USE_FP8_PREFIX:
                p_dot = (p * FP8_MAX).to(v.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p_dot, v) * (v_scale / FP8_MAX)
            else:
                # keep softmax weights p in fp32 for the P·V dot (do not downcast to bf16)
                # on gfx1250; on other platforms restore the original p.to(v.dtype) cast.
                # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
                if IS_GFX1250:
                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
                else:
                    dot = tl.dot(p.to(v.dtype), v)
                acc = acc * re_scale[:, None] + dot * v_scale

            e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    extend_end = 0 if SKIP_EXTEND else cur_block_m_end
    for start_n in range(0, extend_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe)

            if USE_EXP2:
                qk *= sm_scale * LOG2E
            else:
                qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    cur_seq_len_prefix + start_n + offs_n[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            if USE_EXP2:
                re_scale = tl.exp2(e_max - n_e_max)
                p = tl.exp2(qk - n_e_max[:, None])
            else:
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            if USE_FP8_EXTEND:
                p_dot = (p * FP8_MAX).to(v.dtype)
                acc = acc * re_scale[:, None] + tl.dot(p_dot, v) * (1.0 / FP8_MAX)
            else:
                # keep softmax weights p in fp32 for the P·V dot (do not downcast to bf16)
                # on gfx1250; on other platforms restore the original p.to(v.dtype) cast.
                # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
                if IS_GFX1250:
                    dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
                else:
                    dot = tl.dot(p.to(v.dtype), v)
                acc = acc * re_scale[:, None] + dot

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        if USE_EXP2:
            deno += tl.exp2(cur_sink * LOG2E - e_max)
        else:
            deno += tl.exp(cur_sink - e_max)

    # A ragged prefix chunk can be empty for some requests. Represent an empty
    # partial as output=0 and LSE=-inf so merge_state ignores it exactly.
    no_kv = deno == 0.0

    if STORE_LSE:
        offs_lse = (
            cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m
        ) * stride_lse_bs + cur_head * stride_lse_h
        if USE_EXP2:
            lse = tl.log(deno) + e_max * LN2
        else:
            lse = tl.log(deno) + e_max
        lse = tl.where(no_kv, float("-inf"), lse)
        tl.store(LSE_Extend + offs_lse, lse, mask=mask_m)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    deno_safe = tl.where(no_kv, 1.0, deno)
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno_safe[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno_safe[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    extend_seq_lens_cpu=None,
    identity_kv_indices: bool = False,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager

    When ``lse_extend`` is provided, the per-query/head natural-log LSE is also
    written to it (used by DCP to merge partial attention across ranks).
    ``skip_prefix`` / ``skip_extend`` skip the prefix-KV / current-chunk stage
    respectively so DCP can compute those two parts separately.
    ``score_mod`` / ``aux_tensors`` add a custom term to the attention logits;
    see triton_ops/score_mod.py for the contract.
    ``identity_kv_indices`` promises that the prefix buffer is densely packed,
    allowing direct addressing instead of loading an index for every token.
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]
    zero_prefix_shape = (
        head_num == 12
        and k_extend.shape[1] == 12
        and Lq == 192
        and Lk == 192
        and Lv == 128
    )
    absorbed_shape = (
        head_num == 12
        and k_extend.shape[1] == 1
        and Lq == 576
        and Lk == 576
        and Lv == 512
    )
    kimi_k3_shape = zero_prefix_shape or absorbed_shape

    # Match Aiter's opt-in behavior: cast Q, K, and V separately before the
    # native-FP8 zero-prefix kernel at every sequence length.
    use_fp8_zero_prefix = (
        _is_gfx95
        and envs.SGLANG_TRITON_FP8_PREFILL_ATTN.get()
        and zero_prefix_shape
        and q_extend.dtype == torch.bfloat16
        and k_extend.dtype == torch.bfloat16
        and v_extend.dtype == torch.bfloat16
        and k_buffer.dtype == torch.float8_e4m3fn
        and v_buffer.dtype == torch.float8_e4m3fn
        and custom_mask is None
        and is_causal
        and sliding_window_size <= 0
        and logit_cap <= 0
        and xai_temperature_len <= 0
        and sinks is None
        and score_mod is None
        and aux_tensors is None
    )
    if use_fp8_zero_prefix:
        q_extend = q_extend.to(torch.float8_e4m3fn)
        k_extend = k_extend.to(torch.float8_e4m3fn)
        v_extend = v_extend.to(torch.float8_e4m3fn)

    # Get block sizes and configuration for the generic fallback.
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None
    USE_FP8_PREFIX = (
        _is_gfx95
        and kimi_k3_shape
        and k_buffer.dtype == torch.float8_e4m3fn
        and v_buffer.dtype == torch.float8_e4m3fn
    )
    USE_FP8_EXTEND = (
        _is_gfx95
        and zero_prefix_shape
        and k_extend.dtype == torch.float8_e4m3fn
        and v_extend.dtype == torch.float8_e4m3fn
    )
    FP8_MAX = (
        torch.finfo(torch.float8_e4m3fn).max
        if USE_FP8_PREFIX or USE_FP8_EXTEND
        else 1.0
    )
    # At head_dim 192, FP8 operands allow a 128-column tile while BF16 does
    # not fit in LDS. Widen prefix and extend sweeps independently.
    BLOCK_N_ARCH = BLOCK_N
    FP8_BLOCK_N = 128 if BLOCK_N_ARCH < 128 and Lq <= 192 else BLOCK_N_ARCH
    BLOCK_N = FP8_BLOCK_N if USE_FP8_EXTEND else BLOCK_N_ARCH
    BLOCK_N_PREFIX = FP8_BLOCK_N if USE_FP8_PREFIX else BLOCK_N_ARCH
    USE_EXP2 = (
        _is_gfx95
        and kimi_k3_shape
        and logit_cap <= 0
        and xai_temperature_len <= 0
        and score_mod is None
    )
    STORE_LSE = lse_extend is not None
    stride_lse_bs = lse_extend.stride(0) if STORE_LSE else 0
    stride_lse_h = lse_extend.stride(1) if STORE_LSE else 0

    # Compact grid: AMD/HIP-only optimization (parity with flash-attn's ragged-aware
    # launch). Explicitly check _is_hip and allow env var override.
    use_compact_tile_grid = (
        _is_hip and envs.SGLANG_TRITON_COMPACT_EXTEND_ATTENTION.get()
    )
    compact_q_tiles = None
    if use_compact_tile_grid:
        compact_q_tiles = _compact_extend_q_tiles_per_head(
            batch_size=batch_size,
            max_len_extend=max_len_extend,
            total_extend_tokens=q_extend.shape[0],
            block_m=BLOCK_M,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
        )

    use_compact_tile_grid = compact_q_tiles is not None
    if use_compact_tile_grid:
        grid = (compact_q_tiles, head_num)
    else:
        grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = (
        _get_num_stages_for_extend_attention(Lq, Lv, BLOCK_N) if kimi_k3_shape else 1
    )

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        lse_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        k_scale,
        v_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        stride_lse_bs,
        stride_lse_h,
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        batch_size,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        STORE_LSE=STORE_LSE,
        SKIP_PREFIX=skip_prefix,
        SKIP_EXTEND=skip_extend,
        HAS_SINK=HAS_SINK,
        IS_GFX1250=_is_gfx1250,
        STORE_TRANSPOSE=_is_hip,
        USE_COMPACT_TILE_GRID=use_compact_tile_grid,
        USE_EXP2=USE_EXP2,
        USE_FP8_PREFIX=USE_FP8_PREFIX,
        USE_FP8_EXTEND=USE_FP8_EXTEND,
        FP8_MAX=FP8_MAX,
        PAGE_SIZE=page_size,
        IDENTITY_KV_INDICES=identity_kv_indices,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend


@triton.jit
def _fwd_kernel_unified(
    Q,
    O,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_start_pos,
    sm_scale_withk,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    IS_GFX1250: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
):
    """
    Unified 1-stage kernel for deterministic extend attention.
    Both prefix and extend KV are accessed through the unified kv_indices.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence information
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    # Grid axis 2 spans the batch-max extend length; the store is masked by mask_m.
    if cur_block_m * BLOCK_M >= cur_seq_q_len:
        return

    # Load window start position for sliding window attention
    # This is the absolute position of the first key in the window (0 if no sliding window)
    cur_window_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        cur_window_start = tl.load(window_start_pos + cur_seq)

    # Load custom mask start index if using custom mask (for speculative decoding)
    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # XAI temperature handling
    if xai_temperature_len > 0:
        offs_qidx = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m
        xai_temperature_reg = tl.where(
            offs_qidx < xai_temperature_len,
            1.0,
            xai_temperature_len / (offs_qidx + 1.0),
        )

    # Load Q
    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop: process all KV tokens (prefix + extend)
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        # Compute mask
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Apply custom mask if provided
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        # Apply causal mask for extend part
        if IS_CAUSAL and not USE_CUSTOM_MASK:
            # Determine if current KV block is in extend region
            # Only apply causal mask when both Q and K are in extend region
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]

            # Causal mask: q_idx >= (k_idx - prefix_len) when k_idx >= prefix_len
            # For prefix region (k_idx < prefix_len), no causal mask
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,
                True,  # No causal mask for prefix
            )
            final_mask &= causal_mask

        if SLIDING_WINDOW_SIZE > 0:
            # Sliding window mask with correct absolute positions
            # Q absolute position: window_start + prefix_len + q_position_in_extend
            q_abs_pos = (
                cur_window_start
                + cur_seq_prefix_len
                + cur_block_m * BLOCK_M
                + offs_m[:, None]
            )

            # K absolute position: window_start + k_index_in_unified_array
            k_abs_pos = cur_window_start + start_n + offs_n[None, :]

            # Sliding window: query can attend to keys within window_size
            window_mask = q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        # Check if we can skip this tile
        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # Load KV indices
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # Page-aware KV address math (see _fwd_kernel_stage1).
            if PAGE_SIZE == 1:
                # Load K
                offs_buf_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            else:
                page_id = offs_kv_loc // PAGE_SIZE
                tok_in_p = offs_kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
            # dim K>=128 (K=64 ok). This prefix read fires when a radix-cache prefix is
            # reused (prefill reads the cached fp8 KV), and the MLA nope dot has K=512,
            # so we must upcast the fp8 K to q's dtype and dot in bf16 rather than
            # downcasting q to fp8. No-op for a bf16 cache. (Do NOT revert to q.to(fp8).)
            # On all other platforms keep the original q.to(k.dtype) downcast.
            # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
            if IS_GFX1250:
                qk = tl.dot(q, k.to(q.dtype))
            else:
                qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_kpe = (
                        offs_kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                else:
                    offs_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                if IS_GFX1250:
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                else:
                    qk += tl.dot(qpe.to(kpe.dtype), kpe)

            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m)[:, None],
                    start_n + offs_n[None, :],
                    (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m)[:, None],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            # Online softmax
            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            # Load V
            if PAGE_SIZE == 1:
                offs_buf_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            else:
                offs_buf_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + offs_dv[None, :]
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            # keep softmax weights p in fp32 for the P·V dot (do not downcast to bf16)
            # on gfx1250; on other platforms restore the original p.to(v.dtype) cast.
            # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
            if IS_GFX1250:
                dot = tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
            else:
                dot = tl.dot(p.to(v.dtype), v)
            acc = acc * re_scale[:, None] + dot

            e_max = n_e_max

    # Handle sink tokens
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    # Store output
    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / deno[:, None] * v_scale,
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd_unified(
    q,
    o,
    k_buffer,
    v_buffer,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    max_len_extend,
    custom_mask=None,
    mask_indptr=None,
    sm_scale=None,
    logit_cap=0.0,
    is_causal=True,
    sliding_window_size=-1,
    sinks=None,
    window_start_pos=None,
    xai_temperature_len=-1,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
    """
    Unified 1-stage extend attention for deterministic inference.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        o: Output tensor [num_tokens, num_heads, head_dim]
        k_buffer: Key cache buffer
        v_buffer: Value cache buffer
        qo_indptr: Query offsets [batch_size + 1]
        kv_indptr: KV offsets [batch_size + 1] (includes both prefix and extend)
        kv_indices: Unified KV indices (both prefix and extend)
        prefix_lens: Prefix length for each sequence [batch_size]
        max_len_extend: Maximum extend length
        custom_mask: Custom attention mask (for speculative decoding tree attention)
        mask_indptr: Mask offsets [batch_size + 1]
        sm_scale: Softmax scale
        logit_cap: Logit capping value
        is_causal: Whether to apply causal mask
        sliding_window_size: Sliding window size (-1 for no sliding window)
        sinks: Sink tokens
        window_start_pos: Absolute position of first key in sliding window [batch_size]
                         (None if sliding window not used)
        xai_temperature_len: XAI temperature length
    """
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]

    # Get block sizes and configuration
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q.shape[1]
    # head_num lives at dim 1 (3-D) or dim 2 (4-D view).
    kv_head_num = k_buffer.shape[-2]
    kv_group_num = q.shape[1] // kv_head_num

    USE_CUSTOM_MASK = custom_mask is not None
    HAS_SINK = sinks is not None

    # For sliding window attention, window_start_pos tracks the absolute position
    # of the first key in each sequence's window
    if sliding_window_size > 0 and window_start_pos is None:
        # If not provided, assume window starts at position 0
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_kernel_unified[grid](
        q,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale * k_scale,
        v_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        IS_GFX1250=_is_gfx1250,
        PAGE_SIZE=page_size,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


@triton.jit
def _dense_prefill_inner(
    acc,
    deno,
    e_max,
    q,
    qpe,
    K,
    V,
    cur_seq_kv_start,
    cur_kv_head,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    offs_d,
    offs_dpe,
    offs_dv,
    q_pos,
    mask_m,
    mask_d,
    mask_dv,
    start_lo,
    start_hi,
    kv_end,
    qk_scale,
    logit_cap: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_FP8: tl.constexpr,
    LOG2_FP8_MAX: tl.constexpr,
):
    """One online-softmax sweep over ``[start_lo, start_hi)`` of the KV axis.

    Instantiated twice per kernel. ``MASKED=False`` covers the interior, where
    every key is visible to every query row of the block: the predicates are
    gone, so the K/V loads stay contiguous over the head dim and widen to
    dwordx4. ``MASKED=True`` covers only the causal diagonal and the ragged
    tail. Splitting the sweep is what keeps that per-element ``tl.where`` and
    the narrowed loads off the long prefix, which is the bulk of the work.
    """
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(start_lo, start_hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_kn = start_n + offs_n

        offs_k = (
            (cur_seq_kv_start + offs_kn[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        if MASKED:
            mask_n = offs_kn < kv_end
            k = tl.load(K + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        elif EVEN_D:
            k = tl.load(K + offs_k)
        else:
            k = tl.load(K + offs_k, mask=mask_d[:, None], other=0.0)

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_kv_start + offs_kn[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            if MASKED:
                kpe = tl.load(K + offs_kpe, mask=mask_n[None, :], other=0.0)
            else:
                kpe = tl.load(K + offs_kpe)
            qk += tl.dot(qpe, kpe, out_dtype=tl.float32)

        if logit_cap > 0:
            qk *= qk_scale
            qk = logit_cap * tanh(qk / logit_cap)
            qk *= 1.4426950408889634
        else:
            qk *= qk_scale

        if MASKED:
            final_mask = mask_m[:, None] & mask_n[None, :]
            if IS_CAUSAL:
                final_mask &= q_pos[:, None] >= offs_kn[None, :]
            qk = tl.where(final_mask, qk, float("-inf"))
            row_max = tl.max(qk, 1)
            # A fully masked row would poison e_max with -inf; -1e20 keeps the
            # rescale finite and still contributes nothing to deno.
            row_max = tl.where(row_max == float("-inf"), -1e20, row_max)
        else:
            row_max = tl.max(qk, 1)

        n_e_max = tl.maximum(row_max, e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        if USE_FP8:
            # Bias the exponent instead of multiplying P by FP8_MAX after the
            # fact: exp2(qk - m + log2(FP8_MAX)) == exp2(qk - m) * FP8_MAX, so
            # the lift costs a BLOCK_M-wide subtract rather than a
            # BLOCK_M x BLOCK_N one, and skips a rounding step. deno picks up
            # the same constant factor, which cancels against acc at the final
            # divide -- so out_scale drops its 1/FP8_MAX and LSE subtracts
            # log2(FP8_MAX) from e_max.
            p = tl.exp2(qk - (n_e_max - LOG2_FP8_MAX)[:, None])
        else:
            p = tl.exp2(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_kv_start + offs_kn[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        if MASKED:
            v = tl.load(V + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        elif EVEN_D:
            v = tl.load(V + offs_v)
        else:
            v = tl.load(V + offs_v, mask=mask_dv[None, :], other=0.0)

        # P is already lifted off the e4m3 denormal floor by the exponent bias
        # above when USE_FP8; the cast is all that is left.
        #
        # Guarding this rescale on `tl.min(re_scale) < 1.0` (it is exactly 1.0
        # once the running max settles, which is most of a long prefix) was
        # measured at 21% SLOWER: the branch splits the loop body and the
        # pipeliner stops prefetching K/V across iterations. Keep it
        # unconditional.
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)

        e_max = n_e_max

    return acc, deno, e_max


@triton.jit
def _fwd_kernel_dense_prefill(
    Q,
    K,
    V,
    O,
    Lse,
    qo_indptr,
    kv_indptr,
    sm_scale,
    k_scale,
    v_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    kv_group_num: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_LSE: tl.constexpr,
    USE_FP8: tl.constexpr,
    LOG2_FP8_MAX: tl.constexpr,
    EVEN_D: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
):
    """Single-loop dense prefill: K/V hold prefix + current chunk contiguously.

    ``_fwd_kernel`` needs two stages because its prefix lives in the paged
    latent KV cache while its suffix is contiguous. Once the prefix has been
    up-projected into dense per-head K/V (``AttnForwardMethod.MHA_ONE_SHOT``),
    both halves share one base pointer, one dtype and one scale, so the split
    buys nothing and only costs pipelining and registers.

    Causal masking is bottom-right aligned: query ``m`` of a sequence sits at
    absolute position ``prefix_len + m``, where ``prefix_len = kv_len - q_len``.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    # Causal cost grows with block index: block m sweeps prefix_len + m*BLOCK_M
    # keys. Program IDs dispatch roughly in order, so issuing the heavy blocks
    # first lets the cheap ones backfill the tail instead of trailing it.
    cur_block_m = NUM_BLOCKS_M - 1 - tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    cur_seq_q_start = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start
    cur_seq_kv_start = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start
    cur_seq_prefix_len = cur_seq_kv_len - cur_seq_q_len

    # Grid axis 2 spans the batch-max query length; short sequences bail early.
    if cur_block_m * BLOCK_M >= cur_seq_q_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)
    else:
        # Never read: BLOCK_DPE is constexpr, so the whole rope branch is
        # folded away. These only keep the inner helper's signature uniform.
        offs_dpe = offs_d
        qpe = q

    # Absolute position of each query row inside its sequence.
    q_pos = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    if IS_CAUSAL:
        kv_end = tl.minimum(
            cur_seq_kv_len, cur_seq_prefix_len + (cur_block_m + 1) * BLOCK_M
        )
        # The lowest query row of this block sees keys 0..prefix_len+m*BLOCK_M,
        # so every whole BLOCK_N below that bound is unmasked for all rows.
        n_full = ((cur_seq_prefix_len + cur_block_m * BLOCK_M + 1) // BLOCK_N) * BLOCK_N
        n_full = tl.minimum(n_full, kv_end)
    else:
        kv_end = cur_seq_kv_len
        n_full = (kv_end // BLOCK_N) * BLOCK_N

    qk_scale = sm_scale * k_scale
    if logit_cap <= 0:
        qk_scale *= LOG2E

    acc, deno, e_max = _dense_prefill_inner(
        acc,
        deno,
        e_max,
        q,
        qpe,
        K,
        V,
        cur_seq_kv_start,
        cur_kv_head,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        offs_d,
        offs_dpe,
        offs_dv,
        q_pos,
        mask_m,
        mask_d,
        mask_dv,
        0,
        n_full,
        kv_end,
        qk_scale,
        logit_cap,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK_N,
        MASKED=False,
        IS_CAUSAL=IS_CAUSAL,
        EVEN_D=EVEN_D,
        USE_FP8=USE_FP8,
        LOG2_FP8_MAX=LOG2_FP8_MAX,
    )
    acc, deno, e_max = _dense_prefill_inner(
        acc,
        deno,
        e_max,
        q,
        qpe,
        K,
        V,
        cur_seq_kv_start,
        cur_kv_head,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        offs_d,
        offs_dpe,
        offs_dv,
        q_pos,
        mask_m,
        mask_d,
        mask_dv,
        n_full,
        kv_end,
        kv_end,
        qk_scale,
        logit_cap,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK_N,
        MASKED=True,
        IS_CAUSAL=IS_CAUSAL,
        EVEN_D=EVEN_D,
        USE_FP8=USE_FP8,
        LOG2_FP8_MAX=LOG2_FP8_MAX,
    )

    no_kv = deno == 0.0

    if STORE_LSE:
        offs_lse = (
            cur_seq_q_start + cur_block_m * BLOCK_M + offs_m
        ) * stride_lse_bs + cur_head * stride_lse_h
        # e_max is in log2 units because qk carries the folded LOG2E. Under
        # FP8 deno also carries the exponent-bias lift; undoing it in log space
        # is one more constant on the same term.
        if USE_FP8:
            lse = tl.log(deno) + (e_max - LOG2_FP8_MAX) * LN2
        else:
            lse = tl.log(deno) + e_max * LN2
        lse = tl.where(no_kv, float("-inf"), lse)
        tl.store(Lse + offs_lse, lse, mask=mask_m)

    offs_o = (
        (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    # The FP8 lift applied to P divides out between acc and deno, so v_scale
    # is the only surviving factor.
    out_scale = v_scale
    deno_safe = tl.where(no_kv, 1.0, deno)
    tl.store(
        O + offs_o,
        acc * (out_scale / deno_safe[:, None]),
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def can_use_dense_prefill_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    logit_cap: float,
) -> bool:
    """Whether q/k/v may be cast to FP8 ahead of ``dense_prefill_attention_fwd``.

    Deliberately as narrow as the zero-prefix gate in ``extend_attention_fwd``:
    gfx950, BF16 dense inputs, plain causal softmax. Callers are responsible
    for having already rejected custom masks, sinks, SWA and score mods.
    """
    return (
        _is_gfx95
        and envs.SGLANG_TRITON_FP8_PREFILL_ATTN.get()
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and is_causal
        and logit_cap <= 0
    )


def dense_prefill_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    max_len_q: int,
    sm_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    logit_cap: float = 0.0,
    is_causal: bool = True,
    lse: Optional[torch.Tensor] = None,
) -> None:
    """Dense varlen prefill over a fully materialized K/V.

    q/o are addressed by ``qo_indptr``; k/v by ``kv_indptr``, whose
    per-sequence length must be >= the query length. The excess leading rows
    are the cached prefix, which every query of that sequence attends. Writes
    ``o``, and ``lse`` (natural log) when provided.

    Shapes::

        q : [sum(q_len),  H,    Lq]      o : [sum(q_len), H, Lv]
        k : [sum(kv_len), H_kv, Lq]      v : [sum(kv_len), H_kv, Lv]
    """
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk, f"q/k head dims must match, got {Lq} vs {Lk}"
    assert k.shape[0] == v.shape[0], (
        f"k/v token counts must match, got {k.shape[0]} vs {v.shape[0]}"
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size = qo_indptr.shape[0] - 1
    head_num = q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    # Shares the extend-attention tables: same head dims, same archs, and the
    # 192/128 K3 entry is the shape this path exists to serve.
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )
    assert BLOCK_DMODEL + BLOCK_DPE >= Lq, (
        f"tile {BLOCK_DMODEL}+{BLOCK_DPE} cannot cover head dim {Lq}"
    )

    # Both sides of every tl.dot must share a dtype, so a mixed bf16-q/fp8-k
    # pair does not compile. That pair is reachable: on gfx95 with MXFP4
    # kv_b_proj weights, forward_mha_rocm fuses the up-projection with the FP8
    # cast (fused_gemm_afp4wfp4_split_cat) and hands back k/v already in e4m3
    # while q is still bf16. Follow the cheap direction -- promote q rather
    # than upcast the far larger k/v -- and keep the P scaling consistent with
    # it. p is cast to v.dtype, so v decides the underflow lift too.
    fp8 = torch.float8_e4m3fn
    if fp8 in (q.dtype, k.dtype, v.dtype):
        q, k, v = q.to(fp8), k.to(fp8), v.to(fp8)
    else:
        assert q.dtype == k.dtype == v.dtype, (
            f"q/k/v dtypes must match, got {q.dtype}/{k.dtype}/{v.dtype}"
        )
    use_fp8 = q.dtype == fp8
    fp8_max = torch.finfo(fp8).max if use_fp8 else 1.0

    store_lse = lse is not None
    stride_lse_bs = lse.stride(0) if store_lse else 0
    stride_lse_h = lse.stride(1) if store_lse else 0

    extra_kargs = {}
    if _is_hip:
        # No kpack: gfx950 overwrites it to 1 and warns on every launch.
        # matrix_instr_nonkdim=32 was measured and is ~1.4x slower here.
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16}

    num_stages = _get_num_stages_for_extend_attention(Lq, Lv, BLOCK_N)
    if _is_gfx95 and Lq == 192 and Lv == 128 and use_fp8:
        # The shared table is tuned for the two-stage absorbed kernel; this
        # sweep is mask-free over its interior and wants a wider KV tile.
        # FP8 only: at BLOCK_N=128 the BF16 K/V tiles need 168 KB of LDS
        # against a 160 KB limit, so BF16 stays on the narrow tile below.
        # Swept BLOCK_M x BLOCK_N x warps x stages over the three shapes this
        # path actually sees (FP8, ms):
        #
        #                    q16384/p65536   q8192/p32768   q2464/p65536
        #   128/64  w4 st3       7.89            4.00           1.28
        #   128/128 w4 st2       7.12            3.56           1.18
        #   256/128 w4 st2       6.91            3.50           2.08
        #
        # BLOCK_M=256 edges ahead on full chunks but halves the M-block count,
        # which starves the 256 CUs on the short trailing chunk of a request --
        # 1.6x slower there. 128/128 wins everywhere. At BLOCK_N=128 the loop
        # body already covers the load latency, so the deeper pipeline that
        # helped at BLOCK_N=64 no longer pays (7.12 at 2 stages vs 7.24 at 3).
        BLOCK_N = 128
        num_warps = 4
        num_stages = 2
    elif _is_gfx95 and Lq == 192 and Lv == 128:
        # BF16, stuck on BLOCK_N=64 by LDS: there the mask-free interior does
        # pipeline deeper than the table's two stages (10.90ms vs 11.10ms at
        # 16K queries over a 64K prefix). 4 stages does not compile (Triton
        # asserts in its pipeliner).
        num_stages = 3

    num_blocks_m = triton.cdiv(max_len_q, BLOCK_M)
    grid = (batch_size, head_num, num_blocks_m)

    _fwd_kernel_dense_prefill[grid](
        q,
        k,
        v,
        o,
        lse,
        qo_indptr,
        kv_indptr,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        stride_lse_bs,
        stride_lse_h,
        kv_group_num=kv_group_num,
        logit_cap=logit_cap,
        Lq=Lq,
        Lv=Lv,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        STORE_LSE=store_lse,
        USE_FP8=use_fp8,
        LOG2_FP8_MAX=math.log2(fp8_max),
        EVEN_D=(BLOCK_DMODEL + BLOCK_DPE == Lq and BLOCK_DV == Lv),
        NUM_BLOCKS_M=num_blocks_m,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
