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
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging
import math
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.score_mod import unpack_aux_tensors
from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_core_count,
    is_gfx95_supported,
    is_gfx1250_supported,
    is_hip,
)

_is_hip = is_hip()
_is_gfx1250 = _is_hip and is_gfx1250_supported()

logger = logging.getLogger(__name__)


_MIN_BLOCK_KV = 32

# heads per stage-1 tile, shared so the budget's head_tiles cannot drift from the launch
_GROUPED_BLOCK_H = 16


# gfx950 wants 32 where the HIP path otherwise takes 16. That is the model it was picked
# against, not something a sweep isolated: at 16 the first dot is a single 16x16 MFMA
# tile, so the warps only have K=576 to split along and pay a cross-warp reduction every
# KV step, where 32 gives two of them an N tile each. 64 was timed at the batches the
# 4-warp bucket covers and never came out ahead: 3-5% behind at batch 1-3, noise at 4-5.
_MLA_BLOCK_N = 32


class _MlaBucket(NamedTuple):
    """Stage-1 geometry for a batch range. ``batch_max=None`` is the catch-all."""

    num_warps: int
    num_stages: int
    max_splits: int
    batch_max: Optional[int] = None


# gfx950 MLA decode, from a split-count sweep at every captured batch size,
# head_tiles == 1, 68k context (K3 at tp 8). max_splits is where more splits stopped
# paying at small batch, and dividing by batch * head_tiles keeps a smaller tp sane,
# though tuned at tp 8.
_MLA_BUCKETS = (
    _MlaBucket(num_warps=4, num_stages=2, max_splits=112, batch_max=5),
    _MlaBucket(num_warps=2, num_stages=2, max_splits=256, batch_max=24),
    _MlaBucket(num_warps=1, num_stages=1, max_splits=256),
)

# For the paths that must not depend on the batch; the mid bucket sits between the
# other two geometries. Retuning it moves what deterministic inference produces, which
# test_batch_free_geometry_is_pinned guards. max_splits goes unused there.
_MLA_BUCKET_BATCH_FREE = _MLA_BUCKETS[1]

_KEEP_SCHEDULER_SPLITS = None
_CORE_COUNT = {}
_LOGGED_TUNE = False


def _keep_scheduler_splits() -> bool:
    """Whether the caller asked for a specific per-sequence num_kv_splits.

    ``--enable-deterministic-inference`` derives it from a fixed tile size so a
    request's reduction tree cannot depend on its batch mates; a batch-wide count puts
    that back. An explicit tile size or the static-splits env asks for the same thing.
    """
    global _KEEP_SCHEDULER_SPLITS
    if _KEEP_SCHEDULER_SPLITS is None:
        from sglang.srt.runtime_context import get_exec

        try:
            exec_cfg = get_exec()
        except ValueError:
            return False  # not published yet, ask again on the next call
        _KEEP_SCHEDULER_SPLITS = bool(
            exec_cfg.deterministic.enable_deterministic_inference
            or exec_cfg.kernel.triton_attention_split_tile_size
            or envs.SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS.get()
        )
        if _KEEP_SCHEDULER_SPLITS:
            logger.info("MLA decode: keeping the scheduler's num_kv_splits")
    return _KEEP_SCHEDULER_SPLITS


def _grouped_head_tiles(head_num: int, kv_group_num: int) -> int:
    """Stage-1's grid extent along heads."""
    return triton.cdiv(head_num, min(_GROUPED_BLOCK_H, kv_group_num))


def _mla_bucket(batch: int) -> _MlaBucket:
    for bucket in _MLA_BUCKETS[:-1]:
        if batch <= bucket.batch_max:
            return bucket
    return _MLA_BUCKETS[-1]


def _mla_split_budget(num_warps: int, core_count: int) -> int:
    # about one wave of stage-1 workgroups, taking 4 warps to get one per CU and
    # halving the warps to double how many fit. core_count, not a whole MI355X: a CPX
    # partition exposes 32 of the 256
    return core_count * 4 // num_warps


def _mla_core_count(device_index: Optional[int]) -> int:
    count = _CORE_COUNT.get(device_index)
    if count is None:
        count = get_device_core_count(device_index if device_index is not None else 0)
        _CORE_COUNT[device_index] = count
    return count


def _mla_kv_splits(
    batch: int, head_tiles: int, max_kv_splits: int, core_count: int
) -> int:
    """Batch-wide split count for stage-1, or 0 with no device to size it against.

    The budget is a ceiling, not a rounding target: crossing it costs a step, not a
    proportional slice (batch 24, 68k: 21 splits / 504 blocks 358 us, 22 splits /
    528 blocks 528 us). Below it the count stays exact, since each split
    shortens the KV every workgroup walks (batch 136: 7 splits 1628 us, 4 at 2734 us).
    """
    if core_count <= 0:
        return 0
    bucket = _mla_bucket(batch)
    budget = _mla_split_budget(bucket.num_warps, core_count)
    splits = min(max_kv_splits, bucket.max_splits, budget // max(1, batch * head_tiles))
    return max(1, splits)


def _mla_tuning_applies(has_mla: bool, head_dim: int) -> bool:
    # both gates matter: tuned on gfx950 and on Lk=576. Cheapest term first since this
    # runs per layer per decode step, and the env read stays uncached so a test
    # override lands
    return (
        _is_hip
        and has_mla
        and head_dim == 576
        and is_gfx95_supported()
        and envs.SGLANG_MLA_DECODE_TUNE.get()
    )


def _mla_launch_plan(
    q, k_buffer, max_kv_splits: int, has_mla: bool
) -> Tuple[bool, int]:
    """``(take the tuned geometry, batch-wide split count)`` for one decode call.

    Both launches get one decision: stage-2 must merge exactly as many partials as
    stage-1 wrote and a mismatch is silent, so neither the count nor the gate is
    re-derived per launcher. 0 leaves both stages on the scheduler's per-sequence
    counts, their default.
    """
    if not _mla_tuning_applies(has_mla, k_buffer.shape[-1]):
        return False, 0
    if _keep_scheduler_splits():
        return True, 0
    head_num = q.shape[1]
    head_tiles = _grouped_head_tiles(head_num, head_num // k_buffer.shape[-2])
    splits = _mla_kv_splits(
        q.shape[0], head_tiles, max_kv_splits, _mla_core_count(q.device.index)
    )

    global _LOGGED_TUNE
    if splits and not _LOGGED_TUNE:
        _LOGGED_TUNE = True
        logger.info(
            "MLA decode: gfx950 tuned stage-1 geometry, replacing the scheduler's "
            "num_kv_splits and capped by --triton-attention-num-kv-splits "
            "(SGLANG_MLA_DECODE_TUNE=0 to disable)"
        )
    return True, splits


def _extract_kv_strides(buf, page_size: int):
    """Extract (slot_stride, head_stride, page_stride, tok_stride) for a 3-D
    ``[max_slots, head_num, head_dim]`` KV buffer.

    Returns a 4-tuple of ints suitable for passing as ``stride_buf_*bs``,
    ``stride_buf_*h``, ``stride_buf_*page``, ``stride_buf_*tok``.
    """
    if buf.ndim != 3:
        raise ValueError(f"unexpected KV buffer ndim={buf.ndim}, shape={buf.shape}")
    slot_stride = buf.stride(0)
    head_stride = buf.stride(1)
    page_stride = slot_stride * page_size
    tok_stride = slot_stride
    return slot_stride, head_stride, page_stride, tok_stride


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1). For
    # PAGE_SIZE == 1 the address math degenerates and these are unused
    # (Triton specializes the dead branch away at compile time).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Page-aware KV address math. At PAGE_SIZE==1 (legacy
            # / non-shared / shared-at-ps=1), Triton specializes the
            # else-branch away and the SASS is byte-identical to today.
            if PAGE_SIZE == 1:
                offs_buf_k = (
                    kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[None, :]
                )
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[:, None] * stride_buf_kpage
                    + tok_in_p[:, None] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[None, :]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    cur_batch_seq_len - 1,
                    offs_n,
                    cur_batch,
                    cur_head,
                    offs_n < split_kv_end,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    kv_loc[:, None] * stride_buf_vbs
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
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    kv_head_num = k_buffer.shape[-2]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // kv_head_num

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        PAGE_SIZE=page_size,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    IS_GFX1250: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
    forced_kv_splits=0,
    USE_FORCED: tl.constexpr = False,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    # runtime, not constexpr: it only feeds the kv_len_per_split arithmetic below, so
    # a constexpr buys nothing and costs one stage-1 variant per cuda-graph ladder
    # rung (stage-2 does need it at compile time). Any count covers any length since
    # kv_len_per_split rounds cdiv(L, S) up; short sequences leave the tail empty.
    if USE_FORCED:
        kv_splits = forced_kv_splits
    else:
        kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # Hoist loop-invariant base offsets
    base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
    if BLOCK_DPE > 0:
        base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
    if not HAS_MLA:
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        # gfx1250: triton tl.dot(fp8, fp8) returns garbage (~1e34+) for contraction
        # dim K>=128 (verified K=64 ok, K>=128 broken; bf16 fine at all K). The MLA
        # nope QK dot has K=512, so an fp8 KV cache MUST NOT be consumed as an fp8 dot
        # here: keep q in bf16 and upcast the fp8 K to bf16 for the dot. No-op for a
        # bf16 cache. (Do NOT "optimize" this back to q.to(fp8) on gfx1250.)
        # On all other platforms keep the original downcast of q to the KV dtype.
        # TODO: remove this branch once the gfx1250 fp8 tl.dot issue is resolved.
        if IS_GFX1250:
            q_k = q
        else:
            q_k = q.to(K_Buffer.dtype.element_ty)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Page-aware KV address math (see _fwd_kernel_stage1).
            if PAGE_SIZE == 1:
                offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + base_offs_k
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            if IS_GFX1250:
                qk = tl.dot(q_k, k.to(q_k.dtype))
            else:
                qk = tl.dot(q_k, k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                else:
                    offs_buf_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + base_offs_kpe
                    )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    cur_batch_seq_len - 1,
                    offs_n[None, :],
                    cur_batch,
                    cur_head[:, None],
                    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )
            if HAS_MLA:
                v = tl.trans(k)
            else:
                if PAGE_SIZE == 1:
                    offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                else:
                    offs_buf_v = (
                        page_id[:, None] * stride_buf_vpage
                        + tok_in_p[:, None] * stride_buf_vtok
                        + base_offs_v
                    )
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # Keep the softmax weights p in fp32 for the P·V dot (do NOT downcast p to
            # bf16) on gfx1250. The bf16 downcast of p was the accuracy loss vs a torch
            # fp32 SDPA reference (recovers gfx1250 R1 GSM8K ~0.82 -> ~0.92 with
            # attention idealized). On other platforms restore the p.to(v.dtype) cast.
            # TODO: remove this branch once the gfx1250 bf16 P·V issue is resolved.
            if IS_GFX1250:
                acc += tl.dot(p, v.to(tl.float32), out_dtype=tl.float32)
            else:
                acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    tune_mla: bool = False,
    forced_kv_splits: int = 0,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    # 4-D view exposes head_num at dim 2; legacy 3-D exposes
    # it at dim 1.
    kv_head_num = k_buffer.shape[-2]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // kv_head_num

    BLOCK_H = _GROUPED_BLOCK_H
    MAX_KV_SPLITS = max_kv_splits
    head_tiles = _grouped_head_tiles(head_num, kv_group_num)

    extra_kargs = {}
    num_stages = 2
    num_warps = 4
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    if tune_mla:
        # num_warps reorders the fp32 accumulation, so whoever declined the batch-wide
        # count gets a batch-independent geometry too
        bucket = _mla_bucket(batch) if forced_kv_splits else _MLA_BUCKET_BATCH_FREE
        BLOCK, num_warps, num_stages = (
            _MLA_BLOCK_N,
            bucket.num_warps,
            bucket.num_stages,
        )

    # Blocks at or above the split count return immediately, so the grid shrinks too.
    grid = (batch, head_tiles, forced_kv_splits or MAX_KV_SPLITS)

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        HAS_MLA=has_mla,
        USE_PDL=use_pdl,
        IS_GFX1250=_is_gfx1250,
        PAGE_SIZE=page_size,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        forced_kv_splits=forced_kv_splits,
        USE_FORCED=forced_kv_splits > 0,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    v_scale,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    FORCED_KV_SPLITS: tl.constexpr = 0,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    # Same count stage-1 used, or the two disagree about where split i starts. SPLIT_END
    # is a constexpr in both branches: a dynamic bound would merge the same partials
    # (stage-1 leaves the surplus splits masked out) but stops the unrolling, and
    # reassociating the fp32 reduction moves the result a few ULP off stock.
    if FORCED_KV_SPLITS > 0:
        kv_splits = FORCED_KV_SPLITS
        SPLIT_END: tl.constexpr = FORCED_KV_SPLITS
    else:
        kv_splits = tl.load(num_kv_splits + cur_batch)
        SPLIT_END: tl.constexpr = MAX_KV_SPLITS

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in tl.range(0, SPLIT_END, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_scale,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
    use_pdl=False,
    forced_kv_splits: int = 0,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        USE_PDL=use_pdl,
        FORCED_KV_SPLITS=forced_kv_splits,
        num_warps=4,
        num_stages=2,
        **({"launch_pdl": True} if use_pdl else {}),
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
        page_size=page_size,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
    tune_mla, forced_kv_splits = _mla_launch_plan(q, k_buffer, max_kv_splits, has_mla)
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
        has_mla=has_mla,
        use_pdl=use_pdl,
        page_size=page_size,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        tune_mla=tune_mla,
        forced_kv_splits=forced_kv_splits,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        use_pdl=use_pdl,
        forced_kv_splits=forced_kv_splits,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    enable_lean=None,
    lean_Mp=None,
    lean_Lp=None,
    lean_Op=None,
    lean_locks=None,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    # head_num lives at dim 1 (3-D) or dim 2 (4-D shared view).
    kv_head_num = v_buffer.shape[-2]
    kv_group_num = q.shape[1] // kv_head_num

    # Work-Centric (Lean) Attention: a persistent-CTA + work-stealing decode kernel
    # that helps on long sequences where there are many more KV tiles than CUs. The
    # persistent grid is fixed to the device CU count and the kernel derives its own tile
    # schedule from kv_indptr on-device, so this path involves no host sync and is safe to
    # capture in a CUDA graph. Whether Lean pays off for a given shape is decided cheaply by
    # the backend's host-side seqlen gate (lean_decode_seqlen_gate) before we get here.
    # Lean handles both page sizes: the kernel does page-aware address math.
    # ROCm/AMD only: Lean is validated on MI300X/MI355X; CUDA/NVIDIA uses the standard kernel.
    if (
        _is_hip
        and _lean_head_dim_ok(k_buffer.shape[-1], v_buffer.shape[-1])
        and _should_use_lean_decode(
            enable_lean, logit_cap, sinks, xai_temperature_len, score_mod
        )
    ):
        total_programs, XCD_REMAP, NUM_XCDS = _lean_decode_launch_params(
            v_buffer.shape[-2], kv_group_num
        )
        _decode_lean_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            total_programs,
            # Fold k_scale into sm_scale and pass v_scale, exactly as the standard grouped
            # kernel does, so Lean dequantizes fp8 KV consistently (both are 1.0 for bf16/fp16).
            sm_scale * k_scale,
            v_scale,
            XCD_REMAP,
            NUM_XCDS,
            lean_Mp,
            lean_Lp,
            lean_Op,
            lean_locks,
            page_size=page_size,
        )
        return

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            has_mla=has_mla,
            use_pdl=use_pdl,
            page_size=page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )


# ============================================================================
# Work-Centric (Lean) Attention: persistent-CTA + work-stealing decode kernel.
# ============================================================================

_LEAN_BLOCK_M = 16

_NUM_CU = None


def _lean_head_dim_ok(qk_head_dim: int, v_head_dim: int) -> bool:
    """Whether the Lean decode kernel's tiles fit in shared memory for this head dim.

    The non-MLA kernel sets ``BLOCK_DMODEL = next_power_of_2(qk_head_dim)``; at head_dim 256
    (e.g. Gemma-2/3) the K/V tiles overflow the 160 KB LDS budget and the launch raises
    OutOfResources. head_dim <= 128 fits. MLA's rope-split dims (288/576) are special-cased in
    the kernel into a smaller-tiled path and are handled separately. This guard makes the Lean
    dispatch fall back safely instead of crashing, even under an explicit ``enable_lean=True``.
    """
    if qk_head_dim in (288, 576):  # MLA rope-split, special-cased in the kernel
        return True
    return qk_head_dim <= 128 and v_head_dim <= 128


def _lean_num_cus() -> int:
    """Number of compute units on the current device (cached).

    Lean Attention sizes its persistent grid to the hardware CU count so work-stealing can
    fill the GPU. Falls back to 304 (MI300X) if the device cannot be queried.
    """
    global _NUM_CU
    if _NUM_CU is None:
        try:
            _NUM_CU = torch.cuda.get_device_properties(0).multi_processor_count
        except Exception:
            _NUM_CU = 304
    return _NUM_CU


def _lean_decode_block_n(Lk: int) -> int:
    """KV block size for the Lean decode kernel.

    Large head dims (MLA, Lk in {288, 576}) use a small KV block to bound LDS/register
    usage; standard head dims use a large block since decode is memory-bound. The value
    must be identical everywhere it is used so the tile schedule stays consistent.
    """
    if not _is_hip:
        return 64
    return 16 if Lk > 256 else 128


@triton.jit
def remap_xcd(pid, GRID_MN: tl.constexpr, NUM_XCDS: tl.constexpr = 8):
    """Remap program ID across XCDs for AMD MI300X."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid, pids_per_xcd


@triton.jit
def cal_num_split_wgs(
    xcd_pid: tl.int32,
    tile_iter_end: tl.int32,
    cta_end_tile_gid: tl.int32,
    max_tiles_per_wg: tl.int32,
    high_load_wgs: tl.int32,
    num_splits: tl.int32,
):
    zero_i = tl.full((), 0, dtype=tl.int32)
    start_cta = tl.cast(xcd_pid + 1, tl.int32)
    remaining = tl.maximum(tl.cast(tile_iter_end - cta_end_tile_gid, tl.int32), zero_i)
    cap_high = tl.cast(max_tiles_per_wg, tl.int32)
    cap_low = tl.cast(max_tiles_per_wg - 1, tl.int32)
    cap_low = tl.where(cap_low > 0, cap_low, tl.full((), 1, dtype=tl.int32))
    ctas_high_avail = tl.maximum(tl.cast(high_load_wgs, tl.int32) - start_cta, zero_i)
    total_high_capacity = ctas_high_avail * cap_high
    need_high_only = (remaining + cap_high - 1) // cap_high
    rem_after_high = tl.maximum(remaining - total_high_capacity, zero_i)
    need_low_after_high = (rem_after_high + cap_low - 1) // cap_low
    ctas_needed = tl.where(
        remaining <= total_high_capacity,
        need_high_only,
        ctas_high_avail + need_low_after_high,
    )
    max_ctas_allowed = tl.maximum(tl.cast(num_splits - 1, tl.int32), zero_i)
    ctas_to_use = tl.minimum(ctas_needed, max_ctas_allowed)
    k = ctas_to_use
    cap_by_k = tl.where(
        k <= ctas_high_avail,
        k * cap_high,
        total_high_capacity + (k - ctas_high_avail) * cap_low,
    )
    last_cta = start_cta + ctas_to_use
    last_cta = tl.where(ctas_to_use == 0, start_cta - 1, last_cta)
    return last_cta


@triton.jit
def _lean_attention_decode_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    Mp,  # Partial max
    Lp,  # Partial sum
    Op,  # Partial output
    O,  # Final output
    batch_num_block_n,
    locks,
    kv_indptr,
    kv_indices,
    sm_scale,
    v_scale,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vbs,
    stride_buf_vh,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    NUM_HEAD_BLOCKS: tl.constexpr,
    ROWS_PER_XCD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    batch_size: tl.constexpr,
    total_programs: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_rows: tl.constexpr,
    xcd_programs: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """Lean Attention decode kernel - persistent CTA with work stealing.

    The tile schedule (``tiles_per_khead``, ``max_tiles_per_wg``, ``high_load_wgs``,
    ``num_splits``) is computed here on-device from ``kv_indptr`` rather than passed in from
    the host. This keeps the launch free of any host sync (so it is CUDA-graph capturable)
    and lets the schedule adapt to the per-replay sequence length: ``total_programs`` is a
    fixed persistent grid and the work simply re-distributes when the KV length changes.
    """
    current_pid = tl.program_id(0)

    # On-device tile schedule (mirrors the former host-side la_get_num_splits). Reads only
    # GPU state so it is safe under CUDA-graph capture. tiles_per_khead is the number of KV
    # tiles in one row summed over the batch: it MUST match batch_num_block_n (the per-batch
    # cumulative tile count) exactly, so it is read from that array's last entry rather than
    # recomputed as ceil(total_tokens / BLOCK_N) -- those differ whenever a sequence length
    # is not a multiple of BLOCK_N (the common case for a ragged decode batch), which would
    # desync the row<->tile mapping below.
    tiles_per_khead = tl.load(batch_num_block_n + batch_size - 1)
    # Effective rows per XCD (constexpr-folded); total tiles distributed over this XCD.
    eff_rows: tl.constexpr = num_rows // NUM_XCDS if XCD_REMAP else num_rows
    total_tiles = tiles_per_khead * eff_rows
    max_tiles_per_wg = (total_tiles + xcd_programs - 1) // xcd_programs
    max_tiles_per_wg = tl.maximum(max_tiles_per_wg, 1)
    high_load_wgs = total_tiles - (max_tiles_per_wg - 1) * xcd_programs
    # Safe over-estimate of the split count: a row spans at most ceil(tiles/(mtpw-1))+1
    # CTAs; the guarded divisor also covers the max_tiles_per_wg == 1 case.
    split_denom = tl.maximum(max_tiles_per_wg - 1, 1)
    num_splits = 1 + (tiles_per_khead + split_denom - 1) // split_denom

    if XCD_REMAP:
        current_pid, pids_per_xcd = remap_xcd(
            current_pid, GRID_MN=total_programs, NUM_XCDS=NUM_XCDS
        )
        xcd_pid = current_pid % pids_per_xcd
        xcd_id = current_pid // pids_per_xcd
    else:
        xcd_pid = current_pid
        xcd_id = 0
        pids_per_xcd = total_programs

    if xcd_pid < high_load_wgs:
        iter = max_tiles_per_wg * xcd_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            xcd_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    # Use a regular while loop instead of tl.static_range with a dynamic bound to avoid
    # Triton compiler crashes in the Coalesce pass (max_output_tile_cnt is runtime-computed).
    while iter < cta_end_tile_gid:
        tile_row_idx = iter // tiles_per_khead
        tile_idx = tile_row_idx * batch_size
        tile_iter = tile_row_idx * tiles_per_khead

        if batch_size == 1:
            req_size = tl.full((), tiles_per_khead, dtype=tl.int32)
        else:
            req_size = tl.cast(tl.load(batch_num_block_n), tl.int32)
        tile_iter_end = tile_iter + req_size

        for b in range(1, batch_size):
            next_req_size = tl.load(batch_num_block_n + b)
            local_head_iter = iter % tiles_per_khead
            if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                tile_iter = tile_iter + req_size
                tile_idx = tile_idx + b
                tile_iter_end = tile_iter + (next_req_size - req_size)
            req_size = next_req_size

        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
        host_block = iter == tile_iter
        finishing_block = cta_end_tile_gid >= tile_iter_end

        # A tiling "row" is a (kv_head, head_block) pair. For MHA/GQA NUM_HEAD_BLOCKS == 1
        # so a row is just a kv head. For MLA, kv_group_num > BLOCK_M, so a kv head spans
        # NUM_HEAD_BLOCKS head blocks of BLOCK_M query heads each.
        tile_row_idx_global = ROWS_PER_XCD * xcd_id + tile_row_idx
        cur_kv_head = tile_row_idx_global // NUM_HEAD_BLOCKS
        head_block_idx = tile_row_idx_global % NUM_HEAD_BLOCKS
        group_start = cur_kv_head * kv_group_num
        q_head_base = group_start + head_block_idx * BLOCK_M
        tile_batch_idx = tile_idx % batch_size
        cur_batch = tile_batch_idx

        cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
        cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

        # SGLang-style offsets
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < Lk
        mask_dv = offs_dv < Lv

        # Query head block: this row covers BLOCK_M query heads of its kv group, bounded
        # by the group end (group_start + kv_group_num) and the total head count.
        offs_h = q_head_base + tl.arange(0, BLOCK_M)
        mask_h = offs_h < (group_start + kv_group_num)
        mask_h = mask_h & (offs_h < num_query_heads)

        off_q = cur_batch * stride_qbs + offs_h[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(
            Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0
        )  # [BLOCK_M, BLOCK_DMODEL]
        # Cast q to the K buffer dtype so the main dot is a same-dtype MMA. For fp8 KV this
        # makes it dot(fp8, fp8) (triton rejects a bf16xfp8 mix); k_scale is folded into
        # sm_scale to dequantize. For bf16/fp16 KV this is a no-op. Mirrors the standard kernel.
        q_k = q.to(K_Buffer.dtype.element_ty)

        # MLA rope split: the positional-encoding dims live in [BLOCK_DMODEL, Lk).
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < Lk
            off_qpe = (
                cur_batch * stride_qbs + offs_h[:, None] * stride_qh + offs_dpe[None, :]
            )
            qpe = tl.load(
                Q + off_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0
            )

        e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

        local_iter_ptr = local_iter * BLOCK_N
        local_iter_end_ptr = local_iter_end * BLOCK_N
        # Effective token bound: the last tile of a sequence whose length is not a multiple
        # of BLOCK_N is only partially valid. Clamp to cur_batch_seq_len so the KV-index /
        # K / V loads never read past this batch's tokens -- for the final batch that would
        # otherwise run off the end of kv_indices and fault the GPU. For BLOCK_N-aligned
        # sequences this equals local_iter_end_ptr, so the aligned path is unchanged.
        tok_end = tl.minimum(local_iter_end_ptr, cur_batch_seq_len)
        for start_n in range(local_iter_ptr, local_iter_end_ptr, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < tok_end,
                other=0,
            )

            # Load K transposed: [BLOCK_DMODEL, BLOCK_N] so qk = q @ k directly.
            # Page-aware KV address math (mirrors the standard grouped kernel): at
            # PAGE_SIZE==1 the slot index addresses directly; otherwise it splits
            # into (page_id, tok_in_p).
            if PAGE_SIZE == 1:
                offs_buf_k = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + offs_d[:, None]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < tok_end) & (mask_d[:, None]),
                other=0.0,
            )

            qk = tl.dot(q_k, k)  # [BLOCK_M, BLOCK_N]
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_buf_kpe = (
                        kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                else:
                    offs_buf_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < tok_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                # Dequantize the rope-split K to q's dtype for this small dot (matches standard).
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale  # sm_scale carries k_scale (folded by the caller)
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < tok_end),
                qk,
                float("-inf"),
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    kv_loc[:, None] * stride_buf_vbs
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
                mask=(offs_n[:, None] < tok_end) & (mask_dv[None, :]),
                other=0.0,
            )

            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)  # [BLOCK_M, BLOCK_DV]

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        if not host_block:
            mp_ptrs = Mp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            lp_ptrs = Lp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            op_ptrs = (
                Op
                + current_pid * BLOCK_M * BLOCK_DV
                + tl.arange(0, BLOCK_M)[:, None] * BLOCK_DV
                + offs_dv[None, :]
            )
            tl.store(mp_ptrs, e_max, cache_modifier=".wb")
            tl.store(lp_ptrs, e_sum, cache_modifier=".wb")
            tl.store(op_ptrs, acc, mask=mask_dv[None, :], cache_modifier=".wb")
            tl.debug_barrier()
            tl.atomic_xchg(locks + current_pid, 1)
        else:
            if not finishing_block:
                last_cta = cal_num_split_wgs(
                    xcd_pid=xcd_pid,
                    tile_iter_end=tile_iter_end,
                    cta_end_tile_gid=cta_end_tile_gid,
                    max_tiles_per_wg=max_tiles_per_wg,
                    high_load_wgs=high_load_wgs,
                    num_splits=num_splits,
                )
                # Defensive clamp: the partial-result buffers (Mp/Lp/Op/locks) hold one slot
                # per program, and a CTA only ever steals from later CTAs within its own XCD.
                # Clamp to pids_per_xcd so a degenerate schedule (e.g. a forced tiny shape
                # that slips past the host gate) can never index a buffer out of bounds.
                last_cta = tl.minimum(last_cta, pids_per_xcd)
                temp_pid = current_pid
                for cta in range((xcd_pid + 1), last_cta):
                    temp_pid = temp_pid + 1
                    while tl.atomic_cas(locks + temp_pid, 1, 1) != 1:
                        pass
                    mp_ptrs = Mp + temp_pid * BLOCK_M + tl.arange(0, BLOCK_M)
                    lp_ptrs = Lp + temp_pid * BLOCK_M + tl.arange(0, BLOCK_M)
                    op_ptrs = (
                        Op
                        + temp_pid * BLOCK_M * BLOCK_DV
                        + tl.arange(0, BLOCK_M)[:, None] * BLOCK_DV
                        + offs_dv[None, :]
                    )

                    m_cta = tl.load(mp_ptrs)
                    l_cta = tl.load(lp_ptrs)
                    acc_cta = tl.load(op_ptrs, mask=mask_dv[None, :])
                    m_new = tl.maximum(m_cta, e_max)
                    alpha = tl.exp(m_cta - m_new)
                    alpha1 = tl.exp(e_max - m_new)
                    l_new = alpha * l_cta + alpha1 * e_sum
                    acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                    e_max = m_new
                    e_sum = l_new

            # v_scale dequantizes the fp8 V contribution accumulated via dot(p, v); it is 1.0
            # for bf16/fp16 V. Applied once here at the single output-write site (mirrors the
            # standard kernel's `acc / e_sum * v_scale`).
            acc = acc / e_sum[:, None] * v_scale
            offs_o = (
                cur_batch * stride_obs + offs_h[:, None] * stride_oh + offs_dv[None, :]
            )
            tl.store(O + offs_o, acc, mask=mask_h[:, None] & mask_dv[None, :])

        iter = iter + (local_iter_end - local_iter)


def _lean_head_tiles(num_q_heads: int, kv_group_num: int) -> int:
    """Head-tile programs the standard grouped decode kernel launches per (sequence,
    kv-split): ``ceil(num_q_heads / min(16, kv_group_num))``. This is the standard
    kernel's query-head parallelism, which drives how well it already fills the device
    and hence where the Lean-vs-SplitK crossover sits (see :func:`lean_decode_seqlen_gate`).
    """
    block_h = min(16, max(1, kv_group_num))
    return -(-num_q_heads // block_h)  # ceil(num_q_heads / block_h)


def lean_capture_policy(
    num_q_heads: int,
    kv_group_num: int,
    batch: int,
    is_mla: bool = False,
) -> bool:
    """CUDA-graph capture-time bake decision for Lean decode.

    During decode-graph capture ``seq_lens`` are set to the fill value (1), so the
    seq-len based :func:`lean_decode_seqlen_gate` always sees ``avg_len == 1`` and returns
    ``False`` -- baking the *standard* kernel into every captured graph. Since the default
    (auto) path replays those captured graphs, keying the bake on ``seq_lens_sum`` makes
    Lean a no-op under CUDA graphs. But Lean's *fixed* 512-CTA persistent grid derives its
    work schedule on-device from ``kv_indptr`` read at replay (work-stealing), so a baked
    Lean graph still adapts to the real per-step raggedness. The bake decision therefore
    keys only on capture-time-known signals -- ``batch``, the standard kernel's head-tile
    parallelism (``tiles``), and ``is_mla`` -- calibrated to the realistic (ragged) regime.

    Thresholds from ``CALIBRATION.md`` (MI355X, triton 3.7.0, CUDA graphs on, seed 42):

    * MLA (``is_mla``): bake at ``batch >= 8``. ``b1`` is a catastrophic 0.40-0.55x loss
      (the ~128-query-head decode already saturates the CUs, so Lean's fixed grid is pure
      overhead), while ``b >= 8`` is uniform-parity and a 1.09-1.18x ragged win (p99 ITL
      1.24-1.99x lower). This replaces the former blanket ``is_mla -> off``, which was based
      on unrepresentative uniform ``b1`` data.
    * GQA/MHA (``tiles >= 4``): bake at ``batch >= 16`` -- the unconditional-win boundary at
      both 16K and 64K (up to 1.49x on ragged, 1.05-1.13x even on uniform). ``batch < 16``
      is context-split (``b1`` wins at 64K but ``b1-4`` lose at 16K) and cannot be decided
      from batch alone, so it is left to the eager :func:`lean_decode_seqlen_gate`.
    * Heavy TP shard (``tiles < 4``, e.g. Llama-70B @TP=8): never bake. Not calibrated for
      capture and known to regress ~4x at 32K; its rare long-context win still activates via
      the eager seq-len gate (131072 base).
    """
    if batch <= 0:
        return False
    if is_mla:
        return batch >= 8
    if _lean_head_tiles(num_q_heads, kv_group_num) < 4:
        return False
    return batch >= 16


def lean_decode_seqlen_gate(
    num_q_heads: int,
    kv_group_num: int,
    batch: int,
    seq_lens_sum: Optional[int],
    is_mla: bool = False,
) -> bool:
    """Cheap host-side pre-gate for Lean decode (no GPU sync).

    Lean Attention only beats the standard decode kernel for long-enough sequences; for
    short context it both loses and would pay a ``kv_indptr[batch].item()`` host-sync in
    :func:`decode_attention_fwd` just to discover it should fall back. The attention backend
    calls this first, using host-side metadata it already has (``num_q_heads``,
    ``kv_group_num``, ``seq_lens_sum``, ``batch``), so short-context decode skips Lean
    entirely without a sync.

    What actually drives the Lean-vs-SplitK crossover is how well the standard grouped
    kernel already fills the device, i.e. its query-head **parallelism**, not ``kv_group_num``.
    The standard kernel launches ``tiles = ceil(num_q_heads / min(16, kv_group_num))``
    head-tile programs per (sequence, kv-split); when ``tiles`` is large it saturates the CUs
    at short context and Lean wins only much later, while with few query heads per GPU (heavy
    tensor-parallel shards) it under-fills and Lean needs a long context to amortise its
    fixed persistent-grid overhead. Keying the threshold on ``kv_group_num`` alone mispredicts
    this badly: e.g. Llama-3-70B at TP=8 (8 query heads/GPU, ``kv_group_num`` still 8) is 4x
    SLOWER under Lean at 32K, yet the old gate enabled it there. So we tier the base threshold
    on ``tiles`` instead. Thresholds are the crossovers measured by ``benchmark/lean_gate_sweep.py``
    on MI355X (256 CUs); they should scale with the device CU count on other GPUs.

    MLA layers (``is_mla``, i.e. ``qk_head_dim != v_head_dim``) are gated on ``batch`` rather
    than average length: their Lean win is driven by batch raggedness (work-stealing across
    mixed-length requests), not context. Calibration shows ``b1`` loses hard (~0.40-0.55x: the
    ~128-query-head decode already saturates the CUs at batch 1, so Lean's fixed persistent grid
    is pure overhead) while ``b >= 8`` is uniform-parity and a ragged win, so MLA enables at
    ``batch >= 8`` above a small length floor. This replaces the former blanket ``is_mla -> off``
    (which was based on unrepresentative uniform ``b1`` data). In practice MLA models serve under
    CUDA graphs, where :func:`lean_capture_policy` -- not this seq-len gate -- makes the decision.

    The thresholds are set for the END-TO-END crossover, which is LATER than the isolated
    kernel crossover: Lean's decode kernel has a nearly flat per-call cost, so even after the
    standard kernel's attention becomes slower the *whole decode step* only turns over once
    the standard attention has grown enough to clear Lean's flat floor. Measured end-to-end on
    several GQA models, the crossover clusters at ~56-64K largely independent of the exact tile
    count: Qwen2.5-7B (tiles=4) 0.84x@32K, 1.11x@64K, 1.88x@128K; Llama-3.1-8B (tiles=8)
    1.06x@64K, 1.83x@128K; Ministral-8B (tiles=8) 0.86x@32K. So grouped decode uses a single
    64K base and only heavy tensor-parallel shards with very few query-head tiles (tiles<4,
    e.g. Llama-70B @TP=8, whose kernel crossover is already ~128K) push it to 128K. MHA (many
    tiles) uses a lower 16K base (kernel crossover ~8K).

    Thresholds relax as the batch grows, since more concurrent requests fill the persistent
    grid at shorter lengths, and Lean is never enabled below a floor of 4K average tokens,
    keeping the workload clear of the degenerate tiny-tile regime. The relaxation rate is
    tier-dependent, from a saturated batch sweep on MI355X (range-ratio 0.25 ragged, batch =
    concurrency, num_prompts>=6*batch):

    * ``tiles >= 4`` (GQA/MHA): divisor ``batch // 2``. Measured E2E (throughput / median ITL)
      confirms Lean wins well below the old ``batch // 4`` threshold once the batch fills the
      grid. Qwen2.5-7B (tiles=4) @ batch: b4 0.997x/0.97x (neutral -> keep off), b8 1.05x/1.26x,
      b12 1.13x/1.40x, b16 1.20x/1.84x, b32 1.28x/2.31x @ ~18.75K; and @ batch 16 it already
      wins by ~7.5K avg (1.13x/1.39x). Llama-3.1-8B (tiles=8) @ batch 16 wins at every context
      7.5K->30K (1.27-1.31x thrpt, 1.6-2.2x ITL). ``batch // 2`` enables from batch 8 @ ~18K
      and batch 16 @ ~8K while keeping batch 4 conservative (32K threshold, correctly off at
      18.75K where Lean is neutral).
    * ``tiles < 4`` (heavy TP shard): keeps the conservative ``batch // 4``. Its E2E win needs
      very long context (Llama-70B @TP=8 was 4x SLOWER at 32K); the isolated kernel can win at
      high batch/long context but that does not survive the MoE + TP-all-reduce full step, and
      a single-GPU microbench cannot replicate it, so this tier stays protected.
    """
    if batch <= 0:
        return False
    # No CPU length mirror (e.g. gpu-only batches, or the EAGLE draft runner, which leaves
    # seq_lens_sum unset): we cannot judge context length, so fall back to the standard kernel.
    if seq_lens_sum is None:
        return False
    avg_len = seq_lens_sum / batch
    if is_mla:
        # Gate on batch (raggedness proxy), not average length; b1 is a hard loss, b>=8
        # is parity-uniform / ragged-win. The 4K floor keeps degenerate tiny workloads off.
        return batch >= 8 and avg_len >= 4096
    tiles = _lean_head_tiles(num_q_heads, kv_group_num)
    if tiles >= 16:
        base = (
            16384  # MHA / many query heads: standard kernel fills late, Lean wins early
        )
    elif tiles >= 4:
        base = 65536  # typical GQA: measured E2E crossover ~56-64K
    else:
        base = (
            131072  # few query heads/GPU (heavy TP shard): Lean needs very long context
        )
    # Heavy TP shards (tiles<4) relax slowly (batch//4); GQA/MHA relax at batch//2, matching
    # the measured saturated-batch crossovers (see docstring).
    div = batch // 2 if tiles >= 4 else batch // 4
    threshold = max(4096, base // max(1, div))
    return avg_len >= threshold


def _should_use_lean_decode(
    enable_lean: Optional[bool],
    logit_cap: float,
    sinks,
    xai_temperature_len: int,
    score_mod,
) -> bool:
    """Decide whether the Work-Centric (Lean) Attention decode kernel may be used.

    ``enable_lean`` is the resolved activation flag passed by the caller:

    * ``False`` — never use Lean Attention.
    * ``True``  — use Lean Attention (the caller has already decided it is appropriate).
    * ``None``  — do NOT self-enable here. Lean is only beneficial for long sequences and
      its persistent-grid schedule misbehaves on tiny workloads, but this function has no
      cheap way to know the sequence length (reading it would force a host sync that breaks
      CUDA-graph capture). The attention backend resolves ``None`` to ``True``/``False`` via
      :func:`lean_decode_seqlen_gate` using host-side metadata before calling in, so a
      ``None`` that reaches here (e.g. a direct call) conservatively means "off".

    Regardless of the override, Lean Attention is only eligible when the request uses
    none of the features the kernel does not implement. The kernel supports MHA, GQA, and
    MLA (rope split), but ignores logit capping, attention sinks, xAI temperature scaling,
    and score modification, so we fall back to the standard kernel whenever any of those
    are requested rather than silently returning wrong results.
    """
    if not enable_lean:  # False or None
        return False
    if logit_cap and logit_cap > 0:
        return False
    if sinks is not None:
        return False
    if xai_temperature_len and xai_temperature_len > 0:
        return False
    if score_mod is not None:
        return False
    return True


def _lean_decode_launch_params(num_kv_heads, kv_group_num):
    """Lean decode launch parameters that depend only on shape (no seqlen, no sync).

    Returns ``(total_programs, XCD_REMAP, NUM_XCDS)``. ``total_programs`` is the fixed
    persistent-grid size (2× device CU count for better occupancy, rounded to a whole
    number of XCDs when the XCD remap is active). The per-call tile schedule is computed
    inside the kernel from ``kv_indptr``. Shared by :func:`decode_attention_fwd` and the
    test so the grid/XCD decision stays in sync with the kernel.
    """
    num_head_blocks = (kv_group_num + _LEAN_BLOCK_M - 1) // _LEAN_BLOCK_M
    # XCD remap for ROCm only when rows are one-per-kv-head and divisible by 8.
    XCD_REMAP = (num_kv_heads % 8 == 0 and num_head_blocks == 1) if _is_hip else False
    NUM_XCDS = 8 if XCD_REMAP else 1
    # Grid = round(CU_count * multiplier); multiplier defaults to 1.0 (one CTA per CU) and
    # is overridable via SGLANG_FORCE_LEAN_GRID_CU_MULT for grid A/B tuning without a rebuild.
    total_programs = max(
        1, round(_lean_num_cus() * envs.SGLANG_FORCE_LEAN_GRID_CU_MULT.get())
    )
    if XCD_REMAP:
        # The XCD remap requires the grid to be a whole number of XCDs.
        total_programs = max((total_programs // NUM_XCDS) * NUM_XCDS, NUM_XCDS)
    return total_programs, XCD_REMAP, NUM_XCDS


def _decode_lean_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    total_programs,
    sm_scale,  # already folded with k_scale by the caller (matches the standard kernel)
    v_scale,
    XCD_REMAP,
    NUM_XCDS,
    Mp,
    Lp,
    Op,
    locks,
    page_size=1,
):
    """Wrapper for Lean Attention kernel.

    ``total_programs`` is the fixed persistent-grid size (2× device CU count). The kernel
    derives its own tile schedule from ``kv_indptr`` on-device, so no host sync is needed and
    the launch is CUDA-graph capturable. ``Mp``, ``Lp``, ``Op``, ``locks`` are pre-allocated
    persistent-grid partial-result buffers reused across decode steps. ``page_size``
    selects the KV address math over the ``[N, head, dim]`` buffer (strides via
    ``_extract_kv_strides``).
    """
    batch, head_num = q.shape[0], q.shape[1]
    num_kv_heads = k_buffer.shape[-2]
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    kv_group_num = head_num // num_kv_heads

    # MLA rope split: K carries an extra positional-encoding block (Lk > Lv).
    if Lk == 576:
        BLOCK_DMODEL, BLOCK_DPE = 512, 64
    elif Lk == 288:
        BLOCK_DMODEL, BLOCK_DPE = 256, 32
    else:
        BLOCK_DMODEL, BLOCK_DPE = triton.next_power_of_2(Lk), 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    BLOCK_M = _LEAN_BLOCK_M
    BLOCK_N = _lean_decode_block_n(Lk)
    # A kv group wider than BLOCK_M is processed as several head blocks; each
    # (kv_head, head_block) pair is one scheduling "row".
    num_head_blocks = (kv_group_num + BLOCK_M - 1) // BLOCK_M
    num_rows = num_kv_heads * num_head_blocks
    rows_per_xcd = num_rows // NUM_XCDS if XCD_REMAP else num_rows
    xcd_programs = total_programs // NUM_XCDS if XCD_REMAP else total_programs

    # Pre-allocated persistent-grid partial-result buffers (Mp, Lp, Op, locks) are passed
    # in and reused across decode steps; they hold running softmax state for BLOCK_M query
    # heads (one head block of a kv group) per program. Reset locks to zero each call.
    locks.zero_()

    # Prepare batch_num_block_n (cumulative tiles per sequence) over the active batch.
    # seq_len[i] = kv_indptr[i+1] - kv_indptr[i]
    seq_lens = (kv_indptr[1 : batch + 1] - kv_indptr[:batch]).to(
        torch.int64
    )  # use int64 for safe arithmetic
    tiles_per_batch = (seq_lens + (BLOCK_N - 1)) // BLOCK_N
    batch_num_block_n = (
        torch.cumsum(tiles_per_batch, dim=0).to(torch.int32).contiguous()
    )

    max_output_tile_cnt = math.ceil((head_num * batch) / total_programs) + 4

    # Page-aware KV strides. For a 3-D buffer these synthesize page/tok strides so the
    # PAGE_SIZE>1 math collapses to the contiguous slot address; for a 4-D paged buffer they
    # come from the real page/token strides. (See _extract_kv_strides.)
    k_bs, k_h, k_page, k_tok = _extract_kv_strides(k_buffer, page_size)
    v_bs, v_h, v_page, v_tok = _extract_kv_strides(v_buffer, page_size)

    _lean_attention_decode_kernel[(total_programs,)](
        q,
        k_buffer,
        v_buffer,
        Mp,
        Lp,
        Op,
        o,
        batch_num_block_n,
        locks,
        kv_indptr,
        kv_indices,
        sm_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        k_bs,
        k_h,
        k_page,
        k_tok,
        v_bs,
        v_h,
        v_page,
        v_tok,
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        NUM_HEAD_BLOCKS=num_head_blocks,
        ROWS_PER_XCD=rows_per_xcd,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PAGE_SIZE=page_size,
        XCD_REMAP=XCD_REMAP,
        NUM_XCDS=NUM_XCDS,
        batch_size=batch,
        total_programs=total_programs,
        num_query_heads=head_num,
        num_rows=num_rows,
        xcd_programs=xcd_programs,
        max_output_tile_cnt=max_output_tile_cnt,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )
