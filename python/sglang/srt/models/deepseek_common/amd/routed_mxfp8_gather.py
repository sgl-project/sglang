"""Routed-row MXFP8 DP gather for the DP MoE on HIP.

Each rank routes its own rows and quantizes them to MXFP8 before the gather, so a
gathered row is [fp8 hidden | e8m0 group-32 scales | topk ids | topk weights]
(~5.3 KB at hidden 5120) instead of bf16 (10 KB). aiter fused_moe takes the fp8
rows with their scales as pre-quantized activations (its MXFP8 passthrough).

MAX_LEN decode (SGLANG_HIP_DP_ROUTED_MXFP8_GATHER): one packed all-gather. Pad rows
(from ``num_token_non_padded`` on) go out as zero bytes: zero weights, and a zero
hidden prefix that the dispatcher's pad-topk mask recognizes.

SUM_LEN prefill (SGLANG_HIP_DP_ROUTED_MXFP8_GATHER_PREFILL): one grouped all_gatherv
of the fp8 rows, their scales and [ids | weights] straight into the MoE inputs. A
rank's slot can be longer than its rows (buffer-aligned sizes); the tail rows go out
as zero hidden with zero weights.
"""

import logging
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    _note_dp_gather_in_prefill_graph,
    dp_slot_in,
    get_dp_global_num_tokens,
    is_dp_gatherv_active,
    world_dp_gather_enabled,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_ENABLED = is_hip() and envs.SGLANG_HIP_DP_ROUTED_MXFP8_GATHER.get()
_PREFILL_ENABLED = is_hip() and envs.SGLANG_HIP_DP_ROUTED_MXFP8_GATHER_PREFILL.get()
_GROUP = 32
_warned: set = set()


class RoutedMxfp8(NamedTuple):
    hidden_q: torch.Tensor  # [rows, hidden] fp8
    scale: torch.Tensor  # [rows, hidden // 32] e8m0
    topk_output: StandardTopKOutput


def _row_bytes(hidden: int, topk: int) -> int:
    return (hidden + hidden // _GROUP + 8 * topk + 15) // 16 * 16


@triton.jit
def _pack_rows_kernel(
    q_ptr,
    s_ptr,
    ids_ptr,
    w_ptr,
    dst_ptr,
    n_real_ptr,
    H: tl.constexpr,
    G: tl.constexpr,
    K4: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    real = row < tl.load(n_real_ptr)
    in_q = offs < H
    in_s = (offs >= H) & (offs < H + G)
    in_i = (offs >= H + G) & (offs < H + G + K4)
    in_w = (offs >= H + G + K4) & (offs < H + G + 2 * K4)
    v = tl.load(q_ptr + row * H + offs, mask=in_q & real, other=0)
    v = tl.where(
        in_s, tl.load(s_ptr + row * G + (offs - H), mask=in_s & real, other=0), v
    )
    v = tl.where(
        in_i,
        tl.load(ids_ptr + row * K4 + (offs - H - G), mask=in_i & real, other=0),
        v,
    )
    v = tl.where(
        in_w,
        tl.load(w_ptr + row * K4 + (offs - H - G - K4), mask=in_w & real, other=0),
        v,
    )
    tl.store(dst_ptr + row * ROW + offs, v, mask=offs < ROW)


@triton.jit
def _unpack_rows_kernel(
    src_ptr,
    q_ptr,
    s_ptr,
    ids_ptr,
    w_ptr,
    H: tl.constexpr,
    G: tl.constexpr,
    K4: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src_ptr + row * ROW + offs, mask=offs < ROW, other=0)
    tl.store(q_ptr + row * H + offs, v, mask=offs < H)
    tl.store(s_ptr + row * G + (offs - H), v, mask=(offs >= H) & (offs < H + G))
    tl.store(
        ids_ptr + row * K4 + (offs - H - G),
        v,
        mask=(offs >= H + G) & (offs < H + G + K4),
    )
    tl.store(
        w_ptr + row * K4 + (offs - H - G - K4),
        v,
        mask=(offs >= H + G + K4) & (offs < H + G + 2 * K4),
    )


def _skip_reason(
    moe, local: torch.Tensor, forward_batch, global_rows: int, do_shared_local: bool
) -> Optional[str]:
    mode = forward_batch.dp_padding_mode
    if mode is None:
        return None
    max_len = mode.is_max_len()
    if not (_ENABLED if max_len else _PREFILL_ENABLED):
        return None
    # SUM_LEN: every rank must reach the same verdict (it decides the collective each
    # rank issues), so only rank-uniform state may skip there.
    if not do_shared_local:
        return "shared expert not local (SGLANG_DP_SHARED_EXPERT_LOCAL)"
    if getattr(moe, "is_nextn", False):
        return None  # draft layers keep the bf16 gather, not worth a warning
    parallel = get_parallel()
    if parallel.attn_tp_size != 1 or parallel.attn_cp_size > 1:
        return "attention TP / CP"
    if world_dp_gather_enabled():
        return "world DP gather"
    if local.dim() != 2 or local.dtype != torch.bfloat16 or local.shape[1] % _GROUP:
        return f"local rows {tuple(local.shape)} {local.dtype}"
    if not getattr(moe.experts.dispatcher, "use_aiter_moe_runner", False):
        return "MoE runner is not aiter"
    if not getattr(moe.experts, "fp8_activations", False):
        return "routed experts do not take MXFP8 activations"
    if not max_len:
        if not is_dp_gatherv_active():
            return "SUM_LEN gather is not the gatherv path (SGLANG_DP_USE_GATHERV)"
        sizes = get_dp_global_num_tokens()
        if not sizes or sum(sizes) != global_rows:
            return f"gatherv sizes {sizes} do not fill {global_rows} rows"
        return ""
    if not isinstance(forward_batch.num_token_non_padded, torch.Tensor):
        return "no device num_token_non_padded"
    if local.shape[0] == 0:
        return None
    if global_rows != local.shape[0] * parallel.tp_size:
        return f"global rows {global_rows} != {local.shape[0]} x {parallel.tp_size}"
    vl_bias = getattr(moe.gate, "e_score_correction_bias_vl", None)
    if vl_bias is not None and not moe.is_hash:
        from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe

        if deepseek_v2_hip_moe.batch_has_images(forward_batch):
            return None
    return ""


def routed_mxfp8_gather_applies(
    moe, local: torch.Tensor, forward_batch, global_rows: int, do_shared_local: bool
) -> bool:
    if not (_ENABLED or _PREFILL_ENABLED):
        return False
    reason = _skip_reason(moe, local, forward_batch, global_rows, do_shared_local)
    if reason and reason not in _warned:
        _warned.add(reason)
        logger.warning("SGLANG_HIP_DP_ROUTED_MXFP8_GATHER skipped: %s", reason)
    return reason == ""


def _route_local(
    moe,
    local: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    use_vision_topk: bool = False,
):
    from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe

    router_logits, partials = deepseek_v2_hip_moe.forward_gate(
        moe, local, None, fused_gate=not use_vision_topk
    )
    if use_vision_topk:
        from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

        topk_output = vision_topk(moe, router_logits, input_ids)
    else:
        topk_kwargs = {"input_ids": input_ids} if moe.is_hash else {}
        if partials is not None:
            topk_kwargs["router_logits_partials"] = partials
        topk_output = moe.topk(local, router_logits, **topk_kwargs)
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    ids = topk_output.topk_ids.to(torch.int32).contiguous()
    weights = topk_output.topk_weights.to(torch.float32).contiguous()
    return ids, weights, router_logits


def gather_routed_mxfp8(
    moe, local: torch.Tensor, forward_batch, input_ids: Optional[torch.Tensor]
) -> RoutedMxfp8:
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    local = local.contiguous()
    ids, weights, router_logits = _route_local(moe, local, input_ids)
    q, s = per_1x32_mx_quant_hip(
        local, scale=None, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )

    rows, hidden = local.shape
    topk = ids.shape[1]
    groups = hidden // _GROUP
    row_bytes = _row_bytes(hidden, topk)
    block = 1024
    n_real = forward_batch.num_token_non_padded
    tp_group = get_parallel().tp_group
    world = tp_group.world_size

    def stage(dst: torch.Tensor, src) -> None:
        _pack_rows_kernel[(rows, triton.cdiv(row_bytes, block))](
            q.view(torch.uint8),
            s.view(torch.uint8),
            ids.view(torch.uint8),
            weights.view(torch.uint8),
            dst.view(torch.uint8),
            n_real,
            H=hidden,
            G=groups,
            K4=4 * topk,
            ROW=row_bytes,
            BLOCK=block,
        )

    packed_local = local.new_empty((rows, row_bytes // 2))
    packed = local.new_empty((rows * world, row_bytes // 2))
    if not tp_group.aiter_all_gather_staged(packed, packed_local, stage):
        stage(packed_local, None)
        tp_group.all_gather_into_tensor(packed, packed_local)

    total = rows * world
    hidden_q = torch.empty((total, hidden), dtype=dtypes.fp8, device=local.device)
    scale = torch.empty((total, groups), dtype=dtypes.fp8_e8m0, device=local.device)
    g_ids = torch.empty((total, topk), dtype=torch.int32, device=local.device)
    g_weights = torch.empty((total, topk), dtype=torch.float32, device=local.device)
    _unpack_rows_kernel[(total, triton.cdiv(row_bytes, block))](
        packed.view(torch.uint8),
        hidden_q.view(torch.uint8),
        scale.view(torch.uint8),
        g_ids.view(torch.uint8),
        g_weights.view(torch.uint8),
        H=hidden,
        G=groups,
        K4=4 * topk,
        ROW=row_bytes,
        BLOCK=block,
    )
    return RoutedMxfp8(
        hidden_q, scale, StandardTopKOutput(g_weights, g_ids, router_logits=None)
    )


def gather_routed_mxfp8_varlen(
    moe, local: torch.Tensor, forward_batch, input_ids: Optional[torch.Tensor]
) -> RoutedMxfp8:
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe

    _note_dp_gather_in_prefill_graph()
    sizes = get_dp_global_num_tokens()
    slot = sizes[dp_slot_in(sizes)]
    n = min(local.shape[0], slot)
    hidden = local.shape[1]
    groups = hidden // _GROUP
    topk = moe.topk.topk_config.top_k
    device = local.device
    q = torch.empty((slot, hidden), dtype=dtypes.fp8, device=device)
    s = torch.empty((slot, groups), dtype=torch.uint8, device=device)
    meta = torch.empty((slot, 2 * topk), dtype=torch.int32, device=device)
    if n > 0:
        x = local[:n].contiguous()
        use_vision_topk = (
            getattr(moe.gate, "e_score_correction_bias_vl", None) is not None
            and not moe.is_hash
            and deepseek_v2_hip_moe.batch_has_images(forward_batch)
        )
        ids, weights, _ = _route_local(
            moe,
            x,
            input_ids[:n] if input_ids is not None else None,
            use_vision_topk,
        )
        assert ids.shape[1] == topk, f"topk width {ids.shape[1]} != {topk}"
        dynamic_per_group_scaled_quant(
            q[:n], x, s[:n].view(dtypes.fp8_e8m0), _GROUP, shuffle_scale=False
        )
        meta[:n, :topk].copy_(ids)
        meta[:n, topk:].copy_(weights.view(torch.int32))
    if n < slot:
        q[n:].view(torch.uint8).zero_()
        s[n:].zero_()
        meta[n:].zero_()
    g_q, g_s, g_meta = get_parallel().tp_group.all_gatherv(
        [q.view(torch.uint8), s, meta], sizes=sizes
    )
    g_ids = g_meta[:, :topk].contiguous()
    g_weights = g_meta[:, topk:].contiguous().view(torch.float32)
    return RoutedMxfp8(
        g_q.view(dtypes.fp8),
        g_s.view(dtypes.fp8_e8m0),
        StandardTopKOutput(g_weights, g_ids, router_logits=None),
    )
