"""Unified NPU MoE routing entry points.

Every Ascend MoE path — no-A2A ``standard``, ``ascend_ops`` alltoallv and
allgather — permutes tokens into expert order with
``npu_moe_init_routing_v2`` and reverses that permutation with
``npu_moe_finalize_routing``; the paths only differ in which op parameters
they use.  These two functions are the single place that talks to the
torch_npu routing ops, so a future interface change only has to be adapted
here:

- :func:`npu_moe_init_routing`: quantizing pass (``quant_mode``), or
  scale-reuse pass (``scale``) that re-permutes already-quantized tokens
  whose scale traveled with the payload through the a2a communication.
- :func:`npu_moe_finalize_routing`: plain unpermute (no ``topk_weights``)
  or weighted top-k sum, with optional export mode that skips dropped routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import DispatcherOutputDtype

INT8_QUANT_MODE = 1
MXFP8_QUANT_MODE = 3
MXFP4_QUANT_MODE = 9

# Routing quant modes currently implemented by this module are 1 (INT8),
# 3 (MXFP8/E4M3FN), and 9 (MXFP4); -1 means unquantized dispatch.
# The following Ascend routing quant modes are planned for support:
# 0 (static quant), 2 (MXFP8/E5M2), 4/5 (FP8 per-group/E5M2/E4M3FN),
# 6/7/8 (HIF8 direct/per-tensor/per-token), 11/12 (FP8 per-block/E5M2/E4M3FN,
# FP32 3D scales),
# 13 (INT4 dynamic), 14/15 (FP8 per-group with RoundScale+Amax), and
# 16/17 (MXFP8 with RoundScale+Amax/E5M2/E4M3FN).
# Dispatch dtype -> quant mode. Each torch_npu op family defines its own enum
# (the MC2 distribute ops use 0/2/4 for the same formats), so a dtype can map
# to a different mode per op.
ROUTING_QUANT_MODES = {
    DispatcherOutputDtype.BF16: -1,
    DispatcherOutputDtype.INT8: INT8_QUANT_MODE,
    DispatcherOutputDtype.MXFP8: MXFP8_QUANT_MODE,
    DispatcherOutputDtype.MXFP4: MXFP4_QUANT_MODE,
}


def quant_mode_dtype(quant_mode: int) -> torch.dtype:
    """Payload dtype a quant mode puts on the wire.

    Resolved lazily instead of a module-level dict: torch builds without
    float4 support must still import this module on every platform, and fp4
    is only reachable from NPU quantized paths.
    """
    if quant_mode == INT8_QUANT_MODE:
        return torch.int8
    if quant_mode == MXFP8_QUANT_MODE:
        return torch.float8_e4m3fn
    if quant_mode == MXFP4_QUANT_MODE:
        return torch.float4_e2m1fn_x2
    raise ValueError(f"unknown NPU MoE quant mode {quant_mode}")


def normalize_mxfp_scale(scale: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Reshape a flat 2D e8m0 block scale into the pair-split ``[N, M//2, 2]`` form.

    ``npu_dynamic_mx_quant`` and ``npu_moe_init_routing_v2`` emit the scale
    flat while the Ascend grouped matmuls take the pair-split view.  Already
    pair-split scales pass through untouched, so the reshape is idempotent
    and safe on op outputs of either form.
    """
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


@dataclass
class RoutedTokens:
    """Output of :func:`npu_moe_init_routing`.

    ``scale`` is normalized to the form the Ascend grouped matmuls take
    (pair-split e8m0 block scales for the MX modes) and is ``None`` when the
    routing ran unquantized.
    """

    hidden_states: torch.Tensor  # expert-ordered (optionally quantized) rows
    expanded_row_idx: torch.Tensor  # source row of each routed row
    expert_tokens: torch.Tensor  # per-expert token counts, int64
    scale: Optional[torch.Tensor]  # per-token / block scale, None for bf16


def npu_moe_init_routing(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    quant_mode: int = -1,
    scale: Optional[torch.Tensor] = None,
    active_num: Optional[int] = None,
) -> RoutedTokens:
    """Permute tokens into expert order with ``npu_moe_init_routing_v2``.

    ``quant_mode`` selects the wire format: -1 keeps the activations as they
    are, a quant mode fuses the activation quantization into routing (the
    payload comes back quantized, with its scale).  Passing ``scale``
    re-permutes already-quantized tokens instead of quantizing again — their
    scale was produced by an earlier quantizing pass and traveled with the
    payload through the a2a.

    ``active_num`` changes the op's reverse-mapping index semantics; call
    sites pass the exact form each path was validated with.
    """
    if scale is not None:
        if quant_mode not in (INT8_QUANT_MODE, MXFP8_QUANT_MODE, MXFP4_QUANT_MODE):
            raise ValueError(
                "scale reuse requires one of the quant modes "
                f"{(INT8_QUANT_MODE, MXFP8_QUANT_MODE, MXFP4_QUANT_MODE)}, "
                f"got {quant_mode}"
            )
        op_kwargs = dict(
            expert_num=num_experts,
            expert_tokens_num_flag=True,
            expert_tokens_num_type=1,
            row_idx_type=0,
            active_expert_range=[0, num_experts],
            # Already quantized: permute the payload and its traveling scale
            # without quantizing again.
            quant_mode=-1,
            scale=scale,
            x_dtype=quant_mode_dtype(quant_mode),
        )
        if active_num is not None:
            op_kwargs["active_num"] = active_num
        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(hidden_states, topk_ids, **op_kwargs)
        )
        if quant_mode in (MXFP8_QUANT_MODE, MXFP4_QUANT_MODE):
            pertoken_scale = normalize_mxfp_scale(pertoken_scale)
        return RoutedTokens(
            hidden_states,
            expanded_row_idx,
            expert_tokens.to(torch.int64),
            pertoken_scale,
        )

    op_kwargs = dict(
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, num_experts],
        quant_mode=quant_mode,
    )
    if active_num is not None:
        op_kwargs["active_num"] = active_num
    hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch.ops.npu.npu_moe_init_routing_v2(hidden_states, topk_ids, **op_kwargs)
    )
    if quant_mode == -1:
        pertoken_scale = None
    elif quant_mode in (MXFP8_QUANT_MODE, MXFP4_QUANT_MODE):
        pertoken_scale = normalize_mxfp_scale(pertoken_scale)
    return RoutedTokens(
        hidden_states, expanded_row_idx, expert_tokens.to(torch.int64), pertoken_scale
    )


def npu_moe_finalize_routing(
    hidden_states: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 2,
) -> torch.Tensor:
    """Reverse :func:`npu_moe_init_routing` via the torch_npu finalize-routing op.

    ``topk_weights`` applies the router weights (weighted top-k sum) on the
    way; without it the result is a plain unpermute.  ``topk_ids`` selects
    export mode, which skips invalid routes (dropped or duplicated ids).
    """
    if (
        hidden_states.shape[0] > 0
        and expanded_row_idx.shape[0] != hidden_states.shape[0]
    ):
        raise RuntimeError(
            "NPU MoE finalize metadata mismatch: hidden_states has "
            f"{hidden_states.shape[0]} rows but expanded_row_idx has "
            f"{expanded_row_idx.shape[0]} rows"
        )
    return torch.ops.npu.npu_moe_finalize_routing(
        hidden_states.contiguous(),
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.contiguous() if topk_weights is not None else None,
        expanded_src_to_dst_row=expanded_row_idx.to(torch.int32).contiguous(),
        export_for_source_row=topk_ids,
        drop_pad_mode=drop_pad_mode,
    )
