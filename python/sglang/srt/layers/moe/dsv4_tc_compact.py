"""Compact TP-MoE bridge between DeepSeek-V4 prefill graph pieces."""

from __future__ import annotations

import copy
from typing import Optional

import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel
from sglang.srt.utils import is_hip
from sglang.srt.utils.custom_op import register_custom_op

_IS_HIP = is_hip()


def compact_moe_enabled() -> bool:
    """The initial implementation is restricted to one ROCm TP8/DP8 group."""
    if not envs.SGLANG_DSV4_TC_COMPACT_MOE.get() or not _IS_HIP:
        return False
    parallel = get_parallel()
    return (
        parallel.tp_size == 8
        and parallel.attn_dp_size == 8
        and parallel.attn_tp_size == 1
        and parallel.moe_ep_size == 1
        and parallel.pp_size == 1
        and parallel.attn_cp_size == 1
        and get_exec().kernel.attention_backend == "dsv4"
        and get_moe_a2a_backend().is_none()
        and not get_exec().overlap.enable_two_batch_overlap
        and get_exec().graph.cuda_graph_config.prefill.backend == Backend.TC_PIECEWISE
    )


def run_compact_dp_moe(
    hidden_local: torch.Tensor,
    input_ids_local: Optional[torch.Tensor],
    output_local: torch.Tensor,
    *,
    counts: list[int],
    local_rank: int,
    tp_group,
    moe_layer,
    forward_batch,
    counts_gpu: Optional[torch.Tensor] = None,
    layer_id: Optional[int] = None,
) -> None:
    """Run a SUM_LEN MoE between graph pieces, writing a fixed local output.

    Counts are the synchronized, unpadded CPU token counts. All collective sizes
    are resolved here, outside CUDA graph capture. The original ForwardBatch and
    the outer graph's DP allocation metadata are left unchanged.
    """
    if hidden_local.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DSv4 compact MoE must execute as a split graph operation")
    if hidden_local.ndim != 2 or output_local.shape != hidden_local.shape:
        raise ValueError("Compact MoE expects matching [local_bucket, hidden] buffers")
    if len(counts) != tp_group.world_size or not 0 <= local_rank < len(counts):
        raise ValueError("Compact MoE requires the full TP group's token counts")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Compact MoE counts must be nonnegative Python integers")
    local_tokens = counts[local_rank]
    if local_tokens > hidden_local.shape[0]:
        raise ValueError("Real local token count exceeds the attention bucket")
    total_tokens = sum(counts)
    output_local.zero_()
    if total_tokens == 0:
        return

    packed_hidden = hidden_local.new_empty((total_tokens, hidden_local.shape[1]))
    tp_group.all_gatherv(
        hidden_local[:local_tokens].contiguous(), sizes=counts, output=packed_hidden
    )

    packed_ids = None
    if getattr(moe_layer, "is_hash", False):
        if input_ids_local is None or input_ids_local.shape[0] < local_tokens:
            raise ValueError(
                "Hash-routed MoE requires the corresponding local token IDs"
            )
        packed_ids = input_ids_local.new_empty((total_tokens,))
        tp_group.all_gatherv(
            input_ids_local[:local_tokens].contiguous(), sizes=counts, output=packed_ids
        )

    compact_batch = copy.copy(forward_batch)
    compact_batch.global_num_tokens_cpu = list(counts)
    compact_batch.global_num_tokens_gpu = counts_gpu
    compact_batch.global_dp_buffer_len = total_tokens
    compact_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    compact_batch.dp_local_start_pos = None
    compact_batch.dp_local_num_tokens = None

    # A replicated shared expert must be added after the TP sum. Other shared
    # experts (including the recipe's fused shared expert) stay in the TP MoE.
    shared_local = None
    separate_shared = bool(
        getattr(moe_layer, "_shared_expert_tp1", False)
        and getattr(moe_layer, "shared_experts", None) is not None
    )
    if separate_shared and local_tokens:
        shared_local = moe_layer._forward_shared_experts(hidden_local[:local_tokens])

    with get_forward().scoped(mlp_reduce_scatter=True, fuse_mlp_allreduce=False):
        partial = moe_layer(
            packed_hidden,
            compact_batch,
            input_ids=packed_ids,
            input_ids_global=packed_ids,
            skip_shared_experts=separate_shared,
        )
    if partial.shape != packed_hidden.shape:
        raise RuntimeError(
            "Compact MoE must return one partial TP output per real token"
        )
    tp_group.reduce_scatterv(
        partial,
        output=output_local[:local_tokens],
        # Equal sizes use one NCCL reduce-scatter rather than the variable-size
        # implementation's grouped reduction to each rank.
        sizes=None if all(count == counts[0] for count in counts) else counts,
    )
    if shared_local is not None:
        output_local[:local_tokens].add_(shared_local)


@register_custom_op(mutates_args=["output_local"])
@register_split_op()
def dsv4_tc_compact_moe_with_output(
    hidden_local: torch.Tensor,
    input_ids_local: Optional[torch.Tensor],
    real_counts_gpu: torch.Tensor,
    output_local: torch.Tensor,
    layer_id: int,
) -> None:
    context = get_tc_piecewise_forward_context()
    if context is None or context.dp_moe_layers is None:
        raise RuntimeError("Missing DSv4 compact-MoE forward context")
    batch = context.forward_batch
    counts = batch.moe_real_num_tokens_cpu
    if counts is None:
        raise RuntimeError("Real DP counts were not preserved before graph padding")
    run_compact_dp_moe(
        hidden_local,
        input_ids_local,
        output_local,
        counts=counts,
        counts_gpu=real_counts_gpu,
        local_rank=get_parallel().attn_dp_rank,
        tp_group=get_tp_group(),
        moe_layer=context.dp_moe_layers[layer_id],
        forward_batch=batch,
        layer_id=layer_id,
    )
