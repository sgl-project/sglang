"""Single-token BF16 Qwen4 MoE finalize, shared gate and TP4 collective."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.communication.all_reduce_fusion import (
    moe_finalize_shared_gate_all_reduce,
)
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.dp_attention import is_enable_moe_cp_allgather
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_exec, get_lora, get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock


@lru_cache(None)
def _get_decode_comm(tp_comm: CustomAllReduceV2) -> CustomAllReduceV2 | None:
    comm = CustomAllReduceV2(
        tp_comm.group,
        tp_comm.device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    if comm.disabled:
        return None
    register_comm_cleanup(comm)
    return comm


def prepare_qwen4_decode_comm(
    mlp: Qwen2MoeSparseMoeBlock,
) -> CustomAllReduceV2 | None:
    parallel = get_parallel()
    if (
        parallel.tp_size != 4
        or parallel.attn_dp_size != 1
        or parallel.moe_ep_size != 1
        or get_lora().enable_lora
        or get_spec().speculative_algorithm is not None
        or get_exec().overlap.enable_two_batch_overlap
        or is_enable_moe_cp_allgather()
        or not mlp.supports_deferred_finalize
        or mlp.num_experts != 512
        or mlp.experts.w13_weight.dtype != torch.bfloat16
        or mlp.shared_expert_gate is None
        or mlp.enable_shared_expert_fusion
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 10
    ):
        return None
    tp_comm = parallel.tp_group.ca_comm
    if not isinstance(tp_comm, CustomAllReduceV2) or tp_comm.disabled:
        return None
    return _get_decode_comm(tp_comm)


def qwen4_decode_moe(
    hidden: torch.Tensor, mlp: Qwen2MoeSparseMoeBlock, comm: CustomAllReduceV2
) -> torch.Tensor:
    if mlp.alt_stream is not None and get_is_capture_mode():
        current = torch.cuda.current_stream()
        mlp.alt_stream.wait_stream(current)
        # FC2 stays on the consuming stream for its PDL dependency. Both
        # branches read the input and write independently allocated outputs.
        with torch.cuda.stream(mlp.alt_stream):
            gate = shared_expert_gate(hidden, mlp.shared_expert_gate.weight)
            shared = mlp._forward_shared_experts(hidden, apply_gate=False)
        deferred = mlp._forward_router_experts(hidden, defer_finalize=True)
        current.wait_stream(mlp.alt_stream)
        gate.record_stream(current)
        shared.record_stream(current)
    else:
        gate = shared_expert_gate(hidden, mlp.shared_expert_gate.weight)
        shared = mlp._forward_shared_experts(hidden, apply_gate=False)
        deferred = mlp._forward_router_experts(hidden, defer_finalize=True)
    return moe_finalize_shared_gate_all_reduce(
        deferred.gemm2_out,
        deferred.expanded_idx_to_permuted_idx,
        deferred.expert_weights,
        shared,
        gate,
        comm.obj,
    )
