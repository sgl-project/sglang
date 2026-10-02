"""Single-token BF16 Qwen4 MoE finalize, shared gate and TP4 collective."""

from functools import lru_cache

import torch

from sglang.kernels.ops.communication.all_reduce_fusion import (
    moe_finalize_shared_gate_all_reduce,
)
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_exec, get_lora, get_parallel, get_spec


@lru_cache(None)
def _fusion_comm(ca):
    comm = CustomAllReduceV2(
        ca.group,
        ca.device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    if comm.disabled:
        return None
    register_comm_cleanup(comm)
    return comm


def prepare_qwen4_decode_comm(mlp):
    parallel = get_parallel()
    if (
        parallel.tp_size != 4
        or parallel.attn_dp_size != 1
        or parallel.moe_ep_size != 1
        or get_lora().enable_lora
        or get_spec().speculative_algorithm is not None
        or get_exec().overlap.enable_two_batch_overlap
        or not getattr(mlp, "supports_deferred_finalize", False)
        or getattr(mlp, "num_experts", None) != 512
        or mlp.experts.w13_weight.dtype != torch.bfloat16
        or mlp.shared_expert_gate is None
        or mlp.enable_shared_expert_fusion
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 10
    ):
        return None
    ca = parallel.tp_group.ca_comm
    if not isinstance(ca, CustomAllReduceV2) or ca.disabled:
        return None
    return _fusion_comm(ca)


def can_use_qwen4_decode_moe(hidden, mlp, comm):
    return (
        comm is not None
        and not torch.compiler.is_compiling()
        and hidden.shape == (1, 2560)
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and mlp.experts.w13_weight.dtype == torch.bfloat16
    )


def qwen4_decode_moe(hidden, mlp, comm):
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
