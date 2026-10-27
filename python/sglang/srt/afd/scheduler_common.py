"""Shared AFD scheduler initialization and step logging policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.afd_handshake import (
    AFDHandshakeInfo,
    AFDHandshakeResult,
    compute_model_identity,
)
from sglang.srt.afd.comm import (
    AFDCommBackend,
    AFDCommConfig,
    create_afd_comm_backend,
)
from sglang.srt.afd.layers import collect_afd_moe_layers
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_device, get_disagg, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


_VERBOSE_FIRST_STEPS = 5
_LOG_INTERVAL = envs.SGLANG_AFD_LOG_INTERVAL.get()
if _LOG_INTERVAL < 0:
    raise ValueError("SGLANG_AFD_LOG_INTERVAL must be non-negative")


def build_local_handshake_info(scheduler: Scheduler, role: AFDRole) -> AFDHandshakeInfo:
    model_runner = scheduler.tp_worker.model_runner
    moe_layers = collect_afd_moe_layers(model_runner.model, role)
    scattered = get_disagg().afd_ffn_moe_backend == "deepep"
    decode_token_buckets = []
    if role.is_attn():
        wire_token_multiplier = 1 if scattered else get_parallel().tp_size
        decode_token_buckets = [
            model_runner.max_running_requests * wire_token_multiplier
        ]
    return AFDHandshakeInfo(
        role=role.value,
        model_identity=compute_model_identity(get_model().model_path),
        num_ranks=get_parallel().tp_size,
        afd_backend=get_disagg().afd_backend,
        afd_ffn_moe_backend=get_disagg().afd_ffn_moe_backend,
        wire_dtype=str(model_runner.dtype).removeprefix("torch."),
        decode_token_buckets=decode_token_buckets,
        moe_layers=[
            {
                "layer_id": layer.layer_id,
                "hidden_size": layer.hidden_size,
                "top_k": layer.top_k,
                "num_experts": layer.num_experts,
                "num_fused_shared_experts": layer.num_fused_shared_experts,
                "routed_scaling_factor": layer.routed_scaling_factor,
            }
            for layer in moe_layers
        ],
    )


def should_log_step(step_id: int) -> bool:
    return step_id < _VERBOSE_FIRST_STEPS or (
        _LOG_INTERVAL > 0 and step_id % _LOG_INTERVAL == 0
    )


def _afd_device(scheduler: Scheduler) -> str:
    device = str(scheduler.tp_worker.model_runner.device)
    if device == "cuda":
        return f"cuda:{get_device().gpu_id}"
    return device


def init_afd_comm_backend(
    scheduler: Scheduler, role: AFDRole, handshake: AFDHandshakeResult
) -> AFDCommBackend:
    if role.is_attn():
        num_attn_ranks = handshake.local.num_ranks
        num_ffn_ranks = handshake.peer.num_ranks
    else:
        num_attn_ranks = handshake.peer.num_ranks
        num_ffn_ranks = handshake.local.num_ranks
    config = AFDCommConfig(
        role=role,
        num_attn_ranks=num_attn_ranks,
        num_ffn_ranks=num_ffn_ranks,
        instance_rank=get_parallel().tp_rank,
        store_host=handshake.dist_host,
        store_port=handshake.dist_port,
        device=_afd_device(scheduler),
    )
    return create_afd_comm_backend(get_disagg().afd_backend, config)
