"""Shared AFD scheduler initialization and step logging policy."""

from __future__ import annotations

import ctypes
import logging
from typing import TYPE_CHECKING

from sglang.srt.afd.afd_config import (
    AFDRole,
    get_afd_decode_num_tokens_per_bs,
    is_direct_afd_backend,
)
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
from sglang.srt.afd.dispatcher import (
    build_decode_token_buckets,
)
from sglang.srt.afd.layers import collect_afd_moe_layers
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_model,
    get_parallel,
)
from sglang.srt.utils.network import get_local_ip_auto
from sglang.srt.utils.numa_utils import _get_nvml_device_index

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


logger = logging.getLogger(__name__)


# Log the first few steps and then a heartbeat, so bring-up is visible
# without flooding the log at decode rate.
_VERBOSE_FIRST_STEPS = 5
_LOG_INTERVAL = envs.SGLANG_AFD_LOG_INTERVAL.get()
if _LOG_INTERVAL < 0:
    raise ValueError("SGLANG_AFD_LOG_INTERVAL must be non-negative")


def _query_nvshmem_fabric_key(device_id: int) -> tuple[bytes, int] | None:
    try:
        import pynvml
    except ModuleNotFoundError:
        logger.debug("AFD NVSHMEM fabric query skipped: pynvml is unavailable")
        return None

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        nvml_device_id = _get_nvml_device_index(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_id)
        fabric_info = pynvml.c_nvmlGpuFabricInfoV_t()
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(fabric_info))
        if (
            fabric_info.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            or fabric_info.status != pynvml.NVML_SUCCESS
        ):
            return None
        # Some bindings expose this binary UUID as c_char[], which truncates
        # at embedded NUL bytes when read as a Python value.
        cluster_uuid = ctypes.string_at(
            ctypes.addressof(fabric_info) + type(fabric_info).clusterUuid.offset,
            pynvml.NVML_GPU_FABRIC_UUID_LEN,
        )
        if not any(cluster_uuid):
            return None
        return cluster_uuid, int(fabric_info.cliqueId)
    except (AttributeError, pynvml.NVMLError) as exc:
        logger.debug("AFD NVSHMEM fabric query unavailable: %s", exc)
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as exc:
                logger.debug("AFD NVML shutdown failed: %s", exc)


def gather_nvshmem_topology(
    scheduler: Scheduler,
) -> tuple[list[str], list[tuple[bytes, int] | None]]:
    if get_disagg().afd_backend != "nvshmem":
        return [], []
    local = (get_local_ip_auto(), _query_nvshmem_fabric_key(scheduler.ps.gpu_id))
    members = get_parallel().tp_group.all_gather_object(local)
    return [node_id for node_id, _ in members], [key for _, key in members]


def build_local_handshake_info(
    scheduler: Scheduler,
    role: AFDRole,
    rank_node_ids: list[str],
    rank_nvshmem_fabric_keys: list[tuple[bytes, int] | None],
) -> AFDHandshakeInfo:
    moe_layers = collect_afd_moe_layers(
        scheduler.tp_worker.model_runner.model,
        role,
    )
    decode_token_buckets = []
    decode_graph_token_buckets = []
    decode_config = get_exec().graph.cuda_graph_config.decode
    scattered = get_disagg().afd_ffn_moe_backend == "deepep"
    decode_num_tokens_per_bs = (
        get_afd_decode_num_tokens_per_bs() if role.is_attn() else 1
    )
    wire_token_multiplier = decode_num_tokens_per_bs * (
        1 if scattered else scheduler.ps.tp_size
    )
    if role.is_attn():
        if decode_config.backend == Backend.DISABLED:
            local_batch_sizes = decode_config.bs
        else:
            local_batch_sizes, _ = get_batch_sizes_to_capture(
                scheduler.tp_worker.model_runner,
                decode_num_tokens_per_bs,
            )
            decode_graph_token_buckets = [
                batch_size * wire_token_multiplier for batch_size in local_batch_sizes
            ]
        if local_batch_sizes is None:
            raise RuntimeError("AFD requires normalized decode batch sizes")
        decode_token_buckets = build_decode_token_buckets(
            local_batch_sizes=local_batch_sizes,
            max_local_requests=scheduler.tp_worker.model_runner.max_running_requests,
            wire_token_multiplier=wire_token_multiplier,
        )
    elif decode_config.backend == Backend.FULL:
        decode_graph_token_buckets = list(decode_config.bs)

    direct_backend_version = None
    direct_nvfp4_input_scales = []
    direct_deepep_fp8_scale_ue8m0 = None
    if role.is_ffn() and is_direct_afd_backend(get_disagg().afd_backend):
        from sglang.srt.afd.layers import get_direct_nvfp4_input_scales

        direct_nvfp4_input_scales = get_direct_nvfp4_input_scales(moe_layers)
    if get_disagg().afd_backend == "direct_deepep":
        from sglang.srt.afd.direct_deepep import get_direct_deepep_version

        direct_backend_version = get_direct_deepep_version()
        if role.is_ffn() and not direct_nvfp4_input_scales:
            from sglang.srt.layers import deep_gemm_wrapper

            direct_deepep_fp8_scale_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    elif get_disagg().afd_backend == "direct_megamoe":
        from sglang.srt.afd.direct_megamoe import get_direct_megamoe_version

        direct_backend_version = get_direct_megamoe_version()

    return AFDHandshakeInfo(
        role=role.value,
        model_identity=compute_model_identity(get_model().model_path),
        num_ranks=scheduler.ps.tp_size,
        rank_node_ids=rank_node_ids,
        afd_backend=get_disagg().afd_backend,
        afd_ffn_moe_backend=get_disagg().afd_ffn_moe_backend,
        wire_dtype=str(scheduler.tp_worker.model_runner.dtype).removeprefix("torch."),
        direct_backend_version=direct_backend_version,
        direct_nvfp4_input_scales=direct_nvfp4_input_scales,
        direct_deepep_fp8_scale_ue8m0=direct_deepep_fp8_scale_ue8m0,
        direct_deepep_allow_hybrid=(not get_disagg().afd_direct_deepep_disable_hybrid),
        direct_deepep_expert_alignment=(
            envs.SGLANG_AFD_DIRECT_DEEPEP_EXPERT_ALIGNMENT.get()
        ),
        afd_comm_sm_budget=get_disagg().afd_comm_sm_budget,
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
        decode_token_buckets=decode_token_buckets,
        decode_graph_token_buckets=decode_graph_token_buckets,
        decode_num_tokens_per_bs=decode_num_tokens_per_bs,
        disable_cuda_graph_padding=get_exec().graph.disable_cuda_graph_padding,
        decode_num_microbatches=(
            get_disagg().afd_decode_num_microbatches if role.is_attn() else 1
        ),
        requires_eager_decode_communication=(
            role.is_ffn()
            and get_disagg().afd_backend == "nccl"
            and get_disagg().afd_attn_instances > 1
        ),
        rank_nvshmem_fabric_keys=rank_nvshmem_fabric_keys,
    )


def should_log_step(step_id: int) -> bool:
    return step_id < _VERBOSE_FIRST_STEPS or (
        _LOG_INTERVAL > 0 and step_id % _LOG_INTERVAL == 0
    )


def _afd_device(scheduler: Scheduler) -> str:
    device = str(scheduler.tp_worker.model_runner.device)
    if device == "cuda":
        return f"cuda:{scheduler.ps.gpu_id}"
    return device


def init_afd_comm_backend(
    scheduler: Scheduler, role: AFDRole, handshake: AFDHandshakeResult
) -> AFDCommBackend:
    """Create the data-plane backend right after the handshake (fail-fast;
    both instances are alive at this point, so the rendezvous cannot stall
    on a missing peer)."""
    uses_direct_union = is_direct_afd_backend(handshake.local.afd_backend)
    if role.is_attn():
        num_attn_ranks = (
            handshake.total_attn_ranks
            if uses_direct_union
            else handshake.local.num_ranks
        )
        num_local_attn_ranks = handshake.local.num_ranks
        num_ffn_ranks = handshake.peer.num_ranks
    else:
        num_attn_ranks = (
            handshake.total_attn_ranks
            if uses_direct_union
            else handshake.peer.num_ranks
        )
        num_local_attn_ranks = num_attn_ranks
        num_ffn_ranks = handshake.local.num_ranks

    config = AFDCommConfig(
        role=role,
        num_attn_ranks=num_attn_ranks,
        num_ffn_ranks=num_ffn_ranks,
        # Same numbering space as the handshake's num_ranks (= tp_size).
        instance_rank=scheduler.ps.tp_rank,
        store_host=handshake.dist_host,
        store_port=handshake.dist_port,
        device=_afd_device(scheduler),
        attn_instance_rank_offset=(
            handshake.attn_instance_rank_offset
            if uses_direct_union and role.is_attn()
            else 0
        ),
        num_local_attn_ranks=num_local_attn_ranks,
    )
    return create_afd_comm_backend(
        (
            "nccl"
            if is_direct_afd_backend(get_disagg().afd_backend)
            else get_disagg().afd_backend
        ),
        config,
        handshake=handshake,
    )
