from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed import get_world_group
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import get_available_gpu_memory, get_device_memory_capacity

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def is_post_capture_kv_active(
    *, server_args: ServerArgs, is_draft_worker: bool
) -> bool:
    return (
        server_args.post_capture_kv_sizing_planned()
        and current_platform.is_cuda()
        and not is_draft_worker
    )


class PostCaptureKVResize(msgspec.Struct, frozen=True, kw_only=True):
    max_total_num_tokens: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    capped_max_running_requests: Optional[int]


def compute_post_capture_kv_resize(
    model_runner: ModelRunner,
) -> PostCaptureKVResize:
    """Resize the KV pool after capture and return the new sizes for the
    orchestrator to assign. Takes the live ModelRunner because it reads
    post-capture GPU memory + the pool objects it must resize in place."""
    pool = model_runner.token_to_kv_pool
    torch.cuda.synchronize()
    free_gb = get_available_gpu_memory(
        model_runner.device,
        model_runner.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )
    headroom_gb = model_runner.pre_model_load_memory * (
        1 - model_runner.mem_fraction_static
    )
    decode_cuda_graph_config = model_runner.server_args.cuda_graph_config.decode
    decode_max_bs = int(decode_cuda_graph_config.max_bs or 0)
    running_requests = int(model_runner.max_running_requests or decode_max_bs or 1)
    eager_decode_gap = (
        model_runner.server_args.disaggregation_mode != "prefill"
        and decode_cuda_graph_config.backend != Backend.DISABLED
        and decode_max_bs < running_requests
    )
    if eager_decode_gap:
        logger.warning(
            "Post-capture KV sizing: decode CUDA graph max_bs=%d < "
            "max_running_requests=%d; reserving activation headroom",
            decode_max_bs,
            running_requests,
        )
    if eager_decode_gap or mambaish_config(model_runner.model_config) is not None:
        headroom_gb = max(
            headroom_gb,
            model_runner.server_args.mamba_pre_capture_reserve_mb(
                get_device_memory_capacity(model_runner.device)
            )
            / 1024,
        )
    budget_bytes = (
        int(max(0.0, free_gb - headroom_gb) * (1 << 30))
        + pool.post_capture_backed_bytes
    )
    config = model_runner.kv_cache_configurator.config_from_budget(
        budget_bytes, cap_tokens=model_runner.max_total_num_tokens
    )
    pool.finalize_backing(config)
    model_runner.token_to_kv_pool_allocator.resize(config)

    capped_max_running_requests = None
    if model_runner.max_running_requests is not None:
        # Re-calculate max_running_requests for the now smaller pool
        capped_reqs = min(
            model_runner.max_running_requests,
            model_runner.kv_cache_configurator.resolve_max_num_reqs(
                config.max_total_num_tokens
            ),
        )
        if capped_reqs < model_runner.max_running_requests:
            logger.warning(
                "Post-capture KV sizing: max_running_requests %d -> %d",
                model_runner.max_running_requests,
                capped_reqs,
            )
            capped_max_running_requests = capped_reqs
    logger.info(
        "Post-capture KV sizing: max_total_num_tokens=%d, free memory=%.2f GB",
        config.max_total_num_tokens,
        get_available_gpu_memory(model_runner.device, model_runner.gpu_id),
    )
    return PostCaptureKVResize(
        max_total_num_tokens=config.max_total_num_tokens,
        full_max_total_num_tokens=config.full_max_total_num_tokens,
        swa_max_total_num_tokens=config.swa_max_total_num_tokens,
        capped_max_running_requests=capped_max_running_requests,
    )
