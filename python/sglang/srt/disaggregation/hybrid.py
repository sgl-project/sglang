"""
Hybrid PD disaggregation mode.

In hybrid mode, each node performs both prefill and decode locally (like non-PD mode),
but can optionally offload requests to external decode-only nodes when local resources
are under pressure.

Offload decision is based on:
- KV cache watermark: if local KV cache usage exceeds a threshold
- Running request limit: if local decode slots exceed a configured cap
- Long output threshold: if estimated output length exceeds a threshold

Integration:
- The standard process_batch_result_prefill runs first (local decode is default)
- After prefill, `check_and_offload_requests()` is called to divert qualifying
  requests by initiating KV transfer to external decode nodes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def should_offload(scheduler: "Scheduler", req: "Req") -> bool:
    """
    Decide whether a request should be offloaded to an external decode node.

    Returns True if any offload condition is met.
    """
    server_args = scheduler.server_args

    if not server_args.hybrid_external_decode_addresses:
        return False

    # Condition 1: KV cache watermark
    watermark = server_args.hybrid_offload_watermark
    if watermark > 0:
        total_tokens = scheduler.max_total_num_tokens
        available_tokens = scheduler.token_to_kv_pool_allocator.available_size()
        usage_ratio = 1.0 - (available_tokens / total_tokens) if total_tokens > 0 else 0
        if usage_ratio >= watermark:
            logger.debug(
                "Offload triggered by KV watermark: usage=%.2f, rid=%s",
                usage_ratio,
                req.rid,
            )
            return True

    # Condition 2: Local decode limit
    local_limit = server_args.hybrid_local_decode_limit
    if local_limit > 0:
        num_running = len(scheduler.running_batch.reqs) if scheduler.running_batch else 0
        if num_running >= local_limit:
            logger.debug(
                "Offload triggered by local decode limit: running=%d >= %d, rid=%s",
                num_running,
                local_limit,
                req.rid,
            )
            return True

    # Condition 3: Long output threshold
    output_threshold = server_args.hybrid_long_output_threshold
    if output_threshold > 0:
        max_new_tokens = 0
        if hasattr(req, "sampling_params") and req.sampling_params is not None:
            max_new_tokens = getattr(req.sampling_params, "max_new_tokens", 0) or 0
        if max_new_tokens >= output_threshold:
            logger.debug(
                "Offload triggered by long output: max_new_tokens=%d, rid=%s",
                max_new_tokens,
                req.rid,
            )
            return True

    return False


def init_hybrid_disaggregation(scheduler: "Scheduler") -> None:
    """
    Initialize hybrid disaggregation.

    Sets up KV sender infrastructure for offloading while keeping normal decode intact.
    """
    import torch

    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from sglang.srt.mem_cache import kv_cache_builder

    server_args = scheduler.server_args

    draft_token_to_kv_pool, model_config = kv_cache_builder.get_draft_kv_pool(
        draft_worker=scheduler.draft_worker,
        spec_algorithm=scheduler.spec_algorithm,
        server_args=server_args,
        enable_overlap=scheduler.enable_overlap,
    )
    if model_config is None:
        model_config = scheduler.model_config

    buffer_size = scheduler.max_running_requests * 2
    scheduler.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
        buffer_size
    )
    scheduler.disagg_metadata_buffers = MetadataBuffers(
        buffer_size,
        hidden_size=(
            model_config.spec_hidden_size
            if scheduler.spec_algorithm.is_eagle()
            or scheduler.spec_algorithm.is_standalone()
            else 16
        ),
        hidden_states_dtype=(
            model_config.dtype
            if scheduler.spec_algorithm.is_eagle()
            or scheduler.spec_algorithm.is_standalone()
            else torch.float32
        ),
        custom_mem_pool=scheduler.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
    )

    scheduler.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
        token_to_kv_pool=scheduler.token_to_kv_pool_allocator.get_kvcache(),
        draft_token_to_kv_pool=draft_token_to_kv_pool,
        req_to_metadata_buffer_idx_allocator=scheduler.req_to_metadata_buffer_idx_allocator,
        metadata_buffers=scheduler.disagg_metadata_buffers,
        tp_rank=scheduler.ps.tp_rank,
        tp_size=scheduler.ps.tp_size,
        gpu_id=scheduler.ps.gpu_id,
        bootstrap_port=server_args.disaggregation_bootstrap_port,
        gloo_group=scheduler.attn_tp_cpu_group,
        max_total_num_tokens=scheduler.max_total_num_tokens,
        scheduler=scheduler,
        pp_rank=scheduler.ps.pp_rank,
        pp_size=scheduler.ps.pp_size,
        transfer_backend=scheduler.transfer_backend,
    )
    scheduler.disagg_prefill_inflight_queue: List["Req"] = []

    logger.info(
        "Hybrid disaggregation initialized: external_decode=%s, watermark=%.2f, "
        "local_limit=%d, output_threshold=%d",
        server_args.hybrid_external_decode_addresses,
        server_args.hybrid_offload_watermark,
        server_args.hybrid_local_decode_limit,
        server_args.hybrid_long_output_threshold,
    )


def check_and_offload_requests(scheduler: "Scheduler") -> None:
    """
    Called after prefill batch processing in hybrid mode.

    Scans newly added requests in the running batch and offloads those that
    meet the offload criteria by initiating KV transfer.
    """
    if not scheduler.server_args.hybrid_external_decode_addresses:
        return

    if not scheduler.running_batch or not scheduler.running_batch.reqs:
        return

    offload_rids = []
    for req in list(scheduler.running_batch.reqs):
        if len(req.output_ids) == 1 and should_offload(scheduler, req):
            scheduler.running_batch.reqs.remove(req)
            scheduler.disagg_prefill_inflight_queue.append(req)
            scheduler.send_kv_chunk(req, last_chunk=True)
            offload_rids.append(req.rid)

    if offload_rids:
        logger.info(
            "Hybrid: offloaded %d requests to external decode: %s",
            len(offload_rids),
            offload_rids[:5],
        )
