from __future__ import annotations  # noqa: F401

import logging  # noqa: F401
from typing import TYPE_CHECKING, List, Union  # noqa: F401

import torch  # noqa: F401

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: F401
from sglang.srt.environ import envs  # noqa: F401
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: F401
from sglang.srt.managers.io_struct import AbortReq  # noqa: F401
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: F401
from sglang.srt.mem_cache.common import (  # noqa: F401
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.server_args import get_global_server_args  # noqa: F401
from sglang.srt.state_capturer.indexer_topk import (  # noqa: F401
    get_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (  # noqa: F401
    get_global_experts_capturer,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (  # noqa: F401
        EmbeddingBatchResult,
        GenerationBatchResult,
    )

logger = logging.getLogger(__name__)


class SchedulerBatchResultProcessor:
    """``Scheduler.process_batch_result`` hot-path main body. Composition
    target on Scheduler (``self.batch_result_processor``)."""

    def __init__(
        self,
        *,
        is_generation: bool,
        disaggregation_mode,
        enable_hisparse: bool,
        enable_metrics: bool,
        enable_overlap: bool,
        enable_overlap_mlx: bool,
        server_args,
        model_config,
        token_to_kv_pool_allocator,
        tree_cache,
        hisparse_coordinator,
        req_to_token_pool,
        decode_offload_manager,
        metrics_collector,
        draft_worker,
        model_worker,
        logprob_computer,
        output_streamer,
        abort_request,
        report_prefill_stats,
        report_decode_stats,
        update_spec_metrics,
        increment_generated_tokens,
        advance_forward_ct_decode,
    ) -> None:
        self.is_generation = is_generation
        self.disaggregation_mode = disaggregation_mode
        self.enable_hisparse = enable_hisparse
        self.enable_metrics = enable_metrics
        self.enable_overlap = enable_overlap
        self.enable_overlap_mlx = enable_overlap_mlx
        self.server_args = server_args
        self.model_config = model_config
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        self.hisparse_coordinator = hisparse_coordinator
        self.req_to_token_pool = req_to_token_pool
        self.decode_offload_manager = decode_offload_manager
        self.metrics_collector = metrics_collector
        self.draft_worker = draft_worker
        self.model_worker = model_worker
        self.logprob_computer = logprob_computer
        self.output_streamer = output_streamer
        self.abort_request = abort_request
        self.report_prefill_stats = report_prefill_stats
        self.report_decode_stats = report_decode_stats
        self.update_spec_metrics = update_spec_metrics
        self.increment_generated_tokens = increment_generated_tokens
        self.advance_forward_ct_decode = advance_forward_ct_decode
