"""Configuration handoff and CPU placement for the embedded Rust server."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.managers.utils import compute_num_reserved_tokens
from sglang.srt.runtime_context import (
    get_disagg,
    get_model,
    get_observability,
    get_serving,
)
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.rust_extensions._server import ServerArgs

logger = logging.getLogger(__name__)


def _build_server_args(scheduler: Scheduler) -> ServerArgs:
    """The typed launch handoff for the scheduler's embedded Rust server:
    the ``server_args`` fields it reads, the already-resolved
    ``model_config``, and launch-time facts — as the Rust extension's own
    ``ServerArgs`` class. Its constructor takes every field as a required
    keyword (see ``rust/sglang-server/src/message/config.rs``), so a
    missing, extra or mistyped field fails here at boot rather than
    running on a silently-defaulted knob."""
    from sglang.srt.rust_extensions import load_rust_extension

    ext = load_rust_extension("sglang.srt.rust_extensions._server")

    sa = resolving_view(scheduler.server_args)
    mc = scheduler.model_config
    disaggregation_mode = {
        "null": ext.DisaggregationMode.Null,
        "prefill": ext.DisaggregationMode.Prefill,
        "decode": ext.DisaggregationMode.Decode,
    }[get_disagg().disaggregation_mode]
    return ext.ServerArgs(
        model_path=get_model().model_path,
        served_model_name=get_serving().served_model_name,
        tokenizer_path=scheduler.rust_server_tokenizer_path(),
        revision=get_model().revision,
        load_format=get_model().load_format,
        weight_version=get_serving().weight_version,
        host=get_serving().host,
        port=get_serving().port,
        log_level=get_observability().log_level,
        log_level_http=get_observability().log_level_http,
        chat_template=get_serving().chat_template,
        tool_call_parser=get_serving().tool_call_parser,
        reasoning_parser=get_serving().reasoning_parser,
        stream_response_default_include_usage=get_serving().stream_response_default_include_usage,
        tokenizer_worker_num=get_serving().tokenizer_worker_num,
        detokenizer_worker_num=get_serving().detokenizer_worker_num,
        skip_tokenizer_init=get_serving().skip_tokenizer_init,
        incremental_streaming_output=get_serving().incremental_streaming_output,
        disaggregation_mode=disaggregation_mode,
        model_config=ext.ModelConfig(
            context_len=mc.context_len,
            vocab_size=mc.vocab_size,
            is_multimodal=mc.is_multimodal,
            # Resolved default sampling params (generation_config.json when
            # `--sampling-defaults model`, {} otherwise). The rust server
            # consumes these for omitted temperature/top_p in chat
            # conversions instead of hard-coding the OpenAI terminal
            # defaults.
            default_sampling_params=ext.DefaultSamplingParams(
                **mc.get_default_sampling_params()
            ),
        ),
        preferred_sampling_params=(
            json.dumps(get_serving().preferred_sampling_params)
            if get_serving().preferred_sampling_params is not None
            else None
        ),
        allow_auto_truncate=get_serving().allow_auto_truncate,
        enable_return_hidden_states=sa.enable_return_hidden_states,
        # Not a `server_args` field: `TokenizerManager` derives it, and the
        # rust ingress needs the same number for its total-token check.
        num_reserved_tokens=compute_num_reserved_tokens(),
        # Launch-time facts Python's /server_info reports from
        # scheduler_info / the package — stamped here so the rust endpoint
        # can serve them statically (no scheduler round-trip).
        version=__version__,
        max_total_num_tokens=scheduler.max_total_num_tokens,
    )


def _partition_cores(
    mm_workers: int = 0,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Split this rank's allowed cores into ``(launch_cores, server_cores)``.

    Pure computation — no affinity is changed here. Both sets are a subset
    of this rank's NUMA-local cores (when affinity/NUMA bind is on), so the
    partition stays NUMA-local. Returns ``(None, None)`` (server runs
    unpinned, confined only by the process affinity) when the platform has
    no affinity API or too few cores to split.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None, None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except OSError as e:
        logger.warning("rust server: cannot read cpu affinity: %s", e)
        return None, None

    # Need enough cores to reserve launch cores and still pin the pools.
    if len(allowed) < 4:
        logger.info(
            "rust server: only %d cores allowed; running pools unpinned",
            len(allowed),
        )
        return None, None

    # Keep a small slice for the launch loop; cap at 2 (the event loop is
    # effectively serial) and never take more than a quarter of the cores.
    reserve = min(2, len(allowed) // 4)
    launch_cores = allowed[:reserve]
    # Bound the pool instead of taking the whole remainder: this rank's
    # allowed cores are usually the entire NUMA node, shared with the sibling
    # TP ranks' processes, so an unbounded mask lets MM preprocessing bursts
    # preempt a sibling's CUDA-launch thread and inflate every rank's forward
    # through the TP collectives. Measured on Qwen3.5-35B TP4 at one 720p
    # image per request: ~20 ms of ViT wall time on the worst sibling, gone
    # once bounded. The budget covers the CPU-hot threads (MM workers, plus
    # the I/O-shaped tokenizer/ingress/egress/api ones that are rarely all hot
    # at once) and leaves the rest of the node to the scheduler ranks.
    pool_budget = max(8, mm_workers + 4)
    server_cores = allowed[reserve : reserve + pool_budget]
    logger.info(
        "rust server cores=%s, scheduler launch cores=%s",
        server_cores,
        launch_cores,
    )
    return launch_cores, server_cores
