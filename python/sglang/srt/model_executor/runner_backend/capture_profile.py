"""CUDA graph capture profiling shared by graph backends.

The runner supplies naming metadata only; profiler construction, CUDA memory
history, trace export and device-specific summaries live here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.profile_utils import (
    export_cuda_graph_capture_trace,
    graph_capture_profile_dir,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureProfileOptions:
    runner_name: str
    per_shape: bool
    get_trace_tag: Callable[[], Optional[str]]


def create_cuda_capture_profile(options: CaptureProfileOptions):
    if options.per_shape:
        # SGLANG_GRAPH_BATCH_CAPTURE: the profiler is stepped once per
        # captured shape (see FullCudaGraphBackend.capture_one), so
        # on_trace_ready writes one chrome trace per shape.
        rank = get_parallel().tp_rank
        runner_name = options.runner_name
        trace_dir = graph_capture_profile_dir()
        os.makedirs(trace_dir, exist_ok=True)

        def on_trace_ready(prof):
            tag = options.get_trace_tag()
            if tag is None:
                # Reachable only if a subclass overrides _capture_one_stream
                # without tagging; exporting would collide every flush on
                # one name.
                logger.warning(
                    "%s: capture trace flushed with no shape tag; skipping "
                    "export. A subclass likely overrides _capture_one_stream "
                    "without calling _set_profile_trace_tag.",
                    runner_name,
                )
                return
            trace_file = os.path.join(
                trace_dir, f"{runner_name}_{tag}_rank{rank}.json.gz"
            )
            prof.export_chrome_trace(trace_file)
            logger.info(f"Saved trace for {tag} to {trace_file}")

        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Schedule: wait=2 (skip 2 dummy runs), warmup=0, active=1
            # (capture run); repeat=0 repeats the cycle so each captured
            # shape gets its own trace.
            schedule=torch.profiler.schedule(wait=2, warmup=0, active=1, repeat=0),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True,
            on_trace_ready=on_trace_ready,
        )
    else:
        # a single unscheduled pass over the whole
        # capture. The combined trace (if any) is exported in
        # finish_cuda_capture_profile via export_cuda_graph_capture_trace,
        # gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE.
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
    torch.cuda.memory._record_memory_history()
    return profile_context


def finish_cuda_capture_profile(options: CaptureProfileOptions, prof_context):
    torch.cuda.memory._dump_snapshot("cuda_graph_runner_memory_usage.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    log_message = (
        "Sorted by CUDA Time:\n"
        + prof_context.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=10
        )
        + "\n\nSorted by CPU Time:\n"
        + prof_context.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
    )
    logger.info(log_message)

    # single-trace export for the whole capture pass; no-op unless
    # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE is set. In per-bs mode
    # (SGLANG_GRAPH_BATCH_CAPTURE) that env is unset, so this stays a no-op
    # and the per-bs on_trace_ready handles export instead.
    export_cuda_graph_capture_trace(
        prof_context,
        runner_name=options.runner_name,
    )
