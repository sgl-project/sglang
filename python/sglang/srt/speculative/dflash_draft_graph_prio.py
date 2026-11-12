# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_GRAPH_PRIO (default off): capture the DFLASH drafter's decode CUDA graphs on a
priority -1 stream.

On the megakernel line the target KDA commit fold is forked to a side stream and joined late
(SGLANG_OPT_DFLASH_COMMIT_JOIN_LATE), so it runs beside the next draft at the SAME stream priority; at equal
priority the draft kernels queue behind the fold grid's CTA dispatch. CUDA graph kernel nodes keep the priority of
the stream they were CAPTURED on (a graph captured at priority 0 and replayed into a priority -1 stream does not
overlap; captured at -1 it does), so this knob changes only the capture stream of the DFLASH draft decode graphs.

Order-only: the same kernels in the same stream order; only the hardware scheduling priority of the draft graph's
nodes changes, so outputs are bitwise those of the knob-off path. Cost: the draft graphs allocate from their own
capture stream, so they cannot reuse blocks the target graphs freed on the shared capture stream (extra graph-pool
memory; read the post-capture free-memory line at boot).
"""

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

KNOB = "SGLANG_OPT_DFLASH_DRAFT_GRAPH_PRIO"
DRAFT_PRIORITY = -1

_capture_stream = None
_capture_used = False


def draft_graph_prio_requested() -> bool:
    """Fail-closed parse: unset / '' / '0' = off, '1' = on, anything else raises."""
    raw = os.environ.get(KNOB, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{KNOB} must be unset, '0' or '1' (got {raw!r})")


def _is_dflash_draft_runner(model_runner: Any) -> bool:
    spec = getattr(model_runner, "spec_algorithm", None)
    return bool(
        getattr(model_runner, "is_draft_worker", False)
        and spec is not None
        and spec.is_dflash()
    )


def draft_graph_capture_stream(model_runner: Any, default_stream_fn: Callable[[], Any]) -> Any:
    """The stream a decode CUDA graph runner captures on: a priority -1 stream for the DFLASH draft runner when the
    knob is on, else the shared capture stream (default_stream_fn())."""
    global _capture_stream, _capture_used
    if not draft_graph_prio_requested():
        return default_stream_fn()
    if not (_is_dflash_draft_runner(model_runner) and torch.cuda.is_available()):
        return default_stream_fn()
    if _capture_stream is None:
        _capture_stream = torch.cuda.Stream(priority=DRAFT_PRIORITY)
        logger.info(
            "%s ENGAGED: DFLASH draft decode graphs captured on a priority %d stream (kernel nodes keep it at replay)",
            KNOB,
            _capture_stream.priority,
        )
    _capture_used = True
    return _capture_stream


def draft_graph_captured_on_prio() -> bool:
    return _capture_used


def forward_stream_priority(device: Any, spec_algorithm: Any):
    """Priority for a model runner's forward stream: -1 under the knob for a DFLASH server on CUDA (the eager draft prep
    on the forward stream then outranks the join-late fold on its priority-0 side stream), else None (default stream)."""
    if not draft_graph_prio_requested():
        return None
    if not (spec_algorithm is not None and spec_algorithm.is_dflash() and torch.cuda.is_available()):
        return None
    if str(device).split(":")[0] != "cuda":
        return None
    logger.info("%s ENGAGED: forward stream created at priority %d (fold side stream stays at 0)", KNOB, DRAFT_PRIORITY)
    return DRAFT_PRIORITY
