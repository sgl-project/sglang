# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lightweight NVTX annotations for hot SGLang code paths.

Two independent, opt-in gates select which spans are emitted:

* ``SGLANG_ENABLE_NVTX_SCHEDULER`` -- scheduler main-loop stages
  (receive / process inputs / pick next batch / run forward / post-process).
* ``SGLANG_ENABLE_NVTX_OPERATIONS`` -- the batch-overlap operation pipeline
  (per-stage and per-op spans inside the model forward).

Both are off by default. When a gate is off the decorator returns the original
function untouched and the context manager hands back a shared no-op, so
decorated hot paths pay no runtime cost. The decision is made once at import /
decoration time, not per call.
"""

import logging
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Per-subsystem env gates. A subsystem's spans are emitted only when its flag is
# set AND the `nvtx` package is importable; the two checks are folded together
# below so call sites can branch on a single boolean.
NVTX_SCHEDULER_ENABLED = envs.SGLANG_ENABLE_NVTX_SCHEDULER.get()
NVTX_OPERATIONS_ENABLED = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

_nvtx_module = None
if NVTX_SCHEDULER_ENABLED or NVTX_OPERATIONS_ENABLED:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "An SGLANG_ENABLE_NVTX_* flag is set, but the `nvtx` package is "
            "missing. NVTX annotations are disabled."
        )

NVTX_AVAILABLE = _nvtx_module is not None
NVTX_SCHEDULER_ENABLED = NVTX_SCHEDULER_ENABLED and NVTX_AVAILABLE
NVTX_OPERATIONS_ENABLED = NVTX_OPERATIONS_ENABLED and NVTX_AVAILABLE

# Default colors for statically-named spans, so the markers are easy to tell
# apart in Nsight Systems. Call sites with dynamic names (e.g. the operation
# pipeline) pass an explicit `color` instead of relying on this map.
_NVTX_COLOR_MAP = {
    # Scheduler main loop (pipeline order).
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

# Shared no-op handed back on disabled paths so `with nvtx_range(...)` pays no
# per-call generator overhead.
_NULL_CONTEXT = nullcontext()


def _resolve_color(debug_name: str, color: Optional[str]) -> Optional[str]:
    if color is not None:
        return color
    return _NVTX_COLOR_MAP.get(debug_name)


@contextmanager
def _nvtx_range_impl(debug_name: str, color: Optional[str]):
    # record_function carries a non-trivial (~microseconds) cost per call even
    # when no PyTorch profiler is collecting, so only pay it when one is active
    # (e.g. Chrome-trace export). For Nsight-only runs the nvtx.annotate marker
    # alone is enough.
    if torch.autograd._profiler_enabled():
        with torch.autograd.profiler.record_function(debug_name):
            with _nvtx_module.annotate(debug_name, color=color):
                yield
    else:
        with _nvtx_module.annotate(debug_name, color=color):
            yield


def nvtx_range(debug_name: str, *, color: Optional[str] = None, enabled: bool):
    """Return a context manager emitting an NVTX range for ``debug_name``.

    ``enabled`` is the caller subsystem's resolved gate (e.g.
    ``NVTX_OPERATIONS_ENABLED``); when false a shared no-op is returned. ``color``
    overrides the color otherwise looked up from the name->color map.
    """
    if not enabled:
        return _NULL_CONTEXT
    return _nvtx_range_impl(debug_name, _resolve_color(debug_name, color))


def nvtx_annotated_method(
    debug_name: str, *, color: Optional[str] = None, enabled: bool
):
    """Decorate a method so each call emits an NVTX range. See ``nvtx_range``.

    The enable decision is made once at decoration time: when ``enabled`` is
    false the original function is returned untouched, so decorated methods on
    hot paths have zero runtime cost.
    """
    if not enabled:
        return lambda func: func

    color = _resolve_color(debug_name, color)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _nvtx_range_impl(debug_name, color):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-bound per-subsystem helpers so call sites don't repeat the gate: each
# binds its subsystem's resolved enable flag, leaving callers to pass only the
# span name (and optional color).
scheduler_nvtx_method = partial(nvtx_annotated_method, enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(nvtx_range, enabled=NVTX_OPERATIONS_ENABLED)
