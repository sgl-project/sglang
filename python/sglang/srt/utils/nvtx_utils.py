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
"""Lightweight NVTX annotations for the scheduler main loop.

Enabled via the ``SGLANG_ENABLE_NVTX`` environment variable (off by default).
When disabled, the decorator/context manager add zero runtime overhead so they
are safe to leave on hot scheduler paths.

Extended by JoyFuture: added markers for the overlap event loop, running batch
update, and prefill batch selection so the full scheduler pipeline is visible
in Nsight Systems.
"""

import logging
from contextlib import contextmanager, nullcontext
from functools import wraps

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NVTX_ENV_ENABLED = envs.SGLANG_ENABLE_NVTX.get()
_nvtx_module = None
if _NVTX_ENV_ENABLED:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "SGLANG_ENABLE_NVTX is set, but the `nvtx` package is missing. "
            "NVTX annotations are disabled."
        )

NVTX_ENABLED = _nvtx_module is not None

# Colors are assigned per scheduler main-loop stage so the markers are easy to
# distinguish in Nsight Systems.
_NVTX_COLOR_MAP = {
    # === Scheduler main loop (pipeline order) ===
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
    # === Extended markers (JoyFuture) ===
    "scheduler.event_loop_normal": "dark_blue",
    "scheduler.event_loop_overlap": "teal",
    "scheduler.update_running_batch": "orange",
    "scheduler.get_new_batch_prefill": "magenta",
}


@contextmanager
def _nvtx_range_enabled(debug_name: str):
    color = _NVTX_COLOR_MAP.get(debug_name)
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


if NVTX_ENABLED:
    nvtx_range = _nvtx_range_enabled
else:
    # When NVTX is disabled, hand back a shared no-op context manager so hot
    # paths using `with nvtx_range(...)` pay no per-call generator overhead.
    _NULL_CONTEXT = nullcontext()

    def nvtx_range(debug_name: str):
        return _NULL_CONTEXT


def nvtx_annotated_method(debug_name: str):
    # Decide at decoration time. When NVTX is disabled this returns the
    # original function untouched, so decorated methods on hot paths have zero
    # runtime cost.
    if not NVTX_ENABLED:
        return lambda func: func

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(debug_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
