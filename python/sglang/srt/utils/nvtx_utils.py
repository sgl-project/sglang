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
"""Opt-in NVTX annotations gated by the ``SGLANG_ENABLE_NVTX_*`` env flags."""

import logging
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

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

_NVTX_COLOR_MAP = {
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

_NULL_CONTEXT = nullcontext()


def _resolve_color(debug_name: str, color: Optional[str]) -> Optional[str]:
    return color if color is not None else _NVTX_COLOR_MAP.get(debug_name)


@contextmanager
def _nvtx_range_impl(debug_name: str, color: Optional[str]):
    # Only pay record_function's per-call cost when a profiler is collecting;
    # for Nsight-only runs the nvtx marker alone is enough.
    if torch.autograd._profiler_enabled():
        with torch.autograd.profiler.record_function(debug_name):
            with _nvtx_module.annotate(debug_name, color=color):
                yield
    else:
        with _nvtx_module.annotate(debug_name, color=color):
            yield


def nvtx_range(debug_name: str, *, color: Optional[str] = None, enabled: bool):
    if not enabled:
        return _NULL_CONTEXT
    return _nvtx_range_impl(debug_name, _resolve_color(debug_name, color))


def nvtx_annotated_method(
    debug_name: str, *, color: Optional[str] = None, enabled: bool
):
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


scheduler_nvtx_method = partial(nvtx_annotated_method, enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(nvtx_range, enabled=NVTX_OPERATIONS_ENABLED)
