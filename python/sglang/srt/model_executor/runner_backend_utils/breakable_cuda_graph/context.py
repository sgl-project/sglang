# Copyright 2023-2026 SGLang Team
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
"""Runtime state for the breakable CUDA graph runner."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

logger = logging.getLogger(__name__)

_in_breakable_cuda_graph = False


def is_in_breakable_cuda_graph() -> bool:
    return _in_breakable_cuda_graph


@contextmanager
def enable_breakable_cuda_graph():
    """Mark the enclosed scope as inside a BCG capture/replay. Any exception
    raised inside is logged with the BCG-specific failure hint, then re-raised
    for the caller to handle."""
    global _in_breakable_cuda_graph
    _in_breakable_cuda_graph = True
    try:
        yield
    except Exception as exc:
        msg = PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG.format(
            backend=Backend.BREAKABLE, suggestions=BCG_FAILURE_HINT
        )
        logger.error(f"{type(exc).__name__}: {exc}\n{msg}")
        raise
    finally:
        _in_breakable_cuda_graph = False


# Numbered suggestion list plugged into PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG
# when the failing prefill backend is BCG. The overall message template lives
# in runner_backend_utils/__init__.py.
BCG_FAILURE_HINT = (
    "1. change to tc_piecewise by --cuda-graph-backend-prefill=tc_piecewise\n"
    "2. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled\n"
    "3. if it is an OOM problem, set --mem-fraction-static to a smaller value "
    "(e.g., 0.8 or 0.7) or set --cuda-graph-max-bs-prefill to a smaller value "
    "(e.g., 2048)\n"
)
