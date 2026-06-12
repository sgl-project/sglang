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
"""Lightweight per-request lifecycle tracing.

This module emits one structured log line per request lifecycle event
(``received``, ``queued``, ``prefill_start``, ``prefill_end``, ``finished``,
``aborted``) so that developers can correlate a single ``rid`` across the
tokenizer, scheduler, and runtime stages.

It is intentionally minimal: no external dependencies, no extra IDs, no
OpenTelemetry. The emission is gated by the ``SGLANG_TRACE_REQUEST_LIFECYCLE``
environment variable and is a fast no-op when disabled.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from sglang.srt.environ import envs

logger = logging.getLogger("sglang.request_trace")

# Cache the env flag on first use to keep the per-call overhead to a single
# attribute load when tracing is disabled (the common case).
_ENABLED: Optional[bool] = None


def is_request_trace_enabled() -> bool:
    """Return whether request lifecycle tracing is enabled."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(envs.SGLANG_TRACE_REQUEST_LIFECYCLE.get())
    return _ENABLED


def _reset_cache_for_testing() -> None:
    """Force a re-read of the env var. Intended for tests only."""
    global _ENABLED
    _ENABLED = None


def trace_req_event(rid: Any, event: str, **fields: Any) -> None:
    """Emit a single trace event for ``rid`` at lifecycle stage ``event``.

    Extra keyword fields (e.g. ``batch_id``, ``batch_size``,
    ``finish_reason``) are appended to the log line for correlation.
    No-op when ``SGLANG_TRACE_REQUEST_LIFECYCLE`` is not set.
    """
    if not is_request_trace_enabled():
        return

    parts = [f"event={event}", f"rid={rid}", f"ts={time.time():.6f}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info("req_trace " + " ".join(parts))
