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
"""Lightweight per-request stage tracing for latency debugging.

Enable with ``SGLANG_DEBUG_REQUEST_TRACE=1``. When on, each request emits a
line at the following lifecycle points, tagged with the request id(s) and a
high-resolution wall-clock timestamp, in whichever process the stage runs:

    received / tokenize / dispatch / postprocess -> TokenizerManager process
    detokenize                                   -> DetokenizerManager process

The ``phase`` field marks the point within a stage; the values differ by
stage rather than being a uniform start/end pair:

    received     start
    tokenize     start, end
    dispatch     start            (fire-and-forget send; no end)
    postprocess  first, finished  (first output batch; request finished)
    detokenize   first, finished

These lines are prefixed with ``[RTRACE]`` and emitted at INFO level, so they
show up whenever the process log level is INFO or lower (the default). The
Scheduler ``prefill``/``decode`` stages do NOT emit ``[RTRACE]`` lines; they
instead append ``rids=[...]`` to the existing scheduler stats line (gated by
the same env var), which correlates a batch with the ``[RTRACE]`` lines by
``rid``. Join everything across processes on ``rid`` in the combined logs.
This is a debug-only tool; it is a no-op unless the env var is set.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Union

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def req_trace_enabled() -> bool:
    """Whether request tracing is turned on."""
    return envs.SGLANG_DEBUG_REQUEST_TRACE.get()


def req_trace(
    stage: str,
    phase: str,
    rids: Union[str, Iterable[str]],
    extra: str = "",
) -> None:
    """Emit one trace line for ``stage``.

    Args:
        stage: lifecycle stage name, e.g. "received", "tokenize", "dispatch",
            "postprocess", "detokenize".
        phase: the point within the stage, e.g. "start", "end", "first",
            "finished" (which values apply depends on the stage).
        rids: a single request id, or an iterable of request ids (used by the
            batched dispatch path).
        extra: optional extra key=value context appended to the line.

    No-op unless ``SGLANG_DEBUG_REQUEST_TRACE`` is set.
    """
    if not envs.SGLANG_DEBUG_REQUEST_TRACE.get():
        return
    if isinstance(rids, str):
        rid_list = [rids]
    else:
        rid_list = list(rids)
    ts = time.time()
    suffix = f" {extra}" if extra else ""
    logger.info(
        f"[RTRACE] stage={stage} phase={phase} ts={ts:.6f} "
        f"n={len(rid_list)} rids={rid_list}{suffix}"
    )
