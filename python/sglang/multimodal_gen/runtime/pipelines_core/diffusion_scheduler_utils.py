# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def clone_scheduler_runtime(scheduler: Any) -> Any:
    """Create an isolated scheduler runtime from a scheduler template or runtime."""
    return deepcopy(scheduler)


def get_or_create_request_scheduler(
    batch: Req, scheduler_template: Any, *, isolate: bool = False
) -> Any:
    """Return the scheduler runtime for this request.

    Diffusion serving currently executes one request at a time on the normal
    worker path, so reusing the stage-local scheduler preserves warmup caches
    and avoids unnecessary deepcopy overhead. Set ``isolate=True`` only when a
    request can run concurrently or outlive the stage-local scheduler state.
    """
    if batch.scheduler is None:
        batch.scheduler = (
            clone_scheduler_runtime(scheduler_template)
            if isolate
            else scheduler_template
        )
    return batch.scheduler
