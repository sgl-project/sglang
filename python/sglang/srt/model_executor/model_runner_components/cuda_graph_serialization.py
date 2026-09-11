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
"""Lifecycle hooks for CUDA graph serialization (design section 12).

``cuda_graph_setup.capture_cuda_graphs`` calls exactly two functions from
this module, one before the runners are built and one after both captures;
``model_runner.py`` is frozen and gains nothing. Every input is a narrow
keyword argument and every result is a frozen struct, so this module needs no
``ModelRunner`` to be unit tested.

* :func:`plan_graph_serialization` projects the published ``exec.graph``
  leaves into a :class:`GraphSerializationPlan` and forces it off, with the
  reason kept, for the cases the design puts out of scope in v1 (non-CUDA
  devices and spec runners, design section 9.3).
* :func:`force_memory_pool_config` is load step 1 of design section 12 and a
  stub in this draft.
* :func:`finalize_graph_serialization` is step 5 of both lifecycles: it
  collects each runner's ``materializer.finish()`` and stops there in this
  draft.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.model_executor.graph_serialization.plan import (
    GraphSerializationPlan,
    read_plan_from_config,
)
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.graph_serialization.safety import CoverageReport

logger = logging.getLogger(__name__)


def plan_graph_serialization(
    *, device: str, is_draft_worker: bool
) -> GraphSerializationPlan:
    """Resolve this process's graph-serialization plan (design section 13).

    Reads the published bag through ``read_plan_from_config`` (never a
    ``ServerArgs`` instance). ``off`` is returned as is. Otherwise the plan
    is forced off with a reason for a non-CUDA device and for a draft worker,
    whose spec runners are capture-only in v1 (design section 9.3); an
    enabled plan is logged once on rank 0.
    """
    plan = read_plan_from_config()
    if not plan.enabled:
        return plan
    if device != "cuda":
        plan = plan.disabled(f"device {device!r} is not cuda")
    elif is_draft_worker:
        plan = plan.disabled("spec runners are capture-only in v1 (design section 9.3)")
    if not plan.enabled:
        log_info_on_rank0(logger, f"CUDA graph cache disabled: {plan.disabled_reason}")
        return plan
    log_info_on_rank0(
        logger,
        "CUDA graph cache enabled: "
        f"mode={plan.mode.value} dir={plan.cache_dir} "
        f"placement={plan.placement.value} verify={plan.verify.value}",
    )
    return plan


def force_memory_pool_config(plan: GraphSerializationPlan) -> None:
    """Load step 1 (design section 12): reuse the artifact's recorded
    pre-resize ``MemoryPoolConfig`` through a forced branch of
    ``KVCacheConfigurator.configure`` when it fits (collectively decided), so
    KV geometry and shape clamps match the saved graphs (design section 10,
    fact 22)."""
    raise NotImplementedError(
        "force_memory_pool_config: forcing the artifact's recorded pre-resize "
        "MemoryPoolConfig through KVCacheConfigurator.configure is not "
        "implemented in this draft; see DESIGN_cuda_graph_serialization.md "
        "section 10 and section 12 (load step 1)"
    )


def finalize_graph_serialization(
    plan: GraphSerializationPlan, *, prefill_runner: Any, decode_runner: Any
) -> None:
    """Step 5 of both lifecycles (design section 12), after both captures.

    A disabled plan returns at once, so the default server path is unchanged.
    Otherwise each runner that owns a ``materializer`` (an ``EagerRunner`` or
    a ``None`` phase does not) reports through ``finish()``; what follows in a
    full build, the region enumeration, communicator export, fingerprint and
    atomic rank write on save, or the collective communicator restore, shadow
    verify and smoke replay on load, is not in this draft.
    """
    if not plan.enabled:
        return None
    reports: dict[str, Optional[CoverageReport]] = {}
    for name, runner in (("prefill", prefill_runner), ("decode", decode_runner)):
        materializer = getattr(runner, "materializer", None)
        if materializer is None:
            continue
        reports[name] = materializer.finish()
    raise NotImplementedError(
        "finalize_graph_serialization: after the per-runner finish() of "
        f"{sorted(reports) or 'no runner'}, the region enumeration, communicator "
        "export, fingerprint computation, atomic rank write and rank-0 manifest "
        "(save) or the collective communicator restore, shadow verify and smoke "
        "replay (load) are not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 12 (step 5)"
    )
