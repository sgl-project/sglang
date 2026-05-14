# SPDX-License-Identifier: Apache-2.0
"""Model registry for SRT-hosted omni orchestrators.

Runtime transport asks this module for an orchestrator by model name. Model-
specific imports stay here so `sglang.omni.runtime` remains model-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.omni.core.coordinator import OmniCoordinator
    from sglang.srt.managers.scheduler import Scheduler

DEFAULT_OMNI_MODEL_KEY = "sensenova-u1"


def resolve_omni_model_key(model_name: str | None) -> str:
    if model_name is None or not str(model_name).strip():
        return DEFAULT_OMNI_MODEL_KEY

    normalized = str(model_name).strip().lower()
    if "sensenova" in normalized or "u1" in normalized:
        return DEFAULT_OMNI_MODEL_KEY
    raise ValueError(f"Unsupported omni model {model_name!r}")


def get_or_create_omni_orchestrator_from_scheduler(
    *,
    scheduler: "Scheduler",
    model_name: str | None,
) -> "OmniCoordinator":
    model_key = resolve_omni_model_key(model_name)
    cache = scheduler.omni_scheduler_state.orchestrators
    with scheduler.omni_scheduler_state.orchestrator_lock:
        if model_key not in cache:
            # 1. build model-specific wiring outside omni.runtime
            cache[model_key] = _build_orchestrator_from_scheduler(
                scheduler=scheduler,
                model_key=model_key,
            )
    return cache[model_key]


def _build_orchestrator_from_scheduler(
    *,
    scheduler: "Scheduler",
    model_key: str,
) -> "OmniCoordinator":
    # TODO: make a registry
    if model_key == DEFAULT_OMNI_MODEL_KEY:
        from sglang.omni.configs.sensenova_u1 import (
            build_sensenova_u1_orchestrator_from_scheduler,
        )

        return build_sensenova_u1_orchestrator_from_scheduler(
            scheduler=scheduler,
            server_args=scheduler.server_args,
        )
    raise ValueError(f"Unsupported omni model key {model_key!r}")
