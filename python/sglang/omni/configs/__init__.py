# SPDX-License-Identifier: Apache-2.0
"""Model-specific omni wiring helpers."""

from sglang.omni.configs.registry import (
    DEFAULT_OMNI_MODEL_KEY,
    get_or_create_omni_coordinator_from_scheduler,
    resolve_omni_model_key,
)
from sglang.omni.configs.sensenova_u1 import (
    SenseNovaU1OmniPlugin,
    build_sensenova_u1_coordinator,
    build_sensenova_u1_coordinator_from_scheduler,
)

__all__ = [
    "DEFAULT_OMNI_MODEL_KEY",
    "SenseNovaU1OmniPlugin",
    "build_sensenova_u1_coordinator",
    "build_sensenova_u1_coordinator_from_scheduler",
    "get_or_create_omni_coordinator_from_scheduler",
    "resolve_omni_model_key",
]
