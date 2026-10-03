# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams realtime (tick) session configuration.

Same checkpoint contract and defaults as ``CosmosDreamsConfig``. The subclass
exists so ``--pipeline-class-name CosmosDreamsRealtimePipeline`` refines the
model-default config and the ``/v1/realtime_video`` adapter registry, which is
keyed on the pipeline config class, selects the Cosmos-Dreams adapter.
"""

from dataclasses import dataclass, replace

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

# Measured: an 8B Cosmos-Dreams tick at the 480 tier peaks near 36 GB with the
# DiT resident; reloading the DiT per tick costs ~4.7 s, so residency is the
# realtime default whenever a 48 GB card has headroom. Explicit flags still win.
COSMOS_DREAMS_REALTIME_KEEP_RESIDENT_MIN_AVAILABLE_GB = 40


@dataclass
class CosmosDreamsRealtimeConfig(CosmosDreamsConfig):
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return replace(
            super().get_model_deployment_config(),
            keep_resident_min_available_gb=COSMOS_DREAMS_REALTIME_KEEP_RESIDENT_MIN_AVAILABLE_GB,
            keep_resident_components=("dit", "vae"),
        )
