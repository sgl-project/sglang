# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutoDitLayerwiseOffloadConfig:
    enabled_by_default: bool = False
    high_memory_disable_gb: float | None = None


@dataclass(frozen=True)
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: AutoDitLayerwiseOffloadConfig = field(
        default_factory=AutoDitLayerwiseOffloadConfig
    )


# wan/mova auto layerwise offload is a low-memory default; H200-class GPUs can
# keep validated 720p workloads resident and ran faster without layerwise offload
WAN_MOVA_LAYERWISE_OFFLOAD_AUTO_DISABLE_MEM_GB = 130

WAN_MOVA_MODEL_DEPLOYMENT_CONFIG = ModelDeploymentConfig(
    auto_dit_layerwise_offload=AutoDitLayerwiseOffloadConfig(
        enabled_by_default=True,
        high_memory_disable_gb=WAN_MOVA_LAYERWISE_OFFLOAD_AUTO_DISABLE_MEM_GB,
    )
)
