"""
ModelDeploymentConfig provides model-specific config on how to deploy a model optimally

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: bool = False
    # if the available memory is bigger than this value, keep dit resident instead of apply layerwise-offload
    auto_dit_layerwise_offload_high_memory_disable_gb: float | None = None
    fsdp_cfg_auto_min_available_memory_gb: float | None = None
