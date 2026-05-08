from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: bool = False
    auto_dit_layerwise_offload_high_memory_disable_gb: float | None = None
    fsdp_cfg_auto_min_available_memory_gb: float | None = None
