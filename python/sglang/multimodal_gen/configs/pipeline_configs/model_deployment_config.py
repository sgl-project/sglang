"""
ModelDeploymentConfig provides model-specific config on how to deploy a model optimally

"""

from dataclasses import dataclass
from typing import Literal

OffloadComponentName = Literal["dit", "text_encoder", "image_encoder"]


@dataclass(frozen=True)
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: bool = False
    # if the available memory is bigger than this value, keep dit resident instead of apply layerwise-offload
    auto_dit_layerwise_offload_high_memory_disable_gb: float | None = None
    auto_disable_component_offload_min_available_memory_gb: float | None = None
    # keep this explicit because large encoders can OOM even when DiT fits resident
    auto_disable_component_offload_components: tuple[OffloadComponentName, ...] = (
        "dit",
        "text_encoder",
        "image_encoder",
    )
    fsdp_auto_min_available_memory_gb: float | None = None
    fsdp_auto_requires_cfg: bool = True
    fsdp_auto_requires_default_parallelism: bool = True
    auto_enable_cfg_parallel: bool = True
    # degree 1 keeps CFG parallel disabled and leaves GPUs available for SP
    auto_cfg_parallel_degree_by_num_gpus: tuple[tuple[int, int], ...] = ()

    def get_auto_cfg_parallel_degree(self, num_gpus: int) -> int:
        for candidate_num_gpus, cfg_degree in self.auto_cfg_parallel_degree_by_num_gpus:
            if candidate_num_gpus == num_gpus:
                return cfg_degree
        return 2
