"""
ModelDeploymentConfig provides model-specific config on how to deploy a model optimally

"""

from dataclasses import dataclass
from typing import Literal

OffloadComponentName = Literal["dit", "text_encoder", "image_encoder", "vae"]


@dataclass(frozen=True)
class ModelDeploymentConfig:
    dit_layerwise_offload_modes: tuple[Literal["auto", "memory"], ...] = ()
    auto_dit_offload_prefetch_size: float | None = None
    keep_resident_min_available_gb: float | None = None
    # Per-model resident defaults. Auto mode additionally keeps an image DiT
    # resident above the image workload memory threshold; video DiT placement
    # stays with the model's FSDP/layerwise policy.
    keep_resident_components: tuple[OffloadComponentName, ...] = ("vae",)
    fsdp_auto_min_available_memory_gb: float | None = None
    fsdp_auto_requires_cfg: bool = True
    fsdp_auto_requires_default_parallelism: bool = True
    auto_enable_cfg_parallel: bool = True
    # degree 1 keeps CFG parallel disabled and leaves GPUs available for SP
    auto_cfg_parallel_degree_by_num_gpus: tuple[tuple[int, int], ...] = ()
    # torch.compile is model opt-in because it can be slower than eager for
    # diffusion workloads dominated by already-optimized kernels
    speed_mode_enable_torch_compile_by_default: bool = False
    supports_cfg_parallel: bool = True

    def get_auto_cfg_parallel_degree(self, num_gpus: int) -> int:
        for candidate_num_gpus, cfg_degree in self.auto_cfg_parallel_degree_by_num_gpus:
            if candidate_num_gpus == num_gpus:
                return cfg_degree
        return 2
