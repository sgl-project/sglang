from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightLoadPlan:
    """Device plan for checkpoint loading, before runtime residency takes over."""

    # Device used while materializing checkpoint tensors from files.
    checkpoint_load_device: torch.device
    # Device required while running process_weights_after_loading; None means unchanged.
    weight_postprocess_device: torch.device | None = None
    # Delay non-FSDP component CPU offload until after weight postprocessing.
    defer_component_cpu_offload: bool = False

    @classmethod
    def for_component(
        cls,
        *,
        checkpoint_load_device: torch.device,
        needs_device_weight_postprocess: bool,
        component_cpu_offload: bool,
    ) -> "WeightLoadPlan":
        # if on-device weight postprocessing is required, load directly to device to speedup loading
        weight_postprocess_device = (
            checkpoint_load_device if needs_device_weight_postprocess else None
        )
        return cls(
            checkpoint_load_device=checkpoint_load_device,
            weight_postprocess_device=weight_postprocess_device,
            defer_component_cpu_offload=(
                needs_device_weight_postprocess and component_cpu_offload
            ),
        )
