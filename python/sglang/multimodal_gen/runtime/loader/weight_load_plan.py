from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightLoadPlan:
    """Device plan for checkpoint loading, before runtime residency takes over."""

    # Device used while materializing checkpoint tensors from files.
    checkpoint_load_device: torch.device
    # Device required while running process_weights_after_loading; None means unchanged.
    weight_postprocess_device: torch.device | None = None
    # mps layerwise loading retains compatible safetensors tensors as CPU-backed
    # parameters instead of materializing a second unified-memory copy
    mps_layerwise_cpu_staging: bool = False
    # Delay final CPU placement until after device-side weight postprocessing.
    defer_cpu_placement: bool = False
    # keep the complete mapped checkpoint state dict on the load device
    load_full_state_dict_on_device: bool = False

    @classmethod
    def for_component(
        cls,
        *,
        checkpoint_load_device: torch.device,
        needs_device_weight_postprocess: bool,
        component_starts_on_cpu: bool,
        load_full_state_dict_on_device: bool = False,
        mps_layerwise_cpu_staging: bool = False,
    ) -> "WeightLoadPlan":
        # if on-device weight postprocessing is required, load directly to device to speedup loading
        weight_postprocess_device = (
            checkpoint_load_device if needs_device_weight_postprocess else None
        )
        return cls(
            checkpoint_load_device=checkpoint_load_device,
            weight_postprocess_device=weight_postprocess_device,
            defer_cpu_placement=(
                needs_device_weight_postprocess and component_starts_on_cpu
            ),
            load_full_state_dict_on_device=load_full_state_dict_on_device,
            mps_layerwise_cpu_staging=mps_layerwise_cpu_staging,
        )
