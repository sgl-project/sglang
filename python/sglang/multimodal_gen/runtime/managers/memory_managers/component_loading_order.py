"""Memory-aware ordering for pipeline component weight loads.

The pipeline owns component selection, path resolution, and actual loading; this
module only ranks already-selected load specs.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
    is_image_encoder_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
)


@dataclass(frozen=True)
class ComponentLoadSpec:
    """One pipeline component that still needs a real weight load."""

    module_name: str
    load_module_name: str
    transformers_or_diffusers: str
    architecture: str | None
    index: int


def _component_base_name(component_name: str) -> str:
    prefix, separator, suffix = component_name.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return component_name


def _component_variant_priority(component_name: str) -> int:
    _, separator, suffix = component_name.rpartition("_")
    if separator and suffix.isdigit():
        return -int(suffix)
    return 0


def component_load_risk_rank(component_name: str) -> int:
    """Lower rank means the component should be loaded with more free VRAM."""
    candidate_names = (component_name, _component_base_name(component_name))
    if any(is_dit_component_name(name) for name in candidate_names):
        return 0
    if any(is_text_encoder_component_name(name) for name in candidate_names):
        return 1
    if any(is_image_encoder_component_name(name) for name in candidate_names):
        return 2
    if any(is_vae_component_name(name) for name in candidate_names):
        return 3
    return 10


def order_component_load_specs(
    component_specs: list[ComponentLoadSpec],
) -> list[ComponentLoadSpec]:
    # load weight-heavy modules before small CPU-side helpers to reduce startup peak OOMs
    return sorted(
        component_specs,
        key=lambda spec: (
            component_load_risk_rank(spec.load_module_name),
            _component_variant_priority(spec.load_module_name),
            spec.index,
        ),
    )
