from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

import torch.nn as nn

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
    is_dit_component_name,
    is_image_encoder_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
)


class ComponentResidencyMode(str, Enum):
    RESIDENT = "resident"
    COMPONENT_OFFLOAD = "component-offload"
    LAYERWISE_OFFLOAD = "layerwise-offload"


def is_fsdp_managed_module(module: nn.Module) -> bool:
    return module.__class__.__name__.startswith("FSDP")


COMPONENT_RESIDENCY_GROUPS = frozenset(
    {
        LAYERWISE_OFFLOAD_ALL_COMPONENTS,
        LAYERWISE_OFFLOAD_DIT_GROUP,
        LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
        LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
        LAYERWISE_OFFLOAD_VAE_GROUP,
    }
)
COMPONENT_RESIDENCY_GROUP_PRECEDENCE = (
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
)


def normalize_component_residency(
    policies: str | Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str] | None:
    if policies is None:
        return None

    if isinstance(policies, Mapping):
        entries = list(policies.items())
    else:
        raw_policies = [policies] if isinstance(policies, str) else policies
        entries: list[tuple[str, str]] = []
        for raw_policy in raw_policies:
            if not isinstance(raw_policy, str):
                raise ValueError(f"Invalid component residency policy: {raw_policy}.")
            for assignment in raw_policy.split(","):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" not in assignment:
                    raise ValueError(
                        "Component residency policies must use COMPONENT=MODE, got "
                        f"{assignment!r}."
                    )
                selector, mode = assignment.split("=", 1)
                entries.append((selector, mode))

    normalized: dict[str, str] = {}
    valid_modes = {mode.value for mode in ComponentResidencyMode}
    for raw_selector, raw_mode in entries:
        if not isinstance(raw_selector, str) or not isinstance(raw_mode, str):
            raise ValueError(
                f"Invalid component residency policy: {raw_selector}={raw_mode}."
            )
        selector = raw_selector.strip().replace("-", "_").lower()
        mode = raw_mode.strip().replace("_", "-").lower()
        if not selector:
            raise ValueError("Component residency selector cannot be empty.")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid component residency mode {raw_mode!r} for {selector!r}. "
                f"Expected one of: {', '.join(sorted(valid_modes))}."
            )
        normalized[selector] = mode

    return normalized or None


def component_residency_selector_matches(component_name: str, selector: str) -> bool:
    if selector == LAYERWISE_OFFLOAD_ALL_COMPONENTS:
        return True
    if selector == LAYERWISE_OFFLOAD_DIT_GROUP:
        return is_dit_component_name(component_name)
    if selector == LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP:
        return is_text_encoder_component_name(component_name)
    if selector == LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP:
        return is_image_encoder_component_name(component_name)
    if selector == LAYERWISE_OFFLOAD_VAE_GROUP:
        return is_vae_component_name(component_name)
    return component_name == selector


def resolve_explicit_component_residency(
    component_name: str, policies: Mapping[str, str] | None
) -> ComponentResidencyMode | None:
    if not policies:
        return None

    exact_mode = policies.get(component_name)
    if exact_mode is not None:
        return ComponentResidencyMode(exact_mode)

    matching_group_mode = next(
        (
            policies[selector]
            for selector in COMPONENT_RESIDENCY_GROUP_PRECEDENCE
            if selector in policies
            and component_residency_selector_matches(component_name, selector)
        ),
        None,
    )
    selected_mode = matching_group_mode or policies.get(
        LAYERWISE_OFFLOAD_ALL_COMPONENTS
    )
    return ComponentResidencyMode(selected_mode) if selected_mode is not None else None


def resolve_diffusers_pipeline_offload(
    policies: Mapping[str, str] | None,
) -> bool | None:
    if policies is None:
        return None

    if ComponentResidencyMode.LAYERWISE_OFFLOAD.value in policies.values():
        raise ValueError(
            "--component-residency layerwise-offload requires the native "
            "SGLang backend; the diffusers backend exposes only pipeline-wide "
            "model CPU offload"
        )

    pipeline_wide_mode = policies.get(LAYERWISE_OFFLOAD_ALL_COMPONENTS)
    if len(policies) == 1 and pipeline_wide_mode is not None:
        return (
            ComponentResidencyMode(pipeline_wide_mode)
            == ComponentResidencyMode.COMPONENT_OFFLOAD
        )

    raise ValueError(
        "The diffusers backend supports only pipeline-wide residency: use "
        "--component-residency all=resident or all=component-offload"
    )
