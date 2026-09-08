"""Parsing and resolution for component residency CLI modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
    component_base_name,
    is_dit_component_name,
    is_image_encoder_component_name,
    is_text_encoder_component_name,
)

RESIDENT = "resident"
COMPONENT_OFFLOAD = "component-offload"
LAYERWISE_OFFLOAD = "layerwise-offload"
COMPONENT_RESIDENCY_MODES = frozenset(
    (
        RESIDENT,
        COMPONENT_OFFLOAD,
        LAYERWISE_OFFLOAD,
    )
)


class ComponentResidencyError(ValueError):
    """Invalid or unsupported user-selected component residency."""


COMPONENT_RESIDENCY_GROUPS = frozenset(
    (
        LAYERWISE_OFFLOAD_ALL_COMPONENTS,
        LAYERWISE_OFFLOAD_DIT_GROUP,
        LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
        LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
        LAYERWISE_OFFLOAD_VAE_GROUP,
    )
)
COMPONENT_RESIDENCY_GROUP_PRECEDENCE = (
    LAYERWISE_OFFLOAD_DIT_GROUP,
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
)


def normalize_component_residency(
    assignments: str | Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str] | None:
    if assignments is None:
        return None

    if isinstance(assignments, Mapping):
        entries = assignments.items()
    else:
        values = [assignments] if isinstance(assignments, str) else assignments
        parsed_entries: list[tuple[str, str]] = []
        for value in values:
            if not isinstance(value, str):
                raise ComponentResidencyError(
                    f"Invalid component residency assignment: {value!r}"
                )
            for assignment in value.split(","):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" not in assignment:
                    raise ComponentResidencyError(
                        "Component residency must use COMPONENT=MODE, got "
                        f"{assignment!r}"
                    )
                selector, mode = assignment.split("=", 1)
                parsed_entries.append((selector, mode))
        entries = parsed_entries

    normalized: dict[str, str] = {}
    for raw_selector, raw_mode in entries:
        if not isinstance(raw_selector, str) or not isinstance(raw_mode, str):
            raise ComponentResidencyError(
                f"Invalid component residency assignment: {raw_selector!r}={raw_mode!r}"
            )
        selector = raw_selector.strip().replace("-", "_").lower()
        mode = raw_mode.strip().replace("_", "-").lower()
        if not selector:
            raise ComponentResidencyError(
                "Component residency selector cannot be empty"
            )
        if mode not in COMPONENT_RESIDENCY_MODES:
            expected = ", ".join(sorted(COMPONENT_RESIDENCY_MODES))
            raise ComponentResidencyError(
                f"Invalid component residency mode {raw_mode!r} for "
                f"{selector!r}; expected one of: {expected}"
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
        return component_base_name(component_name) in (
            "vae",
            "video_vae",
            "audio_vae",
            "delight_vae",
            "hy3dshape_vae",
            "paint_vae",
        )
    return component_name == selector


def resolve_component_residency_mode(
    component_name: str, assignments: Mapping[str, str] | None
) -> str | None:
    if not assignments:
        return None

    exact_mode = assignments.get(component_name)
    if exact_mode is not None:
        return exact_mode

    for selector in COMPONENT_RESIDENCY_GROUP_PRECEDENCE:
        mode = assignments.get(selector)
        if mode is not None and component_residency_selector_matches(
            component_name, selector
        ):
            return mode
    return assignments.get(LAYERWISE_OFFLOAD_ALL_COMPONENTS)


def resolve_diffusers_pipeline_offload(
    assignments: Mapping[str, str] | None,
) -> bool | None:
    if assignments is None:
        return None
    if LAYERWISE_OFFLOAD in assignments.values():
        raise ComponentResidencyError(
            "--component-residency layerwise-offload requires the native SGLang backend"
        )

    pipeline_mode = assignments.get(LAYERWISE_OFFLOAD_ALL_COMPONENTS)
    if len(assignments) == 1 and pipeline_mode is not None:
        return pipeline_mode == COMPONENT_OFFLOAD
    raise ComponentResidencyError(
        "The diffusers backend supports only pipeline-wide residency; use "
        "--component-residency all=resident or all=component-offload"
    )
