from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

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

RESIDENT_STRATEGY = "resident"
COMPONENT_OFFLOAD_STRATEGY = "component-offload"
LAYERWISE_OFFLOAD_STRATEGY = "layerwise-offload"
RESIDENCY_STRATEGY_NAMES = frozenset(
    {
        RESIDENT_STRATEGY,
        COMPONENT_OFFLOAD_STRATEGY,
        LAYERWISE_OFFLOAD_STRATEGY,
    }
)


def is_fsdp_managed_module(module: nn.Module) -> bool:
    return isinstance(module, FSDPModule)


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
    strategies: str | Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str] | None:
    if strategies is None:
        return None

    if isinstance(strategies, Mapping):
        entries = list(strategies.items())
    else:
        raw_strategies = [strategies] if isinstance(strategies, str) else strategies
        entries: list[tuple[str, str]] = []
        for raw_strategy in raw_strategies:
            if not isinstance(raw_strategy, str):
                raise ValueError(
                    f"Invalid component residency strategy: {raw_strategy}."
                )
            for assignment in raw_strategy.split(","):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" not in assignment:
                    raise ValueError(
                        "Component residency strategies must use COMPONENT=STRATEGY, got "
                        f"{assignment!r}."
                    )
                selector, strategy_name = assignment.split("=", 1)
                entries.append((selector, strategy_name))

    normalized: dict[str, str] = {}
    for raw_selector, raw_strategy_name in entries:
        if not isinstance(raw_selector, str) or not isinstance(raw_strategy_name, str):
            raise ValueError(
                "Invalid component residency strategy: "
                f"{raw_selector}={raw_strategy_name}."
            )
        selector = raw_selector.strip().replace("-", "_").lower()
        strategy_name = raw_strategy_name.strip().replace("_", "-").lower()
        if not selector:
            raise ValueError("Component residency selector cannot be empty.")
        if strategy_name not in RESIDENCY_STRATEGY_NAMES:
            raise ValueError(
                "Invalid component residency strategy "
                f"{raw_strategy_name!r} for {selector!r}. "
                f"Expected one of: {', '.join(sorted(RESIDENCY_STRATEGY_NAMES))}."
            )
        normalized[selector] = strategy_name

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


def resolve_residency_strategy_name(
    component_name: str, strategies: Mapping[str, str] | None
) -> str | None:
    if not strategies:
        return None

    exact_name = strategies.get(component_name)
    if exact_name is not None:
        return exact_name

    matching_group_name = next(
        (
            strategies[selector]
            for selector in COMPONENT_RESIDENCY_GROUP_PRECEDENCE
            if selector in strategies
            and component_residency_selector_matches(component_name, selector)
        ),
        None,
    )
    selected_name = matching_group_name or strategies.get(
        LAYERWISE_OFFLOAD_ALL_COMPONENTS
    )
    return selected_name


def resolve_diffusers_pipeline_offload(
    strategies: Mapping[str, str] | None,
) -> bool | None:
    if strategies is None:
        return None

    if LAYERWISE_OFFLOAD_STRATEGY in strategies.values():
        raise ValueError(
            "--component-residency layerwise-offload requires the native "
            "SGLang backend; the diffusers backend exposes only pipeline-wide "
            "model CPU offload"
        )

    pipeline_wide_name = strategies.get(LAYERWISE_OFFLOAD_ALL_COMPONENTS)
    if len(strategies) == 1 and pipeline_wide_name is not None:
        return pipeline_wide_name == COMPONENT_OFFLOAD_STRATEGY

    raise ValueError(
        "The diffusers backend supports only pipeline-wide residency: use "
        "--component-residency all=resident or all=component-offload"
    )
