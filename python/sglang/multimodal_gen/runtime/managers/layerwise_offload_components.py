from collections.abc import Sequence

LAYERWISE_OFFLOAD_ALL_COMPONENTS = "all"


def normalize_layerwise_offload_components(
    component_names: str | Sequence[str] | None,
) -> list[str] | None:
    if component_names is None:
        return None

    raw_components = (
        [component_names] if isinstance(component_names, str) else component_names
    )
    normalized_components: list[str] = []
    for raw_component in raw_components:
        if not isinstance(raw_component, str):
            raise ValueError(
                f"Invalid layerwise offload component name: {raw_component}."
            )
        for component_name in raw_component.split(","):
            component_name = component_name.strip().replace("-", "_").lower()
            if not component_name:
                continue
            if component_name == LAYERWISE_OFFLOAD_ALL_COMPONENTS:
                return [LAYERWISE_OFFLOAD_ALL_COMPONENTS]
            if component_name not in normalized_components:
                normalized_components.append(component_name)

    return normalized_components or None
