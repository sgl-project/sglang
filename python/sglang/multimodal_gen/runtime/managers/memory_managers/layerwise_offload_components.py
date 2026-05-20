from collections.abc import Collection, Sequence

LAYERWISE_OFFLOAD_ALL_COMPONENTS = "all"
LAYERWISE_OFFLOAD_DIT_GROUP = "dit"
LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP = "text_encoder"
LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP = "image_encoder"
LAYERWISE_OFFLOAD_VAE_GROUP = "vae"
LAYERWISE_OFFLOAD_DEFAULT_GROUP = "default"

# Components whose layerwise policy has been validated as a better default than
# component-level CPU offload when the user has not pinned their placement.
LAYERWISE_OFFLOAD_DEFAULT_GROUP_COMPONENTS = (
    LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_IMAGE_ENCODER_GROUP,
    LAYERWISE_OFFLOAD_VAE_GROUP,
)
DIT_COMPONENT_NAMES = frozenset(
    {
        "transformer",
        "transformer_2",
        "video_dit",
        "video_dit_2",
        "audio_dit",
        "dual_tower_bridge",
    }
)
VAE_COMPONENT_NAMES = frozenset(
    {
        "vae",
        "video_vae",
        "audio_vae",
        "vocoder",
        "spatial_upsampler",
        "condition_image_encoder",
    }
)
DEFAULT_LAYERWISE_VAE_COMPONENT_NAMES = frozenset(
    {
        "vae",
        "video_vae",
        "condition_image_encoder",
    }
)
CPU_OFFLOAD_FLAG_NAMES = (
    "dit_cpu_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
)


def is_dit_component_name(component_name: str) -> bool:
    return component_name in DIT_COMPONENT_NAMES


def is_text_encoder_component_name(component_name: str) -> bool:
    return component_name.startswith("text_encoder") or component_name.endswith(
        "text_encoder"
    )


def is_image_encoder_component_name(component_name: str) -> bool:
    return component_name == "image_encoder"


def is_vae_component_name(component_name: str) -> bool:
    return component_name in VAE_COMPONENT_NAMES


def layerwise_component_matches_selection(
    component_name: str,
    selected_component_name: str,
) -> bool:
    """if the provided component_name (unnormalized, e.g., text_encoder_2)  matches with the selected_component_name (normalized)"""
    if selected_component_name == LAYERWISE_OFFLOAD_TEXT_ENCODER_GROUP:
        return is_text_encoder_component_name(component_name)
    if selected_component_name == LAYERWISE_OFFLOAD_VAE_GROUP:
        # `vae` is a default-policy selector; AV-side decoders remain explicit-only
        return component_name in DEFAULT_LAYERWISE_VAE_COMPONENT_NAMES
    return component_name == selected_component_name


def layerwise_component_matches_any_selection(
    component_name: str,
    selected_component_names: Collection[str],
) -> bool:
    return any(
        layerwise_component_matches_selection(component_name, selected_component_name)
        for selected_component_name in selected_component_names
    )


def cpu_offload_flags_for_layerwise_components(
    component_names: Sequence[str],
) -> tuple[str, ...]:
    component_names = normalize_layerwise_offload_components(component_names) or []
    if LAYERWISE_OFFLOAD_ALL_COMPONENTS in component_names:
        return CPU_OFFLOAD_FLAG_NAMES

    flag_names: list[str] = []
    if LAYERWISE_OFFLOAD_DIT_GROUP in component_names:
        flag_names.append("dit_cpu_offload")

    for component_name in component_names:
        if component_name == LAYERWISE_OFFLOAD_DIT_GROUP:
            continue
        if is_dit_component_name(component_name):
            flag_name = "dit_cpu_offload"
        elif is_text_encoder_component_name(component_name):
            flag_name = "text_encoder_cpu_offload"
        elif is_image_encoder_component_name(component_name):
            flag_name = "image_encoder_cpu_offload"
        elif is_vae_component_name(component_name):
            flag_name = "vae_cpu_offload"
        else:
            continue

        if flag_name not in flag_names:
            flag_names.append(flag_name)

    return tuple(flag_names)


def expand_layerwise_offload_component_group(component_name: str) -> tuple[str, ...]:
    if component_name == LAYERWISE_OFFLOAD_DEFAULT_GROUP:
        return LAYERWISE_OFFLOAD_DEFAULT_GROUP_COMPONENTS
    return (component_name,)


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
            for expanded_component_name in expand_layerwise_offload_component_group(
                component_name
            ):
                if expanded_component_name == LAYERWISE_OFFLOAD_ALL_COMPONENTS:
                    return [LAYERWISE_OFFLOAD_ALL_COMPONENTS]
                if expanded_component_name not in normalized_components:
                    normalized_components.append(expanded_component_name)

    return normalized_components or None
