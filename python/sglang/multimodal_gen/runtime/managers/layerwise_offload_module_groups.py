from collections.abc import Sequence

LAYERWISE_OFFLOAD_MODULE_GROUP_CHOICES = (
    "dit",
    "encoder",
    "bridge",
    "vae",
    "upsampler",
    "vocoder",
    "all",
)
DEFAULT_LAYERWISE_OFFLOAD_MODULE_GROUPS = ("dit",)


def normalize_layerwise_offload_module_groups(
    module_groups: str | Sequence[str] | None,
) -> list[str] | None:
    if module_groups is None:
        return None

    raw_groups = [module_groups] if isinstance(module_groups, str) else module_groups
    normalized_groups: list[str] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, str):
            raise ValueError(
                f"Invalid layerwise offload module group: {raw_group}. "
                f"Must be one of: {', '.join(LAYERWISE_OFFLOAD_MODULE_GROUP_CHOICES)}"
            )
        for group in raw_group.split(","):
            group = group.strip().replace("-", "_").lower()
            if not group:
                continue
            if group not in LAYERWISE_OFFLOAD_MODULE_GROUP_CHOICES:
                raise ValueError(
                    f"Invalid layerwise offload module group: {group}. "
                    f"Must be one of: {', '.join(LAYERWISE_OFFLOAD_MODULE_GROUP_CHOICES)}"
                )
            if group == "all":
                return ["all"]
            if group not in normalized_groups:
                normalized_groups.append(group)

    return normalized_groups or None
