from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.server_args import LORA_MERGE_MODES
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

VALID_TARGETS = ("all", "transformer", "transformer_2", "critic")


def _uses_dtensor_weights(lora_layers: dict[str, BaseLayerWithLoRA]) -> bool:
    return any(isinstance(layer.weight, DTensor) for layer in lora_layers.values())


def _has_active_unmerged_lora(
    lora_layers: dict[str, BaseLayerWithLoRA],
) -> bool:
    return any(
        not layer.merged and not layer.disable_adapter for layer in lora_layers.values()
    )


def _should_merge_lora_for_layers(
    module_name: str,
    lora_layers: dict[str, BaseLayerWithLoRA],
    merge_mode: str,
) -> bool:
    if merge_mode == "dynamic":
        return False
    uses_dtensor_weights = _uses_dtensor_weights(lora_layers)
    if merge_mode == "auto":
        if uses_dtensor_weights:
            logger.info(
                "Using dynamic LoRA for %s because FSDP-sharded weights would require a full-gather merge.",
                module_name,
            )
            return False
        return True
    if uses_dtensor_weights:
        logger.warning(
            "Merging LoRA for %s with FSDP-sharded weights may require full-gather and can OOM.",
            module_name,
        )
    return True


def _resolve_lora_merge_mode(
    merge_weights: bool | None, merge_mode: str | None, server_adapter_merge_mode: str
) -> str:
    if merge_mode is None:
        if merge_weights is not None:
            merge_mode = "merge" if merge_weights else "dynamic"
        else:
            merge_mode = server_adapter_merge_mode
    if merge_mode not in LORA_MERGE_MODES:
        raise ValueError(
            f"Invalid LoRA merge mode: {merge_mode}. Valid modes: {LORA_MERGE_MODES}"
        )
    return merge_mode


def _normalize_lora_params(
    lora_nickname: str | list[str],
    lora_path: str | None | list[str | None],
    strength: float | list[float],
    target: str | list[str],
) -> tuple[list[str], list[str | None], list[float], list[str]]:
    """
    Normalize LoRA parameters to lists for multi-LoRA support.

    Requirements:
    - each nickname must have a corresponding lora_path (no implicit repeat)
    - strength / target if scalar broadcast, else length must match nickname
    """
    # nickname
    if isinstance(lora_nickname, str):
        lora_nicknames = [lora_nickname]
    else:
        lora_nicknames = lora_nickname

    # lora_path: require 1:1 mapping with nickname (no implicit repeat)
    if isinstance(lora_path, list):
        lora_paths = lora_path
    else:
        lora_paths = [lora_path]
    if len(lora_paths) != len(lora_nicknames):
        raise ValueError(
            f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
            f"but lora_path has {len(lora_paths)} items. "
            "Provide one path per nickname."
        )

    # strength and target: allow scalar broadcast, else length must match
    if isinstance(strength, (int, float)):
        strengths = [float(strength)] * len(lora_nicknames)
    else:
        strengths = [float(s) for s in strength]
    if len(strengths) != len(lora_nicknames):
        raise ValueError(
            f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
            f"but strength has {len(strengths)} items"
        )

    if isinstance(target, str):
        targets = [target] * len(lora_nicknames)
    else:
        targets = target

    if len(targets) != len(lora_nicknames):
        raise ValueError(
            f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
            f"but target has {len(targets)} items"
        )

    # Validate targets
    invalid_targets = [t for t in targets if t not in VALID_TARGETS]
    if invalid_targets:
        raise ValueError(
            f"Invalid target(s): {invalid_targets}. Valid targets: {VALID_TARGETS}"
        )

    return lora_nicknames, lora_paths, strengths, targets
