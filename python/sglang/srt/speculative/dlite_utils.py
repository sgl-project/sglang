from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DLITE_ARCHITECTURE_VERSIONS = ("dlite_v1", "dlite_v2")
DLITE_ARCHITECTURE_VERSION = DLITE_ARCHITECTURE_VERSIONS[-1]
DLITE_MODEL_ROLES = ("student", "pivot_q_student")
DLITE_MODEL_ROLE = DLITE_MODEL_ROLES[0]
DLITE_SEQUENTIAL_HEAD = "rnn"


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_dlite_target_layer_ids(
    num_target_layers: int, chs_num_layers: int
) -> tuple[int, ...]:
    """Return the fixed first/last plus evenly spaced DLite CHS layers."""
    layer_count = int(num_target_layers)
    selection_count = int(chs_num_layers)
    if layer_count < 2:
        raise ValueError(
            f"DLite num_target_layers must be at least 2, got {layer_count}."
        )
    if not 2 <= selection_count <= layer_count:
        raise ValueError(
            "DLite chs_num_layers must be in "
            f"[2, {layer_count}], got {selection_count}."
        )

    selected = {0, layer_count - 1}
    middle_count = selection_count - 2
    first_middle, last_middle = 1, layer_count - 2
    if middle_count > 0:
        span = last_middle - first_middle
        for index in range(middle_count):
            fraction = 0.5 if middle_count == 1 else index / (middle_count - 1)
            selected.add(first_middle + round(fraction * span))

    result = tuple(sorted(selected))
    if len(result) != selection_count:
        raise ValueError(
            f"DLite could not select {selection_count} CHS layers from "
            f"{layer_count}: {list(result)}."
        )
    return result


@dataclass(frozen=True)
class DLiteDraftConfig:
    architecture_version: str
    model_role: str
    block_size: int
    chs_num_layers: int
    target_layer_ids: tuple[int, ...]
    sequential_head: str
    sequential_rank: int
    mask_token_id: int
    num_target_layers: int

    @property
    def proposal_length(self) -> int:
        return self.block_size - 1

    @property
    def query_position_offset(self) -> int:
        return 0 if self.architecture_version == "dlite_v1" else 1


def parse_dlite_draft_config(draft_hf_config: Any) -> DLiteDraftConfig:
    architectures = _config_value(draft_hf_config, "architectures", None) or []
    if "DLiteDraftModel" not in architectures:
        raise ValueError(
            "DLite requires draft architectures to contain 'DLiteDraftModel', "
            f"got {architectures!r}."
        )

    raw = _config_value(draft_hf_config, "dlite_config", None) or {}
    required = {
        "architecture_version",
        "chs_num_layers",
        "mask_token_id",
        "model_role",
        "sequential_head",
        "sequential_rank",
        "target_layer_ids",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"DLite dlite_config is missing required fields: {missing}.")

    architecture_version = str(raw["architecture_version"])
    if architecture_version not in DLITE_ARCHITECTURE_VERSIONS:
        raise ValueError(
            "SGLang currently supports released DLite student checkpoints with "
            f"architecture_version in {DLITE_ARCHITECTURE_VERSIONS!r}; got "
            f"{architecture_version!r}."
        )

    model_role = str(raw["model_role"]).lower()
    if model_role not in DLITE_MODEL_ROLES:
        raise ValueError(
            f"DLite inference requires model_role in {DLITE_MODEL_ROLES!r}; "
            f"got {model_role!r}."
        )

    sequential_head = str(raw["sequential_head"]).lower()
    if sequential_head != DLITE_SEQUENTIAL_HEAD:
        raise ValueError(
            f"DLite inference requires sequential_head={DLITE_SEQUENTIAL_HEAD!r}; "
            f"got {sequential_head!r}."
        )

    block_size = int(_config_value(draft_hf_config, "block_size", 0))
    chs_num_layers = int(raw["chs_num_layers"])
    sequential_rank = int(raw["sequential_rank"])
    mask_token_id = int(raw["mask_token_id"])
    num_target_layers = int(_config_value(draft_hf_config, "num_target_layers", 0))
    target_layer_ids = tuple(int(layer_id) for layer_id in raw["target_layer_ids"])

    if block_size < 2:
        raise ValueError(f"DLite block_size must be at least 2, got {block_size}.")
    if sequential_rank <= 0:
        raise ValueError(
            f"DLite sequential_rank must be positive, got {sequential_rank}."
        )
    if mask_token_id < 0:
        raise ValueError(
            f"DLite mask_token_id must be non-negative, got {mask_token_id}."
        )
    if len(target_layer_ids) != chs_num_layers:
        raise ValueError(
            "DLite target_layer_ids length must equal chs_num_layers, got "
            f"{len(target_layer_ids)} and {chs_num_layers}."
        )
    if tuple(sorted(set(target_layer_ids))) != target_layer_ids:
        raise ValueError(
            "DLite target_layer_ids must be sorted and unique, got "
            f"{list(target_layer_ids)}."
        )
    if target_layer_ids[0] < 0 or target_layer_ids[-1] >= num_target_layers:
        raise ValueError(
            "DLite target_layer_ids must be in "
            f"[0, {num_target_layers}), got {list(target_layer_ids)}."
        )
    expected_target_layer_ids = build_dlite_target_layer_ids(
        num_target_layers, chs_num_layers
    )
    if target_layer_ids != expected_target_layer_ids:
        raise ValueError(
            "DLite target_layer_ids must match the released checkpoint's fixed "
            "first/last plus evenly spaced selection, expected "
            f"{list(expected_target_layer_ids)}, got {list(target_layer_ids)}."
        )

    return DLiteDraftConfig(
        architecture_version=architecture_version,
        model_role=model_role,
        block_size=block_size,
        chs_num_layers=chs_num_layers,
        target_layer_ids=target_layer_ids,
        sequential_head=sequential_head,
        sequential_rank=sequential_rank,
        mask_token_id=mask_token_id,
        num_target_layers=num_target_layers,
    )


def validate_dlite_target_compatibility(
    *, target_hf_config: Any, draft_hf_config: Any
) -> DLiteDraftConfig:
    draft = parse_dlite_draft_config(draft_hf_config)

    target_architectures = _config_value(target_hf_config, "architectures", None) or []
    target_model_type = _config_value(target_hf_config, "model_type", None)
    if "Qwen3ForCausalLM" not in target_architectures and target_model_type != "qwen3":
        raise ValueError(
            "DLite currently supports Qwen3ForCausalLM targets only, got "
            f"architectures={target_architectures!r}, model_type={target_model_type!r}."
        )

    comparisons = (
        ("num_hidden_layers", draft.num_target_layers),
        ("hidden_size", int(_config_value(draft_hf_config, "hidden_size", 0))),
        ("vocab_size", int(_config_value(draft_hf_config, "vocab_size", 0))),
    )
    for field, expected in comparisons:
        actual = int(_config_value(target_hf_config, field, 0))
        if actual != expected:
            raise ValueError(
                f"DLite target/draft {field} mismatch: target={actual}, "
                f"draft expects {expected}."
            )

    vocab_size = int(_config_value(target_hf_config, "vocab_size", 0))
    if draft.mask_token_id >= vocab_size:
        raise ValueError(
            "DLite mask_token_id must be present in the target vocabulary, got "
            f"mask_token_id={draft.mask_token_id}, vocab_size={vocab_size}."
        )
    return draft


__all__ = [
    "DLITE_ARCHITECTURE_VERSION",
    "DLITE_ARCHITECTURE_VERSIONS",
    "DLITE_MODEL_ROLE",
    "DLITE_MODEL_ROLES",
    "DLITE_SEQUENTIAL_HEAD",
    "DLiteDraftConfig",
    "build_dlite_target_layer_ids",
    "parse_dlite_draft_config",
    "validate_dlite_target_compatibility",
]
