"""Read local DSpark JSON and safetensors headers without loading tensor payloads.

    .venv/bin/python benchmark/kernels/inspect_dspark_checkpoint.py /path/to/checkpoint

No HF download, remote code, pickle or model import is involved. d2t values and
full loading compatibility require the model loader; headers only expose shapes.
"""

import argparse
import json
import struct
from pathlib import Path

from safetensors import SafetensorError, safe_open

MAX_CONFIG_BYTES = 16 << 20
MAX_HEADER_BYTES = 256 << 20
CONFIG_FIELDS = (
    "architectures",
    "model_type",
    "speculators_model_type",
    "vocab_size",
    "draft_vocab_size",
    "target_vocab_size",
    "hidden_size",
    "target_hidden_size",
    "num_hidden_layers",
    "mask_token_id",
    "markov_rank",
    "markov_head_type",
    "markov_topk",
    "dspark_draft_topk",
    "markov_bias_topk",
    "logit_scale",
    "sample_from_anchor",
    "block_size",
    "aux_hidden_state_layer_ids",
    "eagle_aux_hidden_state_layer_ids",
    "target_layer_ids",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "use_aux_hidden_state",
)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key {name!r}.")
        result[name] = value
    return result


def _inside_checkpoint(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    # 以下为安全注释COSEC：索引文件名及符号链接不能越过用户指定的checkpoint目录。
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(
            f"Checkpoint file escapes its directory or is not a file: {path.name}"
        )
    return resolved


def _read_json(path: Path, limit: int):
    if path.stat().st_size > limit:
        raise ValueError(
            f"JSON file is larger than the {limit}-byte limit: {path.name}"
        )
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"JSON file grew beyond the {limit}-byte limit: {path.name}")
    value = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _read_safetensors_header(path: Path):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors length prefix: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if length > MAX_HEADER_BYTES or length > file_size - 8:
            raise ValueError(f"Invalid/oversized safetensors header: {path.name}")
    # Match the runtime's safe_open loader. Slices expose metadata without
    # reading tensor payloads; the library validates the safetensors format.
    with safe_open(path, framework="pt", device="cpu") as weights:
        return {
            name: {
                "shape": weights.get_slice(name).get_shape(),
                "dtype": weights.get_slice(name).get_dtype(),
            }
            for name in weights.keys()
        }


def inspect_checkpoint(directory: Path) -> dict:
    root = directory.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {root}")
    config = _read_json(
        _inside_checkpoint(root, root / "config.json"), MAX_CONFIG_BYTES
    )
    weight_files = {
        _inside_checkpoint(root, path) for path in root.glob("*.safetensors")
    }
    for index_path in root.glob("*.safetensors.index.json"):
        index = _read_json(_inside_checkpoint(root, index_path), MAX_CONFIG_BYTES)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing weight_map object in {index_path.name}.")
        for name in weight_map.values():
            if not isinstance(name, str) or not name.endswith(".safetensors"):
                raise ValueError("Checkpoint index must refer to safetensors files.")
            weight_files.add(_inside_checkpoint(root, root / name))
    if not weight_files:
        raise ValueError(
            "No local safetensors weights found; pickle/bin checkpoints are not inspected."
        )
    inventory = {}
    shards = []
    for path in sorted(weight_files):
        tensors = _read_safetensors_header(path)
        shards.append(
            {"file": str(path.relative_to(root)), "tensor_count": len(tensors)}
        )
        for name, metadata in tensors.items():
            if name in inventory:
                raise ValueError(
                    f"Duplicate tensor across checkpoint shards: {name!r}."
                )
            inventory[name] = dict(metadata, file=str(path.relative_to(root)))

    def has(fragment):
        return any(fragment in name for name in inventory)

    relevant = {
        name: metadata
        for name, metadata in inventory.items()
        if any(
            fragment in name
            for fragment in (
                "embed_tokens",
                "lm_head",
                "markov_",
                "confidence_head",
                "mask_embedding",
                "d2t",
                "draft_id_to_target_id",
                "t2d",
            )
        )
    }
    return {
        "checkpoint": str(root),
        "config": {name: config.get(name) for name in CONFIG_FIELDS},
        "transformer_layer_config": config.get("transformer_layer_config"),
        "text_config": config.get("text_config"),
        "dspark_config": config.get("dspark_config"),
        "dspark_prefixed_config": {
            key: value for key, value in config.items() if key.startswith("dspark_")
        },
        "speculators_config": config.get("speculators_config"),
        "vocabulary_weights": {
            "has_embedding": has("embed_tokens"),
            "has_lm_head": has("lm_head"),
            "has_d2t": has("d2t") or has("draft_id_to_target_id"),
            "has_mask_embedding": has("mask_embedding"),
        },
        "shards": shards,
        "tensor_count": len(inventory),
        "relevant_weights": relevant,
        "mapping_values_validated": False,
        "validation_scope": "JSON and safetensors headers only; no tensor payloads or model execution.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_checkpoint(args.checkpoint)
    except (OSError, ValueError, SafetensorError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
