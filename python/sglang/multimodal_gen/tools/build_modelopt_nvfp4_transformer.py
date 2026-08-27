"""Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer.

This tool keeps the ModelOpt-exported NVFP4 tensors for most transformer
modules, but can replace a validated subset of numerically sensitive modules
with their original BF16 tensors from the base transformer checkpoint.

It is primarily intended for FLUX.1-dev style ModelOpt NVFP4 exports where:
- the base pipeline should remain separate from the quantized transformer
- fallback BF16 modules are model-family specific
- the serialized FP4 weight byte order may already match the runtime kernel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]

DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS = [
    "transformer_blocks.*.norm1.linear*",
    "transformer_blocks.*.norm1_context.linear*",
    "transformer_blocks.*.ff.net.0.proj*",
    "transformer_blocks.*.ff.net.2*",
    "transformer_blocks.*.ff_context.net.0.proj*",
    "transformer_blocks.*.ff_context.net.2*",
    "single_transformer_blocks.*.norm.linear*",
    "single_transformer_blocks.*.proj_mlp*",
]

_TENSOR_MODULE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
    ".bias",
)


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _find_index_file(model_dir: str) -> str | None:
    for filename in INDEX_FILENAMES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return filename

    matches = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors.index.json")
    )
    return matches[0] if matches else None


def _load_weight_map(model_dir: str) -> tuple[dict[str, str], str | None]:
    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    index_filename = f"{Path(shard_name).stem}.safetensors.index.json"
    return weight_map, index_filename


def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _write_config(model_dir: Path, config: Mapping[str, object]) -> None:
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_non_shard_files(source_dir: str, output_dir: str) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in ignored:
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _load_selected_tensors(
    model_dir: str,
    weight_map: Mapping[str, str],
    tensor_names: Iterable[str],
):
    tensors = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        names_by_file[weight_map[name]].append(name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def _module_name_for_tensor(tensor_name: str) -> str:
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _matches_any_pattern(module_name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if re.fullmatch(regex_str, module_name):
            return True
    return False


def _preset_patterns(pattern_preset: str) -> list[str]:
    if pattern_preset == "none":
        return []
    if pattern_preset == "flux1-nvfp4":
        return list(DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS)
    raise ValueError(f"Unsupported pattern preset: {pattern_preset}")


def _updated_quant_config(
    source_config: Mapping[str, object],
    *,
    fallback_patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    output_config = json.loads(json.dumps(source_config))
    quant_config = output_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("Expected a flat quantization_config dict in config.json.")
    if (
        quant_config.get("quant_method") != "modelopt"
        or "FP4" not in str(quant_config.get("quant_algo", "")).upper()
    ):
        raise ValueError(
            "This tool only supports ModelOpt diffusion NVFP4 exports "
            "(quant_method=modelopt, quant_algo=FP4/NVFP4)."
        )

    ignore_patterns = list(quant_config.get("ignore", []) or [])
    for pattern in fallback_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)

    quant_config["ignore"] = ignore_patterns
    quant_config.setdefault(
        "quant_type", str(quant_config.get("quant_algo", "")).upper()
    )
    quant_config["swap_weight_nibbles"] = swap_weight_nibbles
    return output_config


def build_modelopt_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    modelopt_hf_dir: str,
    output_dir: str,
    pattern_preset: str = "none",
    keep_bf16_patterns: Sequence[str] | None = None,
    swap_weight_nibbles: bool | None = None,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    source_dir = _resolve_transformer_dir(modelopt_hf_dir)
    base_dir = _resolve_transformer_dir(base_transformer_dir)

    patterns = _preset_patterns(pattern_preset)
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)

    resolved_swap_weight_nibbles = (
        swap_weight_nibbles
        if swap_weight_nibbles is not None
        else (False if pattern_preset == "flux1-nvfp4" else True)
    )
    output_config = _updated_quant_config(
        _load_config(source_dir),
        fallback_patterns=patterns,
        swap_weight_nibbles=resolved_swap_weight_nibbles,
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    _copy_non_shard_files(source_dir, str(output_path))
    _write_config(output_path, output_config)

    source_weight_map, index_filename = _load_weight_map(source_dir)
    base_weight_map, _ = _load_weight_map(base_dir)

    fallback_tensor_names = sorted(
        name
        for name in base_weight_map
        if name in source_weight_map
        and _matches_any_pattern(_module_name_for_tensor(name), patterns)
    )
    fallback_tensors = _load_selected_tensors(
        base_dir,
        base_weight_map,
        fallback_tensor_names,
    )
    fallback_modules = {
        _module_name_for_tensor(tensor_name) for tensor_name in fallback_tensor_names
    }

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in source_weight_map.items():
        weights_by_file[filename].append(tensor_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    replaced_tensor_count = 0
    removed_aux_tensor_count = 0

    for filename, tensor_names in sorted(weights_by_file.items()):
        shard_path = os.path.join(source_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})

        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in list(shard_tensors.keys()):
            if "_quantizer." in name:
                del shard_tensors[name]
                removed_aux_tensor_count += 1
                continue

            module_name = _module_name_for_tensor(name)
            if module_name not in fallback_modules:
                continue

            if name in fallback_tensors:
                shard_tensors[name] = fallback_tensors[name]
                replaced_tensor_count += 1
            else:
                del shard_tensors[name]
                removed_aux_tensor_count += 1

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)

        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

    if index_filename is None:
        raise ValueError(
            "Expected a sharded or indexed ModelOpt HF export, but no index file was found."
        )

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": updated_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return {
        "fallback_modules": len(fallback_modules),
        "replaced_tensors": replaced_tensor_count,
        "removed_aux_tensors": removed_aux_tensor_count,
        "output_shards": len(weights_by_file),
        "swap_weight_nibbles": resolved_swap_weight_nibbles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer and "
            "optionally keep selected modules in BF16."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help="Original BF16 transformer directory, or a parent model directory.",
    )
    parser.add_argument(
        "--modelopt-hf-dir",
        required=True,
        help="ModelOpt --hf-ckpt-dir output, or its transformer subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the mixed transformer checkpoint.",
    )
    parser.add_argument(
        "--pattern-preset",
        choices=["none", "flux1-nvfp4"],
        default="none",
        help="Optional model-family BF16 fallback preset.",
    )
    parser.add_argument(
        "--keep-bf16-pattern",
        action="append",
        default=[],
        help=(
            "Glob-style pattern matched against module names without trailing tensor "
            "suffixes such as .weight or .bias."
        ),
    )
    parser.add_argument(
        "--swap-weight-nibbles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether the runtime should swap packed FP4 nibbles before padding. "
            "Defaults to false for --pattern-preset flux1-nvfp4 and true otherwise."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=args.base_transformer_dir,
        modelopt_hf_dir=args.modelopt_hf_dir,
        output_dir=args.output_dir,
        pattern_preset=args.pattern_preset,
        keep_bf16_patterns=args.keep_bf16_pattern,
        swap_weight_nibbles=args.swap_weight_nibbles,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
