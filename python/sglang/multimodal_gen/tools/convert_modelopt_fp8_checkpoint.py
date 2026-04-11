"""Convert a ModelOpt diffusion FP8 export into an SGLang-loadable checkpoint.

The core conversion path is model-agnostic:
- read the ModelOpt diffusers transformer export
- rebuild per-layer `weight_scale` / `input_scale` tensors from `backbone.pt`
- materialize SGLang-native `float8_e4m3fn` weights
- preserve ModelOpt `ignore` layers in their original dtype

Some models still benefit from a small validated BF16 fallback set. Those
fallback profiles are intentionally isolated so the generic FP8 conversion path
remains reusable across future diffusion backbones.

Example:

    python -m sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint \
        --modelopt-hf-dir /tmp/modelopt_flux2_fp8/hf \
        --modelopt-backbone-ckpt /tmp/modelopt_flux2_fp8/ckpt/backbone.pt \
        --base-transformer-dir /path/to/FLUX.2-dev/transformer \
        --output-dir /tmp/modelopt_flux2_fp8/sglang_transformer
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]
FP8_E4M3_MAXBOUND = 448.0
DEFAULT_FLUX2_KEEP_BF16_PATTERNS = [
    r"^time_guidance_embed\.(timestep_embedder|guidance_embedder)\.linear_[12]$",
    r"^double_stream_modulation_(img|txt)\.linear$",
    r"^single_stream_modulation\.linear$",
    r"^x_embedder$",
    r"^context_embedder$",
    r"^norm_out\.linear$",
]
DEFAULT_FLUX1_KEEP_BF16_PATTERNS = [
    r"^transformer_blocks\.\d+\.norm1\.linear$",
    r"^transformer_blocks\.\d+\.norm1_context\.linear$",
    r"^transformer_blocks\.\d+\.ff\.net\.0\.proj$",
    r"^transformer_blocks\.\d+\.ff\.net\.2$",
    r"^transformer_blocks\.\d+\.ff_context\.net\.0\.proj$",
    r"^transformer_blocks\.\d+\.ff_context\.net\.2$",
    r"^single_transformer_blocks\.\d+\.norm\.linear$",
]


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _resolve_backbone_ckpt(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return str(candidate)
    backbone_path = candidate / "backbone.pt"
    if backbone_path.is_file():
        return str(backbone_path)
    raise FileNotFoundError(f"Could not resolve backbone.pt from: {path}")


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


def get_default_keep_bf16_patterns(
    *, model_type: str, class_name: str | None
) -> list[str]:
    if model_type == "flux1":
        return list(DEFAULT_FLUX1_KEEP_BF16_PATTERNS)
    if model_type == "flux2":
        return list(DEFAULT_FLUX2_KEEP_BF16_PATTERNS)
    if model_type == "none":
        return []
    if class_name == "FluxTransformer2DModel":
        return list(DEFAULT_FLUX1_KEEP_BF16_PATTERNS)
    if class_name == "Flux2Transformer2DModel":
        return list(DEFAULT_FLUX2_KEEP_BF16_PATTERNS)
    return []


def should_keep_bf16(
    weight_name: str,
    keep_bf16_patterns: Sequence[str],
) -> bool:
    if not keep_bf16_patterns:
        return False

    module_name = weight_name[:-7] if weight_name.endswith(".weight") else weight_name
    return any(re.search(pattern, module_name) for pattern in keep_bf16_patterns)


def is_ignored_by_modelopt(
    weight_name: str,
    ignore_patterns: Sequence[str],
) -> bool:
    if not ignore_patterns:
        return False

    module_name = weight_name[:-7] if weight_name.endswith(".weight") else weight_name
    for pattern in ignore_patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if re.fullmatch(regex_str, module_name):
            return True
    return False


def build_fp8_scale_map(
    model_state_dict: Mapping[str, torch.Tensor],
    *,
    maxbound: float = FP8_E4M3_MAXBOUND,
) -> dict[str, dict[str, torch.Tensor]]:
    scale_map: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in model_state_dict.items():
        if key.endswith(".weight_quantizer._amax"):
            layer_name = key[: -len(".weight_quantizer._amax")]
            scale_map.setdefault(f"{layer_name}.weight", {})["weight_scale"] = (
                value.detach().to(torch.float32).reshape(1).cpu() / maxbound
            )
        elif key.endswith(".input_quantizer._amax"):
            layer_name = key[: -len(".input_quantizer._amax")]
            scale_map.setdefault(f"{layer_name}.weight", {})["input_scale"] = (
                value.detach().to(torch.float32).reshape(1).cpu() / maxbound
            )

    return {
        weight_name: scale_tensors
        for weight_name, scale_tensors in scale_map.items()
        if {"weight_scale", "input_scale"} <= set(scale_tensors)
    }


def quantize_fp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return weight.contiguous()

    scale = weight_scale.to(weight.device, dtype=torch.float32)
    if scale.numel() != 1:
        raise ValueError(
            "Only per-tensor FP8 scales are supported for diffusion checkpoints, "
            f"got shape {tuple(scale.shape)}."
        )

    quantized = (weight.to(torch.float32) / scale.reshape(1)).to(torch.float8_e4m3fn)
    return quantized.cpu().contiguous()


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
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in tensor_names:
        names_by_file[weight_map[name]].append(name)

    for filename, names in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def convert_modelopt_fp8_checkpoint(
    *,
    modelopt_hf_dir: str,
    modelopt_backbone_ckpt: str,
    output_dir: str,
    base_transformer_dir: str | None = None,
    model_type: str = "auto",
    keep_bf16_patterns: Sequence[str] | None = None,
    maxbound: float = FP8_E4M3_MAXBOUND,
    overwrite: bool = False,
) -> dict[str, int]:
    source_dir = _resolve_transformer_dir(modelopt_hf_dir)
    backbone_ckpt_path = _resolve_backbone_ckpt(modelopt_backbone_ckpt)
    base_dir = (
        _resolve_transformer_dir(base_transformer_dir) if base_transformer_dir else None
    )

    config = _load_config(source_dir)
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError(
            "Expected a flat quantization_config dict in the ModelOpt export."
        )
    if (
        quant_config.get("quant_method") != "modelopt"
        or "FP8" not in str(quant_config.get("quant_algo", "")).upper()
    ):
        raise ValueError(
            "This tool only supports ModelOpt diffusers FP8 exports "
            "(quant_method=modelopt, quant_algo=FP8)."
        )

    class_name = config.get("_class_name")
    ignore_patterns = list(quant_config.get("ignore", []) or [])
    patterns = list(
        get_default_keep_bf16_patterns(model_type=model_type, class_name=class_name)
    )
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)
    if patterns and base_dir is None:
        raise ValueError(
            "BF16 fallback patterns are enabled, but --base-transformer-dir was not provided."
        )

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

    source_weight_map, index_filename = _load_weight_map(source_dir)
    base_weight_map: dict[str, str] = {}
    if base_dir is not None:
        base_weight_map, _ = _load_weight_map(base_dir)

    backbone_state = torch.load(backbone_ckpt_path, map_location="cpu")[
        "model_state_dict"
    ]
    fp8_scale_map = build_fp8_scale_map(backbone_state, maxbound=maxbound)
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    fallback_weight_names = sorted(
        weight_name
        for weight_name in source_weight_map
        if weight_name.endswith(".weight") and should_keep_bf16(weight_name, patterns)
    )
    fallback_tensors = (
        _load_selected_tensors(base_dir, base_weight_map, fallback_weight_names)
        if fallback_weight_names
        else {}
    )
    fallback_scale_names = {
        scale_name
        for weight_name in fallback_weight_names
        for scale_name in (
            weight_name[:-7] + ".weight_scale",
            weight_name[:-7] + ".input_scale",
        )
    }

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for weight_name, filename in source_weight_map.items():
        weights_by_file[filename].append(weight_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    added_scale_count = 0
    preserved_ignored_weight_count = 0

    for filename, names in sorted(weights_by_file.items()):
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
                continue
            if name in fallback_scale_names:
                del shard_tensors[name]
                continue
            if name.endswith(".weight") and is_ignored_by_modelopt(
                name, ignore_patterns
            ):
                preserved_ignored_weight_count += 1
                continue
            if name in fallback_tensors:
                shard_tensors[name] = fallback_tensors[name]
            if (
                name.endswith(".weight")
                and name in fp8_scale_map
                and name not in fallback_tensors
            ):
                scale_tensors = fp8_scale_map[name]
                shard_tensors[name] = quantize_fp8_weight(
                    shard_tensors[name], scale_tensors["weight_scale"]
                )
                shard_tensors[name[:-7] + ".weight_scale"] = scale_tensors[
                    "weight_scale"
                ]
                shard_tensors[name[:-7] + ".input_scale"] = scale_tensors["input_scale"]
                added_scale_count += 2

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)

        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

        del shard_tensors
        gc.collect()

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

    return {
        "quantized_weights": sum(
            1
            for name in source_weight_map
            if name.endswith(".weight")
            and name in fp8_scale_map
            and not is_ignored_by_modelopt(name, ignore_patterns)
        ),
        "bf16_fallback_weights": len(fallback_weight_names),
        "preserved_ignored_weights": preserved_ignored_weight_count,
        "added_scale_tensors": added_scale_count,
        "output_shards": len(weights_by_file),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject FP8 scales from ModelOpt backbone.pt into a diffusers export so "
            "SGLang diffusion can load it natively."
        )
    )
    parser.add_argument(
        "--modelopt-hf-dir",
        required=True,
        help="ModelOpt --hf-ckpt-dir output, or its transformer subdirectory.",
    )
    parser.add_argument(
        "--modelopt-backbone-ckpt",
        required=True,
        help="Path to backbone.pt, or the directory that contains it.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the converted SGLang transformer checkpoint.",
    )
    parser.add_argument(
        "--base-transformer-dir",
        help=(
            "Original BF16 transformer directory (or parent model dir). Required when "
            "BF16 fallback layers are enabled."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "flux1", "flux2", "none"],
        default="auto",
        help=(
            "Optional model-family BF16 fallback profile. 'none' uses the generic "
            "conversion path. 'auto' enables the validated FLUX.1 / FLUX.2 "
            "fallback set when the export config matches those transformer classes."
        ),
    )
    parser.add_argument(
        "--keep-bf16-pattern",
        action="append",
        default=[],
        help=(
            "Regex matched against module names without the trailing .weight. "
            "Matching weights are copied from --base-transformer-dir instead of "
            "staying in FP8."
        ),
    )
    parser.add_argument(
        "--maxbound",
        type=float,
        default=FP8_E4M3_MAXBOUND,
        help="FP8 maxbound used to turn ModelOpt amax into a scale. E4M3 uses 448.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = convert_modelopt_fp8_checkpoint(
        modelopt_hf_dir=args.modelopt_hf_dir,
        modelopt_backbone_ckpt=args.modelopt_backbone_ckpt,
        output_dir=args.output_dir,
        base_transformer_dir=args.base_transformer_dir,
        model_type=args.model_type,
        keep_bf16_patterns=args.keep_bf16_pattern,
        maxbound=args.maxbound,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
