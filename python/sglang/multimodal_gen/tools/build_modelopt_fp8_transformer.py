"""Build an SGLang-loadable ModelOpt FP8 diffusion transformer.

The core conversion path is model-agnostic:
- read the ModelOpt diffusers transformer export
- rebuild per-layer `weight_scale` / `input_scale` tensors from `backbone.pt`
- materialize SGLang-native `float8_e4m3fn` weights
- preserve ModelOpt `ignore` layers in their original dtype

Some models still benefit from a small validated BF16 fallback set. Those
fallback profiles are intentionally isolated so the generic FP8 conversion path
remains reusable across future diffusion backbones.

Example:

    python -m sglang.multimodal_gen.tools.build_modelopt_fp8_transformer \
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

from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    normalize_flat_modelopt_quant_config,
)

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
DEFAULT_LTX2_KEEP_BF16_PATTERNS = [
    r"^(audio_)?adaln_single\.emb\.timestep_embedder\.linear_[12]$",
    r"^(audio_)?adaln_single\.linear$",
    r"^audio_caption_projection\.linear_[12]$",
    r"^audio_patchify_proj$",
    r"^audio_proj_out$",
    r"^av_ca_(a2v_gate|audio_scale_shift|v2a_gate|video_scale_shift)_adaln_single\.emb\.timestep_embedder\.linear_[12]$",
    r"^av_ca_(a2v_gate|audio_scale_shift|v2a_gate|video_scale_shift)_adaln_single\.linear$",
    r"^caption_projection\.linear_[12]$",
    r"^patchify_proj$",
    r"^proj_out$",
    r"^transformer_blocks\.(0|43|44|45|46|47)\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_(q|k|v)$",
    r"^transformer_blocks\.(0|43|44|45|46|47)\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_out\.0$",
    r"^transformer_blocks\.(0|43|44|45|46|47)\.(ff|audio_ff)\.proj_(in|out)$",
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


def _load_first_shard_metadata(
    model_dir: str, weight_map: Mapping[str, str]
) -> dict[str, str]:
    if not weight_map:
        return {}
    first_shard = next(iter(weight_map.values()))
    with safe_open(
        os.path.join(model_dir, first_shard), framework="pt", device="cpu"
    ) as f:
        return dict(f.metadata() or {})


def _module_name_variants(weight_name: str) -> list[str]:
    module_name = weight_name[:-7] if weight_name.endswith(".weight") else weight_name
    variants = [module_name]

    for prefix in ("model.diffusion_model.", "velocity_model."):
        if module_name.startswith(prefix):
            variants.append(module_name[len(prefix) :])

    canonicalized: list[str] = []
    for variant in variants:
        canonicalized.append(
            re.sub(r"(\.audio_ff|\.ff)\.net\.0\.proj$", r"\1.proj_in", variant)
        )
        canonicalized.append(
            re.sub(r"(\.audio_ff|\.ff)\.net\.2$", r"\1.proj_out", variant)
        )
    variants.extend(canonicalized)

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _preferred_module_name(weight_name: str) -> str:
    return _module_name_variants(weight_name)[-1]


def _scale_key_candidates(weight_name: str) -> list[str]:
    candidates = [weight_name]
    if weight_name.startswith("model.diffusion_model."):
        candidates.append(
            "velocity_model." + weight_name[len("model.diffusion_model.") :]
        )
    return candidates


def _resolve_scale_key(
    weight_name: str,
    scale_map: Mapping[str, Mapping[str, torch.Tensor]],
) -> str | None:
    for candidate in _scale_key_candidates(weight_name):
        if candidate in scale_map:
            return candidate
    return None


def _is_ltx2_x0_export(
    *,
    config: Mapping[str, object],
    source_metadata: Mapping[str, str],
    source_weight_map: Mapping[str, str],
) -> bool:
    if config.get("_class_name") != "X0Model":
        return False
    if not any(name.startswith("model.diffusion_model.") for name in source_weight_map):
        return False
    try:
        metadata_config = json.loads(str(source_metadata.get("config", "")))
    except json.JSONDecodeError:
        return False
    return isinstance(metadata_config.get("transformer"), dict)


def _build_output_config(
    *,
    source_config: Mapping[str, object],
    source_metadata: Mapping[str, str],
    quant_config: Mapping[str, object],
    is_ltx2_x0_export: bool,
) -> dict[str, object]:
    if is_ltx2_x0_export:
        metadata_config = json.loads(str(source_metadata["config"]))
        output_config = dict(metadata_config["transformer"])
        output_config["_class_name"] = "LTX2VideoTransformer3DModel"
    else:
        output_config = dict(source_config)

    output_config["quantization_config"] = dict(quant_config)
    return output_config


def _should_keep_ltx2_transformer_key(weight_name: str) -> bool:
    if not weight_name.startswith("model.diffusion_model."):
        return False
    connector_prefixes = (
        "model.diffusion_model.audio_embeddings_connector.",
        "model.diffusion_model.video_embeddings_connector.",
    )
    return not weight_name.startswith(connector_prefixes)


def get_default_keep_bf16_patterns(
    *, model_type: str, class_name: str | None
) -> list[str]:
    if model_type == "ltx2":
        return list(DEFAULT_LTX2_KEEP_BF16_PATTERNS)
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

    return any(
        re.search(pattern, module_name)
        for pattern in keep_bf16_patterns
        for module_name in _module_name_variants(weight_name)
    )


def is_ignored_by_modelopt(
    weight_name: str,
    ignore_patterns: Sequence[str],
) -> bool:
    if not ignore_patterns:
        return False

    for pattern in ignore_patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if any(
            re.fullmatch(regex_str, module_name)
            for module_name in _module_name_variants(weight_name)
        ):
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


def build_modelopt_fp8_transformer(
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
    if quant_config.get("quant_method") != "modelopt":
        raise ValueError(
            "This tool only supports ModelOpt diffusers FP8 exports "
            "(quant_method=modelopt)."
        )

    source_weight_map_all, index_filename = _load_weight_map(source_dir)
    source_metadata = _load_first_shard_metadata(source_dir, source_weight_map_all)
    is_ltx2_export = _is_ltx2_x0_export(
        config=config,
        source_metadata=source_metadata,
        source_weight_map=source_weight_map_all,
    )
    class_name = config.get("_class_name")
    ignore_patterns = list(quant_config.get("ignore", []) or [])
    patterns = list(
        get_default_keep_bf16_patterns(model_type=model_type, class_name=class_name)
    )
    if is_ltx2_export and model_type == "auto":
        patterns.extend(DEFAULT_LTX2_KEEP_BF16_PATTERNS)
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)
    if patterns and base_dir is None and not is_ltx2_export:
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

    if is_ltx2_export:
        source_weight_map = {
            name: filename
            for name, filename in source_weight_map_all.items()
            if _should_keep_ltx2_transformer_key(name)
        }
    else:
        source_weight_map = source_weight_map_all
    base_weight_map: dict[str, str] = {}
    if base_dir is not None:
        base_weight_map, _ = _load_weight_map(base_dir)
    fallback_weight_names = sorted(
        weight_name
        for weight_name in source_weight_map
        if weight_name.endswith(".weight") and should_keep_bf16(weight_name, patterns)
    )
    fallback_weight_names_set = set(fallback_weight_names)

    backbone_state = torch.load(backbone_ckpt_path, map_location="cpu")[
        "model_state_dict"
    ]
    fp8_scale_map = build_fp8_scale_map(backbone_state, maxbound=maxbound)
    quant_algo = str(quant_config.get("quant_algo", "")).upper()
    if quant_algo and "FP8" not in quant_algo:
        raise ValueError(
            "This tool only supports ModelOpt diffusers FP8 exports, "
            f"got quant_algo={quant_config.get('quant_algo')!r}."
        )
    if not quant_algo and not fp8_scale_map:
        raise ValueError(
            "Could not infer an FP8 ModelOpt export: quantization_config.quant_algo "
            "is missing and backbone.pt does not contain FP8 scale tensors."
        )
    effective_quant_config = json.loads(json.dumps(quant_config))
    if not quant_algo:
        effective_quant_config["quant_algo"] = "FP8"
    effective_quant_config = (
        normalize_flat_modelopt_quant_config(effective_quant_config)
        or effective_quant_config
    )

    auto_ignore_modules = sorted(
        {
            _preferred_module_name(weight_name)
            for weight_name in source_weight_map
            if weight_name.endswith(".weight")
            and _resolve_scale_key(weight_name, fp8_scale_map) is None
        }
    )
    fallback_ignore_modules = sorted(
        {_preferred_module_name(weight_name) for weight_name in fallback_weight_names}
    )
    ignore_patterns = sorted(
        {
            *ignore_patterns,
            *auto_ignore_modules,
            *fallback_ignore_modules,
        }
    )
    effective_quant_config["ignore"] = ignore_patterns
    serialized_quant_config = json.dumps(effective_quant_config, sort_keys=True)
    output_config = _build_output_config(
        source_config=config,
        source_metadata=source_metadata,
        quant_config=effective_quant_config,
        is_ltx2_x0_export=is_ltx2_export,
    )

    fallback_tensors = (
        _load_selected_tensors(base_dir, base_weight_map, fallback_weight_names)
        if fallback_weight_names and base_dir is not None
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
        selected_names = set(names)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})

        metadata.setdefault("format", "pt")
        metadata["_class_name"] = str(
            output_config.get("_class_name", metadata.get("_class_name", ""))
        )
        metadata["config"] = json.dumps(output_config, sort_keys=True)
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in list(shard_tensors.keys()):
            if name not in selected_names:
                del shard_tensors[name]
                continue
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
            scale_key = _resolve_scale_key(name, fp8_scale_map)
            if (
                name.endswith(".weight")
                and scale_key is not None
                and name not in fallback_tensors
                and name not in fallback_weight_names_set
            ):
                scale_tensors = fp8_scale_map[scale_key]
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

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, sort_keys=True)

    return {
        "quantized_weights": sum(
            1
            for name in source_weight_map
            if name.endswith(".weight")
            and _resolve_scale_key(name, fp8_scale_map) is not None
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
            "Build an SGLang-loadable ModelOpt FP8 diffusion transformer from a "
            "ModelOpt diffusers export."
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
        choices=["auto", "flux1", "flux2", "ltx2", "none"],
        default="auto",
        help=(
            "Optional model-family BF16 fallback profile. 'none' uses the generic "
            "conversion path. 'auto' enables the validated FLUX.1 / FLUX.2 / LTX-2 "
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
    stats = build_modelopt_fp8_transformer(
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
