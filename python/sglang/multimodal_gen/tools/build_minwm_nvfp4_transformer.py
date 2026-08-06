"""Calibrate and build an SGLang-loadable ModelOpt NVFP4 MinWM transformer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
EXPECTED_MODULE_COUNT = 300
GROUP_SIZE = 16


def minwm_block_linear_names() -> list[str]:
    suffixes = [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    ]
    return [f"blocks.{block}.{suffix}" for block in range(30) for suffix in suffixes]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_packed_state(packed_state: dict[str, torch.Tensor]) -> None:
    for module_name in minwm_block_linear_names():
        required = {
            f"{module_name}.weight": torch.uint8,
            f"{module_name}.weight_scale": torch.float8_e4m3fn,
            f"{module_name}.weight_scale_2": torch.float32,
            f"{module_name}.input_scale": torch.float32,
        }
        for name, dtype in required.items():
            tensor = packed_state.get(name)
            if tensor is None:
                raise ValueError(f"missing packed NVFP4 tensor {name}")
            if tensor.dtype != dtype:
                raise ValueError(f"{name}: expected {dtype}, got {tensor.dtype}")


def materialize_minwm_nvfp4_transformer(
    *,
    input_dir: str,
    packed_state: dict[str, torch.Tensor],
    output_dir: str,
    overwrite: bool = False,
) -> dict:
    source = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(source / "config.json")
    if not (source / INDEX_FILENAME).is_file():
        raise FileNotFoundError(source / INDEX_FILENAME)
    _validate_packed_state(packed_state)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    index = _read_json(source / INDEX_FILENAME)
    source_weight_map = dict(index["weight_map"])
    modules = minwm_block_linear_names()
    missing = sorted(
        name for name in modules if f"{name}.weight" not in source_weight_map
    )
    if missing:
        raise ValueError(
            "MinWM block weights missing from transformer checkpoint: "
            + ", ".join(missing[:10])
        )
    for entry in source.iterdir():
        if (
            entry.name in {"config.json", INDEX_FILENAME}
            or entry.suffix == ".safetensors"
        ):
            continue
        if entry.is_dir():
            shutil.copytree(entry, output / entry.name)
        else:
            shutil.copy2(entry, output / entry.name)

    modules_by_shard: dict[str, list[str]] = defaultdict(list)
    for module_name in modules:
        modules_by_shard[source_weight_map[f"{module_name}.weight"]].append(module_name)

    quant_config = {
        "checkpoint_weight_scale_layout": "linear",
        "group_size": GROUP_SIZE,
        "ignore": [],
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
        "swap_weight_nibbles": False,
    }
    config = _read_json(source / "config.json")
    config["quantization_config"] = quant_config
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    packed_count = 0
    for shard_name in sorted(set(source_weight_map.values())):
        shard_path = source / shard_name
        tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        for module_name in modules_by_shard.get(shard_name, []):
            source_weight = tensors[f"{module_name}.weight"]
            packed_weight = packed_state[f"{module_name}.weight"].cpu().contiguous()
            if packed_weight.shape != (
                source_weight.shape[0],
                source_weight.shape[1] // 2,
            ):
                raise ValueError(
                    f"{module_name}: packed weight shape {tuple(packed_weight.shape)} "
                    f"does not match source shape {tuple(source_weight.shape)}"
                )
            tensors[f"{module_name}.weight"] = packed_weight
            for suffix in ("weight_scale", "weight_scale_2", "input_scale"):
                name = f"{module_name}.{suffix}"
                tensors[name] = packed_state[name].cpu().contiguous()
            packed_count += 1
        save_file(tensors, output / shard_name, metadata=metadata)
        for name, tensor in tensors.items():
            updated_weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    if packed_count != EXPECTED_MODULE_COUNT:
        raise RuntimeError(
            f"expected to pack {EXPECTED_MODULE_COUNT} weights, got {packed_count}"
        )
    with (output / INDEX_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": updated_weight_map},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "format": "sglang-minwm-modelopt-nvfp4-v1",
        "group_size": GROUP_SIZE,
        "output_bytes": total_size,
        "packed_weights": packed_count,
        "swap_weight_nibbles": False,
    }
    with (output / "minwm_nvfp4_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _load_minwm_pipeline(
    *, minwm_root: str, checkpoint: str, pretrained_dir: str, config_path: str
):
    root = Path(minwm_root).resolve()
    wan_root = root / "Wan21"
    os.environ["PRETRAINED_DIR"] = str(Path(pretrained_dir).resolve())
    sys.path.insert(0, str(wan_root))
    sys.path.insert(0, str(root))
    from configs.configuration import PretrainedConfig
    from pipeline import PipelineBase

    config = PretrainedConfig.from_pretrained(str(Path(config_path).resolve()))
    return PipelineBase.from_pretrained(
        config,
        str(Path(checkpoint).resolve()),
        torch.device("cuda"),
        low_memory=False,
    )


def prepare_calibration_kwargs(
    generator, record: dict, *, device: str = "cuda"
) -> dict:
    kwargs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in record.items()
        if name != "output"
    }
    packed_metadata = ("seq_lens", "block_idx", "position_ids")
    if not all(kwargs.get(name) is not None for name in packed_metadata):
        if not hasattr(generator, "make_kv_cache"):
            raise ValueError(
                "calibration record has no packed metadata and the generator "
                "cannot create an inference KV cache"
            )
        kwargs["cache"] = generator.make_kv_cache()
        kwargs.setdefault("self_cache_update", None)
    return kwargs


def build_calibrated_minwm_nvfp4_transformer(
    *,
    input_dir: str,
    output_dir: str,
    minwm_root: str,
    checkpoint: str,
    pretrained_dir: str,
    config_path: str,
    calibration_forward: str,
    overwrite: bool = False,
) -> dict:
    import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.unified_export_hf import (
        _process_quantized_modules,
    )

    pipeline = _load_minwm_pipeline(
        minwm_root=minwm_root,
        checkpoint=checkpoint,
        pretrained_dir=pretrained_dir,
        config_path=config_path,
    )
    generator = pipeline.generator
    record = torch.load(calibration_forward, map_location="cpu", weights_only=True)

    def calibrate(_blocks) -> None:
        if hasattr(generator, "clear_cache"):
            generator.clear_cache()
        kwargs = prepare_calibration_kwargs(generator, record)
        with torch.inference_mode():
            generator(**kwargs)

    mtq.quantize(generator.blocks, mtq.NVFP4_DEFAULT_CFG, calibrate)
    _process_quantized_modules(generator.blocks, torch.bfloat16)
    packed_state = {
        f"blocks.{name}": tensor.detach().cpu()
        for name, tensor in generator.blocks.state_dict().items()
        if "_quantizer." not in name
        and name.endswith(
            (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
        )
    }
    manifest = materialize_minwm_nvfp4_transformer(
        input_dir=input_dir,
        packed_state=packed_state,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    manifest["modelopt_version"] = getattr(modelopt, "__version__", "unknown")
    manifest_path = (
        Path(output_dir).expanduser().resolve() / "minwm_nvfp4_manifest.json"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minwm-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-forward", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_calibrated_minwm_nvfp4_transformer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        minwm_root=args.minwm_root,
        checkpoint=args.checkpoint,
        pretrained_dir=args.pretrained_dir,
        config_path=args.config,
        calibration_forward=args.calibration_forward,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
