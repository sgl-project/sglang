"""Build a calibrated static-FP8 transformer override for MinWM.

The calibration file is produced by ``run_minwm_baseline.py`` and contains
input activation maxima for exactly the 300 projection and FFN linears in the
30 MinWM transformer blocks. Non-block components stay in their source dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0
INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
EXPECTED_MODULE_COUNT = 300
EXPECTED_FFN_MODULE_COUNT = 60


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_calibration(path: Path) -> dict[str, float]:
    data = _read_json(path)
    if data.get("format") != "minwm-static-fp8-calibration-v1":
        raise ValueError(f"unsupported MinWM calibration format in {path}")
    modules = data.get("modules")
    if not isinstance(modules, dict) or len(modules) != EXPECTED_MODULE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODULE_COUNT} calibrated modules, "
            f"found {len(modules) if isinstance(modules, dict) else 0}"
        )
    maxima = {}
    for name, record in modules.items():
        value = float(record["input_amax"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: input_amax must be finite and positive")
        if int(record.get("samples", 0)) <= 0:
            raise ValueError(f"{name}: calibration sample count must be positive")
        maxima[name] = value
    return maxima


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError(
            f"expected a floating-point matrix, got {weight.dtype} {tuple(weight.shape)}"
        )
    amax = weight.detach().abs().amax().to(torch.float32)
    if not torch.isfinite(amax) or float(amax) <= 0:
        raise ValueError("weight maximum must be finite and positive")
    scale = (amax / FP8_MAX).reshape(1).cpu()
    quantized = torch.clamp(weight.to(torch.float32) / scale, -FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).cpu().contiguous(), scale


def build_minwm_static_fp8_transformer(
    *,
    input_dir: str,
    calibration_path: str,
    output_dir: str,
    activation_margin: float = 1.0,
    module_scope: str = "all",
    overwrite: bool = False,
) -> dict:
    source = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    calibration_file = Path(calibration_path).expanduser().resolve()
    if activation_margin < 1.0 or not math.isfinite(activation_margin):
        raise ValueError("activation_margin must be finite and >= 1")
    if module_scope not in {"all", "ffn"}:
        raise ValueError("module_scope must be 'all' or 'ffn'")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(source / "config.json")
    if not (source / INDEX_FILENAME).is_file():
        raise FileNotFoundError(source / INDEX_FILENAME)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    input_amax = _load_calibration(calibration_file)
    if module_scope == "ffn":
        selected_input_amax = {
            name: value for name, value in input_amax.items() if ".ffn." in name
        }
        # minWM maps checkpoint self_attn.* weights to block-level to_{q,k,v,out}
        # modules and cross_attn.* weights below attn2 at runtime.
        ignored_layers = ["to_q", "to_k", "to_v", "to_out", "attn2"]
    else:
        selected_input_amax = input_amax
        ignored_layers = []
    index = _read_json(source / INDEX_FILENAME)
    source_weight_map = dict(index["weight_map"])
    missing = sorted(
        name
        for name in selected_input_amax
        if f"{name}.weight" not in source_weight_map
    )
    if missing:
        raise ValueError(
            "calibrated modules missing from transformer checkpoint: "
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
    for module_name in selected_input_amax:
        modules_by_shard[source_weight_map[f"{module_name}.weight"]].append(module_name)

    quant_config = {
        "activation_scheme": "static",
        "ignored_layers": ignored_layers,
        "quant_method": "fp8",
    }
    config = _read_json(source / "config.json")
    config["quantization_config"] = quant_config
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0
    for shard_name in sorted(set(source_weight_map.values())):
        shard_path = source / shard_name
        tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        metadata.setdefault("format", "pt")
        metadata["quantization_config"] = serialized_quant_config
        for module_name in modules_by_shard.get(shard_name, []):
            weight_name = f"{module_name}.weight"
            quantized, weight_scale = _quantize_weight(tensors[weight_name])
            input_scale = torch.tensor(
                [selected_input_amax[module_name] * activation_margin / FP8_MAX],
                dtype=torch.float32,
            )
            tensors[weight_name] = quantized
            tensors[f"{module_name}.weight_scale"] = weight_scale
            tensors[f"{module_name}.input_scale"] = input_scale
            quantized_count += 1
        save_file(tensors, output / shard_name, metadata=metadata)
        for name, tensor in tensors.items():
            updated_weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    expected_quantized_count = (
        EXPECTED_FFN_MODULE_COUNT if module_scope == "ffn" else EXPECTED_MODULE_COUNT
    )
    if quantized_count != expected_quantized_count:
        raise RuntimeError(
            f"expected to quantize {expected_quantized_count} weights, "
            f"got {quantized_count}"
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

    calibration_sha256 = hashlib.sha256(calibration_file.read_bytes()).hexdigest()
    manifest = {
        "activation_margin": activation_margin,
        "calibration_sha256": calibration_sha256,
        "format": "sglang-minwm-static-fp8-v1",
        "ignored_layers": ignored_layers,
        "module_scope": module_scope,
        "output_bytes": total_size,
        "quantized_weights": quantized_count,
    }
    with (output / "minwm_static_fp8_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activation-margin", type=float, default=1.0)
    parser.add_argument("--module-scope", choices=("all", "ffn"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_minwm_static_fp8_transformer(
        input_dir=args.input_dir,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        activation_margin=args.activation_margin,
        module_scope=args.module_scope,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
