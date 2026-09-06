import glob
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import ComfyFp8Config
from sglang.multimodal_gen.runtime.layers.quantization.comfy_nvfp4 import (
    ComfyNvfp4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod as SrtUnquantizedEmbeddingMethod,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod as SrtUnquantizedLinearMethod,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as SrtVocabParallelEmbedding,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.srt.utils import is_npu

logger = init_logger(__name__)


def process_model_weights_after_loading(
    model: nn.Module,
    process_device: torch.device | None = None,
    *,
    quantized_only: bool = False,
) -> int:
    """Process native and SRT layers once, optionally staging one layer at a time."""
    processed_layers = 0
    for module in model.modules():
        if not isinstance(
            module,
            (
                LinearBase,
                SrtLinearBase,
                VocabParallelEmbedding,
                SrtVocabParallelEmbedding,
            ),
        ):
            continue
        method = module.quant_method
        if method is None:
            continue
        unquantized = isinstance(
            method,
            (
                UnquantizedLinearMethod,
                SrtUnquantizedLinearMethod,
                UnquantizedEmbeddingMethod,
                SrtUnquantizedEmbeddingMethod,
            ),
        )
        if quantized_only and unquantized:
            continue
        if is_npu() and not unquantized:
            torch.npu.config.allow_internal_format = True
        if process_device is None:
            method.process_weights_after_loading(module)
        else:
            with stage_module_for_post_load(module, process_device):
                method.process_weights_after_loading(module)
        if is_npu():
            torch.npu.empty_cache()
        processed_layers += 1
    return processed_layers


def inspect_comfy_quant_markers(
    safetensors_list: list[str],
    param_name_mapper: Callable[[str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read and validate Comfy's tensor-level quantization markers."""
    checkpoint_meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    raw_markers: dict[str, dict[str, Any]] = {}
    marked_dtype_weight_prefixes: set[str] = set()
    global_quant_formats: set[str] = set()

    for path in safetensors_list:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if quant_format := metadata.get("quant_format"):
                global_quant_formats.add(quant_format.lower())
            serialized_metadata = metadata.get("_quantization_metadata")
            if serialized_metadata is not None:
                try:
                    metadata_config = json.loads(serialized_metadata)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid _quantization_metadata in {path}"
                    ) from exc
                if not isinstance(metadata_config, dict):
                    raise ValueError(
                        f"_quantization_metadata in {path} must contain an object"
                    )
                metadata_layers = metadata_config.get("layers")
                if not isinstance(metadata_layers, dict):
                    raise ValueError(
                        f"_quantization_metadata in {path} must contain a layers object"
                    )
                for prefix, marker in metadata_layers.items():
                    if not isinstance(marker, dict):
                        raise ValueError(
                            f"Comfy quantization metadata for {prefix!r} must be an object"
                        )
                    previous = raw_markers.get(prefix)
                    if previous is not None and previous != marker:
                        raise ValueError(
                            f"Conflicting Comfy quantization markers for {prefix!r}"
                        )
                    raw_markers[prefix] = marker
            for key in checkpoint.keys():
                tensor_slice = checkpoint.get_slice(key)
                checkpoint_meta[key] = (
                    tensor_slice.get_dtype(),
                    tuple(tensor_slice.get_shape()),
                )
                if key.endswith(".weight") and tensor_slice.get_dtype() in (
                    "F8_E4M3",
                    "I8",
                    "U8",
                ):
                    marked_dtype_weight_prefixes.add(key.removesuffix(".weight"))
                if not key.endswith(".comfy_quant"):
                    continue
                try:
                    marker = json.loads(checkpoint.get_tensor(key).numpy().tobytes())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid Comfy quantization marker {key!r} in {path}"
                    ) from exc
                if not isinstance(marker, dict):
                    raise ValueError(
                        f"Comfy quantization marker {key!r} must contain a JSON object"
                    )
                prefix = key.removesuffix(".comfy_quant")
                previous = raw_markers.get(prefix)
                if previous is not None and previous != marker:
                    raise ValueError(
                        f"Conflicting Comfy quantization markers for {prefix!r}"
                    )
                raw_markers[prefix] = marker

    if global_quant_formats == {"mxfp8"}:
        for prefix in marked_dtype_weight_prefixes:
            weight_meta = checkpoint_meta[f"{prefix}.weight"]
            scale_meta = checkpoint_meta.get(f"{prefix}.weight_scale")
            weight_dtype, weight_shape = weight_meta
            if (
                weight_dtype != "F8_E4M3"
                or len(weight_shape) != 2
                or weight_shape[1] % 32 != 0
                or scale_meta != ("U8", (weight_shape[0], weight_shape[1] // 32))
            ):
                raise ValueError(
                    f"MXFP8 layer {prefix!r} has incompatible weight/scale metadata: "
                    f"{weight_meta} and {scale_meta}"
                )
            raw_markers.setdefault(prefix, {"format": "mxfp8"})

    missing_markers = marked_dtype_weight_prefixes - raw_markers.keys()
    if missing_markers:
        raise ValueError(
            "Quantized weights are missing comfy_quant metadata: "
            f"{sorted(missing_markers)[:5]}"
        )

    for prefix, marker in raw_markers.items():
        marker_format = marker.get("format")
        required = {f"{prefix}.weight", f"{prefix}.weight_scale"}
        if marker_format == "asym_w4a8_int8":
            required = {
                f"{prefix}.weight",
                f"{prefix}.weight_s_rel",
                f"{prefix}.weight_s_channel",
            }
        if marker_format == "nvfp4":
            required.add(f"{prefix}.weight_scale_2")
        if marker_format not in (
            "float8_e4m3fn",
            "int8_tensorwise",
            "asym_w4a8_int8",
            "convrot_w4a4",
            "nvfp4",
        ):
            continue
        missing = required - checkpoint_meta.keys()
        if missing:
            raise ValueError(
                f"Comfy layer {prefix!r} is missing checkpoint tensors: "
                f"{sorted(missing)}"
            )
        if marker_format == "float8_e4m3fn":
            marker["_activation_scheme"] = (
                "static" if f"{prefix}.input_scale" in checkpoint_meta else "dynamic"
            )
            continue
        if marker_format == "asym_w4a8_int8":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_s_rel"]
            channel_dtype, channel_shape = checkpoint_meta[f"{prefix}.weight_s_channel"]
            group_size = int(marker.get("group_size", 16))
            if group_size < 4:
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has invalid group_size={group_size}"
                )
            if weight_dtype != "I8" or scale_dtype != "F8_E4M3":
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs I8 weights and FP8 "
                    f"group scales, got {weight_dtype} and {scale_dtype}"
                )
            if channel_dtype != "F32":
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs F32 channel scales, "
                    f"got {channel_dtype}"
                )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs a 2D packed weight, "
                    f"got {weight_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            expected_scale_shape = (weight_shape[0], logical_input_size // group_size)
            if scale_shape != expected_scale_shape or channel_shape != (
                weight_shape[0],
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape}, {scale_shape}, and {channel_shape}"
                )
            codebook_key = f"{prefix}.weight_codebook"
            correction_key = f"{prefix}.weight_correction"
            marker["_has_codebook"] = codebook_key in checkpoint_meta
            marker["_has_correction"] = correction_key in checkpoint_meta
            if marker["_has_codebook"] and checkpoint_meta[codebook_key] != (
                "F32",
                (16,),
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} needs an F32[16] codebook"
                )
            expected_correction = (
                logical_input_size // group_size,
                weight_shape[0],
            )
            if marker["_has_correction"] and checkpoint_meta[correction_key] != (
                "F32",
                expected_correction,
            ):
                raise ValueError(
                    f"Comfy W4A8 layer {prefix!r} has an incompatible correction tensor"
                )
            continue
        if marker_format == "convrot_w4a4":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
            if weight_dtype != "I8" or scale_dtype != "F32":
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} needs I8 packed weights and "
                    f"F32 scales, got {weight_dtype} and {scale_dtype}"
                )
            if len(weight_shape) != 2 or scale_shape != (weight_shape[0],):
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape} and {scale_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            convrot_group_size = int(marker.get("convrot_groupsize", 256))
            if convrot_group_size not in (16, 64, 256):
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has unsupported "
                    f"convrot_groupsize={convrot_group_size}"
                )
            if logical_input_size % 64 or logical_input_size % convrot_group_size:
                raise ValueError(
                    f"Comfy W4A4 layer {prefix!r} has input size "
                    f"{logical_input_size}, incompatible with quant_group_size=64 "
                    f"and convrot_groupsize={convrot_group_size}"
                )
            continue
        if marker_format == "nvfp4":
            weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
            scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
            scale_2_dtype, scale_2_shape = checkpoint_meta[f"{prefix}.weight_scale_2"]
            if weight_dtype != "U8" or scale_dtype != "F8_E4M3":
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs U8 packed weights and "
                    f"FP8 block scales, got {weight_dtype} and {scale_dtype}"
                )
            if scale_2_dtype != "F32" or scale_2_shape not in ((), (1,)):
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs a scalar F32 "
                    f"weight_scale_2, got {scale_2_dtype}{scale_2_shape}"
                )
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} needs a 2D packed weight, "
                    f"got {weight_shape}"
                )
            logical_input_size = weight_shape[1] * 2
            expected_scale_shape = (weight_shape[0], logical_input_size // 16)
            if logical_input_size % 16 or scale_shape != expected_scale_shape:
                raise ValueError(
                    f"Comfy NVFP4 layer {prefix!r} has incompatible weight/scale "
                    f"shapes: {weight_shape} and {scale_shape}"
                )
            pre_quant_scale_key = f"{prefix}.pre_quant_scale"
            marker["_has_pre_quant_scale"] = pre_quant_scale_key in checkpoint_meta
            if marker["_has_pre_quant_scale"]:
                pre_scale_dtype, pre_scale_shape = checkpoint_meta[pre_quant_scale_key]
                if pre_scale_dtype not in ("BF16", "F16", "F32") or (
                    pre_scale_shape != (logical_input_size,)
                ):
                    raise ValueError(
                        f"Comfy NVFP4 layer {prefix!r} has an incompatible "
                        f"pre_quant_scale: {pre_scale_dtype}{pre_scale_shape}"
                    )
            continue
        if marker_format != "int8_tensorwise":
            continue
        weight_dtype, weight_shape = checkpoint_meta[f"{prefix}.weight"]
        scale_dtype, scale_shape = checkpoint_meta[f"{prefix}.weight_scale"]
        if weight_dtype == "I8" and scale_dtype == "F32" and scale_shape == ():
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Comfy tensorwise INT8 layer {prefix!r} needs a 2D weight, "
                    f"got {weight_shape}"
                )
            marker["_is_tensorwise_scalar"] = True
            continue
        if weight_dtype != "I8" or scale_dtype != "F32":
            raise ValueError(
                f"Comfy INT8 layer {prefix!r} needs I8 weights and F32 scales, "
                f"got {weight_dtype} and {scale_dtype}"
            )
        if len(weight_shape) != 2 or scale_shape != (weight_shape[0], 1):
            raise ValueError(
                f"Comfy INT8 layer {prefix!r} has incompatible weight/scale "
                f"shapes: {weight_shape} and {scale_shape}"
            )
        marker["_is_rowwise"] = True

    mapped_markers: dict[str, dict[str, Any]] = {}
    for prefix, marker in raw_markers.items():
        mapped_prefix = param_name_mapper(prefix) if param_name_mapper else prefix
        if mapped_prefix in mapped_markers:
            raise ValueError(
                f"Comfy markers collide after parameter mapping at {mapped_prefix!r}"
            )
        mapped_markers[mapped_prefix] = marker
    return mapped_markers


def resolve_comfy_checkpoint_quantization(
    layer_markers: dict[str, dict[str, Any]],
) -> QuantizationConfig | None:
    if not layer_markers:
        return None
    formats = sorted({str(marker.get("format")) for marker in layer_markers.values()})
    if formats == ["int8_tensorwise"]:
        return KitchenInt8Config(layer_markers=layer_markers)
    if formats == ["asym_w4a8_int8"]:
        return KitchenW4A8Config(layer_markers)
    if formats == ["asym_w4a8_int8", "int8_tensorwise"]:
        return KitchenW4A8Config(layer_markers)
    if formats == ["convrot_w4a4"]:
        return KitchenW4A4Config(layer_markers)
    if formats == ["convrot_w4a4", "int8_tensorwise"]:
        return KitchenW4A4Config(layer_markers)
    if formats == ["float8_e4m3fn"]:
        return ComfyFp8Config(layer_markers)
    if formats in (["nvfp4"], ["int8_tensorwise", "nvfp4"]):
        return ComfyNvfp4Config(layer_markers)
    if formats == ["mxfp8"]:
        return MXFP8Config(
            is_checkpoint_fp8_serialized=True,
            layer_markers=layer_markers,
        )
    raise NotImplementedError(
        "Unsupported Comfy quantization format(s): " + ", ".join(formats)
    )


def normalize_flat_modelopt_quant_config(
    quant_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Fill required diffusers fields for flat ModelOpt component configs."""
    if not isinstance(quant_cfg, dict) or quant_cfg.get("quant_method") != "modelopt":
        return quant_cfg

    quant_algo = str(
        quant_cfg.get("quant_algo")
        or quant_cfg.get("quantization", {}).get("quant_algo")
        or ""
    ).upper()
    if not quant_algo:
        return quant_cfg

    normalized = dict(quant_cfg)
    normalized.setdefault("quant_type", quant_algo)
    return normalized


def _infer_nvfp4_group_size_from_tensors(weight, scale) -> Optional[int]:
    """Infer NVFP4 group_size from serialized weight/scale tensor shapes."""
    return _infer_nvfp4_group_size_from_shapes(
        getattr(weight, "shape", ()),
        getattr(scale, "shape", ()),
    )


def _infer_nvfp4_group_size_from_shapes(weight_shape, scale_shape) -> Optional[int]:
    weight_shape = tuple(weight_shape or ())
    scale_shape = tuple(scale_shape or ())
    if len(weight_shape) < 2:
        return None

    input_size = int(weight_shape[1]) * 2
    if input_size <= 0:
        return None

    candidate_num_groups: list[int] = []
    if len(scale_shape) >= 2:
        candidate_num_groups.append(int(scale_shape[-1]))
    elif len(scale_shape) == 1:
        scale_len = int(scale_shape[0])
        if scale_len == int(weight_shape[0]):
            candidate_num_groups.append(1)
        candidate_num_groups.append(scale_len)
    else:
        candidate_num_groups.append(1)

    for num_groups in candidate_num_groups:
        if num_groups <= 0:
            continue
        if input_size % num_groups == 0:
            return input_size // num_groups

    return None


def _read_safetensors_tensor_metadata(file_path: str) -> dict[str, dict[str, Any]]:
    with open(file_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _is_nvfp4_tensor_family(
    module_name: str,
    tensor_metadata: dict[str, dict[str, Any]],
) -> bool:
    weight_metadata = tensor_metadata.get(f"{module_name}.weight")
    scale_metadata = tensor_metadata.get(f"{module_name}.weight_scale")
    if weight_metadata is None or scale_metadata is None:
        return False

    weight_dtype = str(weight_metadata.get("dtype", "")).upper()
    scale_dtype = str(scale_metadata.get("dtype", "")).upper()
    scale_shape = scale_metadata.get("shape", [])
    return weight_dtype == "U8" and "F8_E4M3" in scale_dtype and len(scale_shape) >= 2


def _resolve_quant_method_name(quant_cfg: dict) -> str:
    quant_cfg = normalize_flat_modelopt_quant_config(quant_cfg) or quant_cfg
    quant_method = quant_cfg.get("quant_method")
    if quant_method == "bitsandbytes":
        return "bitsandbytes"
    modelopt_methods = {"modelopt", "modelopt_fp8", "modelopt_fp4"}
    if quant_method not in modelopt_methods:
        return quant_method

    quant_algo = (
        quant_cfg.get("quant_algo")
        or quant_cfg.get("quantization", {}).get("quant_algo")
        or ""
    ).upper()
    if quant_method != "modelopt" and not quant_algo:
        # Preserve explicit legacy configs that select the backend directly.
        # When an algorithm is present below, validate that it agrees.
        return quant_method
    if quant_algo == "MIXED_PRECISION":
        raise ValueError(
            "ModelOpt mixed precision is not supported by the current SGLang diffusion runtime."
        )
    canonical_method = canonicalize_modelopt_quant_algo(quant_algo)
    if (
        quant_method != "modelopt"
        and canonical_method is not None
        and quant_method != canonical_method
    ):
        raise ValueError(
            f"ModelOpt config declares quant_method={quant_method!r}, but "
            f"quant_algo={quant_algo!r} maps to {canonical_method!r}."
        )
    supported_algorithms = {
        "FP8": "modelopt_fp8",
        "NVFP4": "modelopt_fp4",
    }
    runtime_method = supported_algorithms.get(quant_algo)
    if runtime_method is not None:
        return runtime_method
    if canonical_method is not None:
        raise ValueError(
            f"ModelOpt quant_algo={quant_algo!r} maps to {canonical_method!r}, but "
            "that checkpoint algorithm is not supported by the SGLang diffusion runtime. "
            "Supported ModelOpt checkpoint algorithms are FP8 and NVFP4."
        )
    raise ValueError(f"Unsupported ModelOpt quant_algo for diffusion: {quant_algo}")


def _load_quant_cls(quant_cfg: dict):
    quant_method = _resolve_quant_method_name(quant_cfg)
    if not quant_method:
        raise ValueError("Missing quant_method in quantization config.")
    return get_quantization_config(quant_method)


def find_quant_modelslim_config(model_config, component_model_path):
    # Try exact name first, then glob for variant filenames (e.g. after repack)
    quant_config_file = Path(component_model_path, "quant_model_description.json")
    if not quant_config_file.is_file():
        candidates = sorted(
            Path(component_model_path).glob("quant_model_description*.json")
        )
        quant_config_file = candidates[0] if candidates else None

    quant_cfg = None
    if quant_config_file is not None and Path(quant_config_file).is_file():
        with open(quant_config_file) as f:
            quant_cfg = json.load(f)
        # This field is required for flagless model loading but is not present in
        # modelslim model description, so we're adding it here manually.
        quant_cfg["quant_method"] = "modelslim"

    return quant_cfg


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key


def get_quant_config(
    model_config,
    component_model_path: str,
    packed_modules_mapping: Dict[str, List[str]] = {},
    reverse_param_names_mapping: Dict[str, List[str]] = {},
    remap_prefix: Dict[str, str] | None = None,
    quant_ignore_remap: Optional[Dict[str, str]] = None,
) -> QuantizationConfig:
    quant_cfg = find_quant_modelslim_config(model_config, component_model_path)
    if quant_cfg is not None:
        quant_cls = _load_quant_cls(quant_cfg)
        return quant_cls.from_config(quant_cfg, reverse_param_names_mapping)

    checkpoint_quant_spec = resolve_checkpoint_quant_spec(model_config)
    if checkpoint_quant_spec is None:
        return None

    hf_quant_config = normalize_flat_modelopt_quant_config(checkpoint_quant_spec.config)
    quant_cls = _load_quant_cls(hf_quant_config)

    # GGUF doesn't have config file
    if hf_quant_config["quant_method"] == "gguf":
        return quant_cls.from_config({})

    if hf_quant_config is not None:
        hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
        is_modelopt_fp8 = (
            hf_quant_config.get("quant_method") == "modelopt"
            and "FP8" in str(hf_quant_config.get("quant_algo", "")).upper()
        )
        extra_kwargs = (
            {"ignore_remap": quant_ignore_remap}
            if quant_ignore_remap and is_modelopt_fp8
            else {}
        )
        return quant_cls.from_config(hf_quant_config, **extra_kwargs)

    model_name_or_path = model_config["model_path"]
    hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config['quantization_config']['quant_method']}"
        )
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config['quantization_config']['quant_method']}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)
        if remap_prefix is not None and "quantization" in config:
            exclude_modules = [
                replace_prefix(key, remap_prefix)
                for key in config["quantization"]["exclude_modules"]
            ]
            config["quantization"]["exclude_modules"] = exclude_modules
        config["packed_modules_mapping"] = packed_modules_mapping
        return quant_cls.from_config(config)


def handle_fp8_metadata_format(quant_config_dict):
    layers = quant_config_dict.get("layers", {})
    if any(
        isinstance(v, dict) and "float8" in v.get("format", "") for v in layers.values()
    ):
        quant_config_dict["quant_method"] = "fp8"
        quant_config_dict["activation_scheme"] = "dynamic"
    return quant_config_dict


def get_quant_config_from_safetensors_metadata(
    file_path: str,
) -> Optional[QuantizationConfig]:
    """Extract quantization config from a safetensors file's metadata header.
    Returns None if no recognizable quantization metadata is found.
    """
    metadata = get_metadata_from_safetensors_file(file_path)
    if not metadata:
        return None

    quant_config_str = metadata.get("_quantization_metadata")
    quant_config_dict = None
    if quant_config_str:
        try:
            quant_config_dict = json.loads(quant_config_str)
        except Exception:
            quant_config_dict = None

    if quant_config_dict is None:
        quant_config_str = metadata.get("quantization_config")
        if not quant_config_str:
            return None
        try:
            quant_config_dict = json.loads(quant_config_str)
        except Exception:
            return None

    if not quant_config_dict:
        return None

    # handle diffusers fp8 safetensors metadata format
    if (
        "quant_method" not in quant_config_dict
        and "format_version" in quant_config_dict
        and "layers" in quant_config_dict
    ):
        quant_config_dict = handle_fp8_metadata_format(quant_config_dict)

    quant_method = quant_config_dict.get("quant_method")
    if not quant_method:
        return None

    try:
        quant_cls = _load_quant_cls(quant_config_dict)
        config = quant_cls.from_config(quant_config_dict)
        logger.debug(f"Get quantization config from safetensors file: {file_path}")
        return config
    except Exception as _e:
        return None


def get_metadata_from_safetensors_file(file_path: str):
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            return metadata
    except Exception as e:
        logger.warning(e)


def _canonicalize_modulation_exclude(module_name: str) -> str:
    """Map a serialized modulation weight's parent to the runtime linear prefix.

    Qwen-Image wraps the modulation projection in ``nn.Sequential(SiLU, Linear)``,
    so its weights serialize as ``...img_mod.1.weight`` while the runtime
    ReplicatedLinear advertises ``...img_mod`` as its quant/exclusion prefix.
    Strip the trailing Sequential index so a safetensors-inferred BF16 exclude
    entry actually matches the linear (mirrors the ModelOpt FP8 converter, which
    canonicalizes ``.img_mod.1``/``.txt_mod.1`` to ``.img_mod``/``.txt_mod``).
    No-op for any other module name.
    """
    if module_name.endswith((".img_mod.1", ".txt_mod.1")):
        return module_name.removesuffix(".1")
    return module_name


def _build_nvfp4_config_from_safetensors_files(
    file_paths: list[str],
    param_names_mapping_dict: Optional[dict] = None,
    reverse_param_names_mapping_dict: Optional[dict] = None,
    fallback_group_size: Optional[int] = None,
) -> Optional[QuantizationConfig]:
    """Build a single NVFP4 config by aggregating metadata across multiple files.

    Some checkpoints split BF16 fallback layers and NVFP4 layers across multiple
    safetensors. Building the config from only the first matching file can
    incorrectly exclude layers that are quantized in a later shard.
    """
    group_size = None
    quantized_bfl_modules: set[str] = set()
    non_quantized_bfl_modules: set[str] = set()
    files_with_nvfp4_signal: list[str] = []
    checkpoint_uses_packed_qkv = False
    checkpoint_uses_comfy_quant = False
    packed_qkv_pattern = re.compile(
        r"^(double_blocks\.\d+\.(img|txt)_attn\.qkv|single_blocks\.\d+\.linear1)\."
    )

    for file_path in file_paths:
        metadata = get_metadata_from_safetensors_file(file_path)
        quant_config_dict = None
        metadata_signals_nvfp4 = False
        if metadata:
            quant_config_str = metadata.get("_quantization_metadata")
            if quant_config_str:
                try:
                    quant_config_dict = json.loads(quant_config_str)
                except json.JSONDecodeError:
                    quant_config_dict = None
                else:
                    quant_algo = str(quant_config_dict.get("quant_algo", "")).upper()
                    quant_type = str(quant_config_dict.get("quant_type", "")).upper()
                    metadata_signals_nvfp4 = (
                        "NVFP4" in quant_algo
                        or "FP4" in quant_algo
                        or "NVFP4" in quant_type
                    )

        file_quantized_modules: set[str] = set()
        if (
            quant_config_dict is not None
            and "format_version" in quant_config_dict
            and "layers" in quant_config_dict
        ):
            layers = quant_config_dict.get("layers", {})
            metadata_nvfp4_modules = {
                layer_name
                for layer_name, layer_cfg in layers.items()
                if isinstance(layer_cfg, dict) and layer_cfg.get("format") == "nvfp4"
            }
            file_quantized_modules.update(metadata_nvfp4_modules)

        tensor_metadata = _read_safetensors_tensor_metadata(file_path)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            all_keys = set(f.keys())
            if any(packed_qkv_pattern.match(k) for k in all_keys):
                checkpoint_uses_packed_qkv = True
            if any(k.endswith(".comfy_quant") for k in all_keys):
                checkpoint_uses_comfy_quant = True

            # Some ModelOpt NVFP4 exports only store a flat config.json plus
            # per-file metadata without the diffusers `layers` section. Infer
            # quantized modules directly from tensor families in that case.
            # Mixed checkpoints may also contain FP8 fallback layers with scalar
            # `.weight_scale`, so require packed uint8 weights and block scales.
            file_quantized_modules.update(
                key[: -len(".weight_scale")]
                for key in all_keys
                if key.endswith(".weight_scale")
                and _is_nvfp4_tensor_family(
                    key[: -len(".weight_scale")], tensor_metadata
                )
            )

            if file_quantized_modules or metadata_signals_nvfp4:
                files_with_nvfp4_signal.append(file_path)
            quantized_bfl_modules.update(file_quantized_modules)

            if group_size is None:
                for layer_name in sorted(file_quantized_modules):
                    weight_key = f"{layer_name}.weight"
                    scale_key = f"{layer_name}.weight_scale"
                    weight_metadata = tensor_metadata.get(weight_key)
                    scale_metadata = tensor_metadata.get(scale_key)
                    if weight_metadata is not None and scale_metadata is not None:
                        group_size = _infer_nvfp4_group_size_from_shapes(
                            weight_metadata.get("shape"),
                            scale_metadata.get("shape"),
                        )
                        if group_size is not None:
                            break

            for k in sorted(all_keys):
                if not k.endswith(".weight"):
                    continue
                module_name = k[: -len(".weight")]
                if module_name not in file_quantized_modules:
                    non_quantized_bfl_modules.add(module_name)

    if not files_with_nvfp4_signal:
        return None

    if (
        group_size is not None
        and fallback_group_size is not None
        and group_size != fallback_group_size
    ):
        logger.warning(
            "NVFP4 group_size inferred from safetensors (%d) does not match config (%d); "
            "preferring safetensors.",
            group_size,
            fallback_group_size,
        )

    if group_size is None and fallback_group_size is not None:
        logger.info(
            "Falling back to config-derived NVFP4 group_size=%d for %s",
            fallback_group_size,
            ", ".join(files_with_nvfp4_signal),
        )
        group_size = fallback_group_size

    if group_size is None:
        logger.warning(
            "Could not infer group_size from NVFP4 safetensors: %s",
            ", ".join(files_with_nvfp4_signal),
        )
        return None

    exclude_bfl_modules = sorted(non_quantized_bfl_modules - quantized_bfl_modules)

    exclude_modules = []
    mapping_fn = None
    reverse_mapping_fn = None
    if param_names_mapping_dict or reverse_param_names_mapping_dict:
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        if param_names_mapping_dict:
            mapping_fn = get_param_names_mapping(param_names_mapping_dict)
        if reverse_param_names_mapping_dict:
            reverse_mapping_fn = get_param_names_mapping(
                reverse_param_names_mapping_dict
            )

    for module_bfl in exclude_bfl_modules:
        raw_weight_name = f"{module_bfl}.weight"
        if mapping_fn is not None:
            mapped, _, _ = mapping_fn(raw_weight_name)
            if mapped != raw_weight_name:
                exclude_modules.append(
                    mapped[: -len(".weight")] if mapped.endswith(".weight") else mapped
                )
                continue

        if reverse_mapping_fn is not None:
            reverse_mapped, _, _ = reverse_mapping_fn(raw_weight_name)
            if reverse_mapped != raw_weight_name:
                exclude_modules.append(
                    reverse_mapped[: -len(".weight")]
                    if reverse_mapped.endswith(".weight")
                    else reverse_mapped
                )
                continue

        exclude_modules.append(module_bfl)

    exclude_modules = sorted(
        {_canonicalize_modulation_exclude(m) for m in exclude_modules}
    )

    try:
        quant_cls = get_quantization_config("modelopt_fp4")
        checkpoint_uses_swizzled_scales = (
            checkpoint_uses_packed_qkv or checkpoint_uses_comfy_quant
        )
        result = quant_cls.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": group_size,
                "ignore": exclude_modules,
                "checkpoint_uses_packed_qkv": checkpoint_uses_packed_qkv,
                # packed-QKV and Comfy NVFP4 checkpoints store serialized
                # weights/scales in the FlashInfer/CUTLASS checkpoint layout
                "checkpoint_weight_scale_layout": (
                    "swizzled" if checkpoint_uses_swizzled_scales else "linear"
                ),
                "swap_weight_nibbles": checkpoint_uses_swizzled_scales,
                "checkpoint_uses_comfy_quantization": checkpoint_uses_comfy_quant,
            }
        )
        logger.info(
            "Built NVFP4 quant config from %d safetensors: group_size=%d, %d excluded modules, packed_qkv=%s, comfy_quant=%s, scale_layout=%s, swap_nibbles=%s",
            len(files_with_nvfp4_signal),
            group_size,
            len(exclude_modules),
            checkpoint_uses_packed_qkv,
            checkpoint_uses_comfy_quant,
            getattr(result, "checkpoint_weight_scale_layout", "linear"),
            getattr(result, "swap_weight_nibbles", False),
        )
        return result
    except Exception as e:
        logger.warning(
            "Failed to build NVFP4 config from %s: %s",
            ", ".join(files_with_nvfp4_signal),
            e,
        )
        return None


def build_nvfp4_config_from_safetensors(
    file_path: str,
    param_names_mapping_dict: Optional[dict] = None,
    reverse_param_names_mapping_dict: Optional[dict] = None,
    fallback_group_size: Optional[int] = None,
) -> Optional[QuantizationConfig]:
    """Backward-compatible wrapper for a single safetensors file."""
    return _build_nvfp4_config_from_safetensors_files(
        [file_path],
        param_names_mapping_dict,
        reverse_param_names_mapping_dict,
        fallback_group_size,
    )


def build_nvfp4_config_from_safetensors_list(
    file_paths: list[str],
    param_names_mapping_dict: Optional[dict] = None,
    reverse_param_names_mapping_dict: Optional[dict] = None,
    fallback_group_size: Optional[int] = None,
) -> Optional[QuantizationConfig]:
    return _build_nvfp4_config_from_safetensors_files(
        file_paths,
        param_names_mapping_dict,
        reverse_param_names_mapping_dict,
        fallback_group_size,
    )
