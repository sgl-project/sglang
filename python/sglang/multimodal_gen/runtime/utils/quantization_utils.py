import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from safetensors import safe_open

from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    weight_shape = tuple(getattr(weight, "shape", ()))
    scale_shape = tuple(getattr(scale, "shape", ()))
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


def _resolve_quant_method_name(quant_cfg: dict) -> str:
    quant_cfg = normalize_flat_modelopt_quant_config(quant_cfg) or quant_cfg
    quant_method = quant_cfg.get("quant_method")
    if quant_method != "modelopt":
        return quant_method

    quant_algo = (
        quant_cfg.get("quant_algo")
        or quant_cfg.get("quantization", {}).get("quant_algo")
        or ""
    ).upper()
    if quant_algo == "MIXED_PRECISION":
        raise ValueError(
            "ModelOpt mixed precision is not supported by the current SGLang diffusion runtime."
        )
    if "FP8" in quant_algo:
        return "modelopt_fp8"
    if "FP4" in quant_algo or "NVFP4" in quant_algo:
        return "modelopt_fp4"
    raise ValueError(f"Unsupported ModelOpt quant_algo for diffusion: {quant_algo}")


def _load_quant_cls(quant_cfg: dict):
    quant_method = _resolve_quant_method_name(quant_cfg)
    if not quant_method:
        raise ValueError("Missing quant_method in quantization config.")
    return get_quantization_config(quant_method)


def find_quant_modelslim_config(model_config, component_model_path):
    quant_config_file = Path(component_model_path, "quant_model_description.json")
    quant_cfg = None
    if quant_config_file.is_file():
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
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cfg = find_quant_modelslim_config(model_config, component_model_path)
    if quant_cfg is not None:
        quant_cls = _load_quant_cls(quant_cfg)
        return quant_cls.from_config(quant_cfg)

    if "quantization_config" not in model_config:
        return None

    hf_quant_config = normalize_flat_modelopt_quant_config(
        model_config["quantization_config"]
    )
    if hf_quant_config is not None and not isinstance(hf_quant_config, dict):
        hf_quant_config = hf_quant_config.to_dict()
    quant_cls = _load_quant_cls(hf_quant_config)

    # GGUF doesn't have config file
    if hf_quant_config["quant_method"] == "gguf":
        return quant_cls.from_config({})

    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config, "compression_config", None)
    if hf_quant_config is not None:
        hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
        return quant_cls.from_config(hf_quant_config)

    model_name_or_path = model_config["model_path"]
    is_local = os.path.isdir(model_name_or_path)
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
            file_quantized_modules.update(
                layer_name
                for layer_name, layer_cfg in layers.items()
                if isinstance(layer_cfg, dict) and layer_cfg.get("format") == "nvfp4"
            )

        with safe_open(file_path, framework="pt", device="cpu") as f:
            all_keys = set(f.keys())
            if any(packed_qkv_pattern.match(k) for k in all_keys):
                checkpoint_uses_packed_qkv = True

            # Some ModelOpt NVFP4 exports only store a flat config.json plus
            # per-file metadata without the diffusers `layers` section. Infer
            # quantized modules directly from tensor families in that case:
            # quantized modules ship `.weight` + `.weight_scale`, while BF16
            # fallbacks only ship `.weight`.
            file_quantized_modules.update(
                key[: -len(".weight_scale")]
                for key in all_keys
                if key.endswith(".weight_scale")
                and f"{key[: -len('.weight_scale')]}.weight" in all_keys
            )

            if file_quantized_modules or metadata_signals_nvfp4:
                files_with_nvfp4_signal.append(file_path)
            quantized_bfl_modules.update(file_quantized_modules)

            if group_size is None:
                for layer_name in sorted(file_quantized_modules):
                    weight_key = f"{layer_name}.weight"
                    scale_key = f"{layer_name}.weight_scale"
                    if weight_key in all_keys and scale_key in all_keys:
                        w = f.get_tensor(weight_key)
                        s = f.get_tensor(scale_key)
                        group_size = _infer_nvfp4_group_size_from_tensors(w, s)
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
                exclude_modules.append(module_bfl)
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

    exclude_modules = sorted(set(exclude_modules))

    try:
        quant_cls = get_quantization_config("modelopt_fp4")
        result = quant_cls.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": group_size,
                "ignore": exclude_modules,
                "checkpoint_uses_packed_qkv": checkpoint_uses_packed_qkv,
            }
        )
        logger.info(
            "Built NVFP4 quant config from %d safetensors: group_size=%d, %d excluded modules, packed_qkv=%s",
            len(files_with_nvfp4_signal),
            group_size,
            len(exclude_modules),
            checkpoint_uses_packed_qkv,
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
