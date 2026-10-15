# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 package configuration and DiT loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.models.dits.flux3 import Flux3DiTConfig
    from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
        QuantizationConfig,
    )
    from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
        Flux3ImagePackage,
    )


def flux3_image_dit_config_from_package(package: Flux3ImagePackage) -> Flux3DiTConfig:
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )

    source = package.config.dit
    content = {
        mode: channels
        for mode, channels in source.in_channels.items()
        if mode != "text"
    }
    # ``image_rope_scale`` scales the position ids in the pipeline, not the DiT.
    return flux3_image_dit_config(
        in_channels=content,
        sequence={f"x_{mode}": mode for mode in content},
        out_modes=list(source.out_modes),
        vec_in_dim=source.vec_in_dim,
        context_in_dim=source.in_channels["text"],
        hidden_size=source.hidden,
        num_attention_heads=source.heads,
        depth=source.depth,
        depth_single_blocks=source.depth_single,
        axes_dim=tuple(source.axes_dim),
        theta=source.theta,
        mlp_ratio=source.mlp_ratio,
        flow_map=source.flow_map,
    )


def flux3_image_quantization_name(
    quant_config: QuantizationConfig | None,
) -> str | None:
    if quant_config is None:
        return None
    name = quant_config.get_name()
    return {
        "modelopt_fp4": "nvfp4",
        "rowwise_fp8": "rowwise-fp8",
    }.get(name, name)


def resolve_flux3_image_dit_quant_config(
    dit_path: str,
    quantization: str | None = None,
    ignored_layers: list[str] | None = None,
) -> QuantizationConfig | None:
    from sglang.multimodal_gen.runtime.layers.quantization import (
        get_quantization_config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.method_names import (
        canonical_quantization_method,
    )
    from sglang.multimodal_gen.runtime.utils.quantization_utils import (
        get_quant_config_from_safetensors_metadata,
    )

    ckpt_quant = get_quant_config_from_safetensors_metadata(dit_path)
    if not quantization:
        return ckpt_quant
    if ckpt_quant is not None:
        raise ValueError(
            "FLUX.3 checkpoint already has quantization metadata; "
            "do not also set --quantization"
        )
    quantization = canonical_quantization_method(quantization)
    quant_cls = get_quantization_config(quantization)
    if quantization in {"fp8", "mxfp4", "convrot_int8"}:
        return quant_cls(ignored_layers=ignored_layers)
    return quant_cls()


def build_flux3_image_fp8r_quant_config(
    dit_path: str | Path,
    quant_config_path: str | Path,
) -> QuantizationConfig:
    """Translate BFL's sidecar/header FP8R contract to SGLang rowwise FP8."""
    from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
        BF16_FORMAT,
        FP8_FORMAT,
        RowwiseFp8Error,
        build_rowwise_fp8_config_from_metadata,
    )

    dit_path = Path(dit_path)
    quant_config_path = Path(quant_config_path)
    try:
        sidecar = json.loads(quant_config_path.read_text())
    except FileNotFoundError:
        raise ValueError(f"FLUX.3 FP8R config not found: {quant_config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{quant_config_path}: invalid JSON ({exc})") from None

    producer = sidecar.get("producer", {})
    quantization = sidecar.get("quantization", {})
    algorithm = quantization.get("quant_algo")
    excluded = quantization.get("exclude_modules")
    if producer.get("name") != "bfl":
        raise ValueError(f"{quant_config_path}: expected producer.name='bfl'")
    if algorithm != "FP8_PER_CHANNEL_PER_TOKEN":
        raise ValueError(
            f"{quant_config_path}: unsupported quant_algo {algorithm!r}; "
            "expected 'FP8_PER_CHANNEL_PER_TOKEN'"
        )
    if not isinstance(excluded, list) or not all(
        isinstance(name, str) and name for name in excluded
    ):
        raise ValueError(f"{quant_config_path}: exclude_modules must be a string list")

    tensor_metadata = {}
    fp8_layers = set()
    bf16_layers = set()
    other_matrices = []
    with safe_open(str(dit_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("quantization") != "fp8r":
            raise ValueError(
                f"{dit_path}: expected safetensors quantization='fp8r', "
                f"got {metadata.get('quantization')!r}"
            )
        if metadata.get("scale") != "per_row_fp32":
            raise ValueError(
                f"{dit_path}: expected scale='per_row_fp32', "
                f"got {metadata.get('scale')!r}"
            )
        for name in checkpoint.keys():
            tensor = checkpoint.get_slice(name)
            dtype = tensor.get_dtype()
            shape = list(tensor.get_shape())
            tensor_metadata[name] = {"dtype": dtype, "shape": shape}
            if not name.endswith(".weight") or len(shape) != 2:
                continue
            prefix = name.removesuffix(".weight")
            if dtype == "F8_E4M3":
                fp8_layers.add(prefix)
            elif dtype == "BF16":
                bf16_layers.add(prefix)
            else:
                other_matrices.append(f"{name}:{dtype}")

    # Exports differ in which layers stay BF16; the strict load checks the key set.
    if not fp8_layers:
        raise ValueError(f"{dit_path}: no FP8 matrices found")
    if other_matrices:
        raise ValueError(
            f"{dit_path}: matrices must be FP8 or BF16, got {other_matrices[:5]}"
        )
    if set(excluded) != bf16_layers:
        missing = sorted(bf16_layers - set(excluded))
        extra = sorted(set(excluded) - bf16_layers)
        raise ValueError(
            f"{quant_config_path}: exclude_modules disagrees with checkpoint "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    layers = {prefix: {"format": FP8_FORMAT} for prefix in sorted(fp8_layers)}
    layers.update({prefix: {"format": BF16_FORMAT} for prefix in sorted(bf16_layers)})
    config = {
        "format_version": "1.1",
        "layers": layers,
        "quant_configs": {FP8_FORMAT: {"scale_recipe": "rowwise"}},
    }
    result = build_rowwise_fp8_config_from_metadata(config, tensor_metadata)
    if result.input_scale_layers:
        raise RowwiseFp8Error(
            f"{dit_path}: BFL FP8R must use dynamic activations, but serialized "
            f"input scales were found for {sorted(result.input_scale_layers)[:5]}"
        )
    return result


def build_flux3_image_nvfp4_quant_config(
    dit_path: str | Path,
    quant_config_path: str | Path,
) -> QuantizationConfig:
    """Validate BFL's NVFP4 contract and build the existing ModelOpt config."""
    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
    )
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT
    from sglang.multimodal_gen.runtime.utils.quantization_utils import (
        build_nvfp4_config_from_safetensors_list,
    )

    dit_path = Path(dit_path)
    quant_config_path = Path(quant_config_path)
    try:
        sidecar = json.loads(quant_config_path.read_text())
    except FileNotFoundError:
        raise ValueError(
            f"FLUX.3 NVFP4 config not found: {quant_config_path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{quant_config_path}: invalid JSON ({exc})") from None

    producer = sidecar.get("producer", {})
    quantization = sidecar.get("quantization", {})
    algorithm = quantization.get("quant_algo")
    group_size = quantization.get("group_size")
    excluded = quantization.get("exclude_modules")
    if producer.get("name") != "bfl":
        raise ValueError(f"{quant_config_path}: expected producer.name='bfl'")
    if algorithm != "NVFP4":
        raise ValueError(
            f"{quant_config_path}: unsupported quant_algo {algorithm!r}; "
            "expected 'NVFP4'"
        )
    if group_size != 16:
        raise ValueError(
            f"{quant_config_path}: expected group_size=16, got {group_size!r}"
        )
    if not isinstance(excluded, list) or not all(
        isinstance(name, str) and name for name in excluded
    ):
        raise ValueError(f"{quant_config_path}: exclude_modules must be a string list")

    quantized = set()
    bf16_layers = set()
    qkv_input_scales: dict[str, dict[str, float]] = {}
    with safe_open(str(dit_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("quantization") != "nvfp4":
            raise ValueError(
                f"{dit_path}: expected safetensors quantization='nvfp4', "
                f"got {metadata.get('quantization')!r}"
            )
        keys = set(checkpoint.keys())
        quantized = {
            name.removesuffix(".weight_scale_2")
            for name in keys
            if name.endswith(".weight_scale_2")
        }
        for prefix in sorted(quantized):
            expected_names = {
                f"{prefix}.weight",
                f"{prefix}.weight_scale",
                f"{prefix}.weight_scale_2",
                f"{prefix}.input_scale",
            }
            missing = expected_names - keys
            if missing:
                raise ValueError(
                    f"{dit_path}: NVFP4 layer {prefix!r} is missing {sorted(missing)}"
                )
            weight = checkpoint.get_slice(f"{prefix}.weight")
            scale = checkpoint.get_slice(f"{prefix}.weight_scale")
            scale_2 = checkpoint.get_slice(f"{prefix}.weight_scale_2")
            input_scale = checkpoint.get_slice(f"{prefix}.input_scale")
            weight_shape = list(weight.get_shape())
            scale_shape = list(scale.get_shape())
            if (
                weight.get_dtype() != "U8"
                or len(weight_shape) != 2
                or scale.get_dtype() != "F8_E4M3"
                or scale_shape != [weight_shape[0], weight_shape[1] // 8]
                or scale_2.get_dtype() != "F32"
                or list(scale_2.get_shape()) != []
                or input_scale.get_dtype() != "F32"
                or list(input_scale.get_shape()) != []
            ):
                raise ValueError(
                    f"{dit_path}: invalid NVFP4 tensor family for {prefix!r}"
                )
            for leaf in ("q_proj", "k_proj", "v_proj"):
                suffix = f".{leaf}"
                if prefix.endswith(suffix):
                    block = prefix.removesuffix(suffix)
                    qkv_input_scales.setdefault(block, {})[leaf] = float(
                        checkpoint.get_tensor(f"{prefix}.input_scale").item()
                    )
                    break

        other_matrices = []
        for name in keys:
            prefix = name.removesuffix(".weight")
            if not name.endswith(".weight") or prefix in quantized:
                continue
            tensor = checkpoint.get_slice(name)
            if len(tensor.get_shape()) != 2:
                continue
            if tensor.get_dtype() == "BF16":
                bf16_layers.add(prefix)
            else:
                other_matrices.append(f"{name}:{tensor.get_dtype()}")

    # Exports differ in which layers stay BF16; the strict load checks the key set.
    if not quantized:
        raise ValueError(f"{dit_path}: no NVFP4 matrices found")
    if other_matrices:
        raise ValueError(
            f"{dit_path}: matrices must be NVFP4 or BF16, got {other_matrices[:5]}"
        )
    if set(excluded) != bf16_layers:
        missing = sorted(bf16_layers - set(excluded))
        extra = sorted(set(excluded) - bf16_layers)
        raise ValueError(
            f"{quant_config_path}: exclude_modules disagrees with checkpoint "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    for block, scales in qkv_input_scales.items():
        if (
            set(scales) != {"q_proj", "k_proj", "v_proj"}
            or len(set(scales.values())) != 1
        ):
            raise ValueError(
                f"{dit_path}: Q/K/V input scales must match for {block}: {scales}"
            )

    result = build_nvfp4_config_from_safetensors_list(
        [str(dit_path)],
        param_names_mapping_dict=Flux3ImageDiT.param_names_mapping,
        fallback_group_size=16,
    )
    if not isinstance(result, ModelOptFp4Config):
        raise ValueError(f"{dit_path}: could not infer a ModelOpt NVFP4 config")
    if (
        not result.is_checkpoint_nvfp4_serialized
        or result.group_size != 16
        or result.checkpoint_weight_scale_layout != "linear"
        or result.swap_weight_nibbles
        or result.dynamic_input_scale
    ):
        raise ValueError(
            f"{dit_path}: incompatible inferred NVFP4 contract "
            f"(group_size={result.group_size}, "
            f"scale_layout={result.checkpoint_weight_scale_layout!r}, "
            f"swap_weight_nibbles={result.swap_weight_nibbles}, "
            f"dynamic_input_scale={result.dynamic_input_scale})"
        )
    return result


def resolve_flux3_image_package_quant_config(
    package: Flux3ImagePackage,
    quantization: str | None = None,
    ignored_layers: list[str] | None = None,
) -> QuantizationConfig | None:
    """Resolve the serialized quantization contract selected by a package."""
    if package.config.quantization == "fp8r":
        if quantization:
            raise ValueError(
                "FLUX.3 FP8R is serialized in the selected package variant; "
                "do not also set --quantization"
            )
        if package.quant_config is None:
            raise ValueError("FLUX.3 FP8R package has no hf_quant_config.json")
        return build_flux3_image_fp8r_quant_config(
            package.weights, package.quant_config
        )
    if package.config.quantization == "nvfp4":
        if quantization:
            raise ValueError(
                "FLUX.3 NVFP4 is serialized in the selected package variant; "
                "do not also set --quantization"
            )
        if package.quant_config is None:
            raise ValueError("FLUX.3 NVFP4 package has no hf_quant_config.json")
        return build_flux3_image_nvfp4_quant_config(
            package.weights, package.quant_config
        )
    return resolve_flux3_image_dit_quant_config(
        str(package.weights),
        quantization=quantization,
        ignored_layers=ignored_layers,
    )


def select_flux3_image_quant_config(
    quant_config: QuantizationConfig | None,
    config: Flux3DiTConfig,
    *,
    enable_prims_ts: bool = True,
) -> QuantizationConfig | None:
    """Specialize a serialized checkpoint config for the native FLUX.3 DiT.

    Rowwise FP8 and ModelOpt NVFP4 get FLUX.3 methods that own the packed-QKV
    metadata and PrimsTS weight preparation; other configs pass through.
    """
    from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
        Flux3ModelOptFp4Config,
        Flux3PrimsTsConfigMixin,
        Flux3RowwiseFp8Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
        RowwiseFp8Config,
    )

    if quant_config is None:
        return None
    if isinstance(quant_config, Flux3PrimsTsConfigMixin):
        quant_config.enable_prims_ts = enable_prims_ts
        return quant_config
    arch = config.arch_config
    head_dim = arch.hidden_size // arch.num_attention_heads
    if isinstance(quant_config, RowwiseFp8Config):
        return Flux3RowwiseFp8Config.from_base_config(
            quant_config,
            head_dim=head_dim,
            enable_prims_ts=enable_prims_ts,
        )
    if isinstance(quant_config, ModelOptFp4Config):
        return Flux3ModelOptFp4Config.from_base_config(
            quant_config,
            head_dim=head_dim,
            enable_prims_ts=enable_prims_ts,
        )
    return quant_config


def load_flux3_image_dit(
    dit_path: str,
    config: Flux3DiTConfig,
    device: torch.device,
    quant_config: QuantizationConfig | None,
    *,
    enable_prims_ts: bool = True,
    server_args,
):
    """Load a strict FLUX.3 package checkpoint through the standard lifecycle."""
    from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
        Flux3PrimsTsConfigMixin,
    )
    from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
    from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT

    if not os.path.isfile(dit_path):
        raise FileNotFoundError(f"FLUX.3 DiT checkpoint not found: {dit_path}")

    quant_config = select_flux3_image_quant_config(
        quant_config,
        config,
        enable_prims_ts=enable_prims_ts,
    )

    on_cpu = server_args.should_start_component_on_cpu("transformer")
    weight_load_plan = WeightLoadPlan.for_component(
        checkpoint_load_device=torch.device("cpu")
        if on_cpu and quant_config is None
        else device,
        needs_device_weight_postprocess=quant_config is not None,
        component_starts_on_cpu=on_cpu,
        load_full_state_dict_on_device=server_args.should_direct_gpu_weight_load_component(
            "transformer"
        ),
    )
    model = maybe_load_fsdp_model(
        model_cls=Flux3ImageDiT,
        init_params={
            "config": config,
            "hf_config": {},
            "quant_config": quant_config,
        },
        weight_dir_list=[dit_path],
        device=device,
        hsdp_replicate_dim=server_args.hsdp_replicate_dim,
        hsdp_shard_dim=server_args.hsdp_shard_dim,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        component_starts_on_cpu=on_cpu,
        fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
        pin_cpu_memory=server_args.pin_cpu_memory,
        strict=True,
        weight_load_plan=weight_load_plan,
    )
    if isinstance(quant_config, Flux3PrimsTsConfigMixin):
        quant_config.verify_prims_ts_coverage(model)
    return model
