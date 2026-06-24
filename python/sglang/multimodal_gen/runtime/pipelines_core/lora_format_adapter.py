from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional

import torch
from diffusers.loaders import lora_conversion_utils as lcu

logger = logging.getLogger("LoRAFormatAdapter")


class LoRAFormat(str, Enum):
    """Supported external LoRA formats before normalization."""

    STANDARD = "standard"
    NON_DIFFUSERS_SD = "non-diffusers-sd"
    QWEN_IMAGE_STANDARD = "qwen-image-standard"
    XLABS_FLUX = "xlabs-ai"
    KOHYA_FLUX = "kohya-flux"
    WAN = "wan"
    AI_TOOLKIT_FLUX = "ai-toolkit-flux"


def _sample_keys(keys: Iterable[str], k: int = 20) -> list[str]:
    out = []
    for i, key in enumerate(keys):
        if i >= k:
            break
        out.append(key)
    return out


def _has_substring_key(keys: Iterable[str], substr: str) -> bool:
    return any(substr in k for k in keys)


def _has_prefix_key(keys: Iterable[str], prefix: str) -> bool:
    return any(k.startswith(prefix) for k in keys)


def _looks_like_xlabs_flux_key(k: str) -> bool:
    """XLabs FLUX-style keys under double_blocks/single_blocks with lora down/up."""
    if not (k.endswith(".down.weight") or k.endswith(".up.weight")):
        return False

    if not k.startswith(
        (
            "double_blocks.",
            "single_blocks.",
            "diffusion_model.double_blocks",
            "diffusion_model.single_blocks",
        )
    ):
        return False

    return ".processor." in k or ".proj_lora" in k or ".qkv_lora" in k


def _looks_like_kohya_flux(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya FLUX LoRA (flux_lora.py) under lora_unet_double/single_blocks_ prefixes."""
    if not state_dict:
        return False
    keys = state_dict.keys()
    return any(
        k.startswith("lora_unet_double_blocks_")
        or k.startswith("lora_unet_single_blocks_")
        for k in keys
    )


def _looks_like_non_diffusers_sd(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Classic non-diffusers SD LoRA (Kohya/A1111/sd-scripts)."""
    if not state_dict:
        return False
    keys = state_dict.keys()
    return all(
        k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in keys
    )


def _looks_like_wan_lora(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Wan2.2 distill LoRAs (Wan-AI / Wan2.2-Distill-Loras style)."""
    if not state_dict:
        return False

    for k in state_dict.keys():
        if not k.startswith("diffusion_model.blocks."):
            continue
        if ".lora_down" not in k and ".lora_up" not in k:
            continue
        if ".cross_attn." in k or ".self_attn." in k or ".ffn." in k or ".norm3." in k:
            return True

    return False


def _looks_like_qwen_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    if not keys:
        return False
    return _has_prefix_key(keys, "transformer.transformer_blocks.") and (
        _has_substring_key(keys, ".lora.down.weight")
        or _has_substring_key(keys, ".lora.up.weight")
    )


def _looks_like_ai_toolkit_flux_lora(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Detect ai-toolkit/ComfyUI trained Flux LoRA with double_blocks/single_blocks naming.

    Key patterns: double_blocks.{N}.img_attn.proj.lora_A.weight
    """
    keys = list(state_dict.keys())
    if not keys:
        return False

    has_double_blocks = any(
        k.startswith("double_blocks.")
        or k.startswith("base_model.model.double_blocks.")
        for k in keys
    )
    has_single_blocks = any(
        k.startswith("single_blocks.")
        or k.startswith("base_model.model.single_blocks.")
        for k in keys
    )
    has_lora_ab = _has_substring_key(keys, ".lora_A") or _has_substring_key(
        keys, ".lora_B"
    )

    return (has_double_blocks or has_single_blocks) and has_lora_ab


def detect_lora_format_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> LoRAFormat:
    """Classify LoRA format by key patterns only."""
    keys = list(state_dict.keys())
    if not keys:
        return LoRAFormat.STANDARD

    if _looks_like_ai_toolkit_flux_lora(state_dict):
        return LoRAFormat.AI_TOOLKIT_FLUX

    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        return LoRAFormat.STANDARD

    if any(_looks_like_xlabs_flux_key(k) for k in keys):
        return LoRAFormat.XLABS_FLUX
    if _looks_like_kohya_flux(state_dict):
        return LoRAFormat.KOHYA_FLUX

    if _looks_like_wan_lora(state_dict):
        return LoRAFormat.WAN

    if _looks_like_qwen_image(state_dict):
        return LoRAFormat.STANDARD

    if _looks_like_non_diffusers_sd(state_dict):
        return LoRAFormat.NON_DIFFUSERS_SD

    if _has_substring_key(keys, ".lora.down") or _has_substring_key(keys, ".lora_up"):
        return LoRAFormat.NON_DIFFUSERS_SD

    return LoRAFormat.STANDARD


def _convert_qwen_image_standard(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Qwen-Image: transformer.*.lora.down/up -> transformer_blocks.*.lora_A/B."""
    out: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        new_name = name

        if new_name.startswith("transformer."):
            new_name = new_name[len("transformer.") :]

        if new_name.endswith(".lora.down.weight"):
            new_name = new_name.replace(".lora.down.weight", ".lora_A.weight")
        elif new_name.endswith(".lora.up.weight"):
            new_name = new_name.replace(".lora.up.weight", ".lora_B.weight")

        out[new_name] = tensor

    return out


def _convert_non_diffusers_sd_simple(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Generic down/up -> A/B conversion for non-diffusers SD-like formats."""
    out: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        new_name = name

        if "lora_down.weight" in new_name:
            new_name = new_name.replace("lora_down.weight", "lora_A.weight")
        elif "lora_up.weight" in new_name:
            new_name = new_name.replace("lora_up.weight", "lora_B.weight")
        elif new_name.endswith(".lora_down"):
            new_name = new_name.replace(".lora_down", ".lora_A")
        elif new_name.endswith(".lora_up"):
            new_name = new_name.replace(".lora_up", ".lora_B")

        out[new_name] = tensor

    sample = _sample_keys(out.keys(), 20)
    log.info(
        "[LoRAFormatAdapter] after NON_DIFFUSERS_SD simple conversion, "
        "sample keys (<=20): %s",
        ", ".join(sample),
    )
    return out


def _convert_with_diffusers_utils_if_available(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Optional[Dict[str, torch.Tensor]]:
    """Use diffusers.lora_conversion_utils if available."""
    try:
        if hasattr(lcu, "maybe_convert_state_dict"):
            converted = lcu.maybe_convert_state_dict(  # type: ignore[attr-defined]
                state_dict
            )
        else:
            converted = dict(state_dict)

        if not isinstance(converted, dict):
            converted = dict(converted)

        sample = _sample_keys(converted.keys(), 20)
        log.info(
            "[LoRAFormatAdapter] diffusers.lora_conversion_utils converted keys, "
            "sample keys (<=20): %s",
            ", ".join(sample),
        )
        return converted
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[LoRAFormatAdapter] diffusers lora_conversion_utils failed, "
            "falling back to internal converters. Error: %s",
            exc,
        )
        return None


def _convert_via_diffusers_candidates(
    state_dict: Mapping[str, torch.Tensor],
    candidate_names: tuple[str, ...],
    log: logging.Logger,
    unavailable_warning: str,
    no_converter_warning: str,
    success_info: str,
    all_failed_warning: str,
) -> Dict[str, torch.Tensor]:
    """Try multiple named converters in lora_conversion_utils, use the first that works."""
    converters = [
        (n, getattr(lcu, n)) for n in candidate_names if callable(getattr(lcu, n, None))
    ]
    if not converters:
        log.warning(no_converter_warning)
        return dict(state_dict)

    last_err: Optional[Exception] = None

    for name, fn in converters:
        try:
            sd_copy = dict(state_dict)
            out = fn(sd_copy)
            if isinstance(out, tuple) and isinstance(out[0], dict):
                out = out[0]
            if not isinstance(out, dict):
                raise TypeError(f"Converter {name} returned {type(out)}")
            log.info(success_info.format(name=name))
            return out
        except Exception as exc:
            last_err = exc

    log.warning(all_failed_warning.format(last_err=last_err))
    return dict(state_dict)


def _convert_xlabs_ai_via_diffusers(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Convert XLabs FLUX LoRA via diffusers helpers."""
    return _convert_via_diffusers_candidates(
        state_dict,
        (
            "_convert_xlabs_flux_lora_to_diffusers",
            "convert_xlabs_lora_state_dict_to_diffusers",
            "convert_xlabs_lora_to_diffusers",
            "convert_xlabs_flux_lora_to_diffusers",
        ),
        log=log,
        unavailable_warning=(
            "[LoRAFormatAdapter] XLabs FLUX detected but diffusers is unavailable."
        ),
        no_converter_warning=(
            "[LoRAFormatAdapter] No XLabs FLUX converter found in diffusers."
        ),
        success_info="[LoRAFormatAdapter] Converted XLabs FLUX LoRA using {name}",
        all_failed_warning=(
            "[LoRAFormatAdapter] All XLabs FLUX converters failed; "
            "last error: {last_err}"
        ),
    )


def _convert_kohya_flux_via_diffusers(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Convert Kohya FLUX LoRA via diffusers helpers."""
    return _convert_via_diffusers_candidates(
        state_dict,
        (
            "_convert_kohya_flux_lora_to_diffusers",
            "convert_kohya_flux_lora_to_diffusers",
        ),
        log=log,
        unavailable_warning=(
            "[LoRAFormatAdapter] Kohya FLUX detected but diffusers is unavailable."
        ),
        no_converter_warning="[LoRAFormatAdapter] No Kohya FLUX converter found.",
        success_info="[LoRAFormatAdapter] Converted Kohya FLUX LoRA using {name}",
        all_failed_warning=(
            "[LoRAFormatAdapter] Kohya FLUX conversion failed; "
            "last error: {last_err}"
        ),
    )


def _convert_ai_toolkit_flux_lora(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Convert ai-toolkit/ComfyUI trained Flux LoRA to SGLang format.

    Handles the naming convention conversion:
    - double_blocks.{N}.img_attn.qkv -> transformer_blocks.{N}.attn.to_q/k/v
    - double_blocks.{N}.txt_attn.qkv -> transformer_blocks.{N}.attn.add_q/k/v_proj
    - double_blocks.{N}.img_attn.proj -> transformer_blocks.{N}.attn.to_out.0
    - double_blocks.{N}.txt_attn.proj -> transformer_blocks.{N}.attn.to_add_out
    - double_blocks -> transformer_blocks
    - single_blocks -> single_transformer_blocks
    """
    out: Dict[str, torch.Tensor] = {}
    original_state_dict: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        new_name = name
        if new_name.startswith("diffusion_model."):
            new_name = new_name[len("diffusion_model.") :]
        if new_name.startswith("base_model.model."):
            new_name = new_name[len("base_model.model.") :]
        original_state_dict[new_name] = tensor

    num_double_layers = 0
    num_single_layers = 0
    for key in original_state_dict.keys():
        if key.startswith("single_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                num_single_layers = max(num_single_layers, int(parts[1]) + 1)
        elif key.startswith("double_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                num_double_layers = max(num_double_layers, int(parts[1]) + 1)

    lora_keys = ("lora_A", "lora_B")
    attn_types = ("img_attn", "txt_attn")

    for sl in range(num_single_layers):
        single_block_prefix = f"single_blocks.{sl}"
        attn_prefix = f"single_transformer_blocks.{sl}.attn"

        for lora_key in lora_keys:
            linear1_key = f"{single_block_prefix}.linear1.{lora_key}.weight"
            if linear1_key in original_state_dict:
                out[f"{attn_prefix}.to_qkv_mlp_proj.{lora_key}.weight"] = (
                    original_state_dict.pop(linear1_key)
                )

            linear2_key = f"{single_block_prefix}.linear2.{lora_key}.weight"
            if linear2_key in original_state_dict:
                out[f"{attn_prefix}.to_out.{lora_key}.weight"] = (
                    original_state_dict.pop(linear2_key)
                )

    for dl in range(num_double_layers):
        transformer_block_prefix = f"transformer_blocks.{dl}"

        for lora_key in lora_keys:
            for attn_type in attn_types:
                attn_prefix = f"{transformer_block_prefix}.attn"
                qkv_key = f"double_blocks.{dl}.{attn_type}.qkv.{lora_key}.weight"

                if qkv_key not in original_state_dict:
                    continue

                fused_qkv_weight = original_state_dict.pop(qkv_key)

                if lora_key == "lora_A":
                    diff_attn_proj_keys = (
                        ["to_q", "to_k", "to_v"]
                        if attn_type == "img_attn"
                        else ["add_q_proj", "add_k_proj", "add_v_proj"]
                    )
                    for proj_key in diff_attn_proj_keys:
                        out[f"{attn_prefix}.{proj_key}.{lora_key}.weight"] = (
                            fused_qkv_weight
                        )
                else:
                    if fused_qkv_weight.shape[0] % 3 != 0:
                        log.warning(
                            "[LoRAFormatAdapter] QKV weight shape %s not divisible by 3, "
                            "may cause shape mismatch for %s",
                            fused_qkv_weight.shape,
                            qkv_key,
                        )
                    sample_q, sample_k, sample_v = torch.chunk(
                        fused_qkv_weight, 3, dim=0
                    )

                    if attn_type == "img_attn":
                        out[f"{attn_prefix}.to_q.{lora_key}.weight"] = sample_q
                        out[f"{attn_prefix}.to_k.{lora_key}.weight"] = sample_k
                        out[f"{attn_prefix}.to_v.{lora_key}.weight"] = sample_v
                    else:
                        out[f"{attn_prefix}.add_q_proj.{lora_key}.weight"] = sample_q
                        out[f"{attn_prefix}.add_k_proj.{lora_key}.weight"] = sample_k
                        out[f"{attn_prefix}.add_v_proj.{lora_key}.weight"] = sample_v

        proj_mappings = [
            ("img_attn.proj", "attn.to_out.0"),
            ("txt_attn.proj", "attn.to_add_out"),
        ]
        for org_proj, diff_proj in proj_mappings:
            for lora_key in lora_keys:
                original_key = f"double_blocks.{dl}.{org_proj}.{lora_key}.weight"
                if original_key in original_state_dict:
                    diffusers_key = (
                        f"{transformer_block_prefix}.{diff_proj}.{lora_key}.weight"
                    )
                    out[diffusers_key] = original_state_dict.pop(original_key)

    for key, tensor in original_state_dict.items():
        new_key = key.replace("double_blocks.", "transformer_blocks.")
        new_key = new_key.replace("single_blocks.", "single_transformer_blocks.")
        out[new_key] = tensor

    extra_mappings = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "final_layer.linear": "proj_out",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    }

    final_out: Dict[str, torch.Tensor] = {}
    for key, tensor in out.items():
        new_key = key
        for org_key, diff_key in extra_mappings.items():
            if key.startswith(org_key):
                new_key = key.replace(org_key, diff_key, 1)
                break
        final_out[new_key] = tensor

    sample = _sample_keys(final_out.keys(), 20)
    log.info(
        "[LoRAFormatAdapter] after AI_TOOLKIT_FLUX conversion, "
        "sample keys (<=20): %s",
        ", ".join(sample),
    )
    return final_out


def convert_lora_state_dict_by_format(
    state_dict: Mapping[str, torch.Tensor],
    fmt: LoRAFormat,
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Normalize a raw LoRA state_dict into A/B + .weight naming."""
    if fmt == LoRAFormat.QWEN_IMAGE_STANDARD:
        return _convert_qwen_image_standard(state_dict, log)

    if fmt == LoRAFormat.AI_TOOLKIT_FLUX:
        return _convert_ai_toolkit_flux_lora(state_dict, log)

    if fmt == LoRAFormat.XLABS_FLUX:
        converted = _convert_xlabs_ai_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.KOHYA_FLUX:
        converted = _convert_kohya_flux_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.WAN:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)
        return _convert_non_diffusers_sd_simple(maybe, log)

    if fmt == LoRAFormat.STANDARD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)

        if _looks_like_qwen_image(maybe):
            return _convert_qwen_image_standard(maybe, log)

        return maybe

    if fmt == LoRAFormat.NON_DIFFUSERS_SD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)
        return _convert_non_diffusers_sd_simple(maybe, log)

    log.info(
        "[LoRAFormatAdapter] format %s not handled specially, returning as-is",
        fmt,
    )
    return dict(state_dict)


def normalize_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize any supported LoRA format into a single canonical layout."""
    log = logger or globals()["logger"]

    keys = list(state_dict.keys())
    log.info(
        "[LoRAFormatAdapter] normalize_lora_state_dict called, #keys=%d",
        len(keys),
    )
    if keys:
        log.info(
            "[LoRAFormatAdapter] before convert, sample keys (<=20): %s",
            ", ".join(_sample_keys(keys, 20)),
        )

    fmt = detect_lora_format_from_state_dict(state_dict)
    log.info("[LoRAFormatAdapter] detected format: %s", fmt)

    normalized = convert_lora_state_dict_by_format(state_dict, fmt, log)

    norm_keys = list(normalized.keys())
    if norm_keys:
        log.info(
            "[LoRAFormatAdapter] after convert, sample keys (<=20): %s",
            ", ".join(_sample_keys(norm_keys, 20)),
        )

    return normalized
