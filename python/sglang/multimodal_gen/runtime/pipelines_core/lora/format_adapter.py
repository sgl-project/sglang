from __future__ import annotations

import logging
import math
import re
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from diffusers.loaders import lora_conversion_utils as lcu

from sglang.multimodal_gen.runtime.pipelines_core.lora.peft_adapter import (
    apply_peft_config,
    compile_peft_alpha_patterns,
    get_peft_lora_alpha,
    match_peft_alpha_pattern,
    normalize_peft_keys,
)

logger = logging.getLogger("LoRAFormatAdapter")


class LoRAFormat(str, Enum):
    """Supported external LoRA formats before normalization."""

    STANDARD = "standard"
    NON_DIFFUSERS_SD = "non-diffusers-sd"
    QWEN_IMAGE_STANDARD = "qwen-image-standard"
    XLABS_FLUX = "xlabs-ai"
    KOHYA_FLUX = "kohya-flux"
    KOHYA_MINIMAX_H3 = "kohya-minimax-h3"
    KOHYA_QWEN_IMAGE = "kohya-qwen-image"
    KOHYA_Z_IMAGE = "kohya-z-image"
    MUSUBI_WAN = "musubi-wan"
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


_KOHYA_H3_KEY = re.compile(
    r"^lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)"
    r"\.(lora_down|lora_up|alpha)(?:\.weight)?$"
)
_KOHYA_H3_MODULES = {
    "attn_qkv_proj": "attn.qkv_proj",
    "attn_out_proj": "attn.out_proj",
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
}
_KOHYA_H3_SUFFIXES = {
    "lora_down": "lora_A",
    "lora_up": "lora_B",
    "alpha": "alpha",
}


def _looks_like_kohya_minimax_h3(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya `networks.lora_minimax_h3` export: lora_unet_blocks_* fused names."""
    if not state_dict:
        return False
    return any(
        k.startswith("lora_unet_blocks_") and "_attn_qkv_proj." in k for k in state_dict
    )


def _convert_kohya_minimax_h3(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """lora_unet_blocks_N_* + lora_down/up -> native H3 blocks.N.*.lora_A/B."""
    out: Dict[str, torch.Tensor] = {}
    unmatched: list[str] = []

    for name, tensor in state_dict.items():
        match = _KOHYA_H3_KEY.match(name)
        if match is None:
            unmatched.append(name)
            out[name] = tensor
            continue
        layer, module, suffix = match.group(1), match.group(2), match.group(3)
        target = (
            f"blocks.{layer}.{_KOHYA_H3_MODULES[module]}.{_KOHYA_H3_SUFFIXES[suffix]}"
        )
        out[target] = tensor

    sample = _sample_keys(out.keys(), 20)
    log.info(
        "[LoRAFormatAdapter] after KOHYA_MINIMAX_H3 conversion, "
        "unmatched=%d sample keys (<=20): %s",
        len(unmatched),
        ", ".join(sample),
    )
    if unmatched:
        log.warning(
            "[LoRAFormatAdapter] KOHYA_MINIMAX_H3 left %d unmatched keys: %s",
            len(unmatched),
            ", ".join(_sample_keys(unmatched, 8)),
        )
    return out


_MUSUBI_WAN_KEY = re.compile(
    r"^lora_unet_blocks_(\d+)_"
    r"(self_attn_(?:q|k|v|o)|cross_attn_(?:q|k|v|o|k_img|v_img)|ffn_(?:0|2))"
    r"\.(lora_down\.weight|lora_up\.weight|alpha)$"
)


def _looks_like_musubi_wan(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """musubi-tuner Wan LoRA (its default save format): lora_unet_blocks_N_*."""
    keys = list(state_dict.keys())
    return bool(keys) and all(_MUSUBI_WAN_KEY.match(k) for k in keys)


def _convert_musubi_wan(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """lora_unet_blocks_N_self_attn_q.* -> diffusion_model.blocks.N.self_attn.q.*,
    the official Wan layout the WAN path already converts (with alpha)."""
    renamed: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        block, module, suffix = _MUSUBI_WAN_KEY.match(key).groups()
        module = re.sub(r"^(self_attn|cross_attn|ffn)_", r"\1.", module)
        renamed[f"diffusion_model.blocks.{block}.{module}.{suffix}"] = tensor
    log.info("[LoRAFormatAdapter] renamed %d musubi Wan keys", len(renamed))
    return convert_lora_state_dict_by_format(renamed, LoRAFormat.WAN, log)


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


def _looks_like_kohya_qwen_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya/musubi Qwen-Image LoRA: lora_unet_transformer_blocks_N_{img,txt}_{mlp,mod}_*.

    The img/txt mlp and mod names separate it from Kohya exports of other
    MMDiTs that also use diffusers `transformer_blocks` naming.
    """
    return any(
        k.startswith("lora_unet_transformer_blocks_")
        and any(m in k for m in ("_img_mlp_", "_txt_mlp_", "_img_mod_", "_txt_mod_"))
        for k in state_dict
    )


def _looks_like_kohya_z_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya/musubi Z-Image LoRA: lora_unet_{layers,noise_refiner,context_refiner}_N_*."""
    return any(
        k.startswith(
            (
                "lora_unet_layers_",
                "lora_unet_noise_refiner_",
                "lora_unet_context_refiner_",
            )
        )
        and ("_attention_" in k or "_feed_forward_" in k)
        for k in state_dict
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
        k.startswith(("single_blocks.", "base_model.model.single_blocks."))
        and (".linear1." in k or ".linear2." in k)
        for k in keys
    )
    has_lora_ab = _has_substring_key(keys, ".lora_A") or _has_substring_key(
        keys, ".lora_B"
    )

    return (has_double_blocks or has_single_blocks) and has_lora_ab


def _is_qwen_image_dit(dit_class_name: str | None) -> bool:
    # endswith also matches wrapper classes such as FSDPQwenImageTransformer2DModel;
    # Qwen-Image 2.1 (QwenImage21Transformer2DModel) uses a different layout.
    return bool(dit_class_name) and dit_class_name.endswith(
        "QwenImageTransformer2DModel"
    )


def _is_qwen_image21_dit(dit_class_name: str | None) -> bool:
    return bool(dit_class_name) and dit_class_name.endswith(
        "QwenImage21Transformer2DModel"
    )


def _is_z_image_dit(dit_class_name: str | None) -> bool:
    return bool(dit_class_name) and dit_class_name.endswith("ZImageTransformer2DModel")


def detect_lora_format_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    dit_class_name: str | None = None,
) -> LoRAFormat:
    """Classify LoRA format by key patterns, using the loaded DiT class where
    the keys alone are ambiguous."""
    keys = list(state_dict.keys())
    if not keys:
        return LoRAFormat.STANDARD

    if _looks_like_ai_toolkit_flux_lora(state_dict):
        return LoRAFormat.AI_TOOLKIT_FLUX

    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        return LoRAFormat.STANDARD

    if any(_looks_like_xlabs_flux_key(k) for k in keys):
        return LoRAFormat.XLABS_FLUX
    if _looks_like_kohya_minimax_h3(state_dict):
        return LoRAFormat.KOHYA_MINIMAX_H3
    if _looks_like_kohya_flux(state_dict):
        return LoRAFormat.KOHYA_FLUX
    if _looks_like_kohya_qwen_image(state_dict) and (
        dit_class_name is None or _is_qwen_image_dit(dit_class_name)
    ):
        return LoRAFormat.KOHYA_QWEN_IMAGE
    # An attention-only Qwen-Image LoRA has no img/txt mlp or mod names, and its
    # `transformer_blocks_N_attn_*` keys are shared with other MMDiTs.
    if _is_qwen_image_dit(dit_class_name) and (
        _has_prefix_key(keys, "lora_unet_transformer_blocks_")
        or any(k.split(".", 1)[0] in _QWEN_IMAGE_KOHYA_TOP_LEVEL_MODULES for k in keys)
    ):
        return LoRAFormat.KOHYA_QWEN_IMAGE
    if _looks_like_kohya_z_image(state_dict):
        return LoRAFormat.KOHYA_Z_IMAGE
    # A Z-Image LoRA trained only on e.g. adaLN_modulation has no attention or
    # feed-forward names to recognize it by.
    if _is_z_image_dit(dit_class_name) and _has_prefix_key(keys, "lora_unet_"):
        return LoRAFormat.KOHYA_Z_IMAGE
    if _looks_like_musubi_wan(state_dict):
        return LoRAFormat.MUSUBI_WAN

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
        no_converter_warning="[LoRAFormatAdapter] No Kohya FLUX converter found.",
        success_info="[LoRAFormatAdapter] Converted Kohya FLUX LoRA using {name}",
        all_failed_warning=(
            "[LoRAFormatAdapter] Kohya FLUX conversion failed; last error: {last_err}"
        ),
    )


# diffusers' Z-Image converter splits every "_" of a Kohya name; releases before
# these fixups (e.g. the pinned 0.37) leave the names below over-split.
_Z_IMAGE_MODULE_NAME_FIXUPS = {
    "context.refiner.": "context_refiner.",
    "noise.refiner.": "noise_refiner.",
    "adaLN.modulation.": "adaLN_modulation.",
    "all.final.layer.": "all_final_layer.",
    "all.x.embedder.": "all_x_embedder.",
    "cap.embedder.": "cap_embedder.",
    "t.embedder.": "t_embedder.",
}


_KOHYA_DOWN_SUFFIXES = (".lora_down.weight", ".lora.down.weight")


def _strip_default_slot(key: str) -> str:
    if key.endswith(".default.weight"):
        return key.removesuffix(".default.weight") + ".weight"
    return key


def _kohya_alpha_key(module: str) -> str:
    # Dot-format "to.q" modules look their alpha up as "to_q".
    return re.sub(r"to\.([qkv])$", r"to_\1", module) + ".alpha"


def _run_kohya_dit_converter(
    state_dict: Mapping[str, torch.Tensor],
    converter_name: str,
    model_name: str,
    log: logging.Logger,
    module_name_fixups: Mapping[str, str] | None,
) -> Dict[str, torch.Tensor]:
    converted = _convert_via_diffusers_candidates(
        state_dict,
        (converter_name,),
        log=log,
        no_converter_warning=(
            f"[LoRAFormatAdapter] No Kohya {model_name} converter found in diffusers."
        ),
        success_info=(
            f"[LoRAFormatAdapter] Converted Kohya {model_name} LoRA using {{name}}"
        ),
        all_failed_warning=(
            f"[LoRAFormatAdapter] Kohya {model_name} conversion failed; "
            "last error: {last_err}"
        ),
    )
    # A failed conversion returns the input; releases without the fused-qkv
    # split (e.g. 0.37) drop a qkv-only file entirely.
    if not any(k.endswith(".lora_A.weight") for k in converted) or any(
        k.startswith("lora_unet_") for k in converted
    ):
        raise ValueError(
            f"Could not convert this Kohya {model_name} LoRA with the installed "
            "diffusers; it would otherwise load as a no-op"
        )
    out = {}
    for key, tensor in converted.items():
        key = key.removeprefix("transformer.")
        for dotted, underscored in (module_name_fixups or {}).items():
            key = key.replace(dotted, underscored)
        out[key] = tensor
    return out


def _kohya_converted_names(
    downs: Mapping[str, str], run_converter
) -> Dict[str, list[str]]:
    """Map each Kohya module to the converted module names it produces.

    The converter only renames (or splits fused qkv) modules, so a unit-scale
    probe whose A weight holds the module index recovers the Kohya ->
    converted name mapping. A module the converter drops has no entry.
    """
    probe: Dict[str, torch.Tensor] = {}
    modules = list(downs)
    for index, module in enumerate(modules, start=1):
        down_key = downs[module]
        up_key = down_key.replace("down.weight", "up.weight")
        probe[down_key] = torch.full((1, 1), float(index))
        # Three rows so a fused qkv up weight splits into q, k and v.
        probe[up_key] = torch.ones(3, 1)
        probe[_kohya_alpha_key(module)] = torch.tensor(1.0)
    names: Dict[str, list[str]] = {}
    for key, tensor in run_converter(probe).items():
        if key.endswith(".lora_A.weight"):
            module = modules[int(tensor.flatten()[0].item()) - 1]
            names.setdefault(module, []).append(key.removesuffix(".lora_A.weight"))
    return names


def _resolve_kohya_alpha(
    module: str,
    own_alpha: torch.Tensor | None,
    rank: int,
    fill_alpha: int | None,
    pattern_alphas: Mapping[str, int | None],
) -> float:
    """The alpha one Kohya module applies with, checked against its patterns.

    One Kohya alpha covers every module converted from it (q/k/v of a fused
    qkv), so their pattern values and its own alpha must agree.
    """
    if own_alpha is not None:
        default = float(own_alpha.item())
        # Diffusers' converters loop on the alpha / rank scale and never return
        # for a negative one.
        if not math.isfinite(default) or default < 0:
            raise ValueError(
                f"Kohya LoRA alpha of {module!r} must be a non-negative number, "
                f"got {default:g}"
            )
    elif fill_alpha is not None:
        default = float(fill_alpha)
    else:
        default = float(rank)
    resolved = {
        name: default if alpha is None else float(alpha)
        for name, alpha in pattern_alphas.items()
    }
    values = set(resolved.values())
    if own_alpha is not None:
        values.add(default)
    if len(values) > 1:
        own = f"its alpha {default:g}" if own_alpha is not None else "each other"
        raise ValueError(
            f"PEFT alpha_pattern alphas {resolved} for Kohya module "
            f"{module!r} conflict with {own}"
        )
    return values.pop() if values else default


def _split_qkv_converted(module: str, converted_names: Mapping[str, Any]) -> bool:
    """A fused `..._qkv` module that diffusers skips because the same file also
    has its split q/k/v, all of which converted."""
    if not module.endswith("qkv"):
        return False
    base = module[: -len("qkv")]
    return all(
        f"{base}to_{x}" in converted_names or f"{base}to.{x}" in converted_names
        for x in "qkv"
    )


# Qwen-Image modules outside the transformer blocks, which diffusers' Kohya
# converter does not rename.
_QWEN_IMAGE_KOHYA_TOP_LEVEL_MODULES = {
    "lora_unet_img_in": "img_in",
    "lora_unet_txt_in": "txt_in",
    "lora_unet_proj_out": "proj_out",
    "lora_unet_norm_out_linear": "norm_out.linear",
    "lora_unet_time_text_embed_timestep_embedder_linear_1": (
        "time_text_embed.timestep_embedder.linear_1"
    ),
    "lora_unet_time_text_embed_timestep_embedder_linear_2": (
        "time_text_embed.timestep_embedder.linear_2"
    ),
}


def _convert_kohya_dit_via_diffusers(
    state_dict: Mapping[str, torch.Tensor],
    converter_name: str,
    model_name: str,
    log: logging.Logger,
    fill_alpha: int | None = None,
    module_name_fixups: Mapping[str, str] | None = None,
    alpha_patterns: list[tuple[re.Pattern[str], int]] | None = None,
    top_level_modules: Mapping[str, str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Convert a Kohya/musubi DiT LoRA with the matching diffusers helper.

    The helper folds alpha / rank into A/B and returns `transformer.`-prefixed
    diffusers names; the prefix is dropped to match the native DiT names.
    A module without its own `.alpha` gets its PEFT `alpha_pattern` value, else
    `fill_alpha`, else its rank, since older diffusers releases require one.
    A pattern that disagrees with a module's own `.alpha` is rejected. Each
    converted module then records `alpha = rank`, so no alpha is applied twice.
    `top_level_modules` names modules the helper does not handle; they are
    renamed here and keep their alpha. A module the helper drops (e.g. a fused
    qkv on a release that cannot split it) fails the load instead of silently
    leaving part of the LoRA out.
    """

    def run_converter(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return _run_kohya_dit_converter(
            sd, converter_name, model_name, log, module_name_fixups
        )

    # PEFT-saved files can name the adapter slot (`.lora_down.default.weight`).
    # Diffusers drops it before converting; drop it here so alphas resolve per
    # module as well.
    state_dict = {
        _strip_default_slot(key): tensor for key, tensor in state_dict.items()
    }
    downs = {
        key[: -len(suffix)]: key
        for key in state_dict
        for suffix in _KOHYA_DOWN_SUFFIXES
        if key.endswith(suffix)
    }
    top_level = {
        module: name
        for module, name in (top_level_modules or {}).items()
        if module in downs
    }
    block_downs = {m: k for m, k in downs.items() if m not in top_level}
    top_level_keys = {
        key
        for module in top_level
        for key in (
            downs[module],
            downs[module].replace("down.weight", "up.weight"),
            _kohya_alpha_key(module),
        )
    }
    # Anything besides the top-level modules (block LoRA pairs, or e.g. LoKr
    # tensors) goes through the converter, so an adapter it cannot read fails
    # instead of loading only in part.
    run_blocks = any(key not in top_level_keys for key in state_dict)
    converted_names = (
        _kohya_converted_names(block_downs, run_converter) if block_downs else {}
    )
    dropped = sorted(
        module
        for module in set(block_downs) - set(converted_names)
        if not _split_qkv_converted(module, converted_names)
    )
    if dropped:
        raise ValueError(
            f"The installed diffusers drops {len(dropped)} module(s) of this Kohya "
            f"{model_name} LoRA (e.g. {dropped[:3]}); loading the rest would "
            "silently apply only part of it"
        )
    alphas = {}
    for module, down_key in downs.items():
        alpha_key = _kohya_alpha_key(module)
        own_alpha = state_dict.get(alpha_key)
        names = (
            [top_level[module]]
            if module in top_level
            else converted_names.get(module, [])
        )
        pattern_alphas = {
            name: match_peft_alpha_pattern(name, alpha_patterns or []) for name in names
        }
        if own_alpha is not None and own_alpha.numel() != 1:
            if module in top_level:
                raise ValueError(
                    f"Kohya LoRA alpha of {module!r} must be a single number, "
                    f"got shape {tuple(own_alpha.shape)}"
                )
            if any(alpha is not None for alpha in pattern_alphas.values()):
                raise ValueError(f"PEFT alpha_pattern conflicts with {alpha_key!r}")
            continue
        alphas[module] = _resolve_kohya_alpha(
            module,
            own_alpha,
            state_dict[down_key].shape[0],
            fill_alpha,
            pattern_alphas,
        )
        if own_alpha is None:
            state_dict[alpha_key] = torch.tensor(alphas[module])

    out = (
        run_converter({k: v for k, v in state_dict.items() if k not in top_level_keys})
        if run_blocks
        else {}
    )
    for key in [k for k in out if k.endswith(".lora_A.weight")]:
        out[key.removesuffix(".lora_A.weight") + ".alpha"] = torch.tensor(
            float(out[key].shape[0])
        )
    for module, name in top_level.items():
        down_key = downs[module]
        out[f"{name}.lora_A.weight"] = state_dict[down_key]
        out[f"{name}.lora_B.weight"] = state_dict[
            down_key.replace("down.weight", "up.weight")
        ]
        out[f"{name}.alpha"] = torch.tensor(alphas[module])
    return out


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

    def move_alpha(old_prefix: str, new_prefixes: list[str]) -> None:
        # Per-module alpha follows the module it scales; split sections share it.
        alpha = original_state_dict.pop(f"{old_prefix}.alpha", None)
        if alpha is not None:
            for new_prefix in new_prefixes:
                out[f"{new_prefix}.alpha"] = alpha

    for sl in range(num_single_layers):
        single_block_prefix = f"single_blocks.{sl}"
        attn_prefix = f"single_transformer_blocks.{sl}.attn"
        move_alpha(f"{single_block_prefix}.linear1", [f"{attn_prefix}.to_qkv_mlp_proj"])
        move_alpha(f"{single_block_prefix}.linear2", [f"{attn_prefix}.to_out"])

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

        for attn_type in attn_types:
            proj_names = (
                ["to_q", "to_k", "to_v"]
                if attn_type == "img_attn"
                else ["add_q_proj", "add_k_proj", "add_v_proj"]
            )
            move_alpha(
                f"double_blocks.{dl}.{attn_type}.qkv",
                [f"{transformer_block_prefix}.attn.{n}" for n in proj_names],
            )

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
            move_alpha(
                f"double_blocks.{dl}.{org_proj}",
                [f"{transformer_block_prefix}.{diff_proj}"],
            )
            for lora_key in lora_keys:
                original_key = f"double_blocks.{dl}.{org_proj}.{lora_key}.weight"
                if original_key in original_state_dict:
                    diffusers_key = (
                        f"{transformer_block_prefix}.{diff_proj}.{lora_key}.weight"
                    )
                    out[diffusers_key] = original_state_dict.pop(original_key)

    # Leftover double/single block keys (MLP, modulation) keep their original
    # names so the arch config's param_names_mapping resolves them.
    out.update(original_state_dict)

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
        "[LoRAFormatAdapter] after AI_TOOLKIT_FLUX conversion, sample keys (<=20): %s",
        ", ".join(sample),
    )
    return final_out


def convert_lora_state_dict_by_format(
    state_dict: Mapping[str, torch.Tensor],
    fmt: LoRAFormat,
    log: logging.Logger,
    fill_alpha: int | None = None,
    alpha_patterns: list[tuple[re.Pattern[str], int]] | None = None,
) -> Dict[str, torch.Tensor]:
    """Normalize a raw LoRA state_dict into A/B + .weight naming."""
    if fmt == LoRAFormat.QWEN_IMAGE_STANDARD:
        return _convert_qwen_image_standard(state_dict, log)

    if fmt == LoRAFormat.AI_TOOLKIT_FLUX:
        return _convert_ai_toolkit_flux_lora(state_dict, log)

    if fmt == LoRAFormat.XLABS_FLUX:
        converted = _convert_xlabs_ai_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.KOHYA_MINIMAX_H3:
        return _convert_kohya_minimax_h3(state_dict, log)

    if fmt == LoRAFormat.KOHYA_FLUX:
        converted = _convert_kohya_flux_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.KOHYA_QWEN_IMAGE:
        return _convert_kohya_dit_via_diffusers(
            state_dict,
            "_convert_non_diffusers_qwen_lora_to_diffusers",
            "Qwen-Image",
            log,
            fill_alpha,
            alpha_patterns=alpha_patterns,
            top_level_modules=_QWEN_IMAGE_KOHYA_TOP_LEVEL_MODULES,
        )

    if fmt == LoRAFormat.KOHYA_Z_IMAGE:
        return _convert_kohya_dit_via_diffusers(
            state_dict,
            "_convert_non_diffusers_z_image_lora_to_diffusers",
            "Z-Image",
            log,
            fill_alpha,
            _Z_IMAGE_MODULE_NAME_FIXUPS,
            alpha_patterns,
        )

    if fmt == LoRAFormat.MUSUBI_WAN:
        return _convert_musubi_wan(state_dict, log)

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
    *,
    adapter_config: Mapping[str, Any] | None = None,
    dit_class_name: str | None = None,
    lora_alpha: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Normalize any supported LoRA format into a single canonical layout.

    `lora_alpha` is the caller's override for modules that carry no alpha.
    """
    log = logger or globals()["logger"]

    state_dict = normalize_peft_keys(state_dict)
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

    if _is_qwen_image21_dit(dit_class_name) and _has_prefix_key(keys, "lora_unet_"):
        # Kohya names are not mapped to Qwen-Image 2.1 modules yet, and the
        # generic fallback would load the adapter as a no-op.
        raise ValueError(
            "Kohya `lora_unet_*` LoRAs are not supported for Qwen-Image 2.1 yet; "
            "use a diffusers/PEFT-format LoRA"
        )
    fmt = detect_lora_format_from_state_dict(state_dict, dit_class_name)
    log.info("[LoRAFormatAdapter] detected format: %s", fmt)

    fill_alpha = (
        lora_alpha
        if lora_alpha is not None
        else get_peft_lora_alpha(adapter_config or {})
    )
    peft_config = dict(adapter_config or {})
    alpha_patterns = None
    if fmt in (LoRAFormat.KOHYA_QWEN_IMAGE, LoRAFormat.KOHYA_Z_IMAGE):
        # Conversion folds alpha into A/B, so resolve the patterns against the
        # original alphas there instead of the converted `alpha = rank`.
        alpha_patterns = compile_peft_alpha_patterns(peft_config)
        peft_config.pop("alpha_pattern", None)
    normalized = convert_lora_state_dict_by_format(
        state_dict, fmt, log, fill_alpha, alpha_patterns
    )
    normalized = apply_peft_config(normalized, peft_config)

    norm_keys = list(normalized.keys())
    if norm_keys:
        log.info(
            "[LoRAFormatAdapter] after convert, sample keys (<=20): %s",
            ", ".join(_sample_keys(norm_keys, 20)),
        )

    return normalized
