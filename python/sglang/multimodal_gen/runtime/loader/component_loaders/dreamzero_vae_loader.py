# SPDX-License-Identifier: Apache-2.0
"""DreamZero-DROID Wan VAE checkpoint loader."""

from __future__ import annotations

import os

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    iter_prefixed_safetensors,
    load_matching_tensors,
    raise_for_strict_report,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_config import (
    dreamzero_vae_runtime_config_from_checkpoint_config,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DROID_VAE_PREFIX = "action_head.vae.model."


def _remap_residual_block_suffix(suffix: str) -> str | None:
    replacements = (
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        ("shortcut.", "conv_shortcut."),
    )
    for old, new in replacements:
        if suffix.startswith(old):
            return f"{new}{suffix[len(old) :]}"
    return None


def _remap_middle_block(prefix: str, suffix: str) -> str | None:
    index, _, inner = suffix.partition(".")
    if not inner:
        return None
    if index == "0":
        remapped = _remap_residual_block_suffix(inner)
        return f"{prefix}.mid_block.resnets.0.{remapped}" if remapped else None
    if index == "1":
        return f"{prefix}.mid_block.attentions.0.{inner}"
    if index == "2":
        remapped = _remap_residual_block_suffix(inner)
        return f"{prefix}.mid_block.resnets.1.{remapped}" if remapped else None
    return None


def _remap_encoder_downsample(suffix: str) -> str | None:
    index, _, inner = suffix.partition(".")
    if not inner:
        return None

    if inner.startswith("downsamples."):
        inner_index, _, nested = inner[len("downsamples.") :].partition(".")
        if not nested:
            return None
        remapped = _remap_residual_block_suffix(nested)
        if remapped:
            return f"encoder.down_blocks.{index}.resnets.{inner_index}.{remapped}"
        if nested.startswith(("resample.", "time_conv.")):
            return f"encoder.down_blocks.{index}.downsampler.{nested}"
        return None

    remapped = _remap_residual_block_suffix(inner)
    if remapped:
        return f"encoder.down_blocks.{index}.{remapped}"
    if inner.startswith(("resample.", "time_conv.")):
        return f"encoder.down_blocks.{index}.{inner}"
    return None


def _remap_decoder_upsample(suffix: str) -> str | None:
    index, _, inner = suffix.partition(".")
    if not inner:
        return None

    if inner.startswith("upsamples."):
        inner_index, _, nested = inner[len("upsamples.") :].partition(".")
        if not nested:
            return None
        remapped = _remap_residual_block_suffix(nested)
        if remapped:
            return f"decoder.up_blocks.{index}.resnets.{inner_index}.{remapped}"
        if nested.startswith(("resample.", "time_conv.")):
            return f"decoder.up_blocks.{index}.upsampler.{nested}"
        return None

    flat_index = int(index)
    if flat_index < 12:
        block_index = flat_index // 4
        inner_index = flat_index % 4
        if inner_index == 3:
            if inner.startswith(("resample.", "time_conv.")):
                return f"decoder.up_blocks.{block_index}.upsamplers.0.{inner}"
            return None
    else:
        block_index = 3
        inner_index = flat_index - 12

    remapped = _remap_residual_block_suffix(inner)
    if remapped:
        return f"decoder.up_blocks.{block_index}.resnets.{inner_index}.{remapped}"
    return None


def remap_dreamzero_vae_model_key(key: str) -> str | None:
    if key.startswith("encoder.conv1."):
        return f"encoder.conv_in.{key[len('encoder.conv1.') :]}"
    if key.startswith("encoder.downsamples."):
        return _remap_encoder_downsample(key[len("encoder.downsamples.") :])
    if key.startswith("encoder.middle."):
        return _remap_middle_block("encoder", key[len("encoder.middle.") :])
    if key.startswith("encoder.head.0."):
        return f"encoder.norm_out.{key[len('encoder.head.0.') :]}"
    if key.startswith("encoder.head.2."):
        return f"encoder.conv_out.{key[len('encoder.head.2.') :]}"
    if key.startswith("conv1."):
        return f"quant_conv.{key[len('conv1.') :]}"
    if key.startswith("conv2."):
        return f"post_quant_conv.{key[len('conv2.') :]}"
    if key.startswith("decoder.conv1."):
        return f"decoder.conv_in.{key[len('decoder.conv1.') :]}"
    if key.startswith("decoder.middle."):
        return _remap_middle_block("decoder", key[len("decoder.middle.") :])
    if key.startswith("decoder.upsamples."):
        return _remap_decoder_upsample(key[len("decoder.upsamples.") :])
    if key.startswith("decoder.head.0."):
        return f"decoder.norm_out.{key[len('decoder.head.0.') :]}"
    if key.startswith("decoder.head.2."):
        return f"decoder.conv_out.{key[len('decoder.head.2.') :]}"
    return None


def remap_dreamzero_vae_checkpoint_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(_DROID_VAE_PREFIX):
        return None
    return remap_dreamzero_vae_model_key(checkpoint_key[len(_DROID_VAE_PREFIX) :])


def _build_wan_vae_config(runtime_config: dict[str, object]) -> WanVAEConfig:
    arch_kwargs = {
        "base_dim": int(runtime_config["dim"]),
        "decoder_base_dim": int(runtime_config["decoder_dim"]),
        "z_dim": int(runtime_config["z_dim"]),
        "in_channels": int(runtime_config["in_channels"]),
        "out_channels": int(runtime_config["out_channels"]),
        "patch_size": runtime_config["patch_size"],
        "scale_factor_spatial": int(runtime_config["scale_factor_spatial"]),
        "is_residual": bool(runtime_config["is_residual"]),
    }
    if "latents_mean" in runtime_config:
        arch_kwargs["latents_mean"] = tuple(runtime_config["latents_mean"])
        arch_kwargs["latents_std"] = tuple(runtime_config["latents_std"])
    config = WanVAEConfig(arch_config=WanVAEArchConfig(**arch_kwargs))
    config.load_encoder = True
    config.load_decoder = True
    config.use_parallel_encode = False
    config.use_parallel_decode = False
    return config


def build_dreamzero_vae_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroCheckpointLoadReport]:
    runtime_config = dreamzero_vae_runtime_config_from_checkpoint_config(model_path)
    config = _build_wan_vae_config(runtime_config)
    with set_default_torch_dtype(dtype), skip_init_modules():
        if device.type != "cuda":
            model = AutoencoderKLWan(config).to(device=device, dtype=dtype)
        else:
            with torch.cuda.device(device):
                model = AutoencoderKLWan(config).to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)

    if device.type == "cuda":
        with torch.cuda.device(device):
            report = load_matching_tensors(
                model,
                iter_prefixed_safetensors(model_path, _DROID_VAE_PREFIX),
                device=device,
                key_mapper=remap_dreamzero_vae_checkpoint_key,
                report_cls=DreamZeroCheckpointLoadReport,
            )
    else:
        report = load_matching_tensors(
            model,
            iter_prefixed_safetensors(model_path, _DROID_VAE_PREFIX),
            device=device,
            key_mapper=remap_dreamzero_vae_checkpoint_key,
            report_cls=DreamZeroCheckpointLoadReport,
        )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero Wan VAE checkpoint load failed",
    )
    return model, report


class DreamZeroVAELoader(ComponentLoader):
    """Loader entry for DreamZero Wan VAE weights embedded in a full VLA checkpoint."""

    component_names = ["dreamzero_vae"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> nn.Module:
        device = get_local_torch_device()
        model, report = build_dreamzero_vae_from_checkpoint(
            component_model_path,
            device=device,
            dtype=torch.bfloat16,
            strict=True,
        )
        logger.info("Loaded DreamZero VAE checkpoint: %s", report.as_dict())
        return model
