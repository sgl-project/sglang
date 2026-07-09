# SPDX-License-Identifier: Apache-2.0
"""DreamZero-DROID Wan VAE checkpoint loader.

DreamZero uses its original Wan VAE module graph by default. The native SGLang
key-remapping helpers remain available for parity tests, but the runtime path
keeps the original graph to avoid numerical drift from executing a rewritten
module layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
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
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DROID_VAE_PREFIX = "action_head.vae.model."
_WAN22_VAE_NAME = "Wan2.2_VAE.pth"
_WAN21_VAE_NAME = "Wan2.1_VAE.pth"


class DreamZeroVAELoadReport(DreamZeroCheckpointLoadReport):
    include_fallback_impl = True


def _map_residual_block_key(key: str) -> str:
    key = key.replace(".residual.0.gamma", ".norm1.gamma")
    key = key.replace(".residual.2.", ".conv1.")
    key = key.replace(".residual.3.gamma", ".norm2.gamma")
    key = key.replace(".residual.6.", ".conv2.")
    key = key.replace(".shortcut.", ".conv_shortcut.")
    return key


def _map_middle_key(key: str, groot_prefix: str, sglang_prefix: str) -> str:
    if key.startswith(f"{groot_prefix}.0."):
        return _map_residual_block_key(
            key.replace(f"{groot_prefix}.0.", f"{sglang_prefix}.resnets.0.")
        )
    if key.startswith(f"{groot_prefix}.1."):
        return key.replace(f"{groot_prefix}.1.", f"{sglang_prefix}.attentions.0.")
    if key.startswith(f"{groot_prefix}.2."):
        return _map_residual_block_key(
            key.replace(f"{groot_prefix}.2.", f"{sglang_prefix}.resnets.1.")
        )
    return key


def _map_decoder_upsample_key(key: str) -> str:
    prefix = "decoder.upsamples."
    if not key.startswith(prefix):
        return key

    suffix = key[len(prefix) :]
    index_text, rest = suffix.split(".", 1)
    index = int(index_text)
    block_index = min(index // 4, 3)
    local_index = index - block_index * 4
    if local_index < 3:
        mapped = f"decoder.up_blocks.{block_index}.resnets.{local_index}.{rest}"
        return _map_residual_block_key(mapped)
    return f"decoder.up_blocks.{block_index}.upsamplers.0.{rest}"


def remap_dreamzero_vae_model_key(model_key: str) -> str:
    """Map original ``WanVideoVAE.model`` keys to SGLang ``AutoencoderKLWan`` keys."""

    key = model_key
    if key.startswith("model."):
        key = key[len("model.") :]

    key = key.replace("encoder.conv1.", "encoder.conv_in.")
    key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
    key = _map_middle_key(key, "encoder.middle", "encoder.mid_block")
    key = key.replace("encoder.head.0.", "encoder.norm_out.")
    key = key.replace("encoder.head.2.", "encoder.conv_out.")

    if key.startswith("conv1."):
        key = key.replace("conv1.", "quant_conv.", 1)
    if key.startswith("conv2."):
        key = key.replace("conv2.", "post_quant_conv.", 1)

    key = key.replace("decoder.conv1.", "decoder.conv_in.")
    key = _map_middle_key(key, "decoder.middle", "decoder.mid_block")
    key = _map_decoder_upsample_key(key)
    key = key.replace("decoder.head.0.", "decoder.norm_out.")
    key = key.replace("decoder.head.2.", "decoder.conv_out.")

    return _map_residual_block_key(key)


def remap_dreamzero_vae_checkpoint_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(_DROID_VAE_PREFIX):
        return None
    return remap_dreamzero_vae_model_key(checkpoint_key[len(_DROID_VAE_PREFIX) :])


def _resolve_wan_vae_pth(model_path: str | os.PathLike[str]) -> Path | None:
    path = Path(model_path)
    if path.is_file() and path.suffix == ".pth":
        return path
    for file_name in (_WAN22_VAE_NAME, _WAN21_VAE_NAME):
        candidate = path / file_name
        if candidate.is_file():
            return candidate
    return None


def build_dreamzero_vae(
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLWan:
    config = WanVAEConfig(
        use_feature_cache=True,
        use_tiling=False,
        use_temporal_tiling=False,
        use_parallel_tiling=False,
        use_parallel_encode=False,
        use_parallel_decode=False,
    )
    with torch.device("meta"):
        model = AutoencoderKLWan(config).to(dtype=dtype)
    model.eval()
    return model


def load_dreamzero_vae_checkpoint(
    model: AutoencoderKLWan,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroVAELoadReport:
    report = load_matching_tensors(
        model,
        iter_prefixed_safetensors(model_path, _DROID_VAE_PREFIX),
        device=device,
        key_mapper=remap_dreamzero_vae_checkpoint_key,
        report_cls=DreamZeroVAELoadReport,
    )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero VAE checkpoint load failed",
    )
    return report


def build_dreamzero_vae_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroVAELoadReport]:
    from sglang.multimodal_gen.runtime.models.vaes.dreamzero_vae import (
        WanVideoVAE,
        WanVideoVAE38,
    )

    runtime_config = dreamzero_vae_runtime_config_from_checkpoint_config(model_path)
    runtime_target = runtime_config["runtime_target"]
    z_dim = int(runtime_config["z_dim"])
    inner_dim = int(runtime_config["dim"])
    vae_cls = WanVideoVAE38 if runtime_target.endswith("WanVideoVAE38") else WanVideoVAE
    kwargs: dict[str, Any] = {"z_dim": z_dim}
    if vae_cls is WanVideoVAE38:
        kwargs["dim"] = inner_dim

    # Match the original WANPolicyHead construction and post_initialize exactly:
    # instantiate real parameters, then move the complete wrapper to BF16/CUDA
    # before copying checkpoint tensors into the existing parameter objects.
    if device.type == "cuda":
        with torch.cuda.device(device):
            model = vae_cls(**kwargs)
    else:
        model = vae_cls(**kwargs)
    model.to(device=device, dtype=dtype)
    model.eval().requires_grad_(False)

    vae_pth = _resolve_wan_vae_pth(model_path)
    if vae_pth is not None:
        state_dict = torch.load(vae_pth, map_location="cpu")
        incompatible = model.model.load_state_dict(state_dict, strict=strict)
        model.to(device=device, dtype=dtype)
        report = DreamZeroVAELoadReport(
            loaded_keys=list(state_dict.keys()),
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
            shape_mismatches={},
            fallback_impl=f"sglang.dreamzero_vae.{vae_cls.__name__}",
        )
        if strict and (report.missing_keys or report.unexpected_keys):
            raise RuntimeError(
                f"DreamZero original VAE checkpoint load failed: {report.as_dict()}"
            )
        return model, report

    state_dict = model.model.state_dict()
    parameters = dict(model.model.named_parameters())
    buffers = dict(model.model.named_buffers())
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    with torch.no_grad():
        for checkpoint_key, full_tensor in iter_prefixed_safetensors(
            model_path, _DROID_VAE_PREFIX
        ):
            target_name = checkpoint_key[len(_DROID_VAE_PREFIX) :]
            target = parameters.get(target_name)
            if target is None:
                target = buffers.get(target_name)
            expected = state_dict.get(target_name)
            if target is None or expected is None:
                unexpected_keys.append(target_name)
                continue
            if tuple(full_tensor.shape) != tuple(expected.shape):
                shape_mismatches[target_name] = (
                    tuple(expected.shape),
                    tuple(full_tensor.shape),
                )
                continue
            target.copy_(full_tensor.to(device=target.device, dtype=target.dtype))
            loaded_keys.append(target_name)

    report = DreamZeroVAELoadReport(
        loaded_keys=loaded_keys,
        missing_keys=sorted(set(state_dict) - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        fallback_impl=f"sglang.dreamzero_vae.{vae_cls.__name__}",
    )
    if strict and (
        report.missing_keys or report.unexpected_keys or report.shape_mismatches
    ):
        raise RuntimeError(
            f"DreamZero original VAE checkpoint load failed: {report.as_dict()}"
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
