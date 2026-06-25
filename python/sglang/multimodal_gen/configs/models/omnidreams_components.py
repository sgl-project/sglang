# SPDX-License-Identifier: Apache-2.0
"""Component configs for OmniDreams pipeline (text encoder, VAE encoder/decoder).

Provides dataclass configs with impl selection + FP8 acceleration as orthogonal
fields, matching FlashDreams architecture (nested Config with ``setup()`` method).
Each component is independently instantiated, so image_encoder (one-shot first
frame), encoder (per-AR-step HDMap), and decoder are separate module instances.

The module-level ``_load_*`` / ``_resolve_*`` helpers mirror the verified loading
logic from the pipeline (diffusers-safetensors handling, multi-file state dicts,
LightVAE FP8 state, LightTAE remap) so ``setup()`` stays self-contained without
re-implementing — and without breaking — the E2E-validated load paths.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from loguru import logger

from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEArchConfig
from sglang.multimodal_gen.runtime.loader.fsdp_load import set_default_torch_dtype

# --------------------------------------------------------------------------- #
# Native acceleration mode types (folded from native/acceleration.py).       #
# The native/ config-type shell was removed after Phase-1 dropped the        #
# vendored CUDA tree; only the mode Literal + normalizer survive, inlined   #
# here because this is the lower-level module (pipeline_configs imports     #
# from it, so placing the type there would create a circular import).       #
# --------------------------------------------------------------------------- #
NativeAccelerationMode = Literal[
    "disabled", "weight_only_fp8", "fp8_compute", "auto", "required"
]
"""Native DiT acceleration policy (see normalize_native_acceleration_mode)."""

_VALID_NATIVE_MODES: tuple[str, ...] = ("disabled", "weight_only_fp8", "fp8_compute")
_NATIVE_MODE_ALIASES: dict[str, str] = {
    "auto": "disabled",
    "required": "weight_only_fp8",
}


def normalize_native_acceleration_mode(mode: str) -> str:
    """Map ``auto``/``required`` back-compat aliases to real modes and validate.

    The native FP8 DiT path was removed in Phase 1, so ``auto`` no longer has a
    native path to opt into (-> ``disabled``) and ``required`` is satisfied by
    the weight-only FP8 dequant path (-> ``weight_only_fp8``). A warning is
    logged on alias use.
    """
    if mode in _NATIVE_MODE_ALIASES:
        mapped = _NATIVE_MODE_ALIASES[mode]
        logger.warning(
            "native_acceleration mode {!r} is a back-compat alias; mapping to "
            "{!r} (native FP8 DiT removed in Phase 1).",
            mode,
            mapped,
        )
        return mapped
    if mode not in _VALID_NATIVE_MODES:
        raise ValueError(
            f"native_acceleration mode must be one of {_VALID_NATIVE_MODES} "
            f"(or 'auto'/'required' back-compat alias), got {mode!r}"
        )
    return mode

# Canonical latent normalization stats from Wan 2.1 (single source of truth).
_DEFAULT_LATENTS_MEAN = list(WanVAEArchConfig().latents_mean)
_DEFAULT_LATENTS_STD = list(WanVAEArchConfig().latents_std)

# Wan 2.1 VAE weights subdirectory candidates (relative to the model path).
_VAE_RELDIRS = ("vae", "wan_vae", "Wan2.1_VAE")


# --------------------------------------------------------------------------- #
# Shared resolution / loading helpers (verified parity with the pipeline).    #
# --------------------------------------------------------------------------- #
def resolve_wan_vae_path(model_path: str) -> str:
    """Locate the diffusers-format Wan 2.1 VAE weights (dir or file)."""
    base = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    for sub in _VAE_RELDIRS:
        cand = os.path.join(base, sub)
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    for pattern in ("**/*vae*.safetensors", "**/*VAE*.safetensors"):
        matches = sorted(glob.glob(os.path.join(base, pattern), recursive=True))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Diffusers-format Wan 2.1 VAE not found under {base}. Place a "
        f"diffusers Wan VAE (*.safetensors + config.json) under a 'vae/' "
        f"subdirectory or pass checkpoint_path explicitly."
    )


def resolve_light_ckpt_path(
    model_path: str, explicit: str | None, glob_name: str, config_key: str
) -> str:
    """Locate a Light VAE/TAE checkpoint by glob (model dir + parent)."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"{config_key} not found: {explicit}")
    base = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    for root in (base, os.path.dirname(base)):
        matches = sorted(
            glob.glob(os.path.join(root, "**", glob_name), recursive=True)
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"checkpoint ({glob_name}) not found near {base}. Pass "
        f"config.{config_key} explicitly."
    )


def read_vae_state_dict(vae_path: str) -> dict[str, torch.Tensor]:
    """Read a VAE state dict from a diffusers ``*.safetensors`` dir/file."""
    if os.path.isdir(vae_path):
        files = sorted(glob.glob(os.path.join(vae_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No *.safetensors found under '{vae_path}'. Supply a "
                "diffusers-format Wan VAE directory."
            )
        from safetensors.torch import load_file as safetensors_load_file

        state: dict[str, torch.Tensor] = {}
        for f in files:
            state.update(safetensors_load_file(f))
    elif vae_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safetensors_load_file

        state = safetensors_load_file(vae_path)
    else:
        state = torch.load(vae_path, map_location="cpu", weights_only=True)

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def load_wan_vae(
    vae_config: Any,
    vae_path: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Build the SGLang Wan 2.1 VAE and load diffusers-format weights."""
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

    with set_default_torch_dtype(dtype):
        vae = AutoencoderKLWan(vae_config)

    state = read_vae_state_dict(vae_path)
    try:
        vae.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load Wan VAE weights from '{vae_path}' into the "
            "diffusers-format AutoencoderKLWan. Supply the VAE in diffusers "
            "format (a 'vae/' directory with *.safetensors + config.json)."
            f"\nUnderlying error: {exc}"
        ) from exc
    return vae.to(device=device, dtype=dtype).eval()


def _make_vae_config(latents_mean: list[float], latents_std: list[float]) -> Any:
    from sglang.multimodal_gen.configs.models.vaes.wanvae import (
        OmniDreamsVAEArchConfig,
        OmniDreamsVAEConfig,
    )

    # latents_mean/std live on the arch_config (proxied via VAEConfig.__getattr__);
    # they are NOT top-level OmniDreamsVAEConfig constructor args.
    return OmniDreamsVAEConfig(
        arch_config=OmniDreamsVAEArchConfig(
            latents_mean=tuple(latents_mean),
            latents_std=tuple(latents_std),
        )
    )


# --------------------------------------------------------------------------- #
# Component configs.                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class OmniDreamsTextEncoderConfig:
    """Text encoder config (Cosmos-Reason1-7B / Qwen2.5-VL 7B).

    impl="bf16": load HF bf16 model (local ``<model_path>/text_encoder`` dir
        when present, else the pinned HF id + revision).
    impl="fp8_w8a8": load compressed-tensors W8A8 FP8 quantized model. The
        quantization is auto-detected by transformers from ``config.json``.
    """

    impl: Literal["bf16", "fp8_w8a8"] = "bf16"

    # W8A8 quantized model path (required when impl="fp8_w8a8"). Falls back to
    # env SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH.
    fp8_model_path: str | None = None

    # HF model ID + revision for bf16.
    model_id: str = "nvidia/Cosmos-Reason1-7B"
    revision: str = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

    # Resolved at pipeline load time (for local text_encoder dir detection).
    model_path: str | None = None
    device: str = "cuda"

    def setup(self) -> tuple[Any, Any]:
        """Load text encoder + processor. Returns ``(text_encoder, processor)``."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from sglang.multimodal_gen import envs

        if self.impl == "fp8_w8a8":
            src = self.fp8_model_path or envs.SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH
            if src is None or not os.path.isfile(os.path.join(src, "config.json")):
                raise FileNotFoundError(
                    "impl='fp8_w8a8' requires fp8_model_path or "
                    "SGLANG_OMNIDREAMS_TEXT_ENCODER_FP8_PATH pointing to a W8A8 dir"
                )
            revision = None
        else:
            src, revision = self._resolve_bf16_src()

        processor = AutoProcessor.from_pretrained(src, revision=revision)
        text_encoder = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                src,
                revision=revision,
                torch_dtype=torch.bfloat16,
                # quantization_config auto-detected from config.json (W8A8)
            )
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        return text_encoder, processor

    def _resolve_bf16_src(self) -> tuple[str, str | None]:
        if self.model_path and os.path.isdir(self.model_path):
            local = os.path.join(self.model_path, "text_encoder")
            if os.path.isfile(os.path.join(local, "config.json")):
                return local, None
        return self.model_id, self.revision


@dataclass
class _OmniDreamsVAEComponentConfig:
    """Shared fields + WanVAE setup for the encoder/decoder component configs."""

    checkpoint_path: str | None = None

    # FP8 / torch.compile acceleration (semantics depend on the concrete impl).
    native_acceleration: NativeAccelerationMode = "disabled"

    latents_mean: list[float] = field(
        default_factory=lambda: list(_DEFAULT_LATENTS_MEAN)
    )
    latents_std: list[float] = field(
        default_factory=lambda: list(_DEFAULT_LATENTS_STD)
    )

    # Resolved at pipeline load time.
    model_path: str | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def _setup_wanvae(self) -> nn.Module:
        vae_path = self.checkpoint_path or resolve_wan_vae_path(self.model_path or "")
        vae_config = _make_vae_config(self.latents_mean, self.latents_std)
        return load_wan_vae(vae_config, vae_path, self.device, self.dtype)


@dataclass
class OmniDreamsVAEEncoderConfig(_OmniDreamsVAEComponentConfig):
    """VAE encoder config (used for image_encoder and encoder/HDMap roles).

    impl + FP8 acceleration are orthogonal:
    - impl="wanvae": WanVAE 2.1 encoder (full quality, no FP8)
    - impl="lightvae": LightVAE 75%-pruned encoder (lossy speedup, FP8-capable)
    - impl="pixelshuffle": PixelShuffle encoder (planned, not implemented)
    """

    impl: Literal["wanvae", "lightvae", "pixelshuffle"] = "wanvae"

    def setup(self) -> nn.Module:
        """Instantiate the VAE encoder module."""
        if self.impl == "wanvae":
            return self._setup_wanvae()
        if self.impl == "lightvae":
            return self._setup_lightvae()
        if self.impl == "pixelshuffle":
            raise NotImplementedError("PixelShuffle encoder not yet implemented")
        raise ValueError(f"Unknown VAE encoder impl: {self.impl}")

    def _setup_lightvae(self) -> nn.Module:
        from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
            LightVAEEncoder,
        )

        ckpt = self.checkpoint_path or resolve_light_ckpt_path(
            self.model_path or "", None, "*lightvae*.pth", "light_vae_path"
        )

        # Native VAE FP8 was removed with the native CUDA tree; LightVAE always
        # runs the pure-Python bf16 eager encode path. ``native_acceleration``
        # is kept on the config for back-compat but is inert here.
        encoder = LightVAEEncoder(
            checkpoint_path=ckpt,
            latents_mean=list(self.latents_mean),
            latents_std=list(self.latents_std),
            dtype=self.dtype,
        )
        return encoder.to(self.device).eval()


@dataclass
class OmniDreamsVAEDecoderConfig(_OmniDreamsVAEComponentConfig):
    """VAE decoder config.

    impl="wanvae": WanVAE 2.1 decoder (full quality)
    impl="lighttae": LightTAE (TAEHV) decoder (lossy speedup; no FP8 — uses
        bf16 + torch.compile + CUDA Graph)
    """

    impl: Literal["wanvae", "lighttae"] = "wanvae"

    def setup(self) -> nn.Module:
        """Instantiate the VAE decoder module."""
        if self.impl == "wanvae":
            return self._setup_wanvae()
        if self.impl == "lighttae":
            return self._setup_lighttae()
        raise ValueError(f"Unknown VAE decoder impl: {self.impl}")

    def _setup_lighttae(self) -> nn.Module:
        from sglang.multimodal_gen.runtime.models.vaes.taehv import LightTAEDecoder

        ckpt = self.checkpoint_path or resolve_light_ckpt_path(
            self.model_path or "", None, "*lighttae*.pth", "light_tae_path"
        )
        decoder = LightTAEDecoder(checkpoint_path=ckpt, dtype=self.dtype)
        return decoder.to(self.device).eval()
