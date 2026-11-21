# SPDX-License-Identifier: Apache-2.0
"""Optional Tiny AutoEncoder decode while retaining the original VAE encoder."""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.tiny_autoencoder import (
    TinyImageAutoencoder,
    TinyVideoAutoencoder,
)

TINY_VAE_MODELS = {
    "taesd": (TinyImageAutoencoder, 4),
    "taesdxl": (TinyImageAutoencoder, 4),
    "taesd3": (TinyImageAutoencoder, 16),
    "taef1": (TinyImageAutoencoder, 16),
    "taef2": (TinyImageAutoencoder, 128),
    "taeqi2_1": (TinyImageAutoencoder, 64),
    "taehv": (TinyVideoAutoencoder, 16),
    "taew2_1": (TinyVideoAutoencoder, 16),
    "taew2_2": (TinyVideoAutoencoder, 48),
    "taehv1_5": (TinyVideoAutoencoder, 32),
    "taeh3": (TinyVideoAutoencoder, 24),
    "taeltx2": (TinyVideoAutoencoder, 128),
}

# Names used by ComfyUI's preview checkpoint catalog.
TINY_VAE_ALIASES = {
    "lighttaew2_1": "taew2_1",
    "lighttaew2_2": "taew2_2",
    "lighttaehy1_5": "taehv1_5",
    "taeltx_2": "taeltx2",
}


def validate_tiny_vae_config(config):
    if not config.tiny_vae_path and not config.tiny_vae_model:
        return
    if bool(config.tiny_vae_path) != bool(config.tiny_vae_model):
        raise ValueError(
            "--tiny-vae-path and --tiny-vae-model must be supplied together"
        )
    model_name = TINY_VAE_ALIASES.get(config.tiny_vae_model, config.tiny_vae_model)
    if model_name not in TINY_VAE_MODELS:
        raise ValueError(
            f"Unknown TinyVAE model {config.tiny_vae_model!r}; "
            f"choose from {', '.join(TINY_VAE_MODELS)}"
        )
    if model_name == "taeltx2":
        variant = getattr(config.vae_config.arch_config, "video_decoder_variant", None)
        if variant != "ltx_2":
            raise ValueError(
                "taeltx2 weights target LTX-2; LTX-2.3/2.5 require separately trained weights"
            )


def load_tiny_autoencoder(path, model_name, dtype):
    """Restore full checkpoints or decoder-only weights with strict decoder keys."""
    from sglang.multimodal_gen.runtime.loader.utils import initialize_model
    from sglang.multimodal_gen.runtime.weights.source import (
        materialize_weight,
        resolve_weight,
    )

    path = materialize_weight(resolve_weight(path))
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    elif path.endswith((".pth", ".pt", ".bin")):
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("TinyVAE weights must be safetensors, pth, pt or bin")
    model_name = TINY_VAE_ALIASES.get(model_name, model_name)
    model_cls, channels = TINY_VAE_MODELS[model_name]
    model = initialize_model(
        model_cls, {"latent_channels": channels, "load_encoder": False}, dtype
    )
    if not isinstance(state, dict):
        raise ValueError("TinyVAE checkpoint must contain a tensor state dictionary")
    decoder_state = {}
    expected = model.decoder.state_dict()
    extras = []
    for name, tensor in state.items():
        if name.startswith(("encoder.", "taesd_encoder.")) or name in (
            "vae_scale",
            "vae_shift",
        ):
            continue
        if name.startswith("taesd_decoder."):
            name = name.removeprefix("taesd_decoder.")
        elif name.startswith("decoder."):
            name = name.removeprefix("decoder.")
        if name not in expected:
            extras.append(name)
        else:
            decoder_state[name] = tensor
    if extras:
        raise ValueError(f"Unexpected TinyVAE decoder weights: {extras}")
    model.decoder.load_state_dict(decoder_state, strict=True)
    return model.eval().requires_grad_(False)


class TinyVAEDecodeWrapper(nn.Module):
    """Keep the original VAE's raw-latent and pixel contracts at its boundary."""

    # The temporal runner does not use the original decoder's spatial collectives.
    use_parallel_decode = False

    def __init__(self, original, tiny, model_name, pipeline_config):
        super().__init__()
        self.original = original
        self.tiny = tiny
        self.model_name = TINY_VAE_ALIASES.get(model_name, model_name)
        self.pipeline_config = pipeline_config
        if self.model_name in ("taeh3", "taew2_1", "taew2_2"):
            arch = pipeline_config.vae_config.arch_config
            device = next(tiny.parameters()).device
            for name in ("latents_mean", "latents_std"):
                value = torch.tensor(
                    getattr(arch, name), device=device, dtype=torch.float32
                ).view(1, -1, 1, 1, 1)
                self.register_buffer(
                    f"tiny_{name}",
                    value,
                    persistent=False,
                )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("original"), name)

    def forward(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def _normalized_latents(self, latents):
        if self.model_name == "taeltx2":
            # LTX-2 receives raw latents from the AV decoding stage.
            mean = self.original.latents_mean.to(latents).view(1, -1, 1, 1, 1)
            std = self.original.latents_std.to(latents).view(1, -1, 1, 1, 1)
            return (latents - mean) / std
        if self.model_name in ("taeh3", "taew2_1", "taew2_2"):
            mean = self.tiny_latents_mean.to(latents)
            std = self.tiny_latents_std.to(latents)
            return (latents - mean) / std
        if self.model_name == "taef2" and latents.shape[1] == 32:
            b, c, h, w = latents.shape
            latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
            latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(b, 128, h // 2, w // 2)
        scaling, shift = self.pipeline_config.get_decode_scale_and_shift(
            latents.device, latents.dtype, self.original
        )
        if shift is not None:
            latents = latents - shift
        return latents if scaling is None else latents * scaling

    def _pixels(self, latents):
        normalized = self._normalized_latents(latents)
        channels = normalized.shape[1]
        if channels != self.tiny.latent_channels:
            raise ValueError(
                f"{self.model_name} expects {self.tiny.latent_channels} latent channels, got {channels}"
            )
        return self.tiny.decode(normalized.to(next(self.tiny.parameters()).dtype))

    def decode(self, latents, return_dict=False, on_frames=None, **kwargs):
        # Generic and LTX stages consume raw pixels in [-1, 1].
        pixels = self._pixels(latents) * 2 - 1
        if on_frames is not None:
            on_frames(pixels)
        return pixels

    def decode_base(self, latents, on_frames=None, **kwargs):
        # H3's processor owns ImageNet denormalization after decode_base.
        pixels = self._pixels(latents)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = (pixels - mean) / std
        if on_frames is not None:
            on_frames(pixels)
        return pixels

    def prepare_decoder_autocast_weights(self, dtype):
        return 0

    def enable_tiling(self, *args, **kwargs):
        # TinyVAE processes time sequentially. Original encoder tiling stays intact.
        return self.original.enable_tiling(*args, **kwargs)


def with_tiny_vae(vae, server_args, component_name):
    config = server_args.pipeline_config
    if not getattr(config, "tiny_vae_path", None) or component_name not in (
        "vae",
        "video_vae",
    ):
        return vae
    validate_tiny_vae_config(config)
    from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision

    dtype = resolve_decode_precision(server_args, component_name)
    if resolve_decode_precision(server_args, component_name, quality="high") != dtype:
        raise ValueError(
            "TinyVAE requires --vae-decode-precision-high to match the default "
            "decode precision; request-scoped precision changes are unsupported"
        )
    tiny = load_tiny_autoencoder(
        config.tiny_vae_path,
        config.tiny_vae_model,
        dtype,
    )
    tiny = tiny.to(device=next(vae.parameters()).device)
    return TinyVAEDecodeWrapper(vae, tiny, config.tiny_vae_model, config)
