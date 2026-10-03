# SPDX-License-Identifier: Apache-2.0
"""Portable FP32 YuE2 Oobleck VAE with exact-boundary tiled decoding.

Oobleck and SnakeBeta derived from stable-audio-tools a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45.

Copyright (c) 2023 Stability AI; 
Copyright (c) 2022 NVIDIA CORPORATION.

MIT: see THIRD_PARTY_NOTICES.md shipped with this module/model repository.
"""

# NOTE (yiakwy) : same to
# https://github.com/multimodal-art-projection/YuE/tree/main/src/yue2/modeling_vae.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn
from torch.nn.utils import weight_norm
from transformers import PretrainedConfig, PreTrainedModel


def checkpoint(function, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * torch.pow(
        torch.sin(x * alpha), 2
    )


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def get_activation(
    activation: Literal["elu", "snake", "none"], channels=None
) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, act_type):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        residual = x
        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, act_type),
            ResidualUnit(in_channels, in_channels, 3, act_type),
            ResidualUnit(in_channels, in_channels, 9, act_type),
            get_activation(act_type, channels=in_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        upsample_layer = WNConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.layers = nn.Sequential(
            get_activation(act_type, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, act_type),
            ResidualUnit(out_channels, out_channels, 3, act_type),
            ResidualUnit(out_channels, out_channels, 9, act_type),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        if antialias_activation:
            raise ValueError("The released encoder does not use antialias_activation")
        self.in_channels = in_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[-1] * channels),
                WNConv1d(
                    in_channels=c_mults[-1] * channels,
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        snake_type="vanilla",
        antialias_activation=False,
        use_nearest_upsample=False,
        use_filter=False,
        final_tanh=True,
    ):
        super().__init__()
        if antialias_activation or use_nearest_upsample or use_filter:
            raise ValueError("Unsupported option for the released decoder")
        if use_snake and snake_type != "vanilla":
            raise ValueError("The released decoder uses vanilla SnakeBeta")
        self.out_channels = out_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[0] * channels),
                WNConv1d(
                    in_channels=c_mults[0] * channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class YuE2VAEConfig(PretrainedConfig):
    """Configuration shared by YuE2-Vae and YuE2-Vae-legacy.

    Use ``standard`` for listening and ``legacy`` for the paper metric baseline.
    """

    model_type = "yue2_vae"

    _hf_fields = frozenset({
        "model_type", "architectures", "auto_map", "transformers_version",
        "dtype", "torch_dtype", "return_dict", "output_hidden_states",
        "output_attentions", "use_cache", "tie_word_embeddings", "torchscript",
        "is_decoder", "is_encoder_decoder", "add_cross_attention",
        "bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id",
        "attn_implementation",
    })

    def to_dict(self):
        return {key: value for key, value in super().to_dict().items()
                if key in self._hf_fields or key in self._inference_fields}

    _inference_fields = frozenset(['encoder_config', 'decoder_config', 'sample_rate', 'latent_dim', 'downsampling_ratio', 'audio_channels', 'release_variant', 'decode_core_frames', 'decode_halo_frames'])

    def __init__(self, encoder_config=None, decoder_config=None,
                 sample_rate=48000, latent_dim=64, downsampling_ratio=1920,
                 audio_channels=2, release_variant="standard",
                 decode_core_frames=1024, decode_halo_frames=16,
                 **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
        kwargs.setdefault("architectures", ["YuE2VAE"])
        kwargs.setdefault("auto_map", {
            "AutoConfig": "modeling_vae.YuE2VAEConfig",
            "AutoModel": "modeling_vae.YuE2VAE",
        })
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or dict(
            in_channels=2, channels=64, c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6], latent_dim=128, use_snake=True)
        self.decoder_config = decoder_config or dict(
            out_channels=2, channels=64, c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6], latent_dim=64, use_snake=True,
            snake_type="vanilla", use_filter=False, final_tanh=False)
        encoder_fields = {"in_channels", "channels", "latent_dim", "c_mults", "strides",
                          "use_snake", "antialias_activation"}
        decoder_fields = {"out_channels", "channels", "latent_dim", "c_mults", "strides",
                          "use_snake", "snake_type", "antialias_activation",
                          "use_nearest_upsample", "use_filter", "final_tanh"}
        self.encoder_config = {key: value for key, value in self.encoder_config.items() if key in encoder_fields}
        self.decoder_config = {key: value for key, value in self.decoder_config.items() if key in decoder_fields}
        self.sample_rate = int(sample_rate)
        self.latent_dim = int(latent_dim)
        self.downsampling_ratio = int(downsampling_ratio)
        self.audio_channels = int(audio_channels)
        self.release_variant = release_variant
        self.decode_core_frames = int(decode_core_frames)
        self.decode_halo_frames = int(decode_halo_frames)
        if self.decode_core_frames < 1 or self.decode_halo_frames < 0:
            raise ValueError("Invalid VAE core/halo configuration")
        if math.prod(self.decoder_config["strides"]) != self.downsampling_ratio:
            raise ValueError("Decoder strides do not match downsampling_ratio")
        if self.decoder_config["latent_dim"] != self.latent_dim:
            raise ValueError("Decoder input channels do not match latent_dim")


def _dependency_interval(module, low, high):
    """Inclusive input support of an output interval; no waveform blending."""
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in reversed(list(layers)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(f"No audited support rule for {type(module).__name__}")


def _output_length(module, length):
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in layers:
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return ((length - 1) * module.stride[0] - 2 * module.padding[0]
                + module.dilation[0] * (module.kernel_size[0] - 1)
                + module.output_padding[0] + 1)
    if isinstance(module, nn.Conv1d):
        return ((length + 2 * module.padding[0]
                 - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
                // module.stride[0] + 1)
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(f"No audited length rule for {type(module).__name__}")


class YuE2VAE(PreTrainedModel):
    """Strict EMA model; only the decoder needs to reside on an accelerator.

    ``decode_tiled`` preserves finite receptive-field context and writes cropped
    cores to CPU. The mathematical waveform is the full decoder's waveform;
    convolution kernel choices may cause small FP32 rounding differences.
    """

    config_class = YuE2VAEConfig
    base_model_prefix = ""
    main_input_name = "audio"

    def __init__(self, config, decoder_only=False):
        super().__init__(config)
        self.decoder_only = bool(decoder_only)
        if not self.decoder_only:
            self.encoder = OobleckEncoder(**config.encoder_config)
        self.decoder = OobleckDecoder(**config.decoder_config)
        self.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        config=None, decoder_only=False, device="cpu",
                        torch_dtype=None, dtype=None, revision=None, token=None,
                        cache_dir=None, local_files_only=False,
                        force_download=False, subfolder="", **kwargs):
        """Load a local export or Hub repository, selecting decoder tensors.

        Complete exports use unprefixed EMA keys. ``decoder_only=True`` avoids
        constructing the encoder or loading encoder tensors into RAM/GPU.
        """
        from safetensors import safe_open
        requested_dtype = dtype if dtype is not None else torch_dtype
        if requested_dtype not in (None, "auto", "float32", torch.float32):
            raise ValueError("The validated VAE requires FP32; quantize the LM separately")
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            if device_map == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif isinstance(device_map, str):
                device = device_map
            elif isinstance(device_map, dict) and set(device_map) == {""}:
                device = device_map[""]
            else:
                raise ValueError("Use decoder_only=True and a single device for the VAE")
        for name in ("trust_remote_code", "low_cpu_mem_usage", "_from_auto",
                     "_from_pipeline", "_commit_hash", "adapter_kwargs",
                     "_fast_init", "weights_only", "use_safetensors"):
            kwargs.pop(name, None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        if kwargs or model_args:
            raise TypeError(f"Unsupported VAE loading options: {sorted(kwargs)}")
        path = Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(
                str(pretrained_model_name_or_path), revision=revision, token=token,
                cache_dir=cache_dir, local_files_only=local_files_only,
                force_download=force_download,
                allow_patterns=[f"{subfolder + '/' if subfolder else ''}{pattern}"
                                for pattern in ("config.json", "*.safetensors",
                                                "*.safetensors.index.json")]))
        path = path / subfolder
        if config is None:
            config = YuE2VAEConfig.from_pretrained(path, local_files_only=True)
        model = cls(config, decoder_only=decoder_only)
        index = path / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            files = sorted({name for key, name in mapping.items()
                            if not decoder_only or key.startswith("decoder.")})
        else:
            files = ["model.safetensors"]
        state = {}
        for name in files:
            with safe_open(path / name, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not decoder_only or key.startswith("decoder."):
                        if key in state:
                            raise ValueError(f"Duplicate VAE tensor: {key}")
                        state[key] = handle.get_tensor(key)
        expected = set(model.state_dict())
        if set(state) != expected:
            raise ValueError(f"VAE tensor mismatch: missing={sorted(expected-set(state))}, "
                             f"unexpected={sorted(set(state)-expected)}")
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("VAE export contains tensors that are not FP32")
        model.load_state_dict(state, strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if output_loading_info:
            return model, dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[],
                               error_msgs=[])
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        if self.decoder_only:
            raise ValueError("Reload decoder_only=False to save a complete VAE repository")
        if kwargs.get("safe_serialization", True) is False:
            raise ValueError("YuE2 VAE release exports require safetensors")
        return super().save_pretrained(save_directory, *args, **kwargs)

    @property
    def decoder_device(self):
        return next(self.decoder.parameters()).device

    def _latent(self, latent):
        latent = torch.as_tensor(latent)
        if (latent.ndim != 3 or latent.shape[1] != self.config.latent_dim
                or latent.shape[0] < 1 or latent.shape[-1] < 1):
            raise ValueError(f"Expected nonempty [B,{self.config.latent_dim},T] latents")
        if not torch.isfinite(latent).all():
            raise ValueError("VAE latents contain non-finite values")
        if next(self.decoder.parameters()).dtype != torch.float32:
            raise ValueError("VAE decoder weights must remain FP32")
        return latent

    @torch.inference_mode()
    def encode(self, audio, sample=False, generator=None, return_info=False):
        """Encode FP32 stereo audio; posterior mean by default.

        Set ``sample=True`` with a per-request ``torch.Generator`` to reproduce
        stochastic posterior sampling. This audio VAE is not the unreleased
        semantic audio tokenizer semantic tokenizer.
        """
        if self.decoder_only:
            raise RuntimeError("Encoder not loaded; reload with decoder_only=False")
        audio = torch.as_tensor(audio)
        if (audio.ndim != 3 or audio.shape[1] != self.config.audio_channels
                or audio.shape[-1] < self.config.downsampling_ratio):
            raise ValueError("Expected audio [B,2,S] with at least one latent frame")
        if not torch.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        device = next(self.encoder.parameters()).device
        pre = self.encoder(audio.to(device=device, dtype=torch.float32))
        mean, scale = pre.chunk(2, dim=1)
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        if sample:
            noise = torch.randn(mean.shape, dtype=mean.dtype,
                                device=device, generator=generator)
            latent = noise * stdev + mean
        else:
            latent = mean
        if return_info:
            return latent, dict(mean=mean, scale=scale, stdev=stdev)
        return latent

    @torch.inference_mode()
    def decode(self, latent):
        """Full waveform, FP32 [B,2,1920*T-64], without clipping."""
        latent = self._latent(latent)
        with torch.autocast(device_type=self.decoder_device.type, enabled=False):
            return self.decoder(latent.to(device=self.decoder_device, dtype=torch.float32))

    def natural_output_length(self, frames):
        if int(frames) < 1:
            raise ValueError("frames must be positive")
        return _output_length(self.decoder, int(frames))

    def required_halo(self, core_frames=None):
        core_frames = self.config.decode_core_frames if core_frames is None else core_frames
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.decoder, 0, core_frames * ratio - 1)
        return max(0, -low, high - core_frames + 1)

    @torch.inference_mode()
    def decode_tiled(self, latent, core_frames=None, halo_frames=None,
                     output_device="cpu",
                     on_progress: Callable[[int, int], None] | None = None):
        """Decode bounded tiles, retaining exact cores with natural end length.

        Each crop has enough left/right context for every dependency. There is
        no crossfade or boundary smoothing, and no zero padding of final audio.
        CPU output prevents an entire song from accumulating on the GPU.
        ``on_progress(completed, total)`` runs after each existing crop copy;
        no extra synchronization is added. With a CUDA output device, queued
        work may still be executing. Callback exceptions propagate.
        """
        latent = self._latent(latent)
        core_frames = self.config.decode_core_frames if core_frames is None else core_frames
        halo_frames = self.config.decode_halo_frames if halo_frames is None else halo_frames
        if not isinstance(core_frames, int) or core_frames < 1:
            raise ValueError("core_frames must be a positive integer")
        required = self.required_halo(core_frames)
        if not isinstance(halo_frames, int) or halo_frames < required:
            raise ValueError(f"halo_frames must be at least {required} for this decoder")
        frames = latent.shape[-1]
        ratio = self.config.downsampling_ratio
        total = self.natural_output_length(frames)
        audio = torch.empty((latent.shape[0], self.config.audio_channels, total),
                            dtype=torch.float32, device=output_device)
        tiles = (frames + core_frames - 1) // core_frames
        for tile_index, start in enumerate(range(0, frames, core_frames)):
            end = min(frames, start + core_frames)
            left = max(0, start - halo_frames)
            right = min(frames, end + halo_frames)
            tile = self.decode(latent[..., left:right])
            out_start, out_end = start * ratio, min(end * ratio, total)
            crop_start = (start - left) * ratio
            crop = tile[..., crop_start:crop_start + out_end - out_start]
            if crop.shape[-1] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            audio[..., out_start:out_end].copy_(crop.to(output_device))
            del tile, crop
            if on_progress is not None:
                on_progress(tile_index + 1, tiles)
        return audio

    def decode_audio(self, latent, chunked=True, **kwargs):
        return self.decode_tiled(latent, **kwargs) if chunked else self.decode(latent)

    def forward(self, audio, sample=False, generator=None):
        return self.decode(self.encode(audio, sample=sample, generator=generator))


YuE2VAEConfig.register_for_auto_class("AutoConfig")
YuE2VAE.register_for_auto_class("AutoModel")
