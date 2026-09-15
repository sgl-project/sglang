# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 Parallel Decoding Distillation (PDD) Acc LoRA.

Official ``alibaba-pai/MiniMax-H3-Acc-LoRAs`` files are not PEFT-only: a
rank-64 backbone LoRA is paired with a length-``N`` bank of final-layer
heads (``proj_out`` / ``audio_proj_out``). Each Euler step fuses one
``block_size``-wide slice of that bank with a Δt-weighted mean, matching
VideoX-Fun ``apply_pdd_lora`` and ComfyUI PR #15908.

The bank is installed on ``MiniMaxH3FinalLayer`` as named fields
(``_pdd_video_plan`` starts as ``None``). The denoise loop calls
``model.arm_pdd_step(step)`` when that attribute is bound.

Released metadata is ``pdd_num_steps=32``, ``pdd_block_size=4`` (8 NFE).
The stock H3 ``simple`` schedule at 9 sigma points (8 DiT forwards) with
shifts 12 / 3 lands on the 32-grid boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_PDD_CONFIG = {
    "pdd_num_steps": 32,
    "pdd_block_size": 4,
    "lora_rank": 64,
    "lora_alpha": 64.0,
}

_VIDEO_WEIGHT_KEYS = (
    "proj_out.weight",
    "diffusion_model.proj_out.weight",
    "proj_out.set_weight",
    "diffusion_model.proj_out.set_weight",
)
_VIDEO_BIAS_KEYS = (
    "proj_out.bias",
    "diffusion_model.proj_out.bias",
    "proj_out.set_bias",
    "diffusion_model.proj_out.set_bias",
)
_AUDIO_WEIGHT_KEYS = (
    "audio_proj_out.weight",
    "diffusion_model.audio_proj_out.weight",
    "audio_proj_out.set_weight",
    "diffusion_model.audio_proj_out.set_weight",
)
_AUDIO_BIAS_KEYS = (
    "audio_proj_out.bias",
    "diffusion_model.audio_proj_out.bias",
    "audio_proj_out.set_bias",
    "diffusion_model.audio_proj_out.set_bias",
)

# Native H3 video head is 24 * 1 * 2 * 2 = 96; audio head is 32.
_DEFAULT_VIDEO_OUT = 96
_DEFAULT_AUDIO_OUT = 32


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending grid ``0 = t_0 < ... < t_N = 1`` of one MiniMax-H3 schedule."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), sigma)


def pdd_sampling_plan(
    step_sizes: torch.Tensor, start: int, block_size: int
) -> torch.Tensor:
    """Δt-normalized weights over ``[start, start + block_size)``."""
    if block_size < 1:
        raise ValueError(f"PDD block_size must be >= 1, got {block_size}")
    n = int(step_sizes.shape[0])
    start = max(0, min(int(start), n - 1))
    stop = min(start + int(block_size), n)
    if stop <= start:
        raise ValueError(f"PDD interval is empty: start={start}, stop={stop}, n={n}")
    span = step_sizes[start:stop].sum()
    plan = torch.zeros(n, dtype=step_sizes.dtype, device=step_sizes.device)
    plan[start:stop] = step_sizes[start:stop] / span
    return plan


def fuse_pdd_head(
    bank_w: torch.Tensor,
    bank_b: torch.Tensor,
    plan: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend a ``[N, out, in]`` bank into one ``[out, in]`` head."""
    if bank_w.ndim != 3:
        raise ValueError(
            f"PDD weight bank must be 3D [N, out, in], got {tuple(bank_w.shape)}"
        )
    if bank_b.ndim != 2 or bank_b.shape[0] != bank_w.shape[0]:
        raise ValueError(
            f"PDD bias bank must be [N, out] matching {tuple(bank_w.shape)}, "
            f"got {tuple(bank_b.shape)}"
        )
    if plan.numel() != bank_w.shape[0]:
        raise ValueError(f"PDD plan length {plan.numel()} != bank N {bank_w.shape[0]}")
    plan = plan.to(device=bank_w.device, dtype=bank_w.dtype)
    weight = torch.einsum("n,noi->oi", plan, bank_w)
    bias = torch.einsum("n,no->o", plan, bank_b)
    return weight, bias


def pdd_linear(
    hidden: torch.Tensor,
    bank_w: torch.Tensor,
    bank_b: torch.Tensor,
    plan: torch.Tensor,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> torch.Tensor:
    """Fuse the bank and apply a TP-local output projection."""
    weight, bias = fuse_pdd_head(
        bank_w.to(device=hidden.device, dtype=hidden.dtype),
        bank_b.to(device=hidden.device, dtype=hidden.dtype),
        plan,
    )
    if tp_size > 1:
        if weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"PDD out_features {weight.shape[0]} is not divisible by tp_size {tp_size}"
            )
        shard = weight.shape[0] // tp_size
        sl = slice(tp_rank * shard, (tp_rank + 1) * shard)
        weight = weight[sl]
        bias = bias[sl]
    return F.linear(hidden, weight, bias)


def _first_present(
    state: Mapping[str, torch.Tensor], keys: tuple[str, ...]
) -> tuple[str, torch.Tensor] | None:
    for key in keys:
        tensor = state.get(key)
        if tensor is not None:
            return key, tensor
    return None


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(float(value))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _reshape_stacked_head(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim == 3:
        n, out, _inn = weight.shape
        if bias is None:
            raise ValueError(f"PDD key {source} is missing a matching bias bank")
        if bias.ndim == 1:
            if bias.numel() != n * out:
                raise ValueError(
                    f"PDD bias for {source} has shape {tuple(bias.shape)}, "
                    f"expected [{n}, {out}] or [{n * out}]"
                )
            bias = bias.reshape(n, out)
        elif tuple(bias.shape) != (n, out):
            raise ValueError(
                f"PDD bias for {source} has shape {tuple(bias.shape)}, expected [{n}, {out}]"
            )
        return weight, bias
    if weight.ndim != 2:
        raise ValueError(
            f"PDD head {source} must be 2D or 3D, got {tuple(weight.shape)}"
        )
    if weight.shape[0] % out_features != 0:
        raise ValueError(
            f"PDD head {source} rows {weight.shape[0]} are not a multiple of "
            f"out_features={out_features}"
        )
    n = weight.shape[0] // out_features
    if n < 2:
        raise ValueError(f"PDD head {source} is a single head, not a bank")
    if bias is None:
        raise ValueError(f"PDD key {source} is missing a matching bias bank")
    if bias.ndim == 1:
        if bias.numel() != n * out_features:
            raise ValueError(
                f"PDD bias for {source} has {bias.numel()} values, expected {n * out_features}"
            )
        bias = bias.reshape(n, out_features)
    elif tuple(bias.shape) != (n, out_features):
        raise ValueError(
            f"PDD bias for {source} has shape {tuple(bias.shape)}, "
            f"expected [{n}, {out_features}]"
        )
    return weight.reshape(n, out_features, weight.shape[1]), bias


@dataclass
class PDDHeadBank:
    video_weight: torch.Tensor
    video_bias: torch.Tensor
    audio_weight: torch.Tensor
    audio_bias: torch.Tensor
    num_steps: int
    block_size: int
    lora_rank: int
    lora_alpha: float
    consumed_keys: tuple[str, ...]

    @property
    def nfe(self) -> int:
        return self.num_steps // self.block_size


def load_pdd_config(
    metadata: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_PDD_CONFIG)
    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            if isinstance(saved, Mapping):
                for key in config:
                    if key in saved:
                        config[key] = saved[key]
    if metadata:
        for key in config:
            if key in metadata and metadata[key] not in (None, ""):
                config[key] = metadata[key]
    config["pdd_num_steps"] = _as_int(config["pdd_num_steps"], 32)
    config["pdd_block_size"] = _as_int(config["pdd_block_size"], 4)
    config["lora_rank"] = _as_int(config["lora_rank"], 64)
    config["lora_alpha"] = _as_float(config["lora_alpha"], 64.0)
    if (
        config["pdd_block_size"] < 1
        or config["pdd_num_steps"] % config["pdd_block_size"] != 0
    ):
        raise ValueError(
            f"pdd_num_steps={config['pdd_num_steps']} must be divisible by "
            f"pdd_block_size={config['pdd_block_size']}"
        )
    return config


def extract_pdd_payload(
    state_dict: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    video_out_features: int = _DEFAULT_VIDEO_OUT,
    audio_out_features: int = _DEFAULT_AUDIO_OUT,
) -> tuple[dict[str, torch.Tensor], PDDHeadBank] | None:
    """Split an official / Comfy-converted Acc LoRA into backbone + head bank."""
    video = _first_present(state_dict, _VIDEO_WEIGHT_KEYS)
    audio = _first_present(state_dict, _AUDIO_WEIGHT_KEYS)
    if video is None or audio is None:
        return None
    video_key, video_w = video
    audio_key, audio_w = audio
    is_bank = (video_w.ndim == 3 and video_w.shape[0] > 1) or (
        video_w.ndim == 2 and video_w.shape[0] > video_out_features
    )
    if not is_bank:
        return None

    video_bias_hit = _first_present(state_dict, _VIDEO_BIAS_KEYS)
    audio_bias_hit = _first_present(state_dict, _AUDIO_BIAS_KEYS)
    video_b = None if video_bias_hit is None else video_bias_hit[1]
    audio_b = None if audio_bias_hit is None else audio_bias_hit[1]
    video_w, video_b = _reshape_stacked_head(
        video_w, video_b, video_out_features, video_key
    )
    audio_w, audio_b = _reshape_stacked_head(
        audio_w, audio_b, audio_out_features, audio_key
    )
    if video_w.shape[0] != audio_w.shape[0]:
        raise ValueError(
            f"PDD video bank N={video_w.shape[0]} != audio bank N={audio_w.shape[0]}"
        )

    config = load_pdd_config(metadata, config_path)
    num_steps = int(video_w.shape[0])
    if config["pdd_num_steps"] != num_steps:
        logger.info(
            "PDD metadata pdd_num_steps=%s, using bank length %d",
            config["pdd_num_steps"],
            num_steps,
        )
    consumed = [video_key, audio_key]
    if video_bias_hit is not None:
        consumed.append(video_bias_hit[0])
    if audio_bias_hit is not None:
        consumed.append(audio_bias_hit[0])
    bank = PDDHeadBank(
        video_weight=video_w.contiguous(),
        video_bias=video_b.contiguous(),
        audio_weight=audio_w.contiguous(),
        audio_bias=audio_b.contiguous(),
        num_steps=num_steps,
        block_size=int(config["pdd_block_size"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        consumed_keys=tuple(consumed),
    )
    lora_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name not in bank.consumed_keys
    }
    logger.info(
        "PDD Acc LoRA head bank: N=%d block=%d nfe=%d (video %s, audio %s)",
        bank.num_steps,
        bank.block_size,
        bank.nfe,
        tuple(bank.video_weight.shape),
        tuple(bank.audio_weight.shape),
    )
    return lora_state, bank


def _pdd_final_layer(module: nn.Module) -> tuple[nn.Module, nn.Module | None]:
    try:
        return module.final_layer, module
    except AttributeError:
        return module, None


def apply_pdd_head_bank(
    module: nn.Module,
    bank: PDDHeadBank,
    *,
    video_shift: float,
    audio_shift: float,
) -> None:
    """Install the bank on ``module.final_layer`` (or ``module`` itself)."""
    layer, model = _pdd_final_layer(module)
    device = next(layer.parameters()).device
    dtype = torch.float32
    layer.register_buffer(
        "_pdd_video_w",
        bank.video_weight.to(device=device, dtype=dtype),
        persistent=False,
    )
    layer.register_buffer(
        "_pdd_video_b", bank.video_bias.to(device=device, dtype=dtype), persistent=False
    )
    layer.register_buffer(
        "_pdd_audio_w",
        bank.audio_weight.to(device=device, dtype=dtype),
        persistent=False,
    )
    layer.register_buffer(
        "_pdd_audio_b", bank.audio_bias.to(device=device, dtype=dtype), persistent=False
    )
    layer.register_buffer(
        "_pdd_video_dts",
        pdd_time_grid(video_shift, bank.num_steps).diff().float().to(device),
        persistent=False,
    )
    layer.register_buffer(
        "_pdd_audio_dts",
        pdd_time_grid(audio_shift, bank.num_steps).diff().float().to(device),
        persistent=False,
    )
    layer._pdd_num_steps = bank.num_steps
    layer._pdd_block_size = bank.block_size
    layer._pdd_nfe = bank.nfe
    arm_pdd_step(layer, 0)
    if model is not None:
        model.arm_pdd_step = lambda step: arm_pdd_step(layer, step)
    logger.info(
        "Installed PDD heads on final layer: nfe=%d, video_shift=%s, audio_shift=%s. "
        "Use num_inference_steps=%d (%d sigma points / %d DiT forwards).",
        bank.nfe,
        video_shift,
        audio_shift,
        bank.nfe + 1,
        bank.nfe + 1,
        bank.nfe,
    )


def clear_pdd_heads(module: nn.Module) -> None:
    layer, model = _pdd_final_layer(module)
    if model is not None:
        model.arm_pdd_step = None
    for name in (
        "_pdd_video_w",
        "_pdd_video_b",
        "_pdd_audio_w",
        "_pdd_audio_b",
        "_pdd_video_dts",
        "_pdd_audio_dts",
    ):
        if name in layer._buffers:
            delattr(layer, name)
    layer._pdd_num_steps = None
    layer._pdd_block_size = None
    layer._pdd_nfe = None
    layer._pdd_video_plan = None
    layer._pdd_audio_plan = None


def arm_pdd_step(layer: nn.Module, step: int) -> None:
    """Select the block of heads for denoise step ``step`` (0-based NFE index)."""
    if layer._pdd_nfe is None:
        return
    if step < 0:
        raise ValueError(f"PDD step must be >= 0, got {step}")
    if step >= layer._pdd_nfe:
        logger.warning(
            "PDD Acc LoRA is trained for %d NFE; step %d reuses the last block. "
            "Set num_inference_steps=%d.",
            layer._pdd_nfe,
            step,
            layer._pdd_nfe + 1,
        )
        step = layer._pdd_nfe - 1
    start = step * int(layer._pdd_block_size)
    layer._pdd_video_plan = pdd_sampling_plan(
        layer._pdd_video_dts, start, int(layer._pdd_block_size)
    )
    layer._pdd_audio_plan = pdd_sampling_plan(
        layer._pdd_audio_dts, start, int(layer._pdd_block_size)
    )


def _head_tp(head: nn.Module) -> tuple[int, int]:
    inner = head.base_layer if isinstance(head, BaseLayerWithLoRA) else head
    return int(inner.tp_size), int(inner.tp_rank)


def project_pdd_or_base(
    layer: nn.Module,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused (video, audio) if a PDD bank is armed, else ``None``."""
    if layer._pdd_video_plan is None or layer._pdd_audio_plan is None:
        return None
    video_tp, video_rank = _head_tp(layer.video_out)
    audio_tp, audio_rank = _head_tp(layer.audio_out)
    video = pdd_linear(
        hidden,
        layer._pdd_video_w,
        layer._pdd_video_b,
        layer._pdd_video_plan,
        tp_size=video_tp,
        tp_rank=video_rank,
    )
    audio = pdd_linear(
        hidden,
        layer._pdd_audio_w,
        layer._pdd_audio_b,
        layer._pdd_audio_plan,
        tp_size=audio_tp,
        tp_rank=audio_rank,
    )
    return video, audio
