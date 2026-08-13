# SPDX-License-Identifier: Apache-2.0
"""GPU expert placement masks for KT hybrid MoE (prefix / frequency)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_LAYER_MASKS: Optional[torch.Tensor] = None
_LOGICAL_TO_GPU: Optional[torch.Tensor] = None


def is_moe_layer(
    layer_idx: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> bool:
    return layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0


def generate_prefix_masks(
    num_layers: int,
    num_experts: int,
    experts_per_moe_layer: int,
    first_k_dense_replace: int = 0,
    moe_layer_freq: int = 1,
) -> torch.Tensor:
    """Per MoE layer: experts ``0 .. experts_per_moe_layer-1`` on GPU (static default)."""
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    k = min(max(experts_per_moe_layer, 0), num_experts)
    for layer_idx in range(num_layers):
        if is_moe_layer(layer_idx, first_k_dense_replace, moe_layer_freq) and k > 0:
            masks[layer_idx, :k] = True
    return masks


def generate_frequency_masks_per_layer(
    activation_freq: torch.Tensor,
    experts_per_moe_layer: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> torch.Tensor:
    """Per MoE layer: top-``experts_per_moe_layer`` experts by activation frequency."""
    num_layers, num_experts = activation_freq.shape
    k = min(max(experts_per_moe_layer, 0), num_experts)
    masks = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    if k == 0:
        return masks
    freq_cpu = activation_freq.to(device="cpu", dtype=torch.float32)
    for layer_idx in range(num_layers):
        if not is_moe_layer(layer_idx, first_k_dense_replace, moe_layer_freq):
            continue
        _, top = torch.topk(freq_cpu[layer_idx], k=k, largest=True, sorted=False)
        masks[layer_idx, top] = True
    return masks


def build_logical_to_gpu_index(masks: torch.Tensor) -> torch.Tensor:
    """``[num_layers, num_experts]`` bool -> ``[num_layers, num_experts]`` int64 (-1 = CPU)."""
    num_layers, num_experts = masks.shape
    out = torch.full((num_layers, num_experts), -1, dtype=torch.long, device="cpu")
    for layer_idx in range(num_layers):
        logical_ids = torch.nonzero(masks[layer_idx], as_tuple=False).view(-1)
        for slot, logical_id in enumerate(logical_ids.tolist()):
            out[layer_idx, logical_id] = slot
    return out


def load_activation_freq(path: str) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        for key in ("activation_freq", "freq", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, torch.Tensor):
        raise TypeError(
            f"Expected activation_freq tensor in {path}, got {type(data).__name__}"
        )
    return data.to(device="cpu", dtype=torch.float32)


def get_moe_layout_from_server_args(server_args: "ServerArgs") -> Tuple[int, int, int, int]:
    hf = server_args.get_model_config().hf_config
    num_layers = getattr(hf, "num_hidden_layers", None)
    num_experts = getattr(hf, "n_routed_experts", None)
    first_k_dense = getattr(hf, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(hf, "moe_layer_freq", 1) or 1
    if num_layers is None or num_experts is None:
        raise ValueError("Cannot infer MoE layout from model config")
    return num_layers, num_experts, first_k_dense, moe_layer_freq


def ensure_kt_layer_masks(server_args: "ServerArgs") -> None:
    global _LAYER_MASKS, _LOGICAL_TO_GPU
    if _LAYER_MASKS is not None:
        return
    if server_args.kt_weight_path is None:
        return

    num_layers, num_experts, first_k_dense, moe_freq = get_moe_layout_from_server_args(
        server_args
    )
    experts_per = server_args.kt_num_gpu_experts or 0
    strategy = getattr(server_args, "kt_expert_placement_strategy", None) or "prefix"

    if strategy == "frequency":
        freq_path = getattr(server_args, "kt_activation_freq_path", None)
        if not freq_path:
            raise ValueError(
                "--kt-expert-placement-strategy frequency requires "
                "--kt-activation-freq-path"
            )
        freq = load_activation_freq(freq_path)
        if tuple(freq.shape) != (num_layers, num_experts):
            raise ValueError(
                f"activation_freq shape {tuple(freq.shape)} != "
                f"({num_layers}, {num_experts})"
            )
        masks = generate_frequency_masks_per_layer(
            freq, experts_per, first_k_dense, moe_freq
        )
    else:
        if strategy != "prefix":
            logger.warning(
                "[KT] unknown kt_expert_placement_strategy=%r; using prefix",
                strategy,
            )
        masks = generate_prefix_masks(
            num_layers, num_experts, experts_per, first_k_dense, moe_freq
        )

    _LAYER_MASKS = masks
    _LOGICAL_TO_GPU = build_logical_to_gpu_index(masks)
    n_moe = sum(
        1 for i in range(num_layers) if is_moe_layer(i, first_k_dense, moe_freq)
    )
    logger.info(
        "[KT] expert placement strategy=%s experts_per_moe_layer=%d "
        "moe_layers=%d total_gpu_expert_slots=%d",
        strategy,
        experts_per,
        n_moe,
        int(masks.sum().item()),
    )


def get_layer_gpu_experts_mask(layer_idx: int) -> torch.Tensor:
    if _LAYER_MASKS is None:
        raise RuntimeError("KT layer masks not initialized; call ensure_kt_layer_masks")
    return _LAYER_MASKS[layer_idx]


def get_layer_logical_to_gpu_index(layer_idx: int) -> torch.Tensor:
    if _LOGICAL_TO_GPU is None:
        raise RuntimeError("KT layer masks not initialized; call ensure_kt_layer_masks")
    return _LOGICAL_TO_GPU[layer_idx]
