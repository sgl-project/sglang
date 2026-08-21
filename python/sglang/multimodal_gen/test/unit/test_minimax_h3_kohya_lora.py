# SPDX-License-Identifier: Apache-2.0
"""Kohya `networks.lora_minimax_h3` keys map onto native H3 fused layers."""

from collections import defaultdict

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora_format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)

_HIDDEN = 5376
_INNER = 7168
_QKV = 21504
_FC1 = 28672
_FF = 14336
_RANK = 8

_MODULES = (
    ("attn_qkv_proj", _HIDDEN, _QKV, "blocks.{}.attn.qkv_proj"),
    ("attn_out_proj", _INNER, _HIDDEN, "blocks.{}.attn.out_proj"),
    ("mlp_fc1", _HIDDEN, _FC1, "blocks.{}.mlp.fc1"),
    ("mlp_fc2", _FF, _HIDDEN, "blocks.{}.mlp.fc2"),
)


def _kohya_block(layer: int, rank: int = _RANK) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module, in_dim, out_dim, _target in _MODULES:
        prefix = f"lora_unet_blocks_{layer}_{module}"
        tensors[f"{prefix}.alpha"] = torch.tensor(float(rank))
        tensors[f"{prefix}.lora_down.weight"] = torch.randn(rank, in_dim)
        tensors[f"{prefix}.lora_up.weight"] = torch.randn(out_dim, rank)
    return tensors


def _map_adapter(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    arch = MiniMaxH3DiTArchConfig()
    lora_fn = get_param_names_mapping(arch.lora_param_names_mapping)
    param_fn = get_param_names_mapping(arch.param_names_mapping)
    mapped: dict[str, torch.Tensor] = {}
    leftovers: dict[str, dict] = defaultdict(dict)
    for name, weight in state_dict.items():
        name = name.replace(".weight", "")
        name, _, _ = lora_fn(name)
        target, merge_index, num_merge = param_fn(name)
        if merge_index is not None:
            leftovers[target][merge_index] = weight
            if len(leftovers[target]) < num_merge:
                continue
        if target in mapped:
            raise AssertionError(f"duplicate mapped target {target}")
        mapped[target] = weight
    assert leftovers == {}, leftovers
    return mapped


def test_detects_kohya_minimax_h3_before_generic_sd():
    raw = _kohya_block(0)
    assert detect_lora_format_from_state_dict(raw) == LoRAFormat.KOHYA_MINIMAX_H3


def test_normalize_rewrites_to_native_fused_names():
    raw = _kohya_block(12)
    normalized = normalize_lora_state_dict(raw)
    assert detect_lora_format_from_state_dict(normalized) == LoRAFormat.STANDARD
    assert "lora_unet_" not in "".join(normalized)
    assert normalized["blocks.12.attn.qkv_proj.lora_A"].shape == (_RANK, _HIDDEN)
    assert normalized["blocks.12.attn.qkv_proj.lora_B"].shape == (_QKV, _RANK)
    assert normalized["blocks.12.attn.qkv_proj.alpha"].item() == _RANK
    assert normalized["blocks.12.mlp.fc1.lora_B"].shape == (_FC1, _RANK)
    assert normalized["blocks.12.mlp.fc2.lora_A"].shape == (_RANK, _FF)


def test_loader_mapping_accepts_normalized_and_raw_kohya_keys():
    raw = _kohya_block(3)
    expected = {
        "blocks.3.attn.qkv_proj.lora_A",
        "blocks.3.attn.qkv_proj.lora_B",
        "blocks.3.attn.qkv_proj.alpha",
        "blocks.3.attn.out_proj.lora_A",
        "blocks.3.attn.out_proj.lora_B",
        "blocks.3.attn.out_proj.alpha",
        "blocks.3.mlp.fc1.lora_A",
        "blocks.3.mlp.fc1.lora_B",
        "blocks.3.mlp.fc1.alpha",
        "blocks.3.mlp.fc2.lora_A",
        "blocks.3.mlp.fc2.lora_B",
        "blocks.3.mlp.fc2.alpha",
    }

    normalized = normalize_lora_state_dict(raw)
    assert set(_map_adapter(normalized)) == expected

    raw_stripped = {name.replace(".weight", ""): tensor for name, tensor in raw.items()}
    assert set(_map_adapter(raw_stripped)) == expected


def test_full_50_block_export_has_no_unmapped_keys():
    raw = {}
    for layer in range(50):
        raw.update(_kohya_block(layer))
    assert len(raw) == 600
    mapped = _map_adapter(normalize_lora_state_dict(raw))
    assert len(mapped) == 600
    assert all(key.startswith("blocks.") for key in mapped)
    assert not any(key.startswith("lora_unet_") for key in mapped)


def test_does_not_claim_kohya_flux_double_blocks():
    flux_like = {
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.randn(4, 8),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.randn(8, 4),
    }
    assert detect_lora_format_from_state_dict(flux_like) != LoRAFormat.KOHYA_MINIMAX_H3
