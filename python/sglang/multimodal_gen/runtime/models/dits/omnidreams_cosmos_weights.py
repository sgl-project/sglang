# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos streaming weight preparation helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

_PREPARED_SUFFIXES = (
    "self_attn.qkv_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)


def prepare_cosmos_streaming_weights(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Augment a Cosmos state dict with per-block fused self-attention QKV weights.

    Adds ``blocks.{i}.self_attn.qkv_proj.weight`` as
    ``cat([q_proj, k_proj, v_proj], dim=0)`` while preserving the original
    separate Q/K/V keys used by other paths.
    """
    weights = dict(state_dict)
    q_suffix = "self_attn.q_proj.weight"
    for q_key, q_weight in list(weights.items()):
        if not q_key.endswith(q_suffix):
            continue

        prefix = q_key[: -len(q_suffix)]
        fused_key = prefix + "self_attn.qkv_proj.weight"
        if fused_key in weights:
            weights[fused_key] = weights[fused_key].contiguous()
            continue

        k_key = prefix + "self_attn.k_proj.weight"
        v_key = prefix + "self_attn.v_proj.weight"
        if k_key not in weights or v_key not in weights:
            raise KeyError(f"Missing K/V weights for fused QKV key {fused_key!r}")

        weights[fused_key] = torch.cat(
            [q_weight, weights[k_key], weights[v_key]], dim=0
        ).contiguous()

    for key, weight in list(weights.items()):
        if key.endswith(".weight_prepared"):
            weights[key] = weight.contiguous()
            continue
        if key.endswith(_PREPARED_SUFFIXES):
            weights[f"{key}_prepared"] = weight.t().contiguous()

    return {k: v.contiguous() for k, v in weights.items()}
