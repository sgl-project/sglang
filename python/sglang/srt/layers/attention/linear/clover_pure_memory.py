# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Adapted from Diffuspace/Clover-35B-A3B-Preview modeling_clover_moe.py.
import torch


def bd_gdn_local_memory(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    *,
    clean_ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    active_valid_mask: torch.Tensor,
    scale: float,
    softmax_logit_scale: torch.Tensor,
    softmax_history_bias: torch.Tensor,
    softmax_history_c: torch.Tensor,
) -> torch.Tensor:
    """Clover bidirectional softmax preview with a read-only clean-state bucket.

    Normalized q/k determine block attention logits without beta. The history
    bucket logit is bias + history_c * log(valid_block_size / 32), and its
    value is the gate-decayed recurrent readout from clean [B, H, V, K] state.
    """
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(f"Expected query shape [1, T, H, K], got {query.shape}")
    num_tokens = query.shape[1]
    batch_size = cache_indices.numel()
    if num_tokens % batch_size != 0:
        raise ValueError(
            f"BD-GDN token count {num_tokens} is not divisible by batch {batch_size}"
        )
    block_size = num_tokens // batch_size
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    num_value_heads = value.shape[2]
    if num_value_heads % query.shape[2] or num_value_heads % key.shape[2]:
        raise ValueError("BD-GDN value heads must be divisible by query and key heads")
    query = query.repeat_interleave(num_value_heads // query.shape[2], dim=2)
    key = key.repeat_interleave(num_value_heads // key.shape[2], dim=2)

    valid = active_valid_mask.reshape(batch_size, block_size)
    valid_float = valid[:, :, None].float()
    query_float = query.reshape(
        batch_size, block_size, num_value_heads, query.shape[-1]
    ).float()
    key_float = key.reshape(
        batch_size, block_size, num_value_heads, key.shape[-1]
    ).float()
    value_float = value.reshape(
        batch_size, block_size, num_value_heads, value.shape[-1]
    ).float()
    query_float = query_float * torch.rsqrt(
        query_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    key_float = key_float * torch.rsqrt(
        key_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )

    safe_cache_indices = cache_indices.to(torch.long).clamp_min(0)
    clean_state = clean_ssm_states.index_select(0, safe_cache_indices).float()
    active_requests = valid.any(dim=1)
    clean_state = clean_state * active_requests[:, None, None, None].float()

    scaled_query = query_float * scale
    clean_output = torch.einsum(
        "blhk,bhvk->blhv",
        scaled_query,
        clean_state,
    )
    decay = g.reshape(batch_size, block_size, num_value_heads).float().exp()
    history = clean_output * decay[..., None]

    for name, scalar in (
        ("logit_scale", softmax_logit_scale),
        ("history_bias", softmax_history_bias),
        ("history_c", softmax_history_c),
    ):
        if scalar.numel() != num_value_heads:
            raise ValueError(f"Clover {name} must have one entry per value head")
    logit_scale = softmax_logit_scale.reshape(-1).float().to(query_float.device)
    history_bias = softmax_history_bias.reshape(-1).float().to(query_float.device)
    history_c = softmax_history_c.reshape(-1).float().to(query_float.device)
    # [B, H, L(query), S(key)] logits over unscaled l2-normalized q/k.
    logits = torch.einsum("blhk,bshk->bhls", query_float, key_float)
    logits = logits * logit_scale.view(1, -1, 1, 1)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    history_logit = history_bias.view(1, -1, 1, 1).expand(batch_size, -1, block_size, 1)
    valid_block_size = valid.sum(dim=-1).clamp_min(1).float()
    block_size_log_ratio = torch.log(valid_block_size / 32.0)
    history_logit = history_logit + (
        history_c.view(1, -1, 1, 1) * block_size_log_ratio.view(batch_size, 1, 1, 1)
    )
    weights = torch.softmax(torch.cat((logits, history_logit), dim=-1), dim=-1)
    local_output = torch.einsum(
        "bhls,bshv->blhv",
        weights[..., :-1],
        value_float * valid_float[..., None],
    )
    history_weight = weights[..., -1].permute(0, 2, 1)
    output = history_weight[..., None] * history + local_output
    output = output * valid_float[..., None]
    return output.to(query.dtype).reshape(
        1, num_tokens, num_value_heads, value.shape[-1]
    )
