# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)


def shard_sequence_parallel_sequence(
    seqs: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard video tokens and their RoPE frequencies across the SP group."""
    sp_size = get_sp_world_size()
    if sp_size == 1:
        return seqs, freqs
    if seqs.shape[1] != freqs.shape[0]:
        raise ValueError(
            "DreamZero SP requires matching sequence and frequency lengths, got "
            f"{seqs.shape[1]} and {freqs.shape[0]}"
        )
    if seqs.shape[1] % sp_size != 0:
        raise ValueError(
            f"DreamZero sequence length {seqs.shape[1]} is not divisible by "
            f"sequence parallel size {sp_size}"
        )

    seq_len_per_rank = seqs.shape[1] // sp_size
    begin = get_sp_parallel_rank() * seq_len_per_rank
    end = begin + seq_len_per_rank
    return seqs[:, begin:end], freqs[begin:end]


def shard_sequence_parallel_time_embedding(
    e: torch.Tensor,
    sp_seq_len: int,
    action_register_length: int | None,
) -> torch.Tensor:
    """Shard video time embeddings while replicating the action register."""
    sp_size = get_sp_world_size()
    if sp_size == 1:
        return e

    video_seq_len = e.shape[1] - (action_register_length or 0)
    if video_seq_len != sp_seq_len * sp_size:
        raise ValueError(
            "DreamZero SP time embedding length does not match the sharded "
            f"video sequence: video={video_seq_len}, local={sp_seq_len}, "
            f"sp_size={sp_size}"
        )

    begin = get_sp_parallel_rank() * sp_seq_len
    end = begin + sp_seq_len
    e_ret = e[:, begin:end]
    if action_register_length is not None:
        e_ret = torch.cat([e_ret, e[:, -action_register_length:]], dim=1)
    return e_ret


def remove_redundant_action_register(
    tensor: torch.Tensor, action_register_length: int
) -> torch.Tensor:
    """Keep one copy of an action register replicated by SP all-gather."""
    if tensor.dim() < 3:
        raise ValueError("DreamZero gathered SP tensor must have at least 3 dimensions")
    if action_register_length <= 0 or action_register_length > tensor.shape[2]:
        raise ValueError(
            f"Invalid action register length {action_register_length} for "
            f"sequence length {tensor.shape[2]}"
        )

    sp_size, batch, seq_len_with_action = tensor.shape[:3]
    seq_len_without_action = seq_len_with_action - action_register_length
    other_dims = tensor.shape[3:]
    front = tensor[: sp_size - 1, :, :seq_len_without_action]
    front = front.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
        batch, (sp_size - 1) * seq_len_without_action, *other_dims
    )
    last = tensor[sp_size - 1].reshape(batch, seq_len_with_action, *other_dims)
    return torch.cat([front, last], dim=1)


def gather_full_sequence_parallel_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a local sequence and retain the leading SP-rank dimension."""
    sp_size = get_sp_world_size()
    if sp_size == 1:
        return tensor.unsqueeze(0)
    gathered = sequence_model_parallel_all_gather(tensor.contiguous(), dim=0)
    return gathered.reshape(sp_size, *tensor.shape)


def flatten_dim_sp_into_sequence(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten [sp, batch, sequence, ...] into the full sequence layout."""
    if tensor.dim() < 3:
        raise ValueError("DreamZero gathered SP tensor must have at least 3 dimensions")
    sp_size, batch, seq_len = tensor.shape[:3]
    return tensor.permute(1, 0, 2, *range(3, tensor.dim())).reshape(
        batch, seq_len * sp_size, *tensor.shape[3:]
    )


def infer_dreamzero_batch_size(
    inputs: Mapping[str, Any],
    *,
    error_message: str = "Cannot infer DreamZero batch size",
) -> int:
    for value in inputs.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError(error_message)


def infer_dreamzero_model_input_batch_size(model_inputs: Mapping[str, Any]) -> int:
    for key in (
        "images",
        "videos",
        "state",
        "text",
        "text_negative",
    ):
        value = model_inputs.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Cannot infer DreamZero batch size from normalized input")
