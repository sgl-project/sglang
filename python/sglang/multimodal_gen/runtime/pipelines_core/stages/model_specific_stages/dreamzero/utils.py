# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)


def _sp_shard_lengths(seq_len: int) -> list[int]:
    sp_size = get_ulysses_parallel_world_size()
    base, extra = divmod(seq_len, sp_size)
    return [base + (rank < extra) for rank in range(sp_size)]


def _sp_local_slice(seq_lens: list[int]) -> slice:
    rank = get_ulysses_parallel_rank()
    begin = sum(seq_lens[:rank])
    return slice(begin, begin + seq_lens[rank])


def sp_shard_sequence(
    seqs: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if seqs.shape[1] != freqs.shape[0]:
        raise ValueError(
            "DreamZero SP requires matching sequence and frequency lengths, got "
            f"{seqs.shape[1]} and {freqs.shape[0]}"
        )
    if get_ulysses_parallel_world_size() == 1:
        return seqs, freqs, [seqs.shape[1]]

    seq_lens = _sp_shard_lengths(seqs.shape[1])
    local = _sp_local_slice(seq_lens)
    return seqs[:, local], freqs[local], seq_lens


def sp_shard_tensor(tensor: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    if get_ulysses_parallel_world_size() == 1:
        return tensor
    if tensor.shape[1] != sum(seq_lens):
        raise ValueError(
            "DreamZero SP tensor length does not match shard plan: "
            f"got {tensor.shape[1]}, expected {sum(seq_lens)}"
        )
    return tensor[:, _sp_local_slice(seq_lens)]


def sp_gather_tensor(tensor: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    sp_size = get_ulysses_parallel_world_size()
    if sp_size == 1:
        return tensor
    rank = get_ulysses_parallel_rank()
    if tensor.shape[1] != seq_lens[rank]:
        raise ValueError(
            "DreamZero local tensor length does not match shard plan: "
            f"got {tensor.shape[1]}, expected {seq_lens[rank]}"
        )

    max_seq_len = max(seq_lens)
    if tensor.shape[1] < max_seq_len:
        pad = tensor.new_zeros(
            tensor.shape[0], max_seq_len - tensor.shape[1], *tensor.shape[2:]
        )
        tensor = torch.cat([tensor, pad], dim=1)

    gathered = [torch.empty_like(tensor) for _ in range(sp_size)]
    torch.distributed.all_gather(
        gathered,
        tensor.contiguous(),
        group=get_sp_group().ulysses_group,
    )
    return torch.cat(
        [chunk[:, :seq_len] for chunk, seq_len in zip(gathered, seq_lens)],
        dim=1,
    )


def infer_dreamzero_batch_size(tensors: Iterable[Any]) -> int:
    for value in tensors:
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Cannot infer DreamZero batch size")
