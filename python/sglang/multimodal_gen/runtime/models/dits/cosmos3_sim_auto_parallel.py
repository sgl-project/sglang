# SPDX-License-Identifier: Apache-2.0
"""Strict Ulysses sequence parallelism for Cosmos3-Nano-Sim-Auto.

The replay plan addresses tokens in chunk coordinates and attention is
independent per head, so Ulysses changes neither the plan nor the kernels. The
GEN layers run on a padded token shard of the chunk; one all-to-all per layer
gives every rank all of the chunk's queries, keys and values for one contiguous
head group, the rank attends them against the same head group of the ring and
the captions, and a second all-to-all returns the output to the token shard.
Ring and caption K/V therefore hold every token for the rank's head group, and
the ring geometry is the single-GPU one.
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto_attention import (
    ReplayMask,
    replay_attention,
)

KVPair = tuple[torch.Tensor, torch.Tensor]


class UlyssesGroup(msgspec.Struct, frozen=True):
    """The rank's place in the Ulysses group; head groups follow rank order."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 2 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid Ulysses rank {self.rank} of {self.world_size}.")

    def local_heads(self, heads: int) -> int:
        if heads % self.world_size:
            raise ValueError(
                f"{heads} heads are not divisible by the Ulysses degree {self.world_size}."
            )
        return heads // self.world_size

    def head_slice(self, heads: int) -> slice:
        """This rank's contiguous group of ``heads`` (TP-local) heads."""
        size = self.local_heads(heads)
        return slice(self.rank * size, (self.rank + 1) * size)

    def shard(self, num_tokens: int) -> ChunkShard:
        return ChunkShard(group=self, num_tokens=num_tokens)


class ChunkShard(msgspec.Struct, frozen=True):
    """One forward's token split: ``ceil(num_tokens / U)`` tokens per rank, padded at the end."""

    group: UlyssesGroup
    num_tokens: int

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError("A Ulysses shard needs at least one token.")

    @property
    def local_tokens(self) -> int:
        return -(-self.num_tokens // self.group.world_size)

    @property
    def padded_tokens(self) -> int:
        return self.local_tokens * self.group.world_size

    def split(self, tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        """This rank's ``local_tokens`` along ``dim``; the padding repeats the last entry.

        Repeating keeps RoPE positions valid; padded hidden rows never act as
        keys and their outputs are dropped by :meth:`gather`.
        """
        if tensor.shape[dim] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens to shard, got {tensor.shape[dim]}."
            )
        start = self.group.rank * self.local_tokens
        padded = _pad_repeat_last(tensor, self.padded_tokens, dim=dim)
        return padded.narrow(dim, start, self.local_tokens).contiguous()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``[B, local_tokens, ...]`` shards into ``[B, num_tokens, ...]``."""
        self._check_local(tensor)
        gathered = sequence_model_parallel_all_gather(tensor.contiguous(), dim=1)
        return gathered[:, : self.num_tokens]

    def heads_from_tokens(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, local_tokens, H, D]`` Q/K/V -> ``[B, num_tokens, H / U, D]`` of this rank's head group.

        Heads split into contiguous groups in rank order on both sides, so
        query group ``r`` still reads KV group ``r``.
        """
        for tensor in (q, k, v):
            self._check_local(tensor)
        q, k, v = (
            _usp_input_all_to_all(tensor, head_dim=2)[:, : self.num_tokens]
            for tensor in (q, k, v)
        )
        return q, k, v

    def tokens_from_heads(self, output: torch.Tensor) -> torch.Tensor:
        """``[B, num_tokens, H / U, D]`` attention output -> ``[B, local_tokens, H, D]``."""
        if output.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} attention rows, got {output.shape[1]}."
            )
        pad = self.padded_tokens - self.num_tokens
        if pad:
            output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad))
        return _usp_output_all_to_all(output, head_dim=2)

    def _check_local(self, tensor: torch.Tensor) -> None:
        if tensor.shape[1] != self.local_tokens:
            raise ValueError(
                f"Expected {self.local_tokens} local tokens ({self.num_tokens} tokens / "
                f"Ulysses {self.group.world_size}), got {tensor.shape[1]}."
            )


class SpCheck(msgspec.Struct):
    """Debug tally (``SGLANG_DIFFUSION_DEBUG_COSMOS3_MULTIVIEW_SP_CHECK``) over one forward's layers."""

    layers: int = 0
    identical: int = 0
    max_diff: float = 0.0

    def compare(
        self,
        shard: ChunkShard,
        *,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: ReplayMask,
        out: torch.Tensor,
    ) -> None:
        """Full-head attention sliced to this rank's heads must equal ``out`` bitwise (same kernel, same tokens)."""
        full = replay_attention(
            _all_heads(q, shard.group),
            _all_heads(keys, shard.group),
            _all_heads(values, shard.group),
            mask,
        )
        mine = full[:, :, shard.group.head_slice(full.shape[2])]
        self.layers += 1
        self.identical += int(torch.equal(mine, out))
        self.max_diff = max(
            self.max_diff, float((mine.float() - out.float()).abs().max())
        )

    def summary(self, group: UlyssesGroup) -> str:
        return (
            f"Cosmos3-Nano-Sim-Auto SP check (rank {group.rank}/{group.world_size}): "
            f"{self.identical}/{self.layers} layers bitwise identical to full-head attention, "
            f"max|diff| {self.max_diff:.3g}"
        )


def _all_heads(tensor: torch.Tensor, group: UlyssesGroup) -> torch.Tensor:
    """``[B, S, h / U, D]`` of every rank concatenated into ``[B, S, h, D]`` (rank order = head order)."""
    parts = [torch.empty_like(tensor) for _ in range(group.world_size)]
    dist.all_gather(parts, tensor.contiguous(), group=get_sp_group().ulysses_group)
    return torch.cat(parts, dim=2)


def _pad_repeat_last(tensor: torch.Tensor, length: int, *, dim: int) -> torch.Tensor:
    pad = length - tensor.shape[dim]
    if pad == 0:
        return tensor
    last = tensor.narrow(dim, tensor.shape[dim] - 1, 1)
    return torch.cat([tensor, *([last] * pad)], dim=dim)


def current_ulysses_group() -> UlyssesGroup | None:
    """The worker's Ulysses group, or ``None`` at degree 1."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return None
    return UlyssesGroup(rank=get_ulysses_parallel_rank(), world_size=world_size)


def local_head_group(
    layer_kv: Sequence[KVPair], group: UlyssesGroup | None
) -> list[KVPair]:
    """The rank's head group of per-layer ``[1, tokens, H_kv, D]`` K/V (identity without Ulysses)."""
    if group is None:
        return list(layer_kv)
    heads = group.head_slice(layer_kv[0][0].shape[2])
    # Copies, so the full caption K/V is not kept alive by views.
    return [
        (key[:, :, heads].contiguous(), value[:, :, heads].contiguous())
        for key, value in layer_kv
    ]
