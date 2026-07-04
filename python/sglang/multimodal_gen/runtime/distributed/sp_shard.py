# SPDX-License-Identifier: Apache-2.0
"""Unified sequence-parallel shard / pad / gather helpers.

Layout invariant: padding always sits at the end of the LAST rank's local
chunk, so the ulysses-gathered sequence carries one contiguous pad block at its
global tail. `tail_attn_meta` then lets attention skip that block for free
(the pad becomes its own varlen segment - no repacking, no mask compute).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)

# Text shorter than this stays replicated instead of SP-sharded (see
# plan_text_strategy). 0 = always shard when legal; H100 bench showed sharding
# wins from trivial lengths on, so the knob exists only as an escape hatch.
_TEXT_SHARD_MIN = int(os.environ.get("SGLANG_SP_TEXT_SHARD_MIN", "0"))


@dataclass(frozen=True)
class SpShard:
    """Facts of one tail-padded even shard, shared by tensors of that stream."""

    orig_len: int  # real tokens (global)
    local_len: int  # per-rank chunk length (equal on every rank)
    num_pad: int  # pad tokens, all at the last rank's local tail
    sp_size: int
    sp_rank: int

    @property
    def local_pad(self) -> int:
        """Pad rows inside THIS rank's chunk (tail rows of the last rank)."""
        return self.num_pad if self.sp_rank == self.sp_size - 1 else 0

    @property
    def local_real_len(self) -> int:
        return self.local_len - self.local_pad


def plan_shard(seq_len: int) -> SpShard:
    """Shard math only; tensors are sliced separately via `shard_like`."""
    sp_size = get_sp_world_size()
    if sp_size <= 1:
        return SpShard(seq_len, seq_len, 0, 1, 0)
    local_len = (seq_len + sp_size - 1) // sp_size
    return SpShard(
        orig_len=seq_len,
        local_len=local_len,
        num_pad=local_len * sp_size - seq_len,
        sp_size=sp_size,
        sp_rank=get_sp_parallel_rank(),
    )


def shard_like(
    x: torch.Tensor, shard: SpShard, dim: int = 1, pad_mode: str = "zeros"
) -> torch.Tensor:
    """Apply a planned shard to one tensor (RoPE caches use the same plan as
    hidden states so their chunks stay aligned)."""
    if shard.sp_size <= 1:
        return x
    if shard.num_pad > 0:
        if pad_mode == "repeat_last":
            pad = x.narrow(dim, x.shape[dim] - 1, 1)
            pad = pad.expand(
                *[shard.num_pad if i == dim else -1 for i in range(x.dim())]
            )
            x = torch.cat([x, pad], dim=dim)
        else:
            # F.pad pads dims last-to-first: (left, right) pairs from dim -1.
            pads = [0, 0] * (x.dim() - 1 - dim) + [0, shard.num_pad]
            x = F.pad(x, pads)
    return x.narrow(dim, shard.sp_rank * shard.local_len, shard.local_len)


def shard_seq(
    x: torch.Tensor, dim: int = 1, pad_mode: str = "zeros"
) -> tuple[torch.Tensor, SpShard]:
    shard = plan_shard(x.shape[dim])
    return shard_like(x, shard, dim=dim, pad_mode=pad_mode), shard


def gather_seq(local: torch.Tensor, orig_len: int, dim: int = 1) -> torch.Tensor:
    """All-gather an SP-sharded stream and trim the tail padding."""
    if get_sp_world_size() <= 1:
        return local
    full = sequence_model_parallel_all_gather(local.contiguous(), dim=dim)
    if full.shape[dim] > orig_len:
        full = full.narrow(dim, 0, orig_len)
    return full


def shard_seq_prefix(
    x: torch.Tensor, prefix_len: int, shard: SpShard, dim: int = 0
) -> torch.Tensor:
    """Shard only the leading ``prefix_len`` rows (e.g. the text segment of a
    joint RoPE cache) with an existing plan; the remainder is kept as-is."""
    rest = x.shape[dim] - prefix_len
    return torch.cat(
        [
            shard_like(x.narrow(dim, 0, prefix_len), shard, dim=dim),
            x.narrow(dim, prefix_len, rest),
        ],
        dim=dim,
    )


def join_seqs(
    prefix: torch.Tensor, body: torch.Tensor, local_pad: int, dim: int = 1
) -> torch.Tensor:
    """Concatenate ``[prefix, body]`` for joint attention, relocating the
    prefix's ``local_pad`` tail rows behind the body.

    Why: the shard pads the *text* chunk, but the local joint layout is
    [text, image] - after the ulysses gather that pad would sit mid-sequence
    ([... txt_last, PAD, img_last]), and attention can only skip a mid-sequence
    hole by repacking q/k/v. With the pad relocated behind the image, the
    gathered padding forms one global-tail block that the zero-copy varlen
    path (tail_attn_meta) skips for free. Same copy volume as a plain cat.
    """
    if local_pad > 0:
        real = prefix.shape[dim] - local_pad
        return torch.cat(
            [
                prefix.narrow(dim, 0, real),
                body,
                prefix.narrow(dim, real, local_pad),
            ],
            dim=dim,
        )
    return torch.cat([prefix, body], dim=dim)


def split_seqs(
    joint: torch.Tensor, prefix_len: int, local_pad: int, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``join_seqs``: recover ``(prefix, body)`` from the joint
    output, with the pad rows rejoining the prefix tail so the residual text
    stream keeps its per-rank shape (their content is garbage and is excluded
    from every attention, so carrying them along is free)."""
    total = joint.shape[dim]
    if local_pad > 0:
        real = prefix_len - local_pad
        body_end = total - local_pad
        prefix = torch.cat(
            [joint.narrow(dim, 0, real), joint.narrow(dim, body_end, local_pad)],
            dim=dim,
        )
        return prefix, joint.narrow(dim, real, body_end - real)
    return (
        joint.narrow(dim, 0, prefix_len),
        joint.narrow(dim, prefix_len, total - prefix_len),
    )


def should_shard_text(txt_len: int) -> bool:
    """True when the joint-attention text stream should be SP-sharded here
    (see plan_text_strategy for the policy)."""
    return get_sp_world_size() > 1 and plan_text_strategy(txt_len) == "shard"


def tail_attn_meta(
    shard: SpShard,
    batch_size: int,
    device: torch.device,
    image_seq_len: int = 0,
) -> dict | None:
    """Per-request attention meta for a tail-padded shard: `cu_seqlens_tail`
    splits each batch row into [valid | pad] varlen segments over the gathered
    layout, so USPAttention runs varlen FA on the padded q/k/v with zero
    repacking. Built once per request, reused by every block."""
    if shard.sp_size <= 1 or shard.num_pad == 0:
        return None
    seq = shard.sp_size * (shard.local_len + image_seq_len)
    valid = seq - shard.num_pad
    row = torch.tensor([valid, shard.num_pad], dtype=torch.int32, device=device)
    seglens = row.repeat(batch_size)
    cu_seqlens = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seglens, dim=0)
    return {
        "pad_start": valid,
        "pad_end": seq,
        "local_pad": shard.local_pad,
        "cu_seqlens_tail": cu_seqlens,
        "max_seqlen_tail": valid,
    }


def plan_text_strategy(txt_len: int) -> str:
    """Choose "shard" or "replicate" for the joint-attention text stream.
    Padded sharding needs the varlen masked path (unsupported under ring) ->
    replicate there. Measured default is always-shard; the length threshold
    (SGLANG_SP_TEXT_SHARD_MIN) is only an env escape hatch."""
    sp_size = get_sp_world_size()
    if sp_size <= 1:
        return "replicate"
    if txt_len % sp_size != 0 and get_ring_parallel_world_size() > 1:
        return "replicate"
    if txt_len < _TEXT_SHARD_MIN:
        return "replicate"
    return "shard"
