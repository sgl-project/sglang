# SPDX-License-Identifier: Apache-2.0
"""Sequence Parallel helpers for post-training rollout code."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_reduce,
)


def should_do_sp_collective(batch) -> bool:
    return get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False)


def gather_stacked_latents_for_sp(
    pipeline_config,
    batch,
    stacked_latents: torch.Tensor,
) -> torch.Tensor:
    if not should_do_sp_collective(batch):
        return stacked_latents
    if stacked_latents.dim() < 2:
        return stacked_latents
    bsz, t_steps = stacked_latents.shape[0], stacked_latents.shape[1]
    flat_inputs = stacked_latents.flatten(0, 1).contiguous()
    gathered_flat_inputs = pipeline_config.gather_latents_for_sp(
        flat_inputs, batch=batch
    )
    return gathered_flat_inputs.unflatten(0, (bsz, t_steps))


def all_reduce_if_sp_sharded(batch, tensor: torch.Tensor) -> torch.Tensor:
    if not should_do_sp_collective(batch):
        return tensor
    tensor = tensor.to(get_local_torch_device())
    sequence_model_parallel_all_reduce(tensor)
    return tensor


def all_gather_if_sp_sharded(batch, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if not should_do_sp_collective(batch):
        return x
    x = x.to(get_local_torch_device()).contiguous()
    return sequence_model_parallel_all_gather(x, dim=dim)


def maybe_trim_sp_rope_seq_for_batch(batch, rope: torch.Tensor) -> torch.Tensor:
    raw = getattr(batch, "raw_latent_shape", None)
    if raw is None or len(raw) < 2:
        return rope
    target = int(raw[1])
    if rope.shape[0] > target:
        return rope[:target]
    return rope
