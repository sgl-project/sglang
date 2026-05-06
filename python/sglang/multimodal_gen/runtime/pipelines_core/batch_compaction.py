# SPDX-License-Identifier: Apache-2.0
"""Batch compaction helpers for cancelled generation requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.cancellation import (
    RequestCancelledError,
    get_cancel_reason,
    is_request_cancelled,
    raise_if_cancelled,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_world_group,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

# Req fields indexed by prompt/request.
PROMPT_BATCH_FIELDS = (
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_attention_mask",
    "negative_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
    "pooled_embeds",
    "neg_pooled_embeds",
    "clip_embedding_pos",
    "clip_embedding_neg",
    "prompt_seq_lens",
    "negative_prompt_seq_lens",
    "audio_prompt_embeds",
    "negative_audio_prompt_embeds",
)

# Req fields indexed by generated output.
OUTPUT_BATCH_FIELDS = (
    "seeds",
    "generator",
    "latents",
    "y",
    "latent_ids",
    "audio_latents",
    "audio_noise",
    "noise_pred",
    "image_latent",
    "condition_image_latent_ids",
)

# Denoising context fields.
CTX_PROMPT_FIELDS = (
    "image_kwargs",
    "pos_cond_kwargs",
    "neg_cond_kwargs",
)

CTX_OUTPUT_FIELDS = (
    "extra_step_kwargs",
    "guidance",
    "latents",
    "audio_latents",
    "denoise_mask",
    "clean_latent",
    "last_denoised_video",
    "last_denoised_audio",
    "current_latents",
    "kv_cache1",
    "crossattn_cache",
    "trajectory_latents",
    "trajectory_audio_latents",
)

SCHEDULER_BATCH_STATE_FIELDS = (
    "model_outputs",
    "last_sample",
    "ets",
    "derivatives",
    "cur_sample",
    "prev_sample",
)


@dataclass(slots=True)
class DynamicBatchLayout:
    """Dynamic-batch request layout stored in Req.extra.

    `active_request_indices` points into `original_request_ids`, so the original
    request order is preserved after repeated compactions.
    """

    original_request_ids: list[str]
    active_request_indices: list[int]
    num_outputs_per_request: list[int]

    @classmethod
    def from_req(cls, batch: Req) -> "DynamicBatchLayout | None":
        extra = batch.extra if isinstance(batch.extra, dict) else {}
        original_request_ids = extra.get("dynamic_batch_original_request_ids")
        if original_request_ids is None:
            original_request_ids = extra.get("dynamic_batch_request_ids")
        if not original_request_ids:
            return None

        original_request_ids = [str(request_id) for request_id in original_request_ids]
        active_request_indices = extra.get("dynamic_batch_active_request_indices")
        if active_request_indices is None:
            active_request_indices = list(range(len(original_request_ids)))
        active_request_indices = [int(index) for index in active_request_indices]

        num_outputs_per_request = extra.get("dynamic_batch_num_outputs_per_request")
        if not num_outputs_per_request or len(num_outputs_per_request) != len(
            original_request_ids
        ):
            num_outputs_per_request = [
                int(batch.num_outputs_per_prompt)
                for _ in range(len(original_request_ids))
            ]
        else:
            num_outputs_per_request = [int(count) for count in num_outputs_per_request]

        if len(set(active_request_indices)) != len(active_request_indices) or any(
            index < 0 or index >= len(original_request_ids)
            for index in active_request_indices
        ):
            raise ValueError(
                "Invalid dynamic_batch_active_request_indices: "
                f"{active_request_indices}"
            )

        return cls(
            original_request_ids=original_request_ids,
            active_request_indices=active_request_indices,
            num_outputs_per_request=num_outputs_per_request,
        )

    @property
    def active_request_ids(self) -> list[str]:
        return [
            self.original_request_ids[index] for index in self.active_request_indices
        ]

    @property
    def active_num_outputs(self) -> int:
        return sum(
            self.num_outputs_per_request[index] for index in self.active_request_indices
        )

    def kept_output_indices(self, keep_prompt_indices: list[int]) -> list[int]:
        """Return output indices for kept prompt indices.

        Prompt indices and output indices differ when a request produces more
        than one output.
        """
        keep_prompt_indices = set(keep_prompt_indices)
        keep_output_indices: list[int] = []
        output_start = 0
        for local_index, original_index in enumerate(self.active_request_indices):
            req_output_count = self.num_outputs_per_request[original_index]
            if local_index in keep_prompt_indices:
                keep_output_indices.extend(
                    range(output_start, output_start + req_output_count)
                )
            output_start += req_output_count
        return keep_output_indices


@dataclass(slots=True)
class BatchCompactionContext:
    latents: torch.Tensor
    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]
    neg_cond_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)
    guidance: torch.Tensor | None = None
    scheduler: Any | None = None
    trajectory_latents: list[torch.Tensor] = field(default_factory=list)
    current_latents: torch.Tensor | None = None
    kv_cache1: list | None = None
    crossattn_cache: list | None = None


@dataclass(slots=True)
class BatchCompactionResult:
    old_request_count: int = 0
    new_request_count: int = 0

    @property
    def compacted(self) -> bool:
        return self.old_request_count > self.new_request_count


def _is_shape_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, int) for item in value)


def _slice_tensor_first_dim(value: torch.Tensor, indices: list[int]) -> torch.Tensor:
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=value.device)
    return value.index_select(0, index_tensor)


def _has_nested_batch_axis(value: Any, expected_first_dim: int) -> bool:
    """Check whether a nested value contains a batch-aligned child."""
    if isinstance(value, torch.Tensor):
        return value.ndim > 0 and int(value.shape[0]) == expected_first_dim
    if isinstance(value, dict):
        return any(
            _has_nested_batch_axis(item, expected_first_dim) for item in value.values()
        )
    if isinstance(value, list):
        if _is_shape_tuple(value):
            return False
        if len(value) == expected_first_dim:
            return True
        return any(_has_nested_batch_axis(item, expected_first_dim) for item in value)
    if isinstance(value, tuple):
        if _is_shape_tuple(value):
            return False
        return any(_has_nested_batch_axis(item, expected_first_dim) for item in value)
    return False


def slice_batch_axis(value: Any, indices: list[int], expected_first_dim: int) -> Any:
    """Slice nested values along the current batch axis.

    Tensors are sliced when dim 0 matches `expected_first_dim`. Lists with the
    same length are sliced as batches unless they contain nested batched values.
    Shape tuples are left unchanged.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == expected_first_dim:
            return _slice_tensor_first_dim(value, indices)
        return value

    if isinstance(value, dict):
        return {
            key: slice_batch_axis(item, indices, expected_first_dim)
            for key, item in value.items()
        }

    if isinstance(value, list):
        if len(value) == expected_first_dim:
            if any(_has_nested_batch_axis(item, expected_first_dim) for item in value):
                return [
                    slice_batch_axis(item, indices, expected_first_dim)
                    for item in value
                ]
            return [value[index] for index in indices]
        return [slice_batch_axis(item, indices, expected_first_dim) for item in value]

    if isinstance(value, tuple):
        if _is_shape_tuple(value):
            return value
        return tuple(
            slice_batch_axis(item, indices, expected_first_dim) for item in value
        )

    return value


def _cancel_mask_across_workers(
    request_ids: list[str],
    server_args: Any,
    device: torch.device,
) -> list[bool]:
    """All-reduce cancel bits so every rank compacts the same requests."""
    cancel_mask = [
        1 if is_request_cancelled(request_id, server_args) else 0
        for request_id in request_ids
    ]
    if not cancel_mask:
        return []

    try:
        world_group = get_world_group()
    except AssertionError:
        return [bool(item) for item in cancel_mask]

    mask_tensor = torch.tensor(cancel_mask, dtype=torch.int32, device=device)
    if world_group.world_size > 1:
        mask_tensor = world_group.all_reduce(mask_tensor)
    return [bool(item) for item in mask_tensor.cpu().tolist()]


def _context_device(ctx: Any) -> torch.device:
    latents = getattr(ctx, "latents", None)
    if isinstance(latents, torch.Tensor):
        return latents.device
    return get_local_torch_device()


def _compact_scheduler_state(
    scheduler: Any,
    keep_output_indices: list[int],
    old_output_count: int,
) -> None:
    """Slice scheduler history buffers with the output batch."""
    if scheduler is None:
        return
    for name in SCHEDULER_BATCH_STATE_FIELDS:
        if hasattr(scheduler, name):
            setattr(
                scheduler,
                name,
                slice_batch_axis(
                    getattr(scheduler, name),
                    keep_output_indices,
                    old_output_count,
                ),
            )


def _slice_attr(obj: Any, name: str, indices: list[int], size: int) -> None:
    if hasattr(obj, name):
        setattr(obj, name, slice_batch_axis(getattr(obj, name), indices, size))


def compact_batch(
    batch: Req,
    ctx: Any,
    server_args: Any,
) -> BatchCompactionResult:
    """Remove cancelled requests from an active batch.

    Prompt-aligned state is sliced by request index. Output-aligned state is
    sliced by generated-output index. Raises when no active request remains.
    """
    layout = DynamicBatchLayout.from_req(batch)
    if layout is None:
        return BatchCompactionResult()
    if len(layout.active_request_indices) <= 1:
        raise_if_cancelled(batch, server_args)
        return BatchCompactionResult(
            old_request_count=len(layout.active_request_indices),
            new_request_count=len(layout.active_request_indices),
        )

    cancel_mask = _cancel_mask_across_workers(
        layout.active_request_ids,
        server_args,
        _context_device(ctx),
    )
    if not any(cancel_mask):
        return BatchCompactionResult(
            old_request_count=len(layout.active_request_indices),
            new_request_count=len(layout.active_request_indices),
        )

    first_cancelled_id = next(
        request_id
        for request_id, cancelled in zip(layout.active_request_ids, cancel_mask)
        if cancelled
    )
    keep_prompt_indices = [
        index for index, cancelled in enumerate(cancel_mask) if not cancelled
    ]
    if not keep_prompt_indices:
        raise RequestCancelledError(
            request_id=first_cancelled_id,
            reason=get_cancel_reason(first_cancelled_id, server_args),
        )

    old_prompt_count = len(layout.active_request_indices)
    old_output_count = layout.active_num_outputs
    keep_output_indices = layout.kept_output_indices(keep_prompt_indices)
    next_active_indices = [
        layout.active_request_indices[index] for index in keep_prompt_indices
    ]

    # Prompt-aligned state.
    batch.prompt = slice_batch_axis(
        batch.prompt,
        keep_prompt_indices,
        old_prompt_count,
    )
    for name in PROMPT_BATCH_FIELDS:
        _slice_attr(batch, name, keep_prompt_indices, old_prompt_count)

    # Output-aligned state.
    for name in OUTPUT_BATCH_FIELDS:
        _slice_attr(batch, name, keep_output_indices, old_output_count)

    # Request index map used by output splitting.
    batch.extra["dynamic_batch_request_ids"] = [
        layout.original_request_ids[index] for index in next_active_indices
    ]
    batch.extra["dynamic_batch_active_request_indices"] = next_active_indices
    if "dynamic_batch_seeds" in batch.extra:
        batch.extra["dynamic_batch_seeds"] = slice_batch_axis(
            batch.extra["dynamic_batch_seeds"],
            keep_prompt_indices,
            old_prompt_count,
        )
    if "dynamic_batch_output_paths" in batch.extra:
        batch.extra["dynamic_batch_output_paths"] = slice_batch_axis(
            batch.extra["dynamic_batch_output_paths"],
            keep_output_indices,
            old_output_count,
        )

    for name in CTX_PROMPT_FIELDS:
        _slice_attr(ctx, name, keep_prompt_indices, old_prompt_count)
    for name in CTX_OUTPUT_FIELDS:
        _slice_attr(ctx, name, keep_output_indices, old_output_count)

    # Mirror compacted latents back to Req.
    if hasattr(ctx, "latents"):
        batch.latents = ctx.latents
    if hasattr(ctx, "audio_latents"):
        batch.audio_latents = ctx.audio_latents

    _compact_scheduler_state(
        getattr(ctx, "scheduler", None),
        keep_output_indices,
        old_output_count,
    )
    _compact_scheduler_state(
        getattr(ctx, "audio_scheduler", None),
        keep_output_indices,
        old_output_count,
    )

    if isinstance(getattr(ctx, "latents", None), torch.Tensor):
        batch.raw_latent_shape = ctx.latents.shape
    if isinstance(getattr(batch, "audio_latents", None), torch.Tensor):
        batch.raw_audio_latent_shape = batch.audio_latents.shape

    return BatchCompactionResult(
        old_request_count=old_prompt_count,
        new_request_count=len(next_active_indices),
    )
