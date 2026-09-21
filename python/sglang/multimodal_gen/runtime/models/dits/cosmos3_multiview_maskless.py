# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Maskless multiview GEN attention for Cosmos3 exports with ``backend: maskless``.

The ``decomposed`` scope is computed as unmasked varlen dense passes merged by
log-sum-exp instead of one masked kernel over ``[UND | GEN]``:

* **same view**: every ``(view axis, view)`` group over all of its frames, the
  control item's tokens and the target's together;
* **cross instant**: every frame instant across views, sensor tokens only
  (control tokens are sliced out), with LiDAR sweeps quantized onto the camera
  frame grid by maximum overlap;
* **gen->und**: each same-view group against the captions it reads (camera:
  its own caption; LiDAR: all captions or none), or every GEN token against
  the whole caption when the checkpoint packs a single one.

A query's own ``(view, frame)`` cell is in both sensor passes, so its keys carry
twice the softmax weight they have under the mask. That is the attention the
maskless exports were trained with; it is not a drop-in for the masked backends.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MASKLESS_BACKEND = "maskless"
# Varlen kernels that return the per-row log-sum-exp the merge needs. "auto" picks
# FA4 on Blackwell (sm100+), FA3 on Hopper (sm90) and torch's FA2 elsewhere; the
# three agree to bf16 rounding (measured 1 ulp on GB200 and H200), so this is a
# speed knob, not a semantics one.
MASKLESS_KERNELS = ("auto", "fa4", "fa3", "fa2")
MASKLESS_KERNEL_ENV_VAR = "SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_MASKLESS_KERNEL"
# LiDAR sweeps and camera frames live on separate view axes; a LiDAR sweep is a
# one-view item on axis 1 so its group is distinct from camera view 0.
_CAMERA_AXIS = 0
_LIDAR_AXIS = 1


def maskless_unavailable_reason(
    *,
    attention_scope: str,
    decomposed_temporal_window_seconds: float | None,
    control_attends_sensor: bool,
) -> str | None:
    """Why the decomposition cannot serve a layout, or ``None`` when it can."""
    if attention_scope not in ("decomposed", "same_view"):
        return (
            f"attention_scope={attention_scope!r} is not a partition of the GEN "
            "stream; the maskless backend expresses only 'decomposed' or 'same_view'."
        )
    if decomposed_temporal_window_seconds is not None:
        return (
            "decomposed_temporal_window_seconds must be null: the maskless backend "
            "attends whole frame instants, not a temporal window."
        )
    if not control_attends_sensor:
        return (
            "control_attends_sensor must be true: the same-view pass is symmetric, "
            "so a control item always reads its target."
        )
    return None


class MultiviewMasklessPlan(msgspec.Struct, frozen=True, eq=False):
    """Varlen partitions of one sample's GEN stream, tiled over the batch.

    Every gather indexes the flattened ``[B * gen_tokens]`` stream; ``None``
    means the groups already tile the stream in packed order. Offsets are int32
    cumulative lengths in the layout the varlen kernels take.
    """

    gen_tokens: int
    und_tokens: int
    batch_size: int
    same_view_gather: torch.Tensor | None
    same_view_offsets: torch.Tensor
    same_view_max_len: int
    cross_view_gather: torch.Tensor | None
    cross_view_offsets: torch.Tensor | None
    cross_view_max_len: int
    #: Query rows of the gen->und pass; ``None`` keys every GEN token per sample.
    caption_q_gather: torch.Tensor | None
    caption_q_offsets: torch.Tensor
    caption_q_max_len: int
    #: Key rows of the gen->und pass into the flattened UND stream.
    caption_kv_gather: torch.Tensor | None
    caption_kv_offsets: torch.Tensor
    caption_kv_max_len: int


def _cumulative_offsets(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    if lengths:
        offsets[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return offsets


def _partition(group_ids: torch.Tensor) -> tuple[torch.Tensor | None, list[int]]:
    """``(gather or None, group lengths)`` from per-token group ids."""
    gather = torch.argsort(group_ids, stable=True)
    _, inverse = torch.unique_consecutive(group_ids[gather], return_inverse=True)
    lengths = torch.bincount(inverse).tolist()
    identity = torch.equal(
        gather, torch.arange(group_ids.numel(), device=group_ids.device)
    )
    return (None if identity else gather), lengths


def _item_axis(item: MaskItem) -> int:
    return _LIDAR_AXIS if item.is_lidar else _CAMERA_AXIS


def _caption_run(
    item: MaskItem,
    view: int,
    caption_lengths: Sequence[int],
    lidar_attends_captions: bool,
    device: torch.device,
) -> torch.Tensor:
    """UND rows one same-view group reads under per-view captions."""
    starts = [0]
    for length in caption_lengths:
        starts.append(starts[-1] + length)
    if item.is_lidar:
        if not lidar_attends_captions:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.arange(starts[-1], device=device)
    if view >= len(caption_lengths):
        raise ValueError(
            f"Cosmos3 multiview camera view {view} has no caption: only "
            f"{len(caption_lengths)} per-view captions were packed."
        )
    return torch.arange(starts[view], starts[view + 1], device=device)


def build_multiview_maskless_plan(
    layout: MultiviewLayout,
    *,
    und_tokens: int,
    batch_size: int,
    device: torch.device,
) -> MultiviewMasklessPlan:
    """Partition ``layout.items`` for the three maskless passes."""
    reason = maskless_unavailable_reason(
        attention_scope=layout.attention_scope,
        decomposed_temporal_window_seconds=layout.decomposed_temporal_window_seconds,
        control_attends_sensor=layout.control_attends_sensor,
    )
    if reason is not None:
        raise ValueError(f"Cosmos3 multiview maskless backend: {reason}")
    items = layout.items
    if not items or all(item.is_control for item in items):
        raise ValueError(
            "Cosmos3 multiview maskless plan needs at least one target item."
        )
    if batch_size <= 0 or und_tokens <= 0:
        raise ValueError(
            "Cosmos3 multiview maskless plan needs positive batch_size and und_tokens: "
            f"batch_size={batch_size}, und_tokens={und_tokens}."
        )
    caption_lengths = tuple(layout.caption_lengths)
    if caption_lengths and sum(caption_lengths) != und_tokens:
        raise ValueError(
            "Cosmos3 multiview caption_lengths must partition the UND tokens: "
            f"lengths={list(caption_lengths)}, und_tokens={und_tokens}."
        )
    per_view_captions = len(caption_lengths) > 1
    gen_tokens = layout.gen_tokens
    # A sample owning one same-view group (a single camera, no LiDAR) has
    # nothing to reach across instants; folding it would only double-weight
    # every query's own instant.
    single_group = len({_item_axis(item) for item in items}) == 1 and all(
        item.num_views == 1 for item in items
    )
    fold_instants = layout.attention_scope == "decomposed" and not single_group
    # The camera item anchors the frame grid so a joint sample's camera tokens
    # keep the frame indices a camera-only sample gives them.
    anchor_rate = items[0].seconds_per_frame

    view_group: dict[tuple[int, int], int] = {}
    group_item: dict[int, tuple[MaskItem, int]] = {}
    view_ids: list[torch.Tensor] = []
    instant_ids: list[torch.Tensor] = []
    sensor_positions: list[torch.Tensor] = []
    caption_reader_positions: list[torch.Tensor] = []
    position = 0
    for item in items:
        frames = item.token_shape[0] // item.num_views
        spatial = item.token_shape[1] * item.token_shape[2]
        axis = _item_axis(item)
        ids = []
        for view in range(item.num_views):
            group = view_group.setdefault((axis, view), len(view_group))
            group_item.setdefault(group, (item, view))
            ids.append(group)
        view_ids.append(
            torch.tensor(ids, device=device).repeat_interleave(frames * spatial)
        )
        if fold_instants and not item.is_control:
            frame_ids = torch.arange(frames, device=device, dtype=torch.float64)
            # Midpoint quantization is maximum-overlap assignment of a latent
            # frame span onto the anchor grid; the epsilon settles exact ties.
            instants = torch.floor(
                (frame_ids + 0.5) * (item.seconds_per_frame / anchor_rate) + 1e-6
            ).long()
            instant_ids.append(
                instants.repeat_interleave(spatial).repeat(item.num_views)
            )
            sensor_positions.append(
                torch.arange(position, position + item.num_tokens, device=device)
            )
        if not (item.is_lidar and not layout.lidar_attends_captions):
            caption_reader_positions.append(
                torch.arange(position, position + item.num_tokens, device=device)
            )
        position += item.num_tokens
    if not caption_reader_positions:
        raise ValueError(
            "Cosmos3 multiview maskless plan: no GEN token reads a caption; "
            "generation without text conditioning is not an attention this builds."
        )

    same_view_gather, same_view_lens = _partition(torch.cat(view_ids))

    cross_view_gather = cross_view_offsets = None
    cross_view_lens: list[int] = []
    if instant_ids:
        sensor_index = torch.cat(sensor_positions)
        order, cross_view_lens = _partition(torch.cat(instant_ids))
        cross_view_gather = sensor_index if order is None else sensor_index[order]

    if per_view_captions:
        # The gen->und pass borrows the same-view partition for its queries and
        # keys each group against its own caption run; groups reading nothing
        # leave the pass rather than being handed zero keys.
        packed_order = (
            torch.arange(gen_tokens, device=device)
            if same_view_gather is None
            else same_view_gather
        )
        q_runs: list[torch.Tensor] = []
        kv_runs: list[torch.Tensor] = []
        start = 0
        for group, length in enumerate(same_view_lens):
            item, view = group_item[group]
            run = _caption_run(
                item, view, caption_lengths, layout.lidar_attends_captions, device
            )
            if run.numel():
                q_runs.append(packed_order[start : start + length])
                kv_runs.append(run)
            start += length
        caption_q_gather: torch.Tensor | None = torch.cat(q_runs)
        caption_q_lens = [int(run.numel()) for run in q_runs]
        caption_kv_gather: torch.Tensor | None = torch.cat(kv_runs)
        caption_kv_lens = [int(run.numel()) for run in kv_runs]
    else:
        readers = torch.cat(caption_reader_positions)
        caption_q_gather = None if readers.numel() == gen_tokens else readers
        caption_q_lens = [int(readers.numel())]
        caption_kv_gather = None
        caption_kv_lens = [und_tokens]

    def _tile(gather: torch.Tensor | None, tokens: int) -> torch.Tensor | None:
        if gather is None and batch_size == 1:
            return None
        base = torch.arange(tokens, device=device) if gather is None else gather
        shifts = torch.arange(batch_size, device=device) * tokens
        return (base.unsqueeze(0) + shifts.unsqueeze(1)).reshape(-1)

    return MultiviewMasklessPlan(
        gen_tokens=gen_tokens,
        und_tokens=und_tokens,
        batch_size=batch_size,
        same_view_gather=_tile(same_view_gather, gen_tokens),
        same_view_offsets=_cumulative_offsets(same_view_lens * batch_size, device),
        same_view_max_len=max(same_view_lens),
        cross_view_gather=(
            _tile(cross_view_gather, gen_tokens)
            if cross_view_gather is not None
            else None
        ),
        cross_view_offsets=(
            _cumulative_offsets(cross_view_lens * batch_size, device)
            if cross_view_lens
            else None
        ),
        cross_view_max_len=max(cross_view_lens, default=0),
        caption_q_gather=_tile(caption_q_gather, gen_tokens),
        caption_q_offsets=_cumulative_offsets(caption_q_lens * batch_size, device),
        caption_q_max_len=max(caption_q_lens),
        caption_kv_gather=_tile(caption_kv_gather, und_tokens),
        caption_kv_offsets=_cumulative_offsets(caption_kv_lens * batch_size, device),
        caption_kv_max_len=max(caption_kv_lens),
    )


def _varlen_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager float32 varlen attention with LSE, for CPU tests and as the oracle."""
    group = q.shape[1] // k.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.full(q.shape[:2], torch.finfo(torch.float32).min, device=q.device)
    for start_q, end_q, start_k, end_k in zip(
        cu_q[:-1].tolist(), cu_q[1:].tolist(), cu_k[:-1].tolist(), cu_k[1:].tolist()
    ):
        if end_q == start_q or end_k == start_k:
            continue
        qs = q[start_q:end_q].float()
        ks = k[start_k:end_k].float().repeat_interleave(group, dim=1)
        vs = v[start_k:end_k].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        lse[start_q:end_q] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        out[start_q:end_q] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs)
    return out, lse


VarlenKernel = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        float,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


def _fa2_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
    out, lse, *_ = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, max_q, max_k, 0.0, False, False, scale=scale
    )
    return out, lse.transpose(0, 1)


@functools.lru_cache(maxsize=None)
def _import_kernel(name: str) -> VarlenKernel:
    """The varlen kernel ``name`` as ``(q, k, v, cu_q, cu_k, max_q, max_k, scale)``.

    Every kernel returns ``([N, H, D], [N, H] fp32 LSE)``; FA3 and FA4 hand back the
    LSE as ``[H, N]`` and are transposed here. Raises ImportError when the package
    is missing.
    """
    if name == "fa2":
        return _fa2_varlen
    if name == "fa3":
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        def _fa3_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
            # Returns (out, softmax_lse[H, N], *accumulators) when the LSE is requested.
            out, lse, *_ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                max_q,
                max_k,
                softmax_scale=scale,
                causal=False,
                return_softmax_lse=True,
            )
            return out, lse.transpose(0, 1)

        return _fa3_varlen
    if name == "fa4":
        from flash_attn.cute import flash_attn_varlen_func as fa4_varlen

        def _fa4_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
            out, lse = fa4_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                softmax_scale=scale,
                causal=False,
                return_lse=True,
            )
            return out, lse.transpose(0, 1)

        return _fa4_varlen
    raise ValueError(
        f"Unknown maskless kernel {name!r}; expected one of {list(MASKLESS_KERNELS)}."
    )


# FA4's CuTe kernels target sm90 and sm100+; its sm80 fallback fails to compile
# (seen on Ada), so "auto" never picks it below Hopper and an explicit request is
# refused there instead of failing inside the first forward.
_FA4_MIN_CAPABILITY_MAJOR = 9
_FA3_MIN_CAPABILITY_MAJOR = 8

_resolved_kernels: dict[tuple[str, int, str], str] = {}


def resolve_maskless_kernel(
    device: torch.device,
    requested: str | None = None,
    *,
    capability_major: int | None = None,
) -> str:
    """Pick the varlen kernel for ``device``; env override, else by compute capability."""
    if requested is None:
        requested = envs.SGLANG_DIFFUSION_COSMOS3_MULTIVIEW_MASKLESS_KERNEL or "auto"
    if requested not in MASKLESS_KERNELS:
        raise ValueError(
            f"{MASKLESS_KERNEL_ENV_VAR} must be one of {list(MASKLESS_KERNELS)}, got {requested!r}."
        )
    if capability_major is None:
        if device.type != "cuda":
            return "fa2"
        capability_major = torch.cuda.get_device_capability(device)[0]
    key = (str(device), capability_major, requested)
    cached = _resolved_kernels.get(key)
    if cached is not None:
        return cached
    if requested == "auto":
        preferred = (
            "fa4"
            if capability_major >= 10
            else "fa3"
            if capability_major == 9
            else "fa2"
        )
        try:
            _import_kernel(preferred)
            kernel = preferred
        except ImportError as exc:
            logger.warning(
                "Cosmos3 maskless attention: %s is unavailable (%s); falling back to torch FA2.",
                preferred,
                exc,
            )
            kernel = "fa2"
    else:
        minimum = {
            "fa4": _FA4_MIN_CAPABILITY_MAJOR,
            "fa3": _FA3_MIN_CAPABILITY_MAJOR,
        }.get(requested, 0)
        if capability_major < minimum:
            raise ValueError(
                f"Cosmos3 maskless kernel {requested!r} needs compute capability {minimum}.x or "
                f"newer; this device is {capability_major}.x."
            )
        _import_kernel(
            requested
        )  # an explicit choice fails loudly rather than falling back
        kernel = requested
    _resolved_kernels[key] = kernel
    logger.info(
        "Cosmos3 maskless attention kernel: %s (%s, sm%d0)",
        kernel,
        torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        capability_major,
    )
    return kernel


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    kernel: str = "fa2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unmasked varlen attention; returns ``([N, H, D], [N, H] fp32 LSE)``."""
    if q.device.type != "cuda":
        return _varlen_attention_reference(q, k, v, cu_q, cu_k, scale)
    return _import_kernel(kernel)(q, k, v, cu_q, cu_k, max_q, max_k, scale)


def _scatter(
    out: torch.Tensor,
    lse: torch.Tensor,
    gather: torch.Tensor | None,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-major back to packed order; rows the pass never saw weigh nothing."""
    if gather is None:
        return out, lse
    out_full = out.new_zeros((tokens, *out.shape[1:]))
    lse_full = lse.new_full((tokens, lse.shape[1]), torch.finfo(lse.dtype).min)
    out_full[gather] = out
    lse_full[gather] = lse
    return out_full, lse_full


def merge_attentions(
    outputs: Sequence[torch.Tensor], lses: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Combine passes over one query set as if their keys had been concatenated."""
    stacked_lse = torch.stack([lse.float() for lse in lses])  # [P, N, H]
    peak = stacked_lse.amax(dim=0)
    weights = torch.exp(stacked_lse - peak)  # [P, N, H]
    merged = sum(
        weight.unsqueeze(-1) * out.float() for weight, out in zip(weights, outputs)
    )
    return merged / weights.sum(dim=0).unsqueeze(-1)


def _plan_for(
    context: MultiviewAttentionContext,
    *,
    und_tokens: int,
    batch_size: int,
    device: torch.device,
) -> MultiviewMasklessPlan:
    key: tuple[Any, ...] = (
        MASKLESS_BACKEND,
        context.layout.cache_key(),
        und_tokens,
        batch_size,
        str(device),
    )
    plan = context.mask_cache.get(key)
    if not isinstance(plan, MultiviewMasklessPlan):
        plan = build_multiview_maskless_plan(
            context.layout, und_tokens=und_tokens, batch_size=batch_size, device=device
        )
        context.mask_cache[key] = plan
    return plan


def multiview_maskless_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
    kernel: str | None = None,
) -> torch.Tensor:
    """Three unmasked passes merged by LSE. Inputs ``[B, S, H, D]``; returns the same.

    ``kernel`` pins one of ``MASKLESS_KERNELS``; ``None`` resolves it from the env
    override and the device once per process.
    """
    batch_size, gen_tokens = q.shape[:2]
    if k.shape[:2] != (batch_size, gen_tokens) or k.shape != v.shape:
        raise ValueError(
            "Cosmos3 multiview q/k/v sequence geometry must match before GQA: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if k_und.shape != v_und.shape or k_und.shape[0] != batch_size:
        raise ValueError(
            "Cosmos3 multiview UND key/value geometry mismatch: "
            f"k_und={tuple(k_und.shape)}, v_und={tuple(v_und.shape)}."
        )
    if gen_tokens != context.layout.gen_tokens:
        raise ValueError(
            f"Cosmos3 multiview GEN length {gen_tokens} does not match the layout's "
            f"{context.layout.gen_tokens} tokens."
        )
    plan = _plan_for(
        context, und_tokens=k_und.shape[1], batch_size=batch_size, device=q.device
    )
    scale = 1.0 / math.sqrt(q.shape[-1])
    kernel = resolve_maskless_kernel(q.device, kernel)
    tokens = batch_size * gen_tokens
    q_flat = q.reshape(tokens, *q.shape[2:])
    k_flat = k.reshape(tokens, *k.shape[2:])
    v_flat = v.reshape(tokens, *v.shape[2:])
    k_und_flat = k_und.reshape(-1, *k_und.shape[2:])
    v_und_flat = v_und.reshape(-1, *v_und.shape[2:])

    def _select(tensor: torch.Tensor, gather: torch.Tensor | None) -> torch.Tensor:
        return tensor if gather is None else tensor[gather]

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []

    gather = plan.same_view_gather
    out, lse = _varlen_attention(
        _select(q_flat, gather),
        _select(k_flat, gather),
        _select(v_flat, gather),
        plan.same_view_offsets,
        plan.same_view_offsets,
        plan.same_view_max_len,
        plan.same_view_max_len,
        scale,
        kernel,
    )
    out, lse = _scatter(out, lse, gather, tokens)
    outputs.append(out)
    lses.append(lse)

    if plan.cross_view_offsets is not None:
        gather = plan.cross_view_gather
        out, lse = _varlen_attention(
            _select(q_flat, gather),
            _select(k_flat, gather),
            _select(v_flat, gather),
            plan.cross_view_offsets,
            plan.cross_view_offsets,
            plan.cross_view_max_len,
            plan.cross_view_max_len,
            scale,
            kernel,
        )
        out, lse = _scatter(out, lse, gather, tokens)
        outputs.append(out)
        lses.append(lse)

    out, lse = _varlen_attention(
        _select(q_flat, plan.caption_q_gather),
        _select(k_und_flat, plan.caption_kv_gather),
        _select(v_und_flat, plan.caption_kv_gather),
        plan.caption_q_offsets,
        plan.caption_kv_offsets,
        plan.caption_q_max_len,
        plan.caption_kv_max_len,
        scale,
        kernel,
    )
    out, lse = _scatter(out, lse, plan.caption_q_gather, tokens)
    outputs.append(out)
    lses.append(lse)

    merged = merge_attentions(outputs, lses).to(q.dtype)
    return merged.reshape(batch_size, gen_tokens, *q.shape[2:])
