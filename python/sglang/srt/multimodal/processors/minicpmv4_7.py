# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLang multimodal processor for MiniCPM-V 4.7.

The canvas M-RoPE layout ports the checkpoints' reference
``mrope_minicpmv4_7.py`` and must assign exactly the same positions.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch

from sglang.srt.models.minicpmv import MiniCPMV4_7ForConditionalGeneration
from sglang.srt.multimodal.processors.minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMV4_6MultimodalProcessor,
)

logger = logging.getLogger(__name__)

_SCOPED_KWARGS_KEYS = ("images_kwargs", "videos_kwargs", "audio_kwargs")
_STRUCTURAL_KEYS = ("im_start_id", "im_end_id", "slice_start_id", "slice_end_id")

# (start, end, index into target_sizes) of one run of visual tokens.
_Span = Tuple[int, int, int]
_Frame = Tuple[_Span, List[_Span]]


def _parse_mm_processor_kwargs(raw) -> dict:
    """Flatten HF-style ``images_kwargs`` / ``videos_kwargs``; top-level keys win."""
    if not isinstance(raw, dict):
        return {}
    flat = {}
    for key in _SCOPED_KWARGS_KEYS:
        if isinstance(raw.get(key), dict):
            flat.update(raw[key])
    flat.update({k: v for k, v in raw.items() if k not in _SCOPED_KWARGS_KEYS})
    return flat


def _llm_grid(n_tokens: int, vit_h: int, vit_w: int) -> Tuple[int, int]:
    """LLM-token grid for ``n_tokens`` merged from a ``vit_h x vit_w`` patch grid."""
    if n_tokens <= 0 or vit_h <= 0 or vit_w <= 0:
        return 0, 0
    vit_total = vit_h * vit_w
    if vit_total == n_tokens:
        return vit_h, vit_w
    if vit_total < n_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / n_tokens)))
    if factor > 0 and vit_h % factor == 0 and vit_w % factor == 0:
        height, width = vit_h // factor, vit_w // factor
        if height * width == n_tokens:
            return height, width
    ratio = vit_h / vit_w
    width = max(1, int(round(math.sqrt(n_tokens / ratio))))
    height = n_tokens // width
    if height * width == n_tokens and height > 0:
        return height, width
    return 0, 0


def build_image_bounds(input_ids: torch.Tensor, token_ids: dict) -> torch.Tensor:
    """``[start, end)`` of every thumbnail / slice token run, shape (n, 2)."""
    start_ids = [token_ids[k] for k in ("im_start_id", "slice_start_id")]
    end_ids = [token_ids[k] for k in ("im_end_id", "slice_end_id")]
    starts = torch.isin(input_ids, torch.tensor(start_ids)).nonzero().flatten() + 1
    ends = torch.isin(input_ids, torch.tensor(end_ids)).nonzero().flatten()
    n = min(len(starts), len(ends))
    return torch.stack([starts[:n], ends[:n]], dim=-1)


def _frame_end(ids: List[int], frame: _Frame, structural: set, limit: int) -> int:
    thumbnail, slices = frame
    end = slices[-1][1] if slices else thumbnail[1]
    while end < limit and ids[end] in structural:
        end += 1
    return min(end, limit)


def _group_frames(
    ids: List[int], bounds: List[List[int]], token_ids: dict, structural: set
) -> List[List[_Frame]]:
    """Pair each thumbnail with its slices, then merge frames separated only by
    structural / newline tokens into one video group."""
    frames: List[_Frame] = []
    for index, (start, end) in enumerate(bounds):
        if frames and ids[start - 1] == token_ids["slice_start_id"]:
            frames[-1][1].append((start, end, index))
        else:
            frames.append(((start, end, index), []))

    gap_ok = structural | {token_ids["newline_id"]}
    groups = [[frames[0]]]
    for prev, frame in zip(frames, frames[1:]):
        gap = range(_frame_end(ids, prev, structural, len(ids)), frame[0][0] - 1)
        if all(ids[p] in gap_ok for p in gap):
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _fill_text(positions: torch.Tensor, start: int, end: int, pos: int) -> int:
    if end <= start:
        return pos
    positions[:, start:end] = torch.arange(pos, pos + end - start)
    return pos + end - start


def _fill_grid(
    positions: torch.Tensor,
    start: int,
    base: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> None:
    end = start + len(rows) * len(cols)
    positions[0, start:end] = base
    positions[1, start:end] = rows.repeat_interleave(len(cols)) + base
    positions[2, start:end] = cols.repeat(len(rows)) + base


def _place_frame(
    positions: torch.Tensor,
    frame: _Frame,
    frame_end: int,
    base: int,
    sizes: List[List[int]],
) -> int:
    """Lay one thumbnail and its slice grid out on a shared canvas at ``base``;
    returns the first position after the canvas."""
    (t_start, t_end, t_index), slices = frame
    im_start = t_start - 1
    n_thumb = t_end - t_start
    th, tw = _llm_grid(n_thumb, *sizes[t_index])

    if slices:
        cols = len(slices)
        for k in range(len(slices) - 1):
            if slices[k + 1][0] - slices[k][1] > 2:
                cols = k + 1
                break
        rows = len(slices) // cols
        if rows * cols != len(slices):
            rows, cols = 1, len(slices)
        s0_start, s0_end, s0_index = slices[0]
        sh, sw = _llm_grid(s0_end - s0_start, *sizes[s0_index])
        canvas_h, canvas_w = rows * sh, cols * sw
    else:
        canvas_h, canvas_w = max(th, 1), max(tw, 1)

    positions[:, im_start:frame_end] = base
    positions[1:, im_start] = max(base - 1, 0)
    if t_end < frame_end:
        positions[1, t_end] = base + canvas_h
        positions[2, t_end] = base + canvas_w

    if th > 0:
        if slices and canvas_h > 0 and canvas_w > 0:
            # The thumbnail is stretched over the full slice canvas.
            t_rows = torch.linspace(0, canvas_h - 1, th).round().long()
            t_cols = torch.linspace(0, canvas_w - 1, tw).round().long()
        else:
            t_rows, t_cols = torch.arange(th), torch.arange(tw)
        _fill_grid(positions, t_start, base, t_rows, t_cols)
    else:
        _fill_text(positions, t_start, t_end, base)

    for k, (s_start, s_end, s_index) in enumerate(slices):
        sh_k, sw_k = _llm_grid(s_end - s_start, *sizes[s_index])
        h_off, w_off = (k // cols) * sh, (k % cols) * sw
        positions[1, s_start - 1] = base + h_off
        positions[2, s_start - 1] = base + w_off
        if s_end < frame_end:
            positions[1, s_end] = base + h_off + max(sh_k - 1, 0)
            positions[2, s_end] = base + w_off + max(sw_k - 1, 0)
        if sh_k > 0:
            rows_k = torch.arange(sh_k) + h_off
            cols_k = torch.arange(sw_k) + w_off
            _fill_grid(positions, s_start, base, rows_k, cols_k)
        else:
            _fill_text(positions, s_start, s_end, base)

    # Row-separating newlines sit one column past the canvas, on the row just ended.
    for k in range(len(slices) - 1):
        gap_start, gap_end = slices[k][1] + 1, slices[k + 1][0] - 1
        if gap_end > gap_start:
            positions[1, gap_start:gap_end] = base + (k // cols + 1) * sh - 1
            positions[2, gap_start:gap_end] = base + cols * sw

    return base + max(canvas_h, canvas_w) + 1


def compute_canvas_positions(
    input_ids: torch.Tensor,
    image_bounds: torch.Tensor,
    target_sizes: List[List[int]],
    token_ids: dict,
) -> torch.Tensor:
    """(3, seq_len) ``[temporal, height, width]`` canvas M-RoPE positions."""
    ids = input_ids.tolist()
    seq_len = len(ids)
    positions = torch.arange(seq_len).expand(3, -1).clone()
    if len(image_bounds) == 0:
        return positions

    structural = {token_ids[k] for k in _STRUCTURAL_KEYS}
    pos = cursor = 0
    for frames in _group_frames(ids, image_bounds.tolist(), token_ids, structural):
        group_start = frames[0][0][0] - 1
        group_end = _frame_end(ids, frames[-1], structural, seq_len)
        pos = _fill_text(positions, cursor, group_start, pos)
        frame_cursor = group_start
        for frame in frames:
            im_start = frame[0][0] - 1
            if im_start > frame_cursor:
                # The extra step keeps the next frame's <image> halo off the gap.
                pos = _fill_text(positions, frame_cursor, im_start, pos) + 1
            frame_end = _frame_end(ids, frame, structural, group_end)
            pos = _place_frame(positions, frame, frame_end, pos, target_sizes)
            frame_cursor = frame_end
        cursor = group_end
    _fill_text(positions, cursor, seq_len, pos)
    return positions


def _sequential_positions(seq_len: int) -> torch.Tensor:
    return torch.arange(seq_len).expand(3, -1)


def _rope_delta(positions: torch.Tensor) -> torch.Tensor:
    """Decode offset: the first generated token sits one past the last prompt position."""
    delta = int(positions.amax()) + 1 - positions.shape[1]
    return torch.tensor([[delta]], dtype=torch.int64)


class MiniCPMV4_7MultimodalProcessor(MiniCPMV4_6MultimodalProcessor):
    models = [MiniCPMV4_7ForConditionalGeneration]
    # 4.7 was trained without ``<image_id>`` in front of videos.
    video_image_id = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.canvas_token_ids = self._resolve_canvas_token_ids(hf_config)

    def _image_processor_for(self, request_obj) -> MiniCPMV4_6ImageProcessor:
        kwargs = _parse_mm_processor_kwargs(
            getattr(request_obj, "mm_processor_kwargs", None)
        )
        default = self.image_processor

        downsample_mode = kwargs.get("downsample_mode") or default.downsample_mode
        if downsample_mode not in ("4x", "16x"):
            logger.warning("Ignoring unknown downsample_mode %r", downsample_mode)
            downsample_mode = default.downsample_mode
        max_slice_nums = kwargs.get("max_slice_nums")
        if max_slice_nums is None:
            max_slice_nums = default.max_slice_nums
        try:
            max_slice_nums = int(max_slice_nums)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric max_slice_nums %r", max_slice_nums)
            max_slice_nums = default.max_slice_nums

        if (downsample_mode, max_slice_nums) == (
            default.downsample_mode,
            default.max_slice_nums,
        ):
            return default
        return MiniCPMV4_6ImageProcessor(
            max_slice_nums=max_slice_nums,
            scale_resolution=default.scale_resolution,
            patch_size=default.patch_size,
            downsample_mode=downsample_mode,
        )

    def _resolve_canvas_token_ids(self, hf_config) -> Optional[dict]:
        mrope_mode = str(getattr(hf_config, "mrope_mode", "")).strip().lower()
        if (
            not getattr(hf_config, "uses_mrope_canvas", False)
            and mrope_mode != "canvas"
        ):
            return None

        # Byte-level BPE vocabularies store "\n" as "\u010a", so encode it.
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        from_tokenizer = {
            "im_start_id": self.image_start_id,
            "im_end_id": self.image_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
            "newline_id": newline_ids[0] if len(newline_ids) == 1 else None,
        }
        token_ids = {
            "im_start_id": getattr(hf_config, "image_start_id", None),
            "im_end_id": getattr(hf_config, "image_end_id", None),
            "slice_start_id": getattr(hf_config, "slice_start_id", None),
            "slice_end_id": getattr(hf_config, "slice_end_id", None),
            "newline_id": getattr(hf_config, "newline_id", None),
        }
        token_ids = {
            k: from_tokenizer[k] if v is None else v for k, v in token_ids.items()
        }
        if None in token_ids.values():
            logger.warning("Canvas M-RoPE disabled: token ids missing (%s)", token_ids)
            return None
        return {k: int(v) for k, v in token_ids.items()}

    def compute_mrope_positions(self, input_ids, mm_items):
        ids = torch.tensor(list(input_ids), dtype=torch.long)
        positions = self._canvas_positions(ids, mm_items)
        # The Qwen3.5 backbone reads mrope_positions for every multimodal request.
        if positions is None:
            positions = _sequential_positions(len(ids))
        return positions, _rope_delta(positions)

    def _canvas_positions(self, ids, mm_items) -> Optional[torch.Tensor]:
        if self.canvas_token_ids is None:
            return None
        # Items are appended images-first; the canvas walk needs them in prompt order.
        ordered = sorted(mm_items, key=lambda item: item.offsets[0][0])
        target_sizes = [torch.as_tensor(item.tgt_size[0]).tolist() for item in ordered]
        image_bounds = build_image_bounds(ids, self.canvas_token_ids)
        if len(image_bounds) != len(target_sizes):
            logger.warning(
                "Canvas M-RoPE: %d image bounds vs %d patch grids; "
                "using sequential positions",
                len(image_bounds),
                len(target_sizes),
            )
            return None
        return compute_canvas_positions(
            ids, image_bounds, target_sizes, self.canvas_token_ids
        )


__all__ = ["MiniCPMV4_7MultimodalProcessor"]
