# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-sequence materialization from the validated workspace
builder, covering fl2va and t2va layouts.

Layout: [text L | imgvid_cond C | audio A(=t*2ch) | video_target V | pad P].
Builder rules:
- block-derived position infos, update masks, token tags, and cu_seqlens
- img_position_ids fp64 grid: text rows (row_idx,0,0); video/cond t counter
  continues text_len with temporal interp spans (frame_rescale 5/3 x
  frame_per_token (1,4,4,4,4)); each spatial sqrt_area axis uses evenly spaced
  coordinates excluding the right endpoint, then scales them by INTERP;
  audio channel-major blocks pinned to the w-grid extremes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _keyframe_cond_frame_indices(
    *,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if not include_keyframe_cond:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted when keyframe cond is not included"
            )
        return []
    if keyframe_frame_indices is None:
        raise ValueError("strict fl2va packed layout requires keyframe_frame_indices")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in keyframe_frame_indices
    ):
        raise ValueError(
            "strict fl2va packed layout requires integer keyframe_frame_indices"
        )
    out = list(keyframe_frame_indices)
    if tuple(out) not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "strict fl2va packed layout requires keyframe_frame_indices in "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {out!r}"
        )
    return out


def _resolve_keyframe_frame_indices(
    frame_indices: Sequence[int],
    *,
    frame_count: int | None,
) -> list[int]:
    if frame_indices and frame_count is None:
        raise ValueError(
            "frame_count is required when keyframe_frame_indices are provided"
        )
    if frame_count is None:
        return []
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    seen: dict[int, int] = {}
    resolved: list[int] = []
    for block_index, semantic_index in enumerate(frame_indices):
        if semantic_index == -1:
            resolved_index = frame_count - 1
        elif 0 <= semantic_index < frame_count:
            resolved_index = semantic_index
        else:
            raise ValueError(
                f"keyframe frame index {semantic_index} must be -1 or in "
                f"[0, {frame_count})"
            )
        previous = seen.get(resolved_index)
        if previous is not None:
            raise ValueError(
                f"keyframe frame index at block {block_index} resolves to "
                f"{resolved_index}, already bound by block {previous}"
            )
        seen[resolved_index] = block_index
        resolved.append(resolved_index)
    return resolved


def _temporal_position_span(temporal_length: int) -> float:
    """Temporal position span for patch_t=1, in fp64.

    NOTE: intentionally NOT merged with ``_video_t_span``. This variant sums
    via numpy (pairwise summation), matching the fl2va anchor computation,
    while ``_video_t_span`` sums sequentially, matching the ref2va
    t-origin accumulation. The two orders diverge in the last ulp
    from n=16 onward, so each path must keep its own summation order.
    """
    spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
    for token_index in range(_T_GROUP):
        spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Build the packed-sequence structural fields for one CFG branch.

    The used length is padded up to a multiple of 64.
    """
    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    cond_frame_indices = _keyframe_cond_frame_indices(
        include_keyframe_cond=include_keyframe_cond,
        keyframe_frame_indices=keyframe_frame_indices,
    )
    resolved_cond_frame_indices = _resolve_keyframe_frame_indices(
        cond_frame_indices,
        frame_count=frame_count,
    )
    cond_rows = len(cond_frame_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = (
        torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_img_pos])
        if cond_rows
        else target_img_pos
    )
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = t_grid[:, None]
    video_g[:, :, 1:] = frame[None]
    for block_index, pixel_index in enumerate(resolved_cond_frame_indices):
        sl = slice(
            cond_sl.start + block_index * frame_rows,
            cond_sl.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = (
                float(text_len) + _temporal_position_span(latent_t) - _FRAME_RESCALE
            )
        else:
            raise ValueError(
                "fl2va packed layout only supports first/last keyframe anchors, "
                f"got resolved frame index {pixel_index}"
            )
        g[sl, 0] = cond_t
        g[sl, 1:] = frame
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)  # PADDING
    token_tags[text_sl] = 1  # TEXT (fl2va image-segment override happens upstream)
    token_tags[audio_sl] = 2  # AUDIO
    token_tags[img_pos] = 0  # VIDEO

    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": cu,
    }


def _positive_int(
    block: Mapping[str, object],
    key: str,
    path: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        predicate = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path}.{key} must be {predicate}")
    return int(value)


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 1.0 / 2.0
    right = left + ratio * 1.0
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _video_t_span(n: int) -> float:
    # Sequential fp64 summation on purpose — see _temporal_position_span for
    # why the two span implementations must not be unified.
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n))


def _range_for_slice(sl: slice) -> torch.Tensor:
    return torch.arange(sl.start, sl.stop, dtype=torch.long)


def _cat_ranges(parts: list[torch.Tensor]) -> torch.Tensor:
    if len(parts) == 1:
        return parts[0]
    if parts:
        return torch.cat(parts)
    return torch.empty(0, dtype=torch.long)


def minimax_h3_packed_sequence_ref2va_blocks(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    ref_blocks: Sequence[Mapping[str, object]],
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    frame_count: int | None = None,
    audio_channel: int = 2,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """General ref2va-family packed layout.

    ``ref_blocks`` are consumed in request/plan order:
    - ``{"kind": "image", "latent_h": H, "latent_w": W}``
    - ``{"kind": "audio", "ref_audio_t": T}``
    - ``{"kind": "video"|"video_audio", "ref_audio_t": T,
       "latent_t": RT, "latent_h": RH, "latent_w": RW}``

    Optional first/last keyframes are packed immediately after text, matching
    Comfy's hybrid Ref2VA + guide layout. Video-bearing blocks pack their audio
    rows immediately before their video rows; both share the same temporal
    origin and advance by the longer of the audio and video spans. Standalone
    audio advances the target origin by its own T, and image blocks advance it
    by one integer slot.
    """
    if not isinstance(ref_blocks, Sequence) or isinstance(ref_blocks, (str, bytes)):
        raise ValueError("ref_blocks must be a sequence")

    parsed: list[dict[str, object]] = []
    ref_visual_rows = 0
    ref_audio_rows = 0
    for index, raw in enumerate(ref_blocks):
        path = f"ref_blocks[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be an object")
        kind = raw.get("kind", raw.get("type"))
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{path}.kind must be a non-empty string")
        if kind == "image":
            rh = _positive_int(raw, "latent_h", path)
            rw = _positive_int(raw, "latent_w", path)
            rows = (rh // _PATCH_H) * (rw // _PATCH_W)
            item = {"kind": kind, "latent_h": rh, "latent_w": rw, "rows": rows}
            ref_visual_rows += rows
        elif kind == "audio":
            rt = _positive_int(raw, "ref_audio_t", path, allow_zero=True)
            rows = rt * audio_channel
            item = {"kind": kind, "ref_audio_t": rt, "audio_rows": rows}
            ref_audio_rows += rows
        elif kind in ("video", "video_audio"):
            rt = _positive_int(raw, "ref_audio_t", path, allow_zero=True)
            vt = _positive_int(raw, "latent_t", path)
            vh = _positive_int(raw, "latent_h", path)
            vw = _positive_int(raw, "latent_w", path)
            frame_rows = (vh // _PATCH_H) * (vw // _PATCH_W)
            audio_rows = rt * audio_channel
            video_rows = vt * frame_rows
            item = {
                "kind": kind,
                "ref_audio_t": rt,
                "latent_t": vt,
                "latent_h": vh,
                "latent_w": vw,
                "frame_rows": frame_rows,
                "audio_rows": audio_rows,
                "video_rows": video_rows,
            }
            ref_audio_rows += audio_rows
            ref_visual_rows += video_rows
        else:
            raise ValueError(f"{path}.kind unsupported for ref2va: {kind!r}")
        parsed.append(item)

    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    keyframe_indices = _keyframe_cond_frame_indices(
        include_keyframe_cond=keyframe_frame_indices is not None,
        keyframe_frame_indices=keyframe_frame_indices,
    )
    resolved_keyframe_indices = _resolve_keyframe_frame_indices(
        keyframe_indices,
        frame_count=frame_count,
    )
    keyframe_rows = len(keyframe_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    ref_rows = ref_visual_rows + ref_audio_rows
    used = text_len + keyframe_rows + ref_rows + audio_rows + video_rows
    if seq_len is None:
        seq_len = (
            (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
            // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
            * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        )
    if seq_len < used:
        raise ValueError(f"seq_len {seq_len} < used rows {used}")

    text_sl = slice(0, text_len)
    keyframe_sl = slice(text_len, text_len + keyframe_rows)
    cursor = keyframe_sl.stop
    block_slices: list[dict[str, object]] = []
    for item in parsed:
        kind = str(item["kind"])
        if kind == "image":
            rows = int(item["rows"])
            visual_sl = slice(cursor, cursor + rows)
            cursor = visual_sl.stop
            block_slices.append({**item, "visual_sl": visual_sl})
        elif kind == "audio":
            rows = int(item["audio_rows"])
            audio_sl = slice(cursor, cursor + rows)
            cursor = audio_sl.stop
            block_slices.append({**item, "audio_sl": audio_sl})
        else:
            a_rows = int(item["audio_rows"])
            v_rows = int(item["video_rows"])
            audio_sl = slice(cursor, cursor + a_rows)
            visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)
            cursor = visual_sl.stop
            block_slices.append({**item, "audio_sl": audio_sl, "visual_sl": visual_sl})

    audio_sl = slice(cursor, cursor + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    ref_img_pos_parts: list[torch.Tensor] = []
    ref_audio_pos_parts: list[torch.Tensor] = []
    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    target_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, target_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, target_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    target_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    t_cursor = float(text_len)
    for item in block_slices:
        kind = str(item["kind"])
        if kind == "image":
            visual_sl = item["visual_sl"]
            assert isinstance(visual_sl, slice)
            ref_img_pos_parts.append(_range_for_slice(visual_sl))
            rh = int(item["latent_h"])
            rw = int(item["latent_w"])
            area = np.sqrt(rh * rw)
            ref_hh, ref_ww = torch.meshgrid(
                _axis_from_sqrt_area(rh, _PATCH_H, area),
                _axis_from_sqrt_area(rw, _PATCH_W, area),
                indexing="ij",
            )
            g[visual_sl, 0] = t_cursor
            g[visual_sl, 1] = ref_hh.reshape(-1)
            g[visual_sl, 2] = ref_ww.reshape(-1)
            t_cursor += 1.0
        elif kind == "audio":
            audio_ref_sl = item["audio_sl"]
            assert isinstance(audio_ref_sl, slice)
            ref_t = int(item["ref_audio_t"])
            ref_audio_pos_parts.append(_range_for_slice(audio_ref_sl))
            ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
            g[audio_ref_sl, 0] = ref_t_grid.repeat(audio_channel)
            if ref_t:
                g[audio_ref_sl.start : audio_ref_sl.start + ref_t, 2] = float(w_grid[0])
                g[audio_ref_sl.start + ref_t : audio_ref_sl.stop, 2] = float(w_grid[-1])
            t_cursor += float(ref_t)
        else:
            audio_ref_sl = item["audio_sl"]
            visual_sl = item["visual_sl"]
            assert isinstance(audio_ref_sl, slice)
            assert isinstance(visual_sl, slice)
            ref_t = int(item["ref_audio_t"])
            vt = int(item["latent_t"])
            vh = int(item["latent_h"])
            vw = int(item["latent_w"])
            ref_audio_pos_parts.append(_range_for_slice(audio_ref_sl))
            ref_img_pos_parts.append(_range_for_slice(visual_sl))

            ref_area = np.sqrt(vh * vw)
            rv_h_grid = _axis_from_sqrt_area(vh, _PATCH_H, ref_area)
            rv_w_grid = _axis_from_sqrt_area(vw, _PATCH_W, ref_area)
            rv_hh, rv_ww = torch.meshgrid(rv_h_grid, rv_w_grid, indexing="ij")

            ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
            g[audio_ref_sl, 0] = ref_t_grid.repeat(audio_channel)
            if ref_t:
                g[audio_ref_sl.start : audio_ref_sl.start + ref_t, 2] = float(
                    rv_w_grid[0]
                )
                g[audio_ref_sl.start + ref_t : audio_ref_sl.stop, 2] = float(
                    rv_w_grid[-1]
                )

            rv_frame = torch.stack([rv_hh.reshape(-1), rv_ww.reshape(-1)], dim=-1)
            rv_g = g[visual_sl].view(vt, int(item["frame_rows"]), 3)
            rv_g[:, :, 0] = _video_t_grid(vt, t_cursor)[:, None]
            rv_g[:, :, 1:] = rv_frame[None]
            t_cursor += max(float(ref_t), _video_t_span(vt))

    audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = _video_t_grid(latent_t, t_cursor)[:, None]
    video_g[:, :, 1:] = target_frame[None]

    for block_index, pixel_index in enumerate(resolved_keyframe_indices):
        sl = slice(
            keyframe_sl.start + block_index * frame_rows,
            keyframe_sl.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            cond_t = t_cursor
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = t_cursor + _temporal_position_span(latent_t) - _FRAME_RESCALE
        else:
            raise ValueError(
                "hybrid ref2va layout only supports first/last keyframe anchors, "
                f"got resolved frame index {pixel_index}"
            )
        g[sl, 0] = cond_t
        g[sl, 1:] = target_frame

    keyframe_img_pos = _range_for_slice(keyframe_sl)
    target_img_pos = _range_for_slice(video_sl)
    target_audio_pos = _range_for_slice(audio_sl)
    img_pos = _cat_ranges([keyframe_img_pos] + ref_img_pos_parts + [target_img_pos])
    audio_pos = _cat_ranges(ref_audio_pos_parts + [target_audio_pos])

    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[keyframe_rows + ref_visual_rows :] = True
    audio_update_mask = torch.zeros(audio_pos.shape[0], dtype=torch.bool)
    audio_update_mask[ref_audio_rows:] = True
    text_pos = torch.arange(0, text_len)

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)  # PADDING
    token_tags[text_sl] = 1  # TEXT
    token_tags[audio_pos] = 2  # AUDIO (refs + target)
    token_tags[img_pos] = 0  # VIDEO (refs + target)

    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": cu,
    }


__all__ = [
    "minimax_h3_packed_sequence",
    "minimax_h3_packed_sequence_ref2va_blocks",
]
