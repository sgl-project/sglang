# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen presentation building.

Builds the positive presentation token stream:
- fl2va: '<Picture 1>: ' label + vision block (<|vision_start|> +
  N*<|image_pad|> + <|vision_end|>) + prompt text.
- t2va: prompt text only (no vision block).
Prompt text passes through verbatim (no stripping or rewriting).

All presentation variants are emitted through the shared ``_Presentation``
accumulator so ids and AdaLN token tags cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

_TEXT_TAG = 1
_VIDEO_TAG = 0


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> list[int]:
    return (
        [tokenizer.convert_tokens_to_ids(VISION_START)]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count)
        + [tokenizer.convert_tokens_to_ids(VISION_END)]
    )


class _Presentation:
    """Accumulates aligned (ids, token_tags) presentation segments."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: list[int]) -> None:
        self.ids += token_ids
        self.tags += [_TEXT_TAG] * len(token_ids)

    def vision(self, token_ids: list[int]) -> None:
        self.ids += token_ids
        self.tags += [_VIDEO_TAG] * len(token_ids)

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def _timestamped_video_blocks(
    presentation: _Presentation,
    tokenizer: Any,
    *,
    counts: Sequence[int],
    timestamps: Sequence[float],
    context: str,
) -> None:
    """Emit per-temporal-block ``<{t:.1f} seconds>`` text + VIDEO vision."""

    counts = [int(value) for value in counts]
    timestamps = [float(value) for value in timestamps]
    if not counts or len(counts) != len(timestamps):
        raise ValueError(f"{context}video block token counts and timestamps must align")
    for count, timestamp in zip(counts, timestamps):
        if count <= 0:
            raise ValueError(f"{context}video block token count must be positive")
        presentation.text(_text_ids(tokenizer, f"<{timestamp:.1f} seconds>"))
        presentation.vision(_vision_block_ids(tokenizer, VIDEO_PAD, count))


def minimax_h3_text_only_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    """t2va presentation: verbatim prompt, no special tokens."""
    if not prompt:
        raise ValueError("prompt must be non-empty")
    return torch.tensor(_text_ids(tokenizer, prompt), dtype=torch.long)


def minimax_h3_multi_image_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not image_token_counts:
        raise ValueError("image_token_counts must be non-empty")
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        if int(count) <= 0:
            raise ValueError("image_token_count must be positive")
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def minimax_h3_ref2va_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: list[tuple[str, int]],
    image_token_count: int | list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ref2va positive presentation:

    per condition in request order — image i: ``<Picture i>: `` label followed
    by the vision block; audio j: ``<Audio j>: `` label only (audio content
    never enters Qwen) — then the verbatim prompt. Returns (ids, token_tags)
    with the vision block tagged VIDEO(0) and everything else TEXT(1).

    condition_labels: [("image", 1), ("audio", 1), ...] with 1-based ordinals
    per type.
    """
    return minimax_h3_ref2va_video_presentation(
        tokenizer,
        prompt=prompt,
        condition_labels=condition_labels,
        image_token_count=image_token_count,
        video_block_token_counts=None,
        video_block_timestamps=None,
    )


def _as_int_list(value: int | Sequence[int] | None, *, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an int or a sequence of ints")
    return [int(item) for item in value]


def _as_nested_int_list(
    value: Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    name: str,
) -> list[list[int]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        out: list[list[int]] = []
        for group in value:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise ValueError(f"{name} must not mix nested and flat entries")
            out.append([int(item) for item in group])
        return out
    return [[int(item) for item in value]]


def _as_nested_float_list(
    value: Sequence[float] | Sequence[Sequence[float]] | None,
    *,
    name: str,
) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        out: list[list[float]] = []
        for group in value:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise ValueError(f"{name} must not mix nested and flat entries")
            out.append([float(item) for item in group])
        return out
    return [[float(item) for item in value]]


def minimax_h3_ref2va_video_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: list[tuple[str, int]],
    image_token_count: int | list[int] | None,
    video_block_token_counts: list[int] | list[list[int]] | None,
    video_block_timestamps: list[float] | list[list[float]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ref2va (optionally with video refs) positive presentation:

    per condition in request order —
    - image i:  ``<Picture i>: `` label + one image vision block;
    - audio j:  ``<Audio j>: `` label only (audio content never enters Qwen);
    - video k:  ``<Video k>: `` label, then per temporal block a timestamp
      text ``<{t:.1f} seconds>`` followed by a VIDEO vision block
      (<|vision_start|> + <|video_pad|> x n + <|vision_end|>). Timestamps are
      the mean of each merged frame pair (Qwen3VL temporal merge 2; odd frame
      counts repeat the last frame), emitting the
      ``<0.2 seconds>`` ..
      ``<4.0 seconds>`` sequence — note Python bankers-rounding at .1f.
    then the verbatim prompt. Vision blocks are tagged VIDEO(0), everything
    else TEXT(1).
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    presentation = _Presentation()
    image_token_counts = _as_int_list(image_token_count, name="image_token_count")
    video_counts_by_ref = _as_nested_int_list(
        video_block_token_counts,
        name="video_block_token_counts",
    )
    video_timestamps_by_ref = _as_nested_float_list(
        video_block_timestamps,
        name="video_block_timestamps",
    )
    if len(video_counts_by_ref) != len(video_timestamps_by_ref):
        raise ValueError("video block token counts and timestamps must align")
    image_seen = 0
    video_seen = 0
    for cond_type, ordinal in condition_labels:
        if cond_type == "image":
            image_seen += 1
            if image_seen > len(image_token_counts):
                raise ValueError("image_token_count required for an image reference")
            count = int(image_token_counts[image_seen - 1])
            if count <= 0:
                raise ValueError("image_token_count required for an image reference")
            presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
            presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
        elif cond_type == "audio":
            presentation.text(_text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif cond_type == "video":
            video_seen += 1
            if video_seen > len(video_counts_by_ref):
                raise ValueError(
                    "video reference requires block token counts and timestamps"
                )
            counts = video_counts_by_ref[video_seen - 1]
            timestamps = video_timestamps_by_ref[video_seen - 1]
            if not counts or not timestamps:
                raise ValueError(
                    "video reference requires block token counts and timestamps"
                )
            presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
            _timestamped_video_blocks(
                presentation,
                tokenizer,
                counts=counts,
                timestamps=timestamps,
                context="",
            )
        else:
            raise ValueError(f"unsupported ref2va condition type {cond_type!r}")
    if image_seen != len(image_token_counts):
        raise ValueError("unused image_token_count entries")
    if video_seen != len(video_counts_by_ref):
        raise ValueError("unused video block token count entries")
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


__all__ = [
    "minimax_h3_multi_image_presentation",
    "minimax_h3_ref2va_presentation",
    "minimax_h3_ref2va_video_presentation",
    "minimax_h3_text_only_ids",
]
