# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import torch
from PIL import Image

_CANVAS_COLOR = (255, 255, 255)

FIGURE_ONE = "<Figure 1>"
_REFERENCE_HINT = f"The first frame refers to {FIGURE_ONE}"


def resize_pad(image: Image.Image, *, height: int, width: int) -> Image.Image:
    """Letterbox, never crop: the image conditions the whole clip rather than becoming a frame of it."""
    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"invalid image size {image.size}")

    scale = min(width / source_w, height / source_h)
    target_w = max(1, round(source_w * scale))
    target_h = max(1, round(source_h * scale))
    resized = image.convert("RGB").resize(
        (target_w, target_h), resample=Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (width, height), _CANVAS_COLOR)
    canvas.paste(resized, ((width - target_w) // 2, (height - target_h) // 2))
    return canvas


def target_size(
    image: Image.Image, *, generation_height: int, generation_width: int
) -> tuple[int, int]:
    long_edge = max(generation_height, generation_width)
    if image.width > image.height:
        return round(image.height * long_edge / image.width), long_edge
    return long_edge, round(image.width * long_edge / image.height)


def resample_to(image: Image.Image, height: int, width: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def aligned_size(height: int, width: int, *, align: int) -> tuple[int, int]:
    """``align`` is TWICE the VAE spatial compression ratio, not the ratio itself."""
    return max(align, height - height % align), max(align, width - width % align)


def ensure_figure_phrase(prompt: str) -> str:
    """Without the phrase the pooled embedding is zero and the image conditions nothing, with no error."""
    if FIGURE_ONE in prompt:
        return prompt
    try:
        parsed = json.loads(prompt)
        if not isinstance(parsed, dict):
            raise ValueError("prompt JSON is not an object")
    except (json.JSONDecodeError, TypeError, ValueError):
        return prompt + f"reference_layer:{_REFERENCE_HINT}"
    parsed["reference_layer"] = [_REFERENCE_HINT]
    return json.dumps(parsed, ensure_ascii=False)


def pool_figure_embedding(
    *,
    prompt: str,
    tokenizer,
    prompt_embeds: torch.Tensor,
    phrase: str = FIGURE_ONE,
) -> torch.Tensor:
    """Uses character offsets, not a token-id search, because the phrase spans several subwords."""
    width = prompt_embeds.shape[-1]
    zero = prompt_embeds.new_zeros(width)

    start_char = prompt.find(phrase)
    if start_char < 0:
        return zero
    end_char = start_char + len(phrase)

    encoded = tokenizer(
        [prompt], return_tensors="pt", return_offsets_mapping=True, truncation=False
    )
    offsets = encoded["offset_mapping"][0]

    rows: list[int] = []
    for index, (begin, end) in enumerate(offsets.tolist()):
        # Special tokens report a (0, 0) span; keeping them would pool padding.
        if begin == 0 and end == 0 and index != 0:
            continue
        if max(begin, start_char) < min(end, end_char):
            rows.append(index)
    if not rows:
        return zero

    selection = torch.tensor(rows, device=prompt_embeds.device)
    return prompt_embeds.index_select(0, selection).mean(dim=0)
