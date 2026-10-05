"""Layout-conditioning helpers for HiDream-O1-Image.

Ported from the official HiDream-O1-Image models/utils.py: parse xxyy
relative layout boxes and render the bordered reference images plus the
black layout image the model was trained on.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

MAX_BOX = 5

DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 180, 0),
    (0, 0, 255),
    (204, 180, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def load_layout_bboxes(layout_bboxes: str) -> Any:
    """Load layout boxes from either a JSON string or a JSON file path."""
    if os.path.exists(layout_bboxes):
        with open(layout_bboxes, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(layout_bboxes)


def _unwrap_boxes(data: Any) -> Any:
    if isinstance(data, dict):
        for key in ("layout_bboxes", "bboxes", "boxes", "bbox_list"):
            if key in data:
                return data[key]
    return data


def _as_bbox_and_text(item: Any) -> Tuple[Sequence[float], str]:
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box")
        text = str(item.get("text") or item.get("label") or "")
        if bbox is None:
            raise ValueError(f"Missing bbox in layout item: {item!r}")
        return bbox, text
    if isinstance(item, (list, tuple)) and len(item) == 4:
        return item, ""
    raise ValueError(f"Unsupported layout bbox item: {item!r}")


def _xxyy_relative_to_absolute_bbox(
    bbox: Sequence[float], width: int, height: int
) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox!r}")
    x1, x2, y1, y2 = [float(v) for v in bbox]

    # xxyy relative [x1, x2, y1, y2]; 0-1 intended, 0-100 accepted
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_abs <= 100.0:
        x1, x2 = x1 / 100.0 * width, x2 / 100.0 * width
        y1, y2 = y1 / 100.0 * height, y2 / 100.0 * height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bbox after scaling/clamping: {[x1, y1, x2, y2]!r}"
        )
    return [x1, y1, x2, y2]


def parse_layout_bboxes(
    layout_bboxes: Any, width: int, height: int
) -> List[Dict[str, Any]]:
    """Convert xxyy relative layout boxes into the bbox layout format."""
    raw_boxes = _unwrap_boxes(layout_bboxes)
    if not isinstance(raw_boxes, list):
        raise ValueError(
            "layout_bboxes must be a list, or a dict containing one of: "
            "layout_bboxes/bboxes/boxes"
        )

    parsed = []
    for idx, item in enumerate(raw_boxes):
        bbox, text = _as_bbox_and_text(item)
        parsed.append(
            {
                "bbox": _xxyy_relative_to_absolute_bbox(bbox, width, height),
                "color": "",
                "text": text,
                "image": None,
                "_orig_idx": idx,
            }
        )
    return parsed


def _bbox_area(item: Dict[str, Any]) -> int:
    x1, y1, x2, y2 = item["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _get_render_params(image_width: int, image_height: int) -> Tuple[int, int]:
    edge = math.sqrt(image_width * image_height)
    max_font_size = int(edge * 0.07)
    max_bbox_line_width = int(edge * 0.05)
    return max_font_size, max_bbox_line_width


def draw_bbox_layout(
    bbox_list: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    max_bbox: int = MAX_BOX,
    max_bbox_line_width: int | None = None,
    bbox_line_gap: int | None = None,
    return_color: bool = False,
):
    """Draw a black layout image with colored boxes, matching the
    training-side layout style."""
    if max_bbox_line_width is None:
        _, max_bbox_line_width = _get_render_params(image_width, image_height)
    if bbox_line_gap is None:
        bbox_line_gap = max(1, max_bbox_line_width // max_bbox)

    image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    color_list = [None] * len(bbox_list)
    sorted_bboxes = sorted(bbox_list, key=_bbox_area, reverse=True)[:max_bbox]

    for sorted_idx, item in enumerate(sorted_bboxes):
        color = DEFAULT_COLORS[sorted_idx % len(DEFAULT_COLORS)]
        orig_idx = int(item.get("_orig_idx", sorted_idx))
        if 0 <= orig_idx < len(color_list):
            color_list[orig_idx] = color
        line_width = max(max_bbox_line_width - sorted_idx * bbox_line_gap, 5)
        draw.rectangle(
            [int(v) for v in item["bbox"]], outline=color, width=line_width
        )

    if return_color:
        return image, color_list
    return image


def _add_outer_border_keep_size(
    pil: Image.Image, color: Sequence[int], width: int
) -> Image.Image:
    """Draw a border inside the image without changing its size."""
    img = pil.convert("RGB").copy()
    color_tuple = tuple(int(c) for c in color)
    width = max(0, int(width))
    if width == 0:
        return img

    draw = ImageDraw.Draw(img)
    w, h = img.size
    for t in range(width):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color_tuple)
    return img


def create_layout_reference_images(
    ref_pils: Sequence[Image.Image],
    layout_bboxes: Any,
    image_width: int,
    image_height: int,
    ref_max_size: int | None = None,
    patch_size: int = 32,
) -> List[Image.Image]:
    """Create bordered ref images plus one layout image (len(ref_pils) + 1)."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream.generation import (
        resize_pilimage,
    )

    parsed_boxes = parse_layout_bboxes(layout_bboxes, image_width, image_height)
    layout_image, color_list = draw_bbox_layout(
        parsed_boxes,
        image_width=image_width,
        image_height=image_height,
        return_color=True,
    )

    output_refs: List[Image.Image] = []
    for idx, ref in enumerate(ref_pils):
        if ref_max_size is not None:
            ref = resize_pilimage(ref, ref_max_size, patch_size)
        color = (
            color_list[idx]
            if idx < len(color_list) and color_list[idx] is not None
            else DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        )
        line_width = int(math.sqrt(ref.width * ref.height) * 0.04)
        bordered = _add_outer_border_keep_size(ref, color, line_width)
        output_refs.append(bordered)
    output_refs.append(layout_image)
    return output_refs
