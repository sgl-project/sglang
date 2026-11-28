"""Image-only Qwen-VL metadata for the two opt-in P/D experiments.

``request`` accepts grid-only image refs at decode. ``transfer`` accepts an
already expanded token prompt and restores metadata from prefill's registered
metadata buffers. Neither path may use a decode prefix cache: a caller's grid
is not proof of content identity, and the transfer identity arrives after KV
allocation. Ordinary serving keeps its existing behavior.
"""

from __future__ import annotations

import hashlib
import struct
from array import array

import msgspec

from sglang.srt.environ import envs


def media_mode() -> str:
    mode = envs.SGLANG_PD_MEDIA_METADATA_MODE.get()
    if mode not in ("original", "request", "transfer"):
        raise ValueError(f"Unknown P/D media metadata mode: {mode}")
    return mode


def validate_server_args(args) -> None:
    if media_mode() == "original":
        return
    if args.disaggregation_mode not in ("prefill", "decode"):
        raise ValueError("P/D media metadata requires a prefill or decode worker")
    if args.disaggregation_mode == "decode" and not args.disable_radix_cache:
        raise ValueError(
            "P/D media metadata currently requires --disable-radix-cache on decode"
        )
    if args.speculative_algorithm is not None:
        raise ValueError(
            "P/D media metadata currently requires non-speculative decoding"
        )


def grid_row(value, *, merge_size: int) -> tuple[int, int, int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            "image_grid_thw must contain exactly one [t, h, w] row per image"
        )
    if any(type(x) is not int or x <= 0 for x in value):
        raise ValueError("image grid dimensions must be positive integers")
    t, h, w = value
    if t != 1 or merge_size <= 0 or h % merge_size or w % merge_size:
        raise ValueError(
            "image grid must have t=1 and h/w divisible by spatial_merge_size"
        )
    return t, h, w


def expand_image_prompt(ids, grids, image_token_id: int, merge_size: int, limit: int):
    """Expand exactly one placeholder per image, with a bounded allocation."""
    counts = [t * h * w // merge_size**2 for t, h, w in grids]
    if sum(x == image_token_id for x in ids) != len(grids):
        raise ValueError("Image placeholder count does not match metadata items")
    size = len(ids) - len(grids) + sum(counts)
    if size >= limit:
        raise ValueError(
            f"Expanded image prompt ({size} tokens) exceeds context limit {limit}"
        )
    expanded, spans = [], []
    next_item = 0
    for token in ids:
        if token != image_token_id:
            expanded.append(int(token))
            continue
        count = counts[next_item]
        start = len(expanded)
        expanded.extend([image_token_id] * count)
        spans.append((start, start + count - 1))
        next_item += 1
    return expanded, spans


def image_mrope(length: int, grids, spans, merge_size: int):
    """Qwen image-only position rule, independent of pixel preprocessing."""
    import torch

    positions = torch.empty((3, length), dtype=torch.int64)
    cursor = base = 0
    if len(grids) != len(spans):
        raise ValueError("Grid and span counts differ")
    for grid, (start, end) in zip(grids, spans):
        t, h, w = grid_row(grid, merge_size=merge_size)
        h, w = h // merge_size, w // merge_size
        if not cursor <= start <= end < length or end - start + 1 != t * h * w:
            raise ValueError("Image span is inconsistent with its grid or prompt")
        text_len = start - cursor
        positions[:, cursor:start] = torch.arange(base, base + text_len)
        base += text_len
        positions[0, start : end + 1] = base
        positions[1, start : end + 1] = torch.arange(h).repeat_interleave(w) + base
        positions[2, start : end + 1] = torch.arange(w).repeat(h) + base
        base += max(t, h, w)
        cursor = end + 1
    positions[:, cursor:] = torch.arange(base, base + length - cursor)
    return positions, torch.tensor([[base - cursor]], dtype=torch.int64)


def request_image_metadata(image_data, input_ids, *, image_token_id, merge_size, limit):
    """Return None for normal media; reject mixed metadata/pixel requests."""
    items = image_data if isinstance(image_data, list) else [image_data]
    metadata = [
        isinstance(x, dict) and not x.get("url") and "image_grid_thw" in x
        for x in items
    ]
    if not any(metadata):
        return None
    if not all(metadata):
        raise ValueError("Cannot mix metadata-only and full-media image items")
    from sglang.srt.managers.req_types import (
        MultimodalDataItem,
        MultimodalProcessorOutput,
    )
    from sglang.srt.multimodal.modality import Modality

    grids = [grid_row(x["image_grid_thw"], merge_size=merge_size) for x in items]
    ids, spans = expand_image_prompt(
        input_ids, grids, image_token_id, merge_size, limit
    )
    positions, delta = image_mrope(len(ids), grids, spans, merge_size)
    mm_items = []
    for grid, span in zip(grids, spans):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[span],
            model_specific_data={"image_grid_thw": [list(grid)]},
        )
        # This identity namespaces geometry only. The decode cache is disabled
        # for this mode, so it cannot authorize content reuse.
        item.set_identity(
            hashlib.sha256(b"pd-grid-only-v1" + msgspec.msgpack.encode(grid)).digest()
        )
        mm_items.append(item)
    return MultimodalProcessorOutput(
        input_ids=ids,
        mm_items=mm_items,
        im_token_id=image_token_id,
        mrope_positions=positions,
        mrope_position_delta=delta,
    )


class ImageMetadata(msgspec.Struct, forbid_unknown_fields=True):
    grid: tuple[int, int, int]
    span: tuple[int, int]
    identity: bytes


class MediaMetadata(msgspec.Struct, forbid_unknown_fields=True):
    version: int
    length: int
    image_token_id: int
    merge_size: int
    prompt_digest: bytes
    images: list[ImageMetadata]


def canonical_digest(ids, spans, image_token_id):
    canonical = list(ids)
    for start, end in spans:
        if not 0 <= start <= end < len(canonical):
            raise ValueError("Image span lies outside the prompt")
        canonical[start : end + 1] = [image_token_id] * (end - start + 1)
    return hashlib.sha256(struct.pack(f"<{len(canonical)}q", *canonical)).digest()


def export_metadata(req, *, merge_size: int, capacity: int) -> bytes:
    mm = req.multimodal_inputs
    if mm is None or not mm.mm_items:
        return b""
    images = []
    for item in mm.mm_items:
        if not item.is_image() or not item.offsets or len(item.offsets) != 1:
            raise ValueError(
                "P/D metadata transfer currently supports images with one span each"
            )
        grid = grid_row(
            item.model_specific_data.get("image_grid_thw"), merge_size=merge_size
        )
        if item.identity is None or len(item.identity) != 32:
            raise ValueError("Prefill must compute each image's full content identity")
        images.append(ImageMetadata(grid, tuple(item.offsets[0]), item.identity))
    spans = [x.span for x in images]
    cursor = 0
    for image in images:
        start, end = image.span
        t, h, w = image.grid
        if (
            not cursor <= start <= end < len(req.origin_input_ids)
            or end - start + 1 != t * h * w // merge_size**2
        ):
            raise ValueError(
                "Prefill image span is inconsistent with its grid or prompt"
            )
        cursor = end + 1
    meta = MediaMetadata(
        1,
        len(req.origin_input_ids),
        mm.im_token_id,
        merge_size,
        canonical_digest(req.origin_input_ids, spans, mm.im_token_id),
        images,
    )
    payload = msgspec.msgpack.encode(meta)
    if len(payload) > capacity:
        raise ValueError(
            f"P/D media metadata needs {len(payload)} bytes; capacity is {capacity}"
        )
    return payload


def import_metadata(payload: bytes, req, *, merge_size: int, image_token_id: int):
    """Validate before mutating the request; KV has already been allocated."""
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
    from sglang.srt.multimodal.modality import Modality

    ids = list(req.origin_input_ids)
    if not payload:
        if image_token_id in ids:
            raise ValueError("Prefill did not send metadata for decode's image prompt")
        return
    meta = msgspec.msgpack.decode(payload, type=MediaMetadata)
    if (meta.version, meta.length, meta.image_token_id, meta.merge_size) != (
        1,
        len(ids),
        image_token_id,
        merge_size,
    ):
        raise ValueError(
            "P/D media metadata version, prompt length or model geometry differs"
        )
    spans = [image.span for image in meta.images]
    grids = [image.grid for image in meta.images]
    positions, delta = image_mrope(len(ids), grids, spans, merge_size)
    if canonical_digest(ids, [], image_token_id) != meta.prompt_digest:
        raise ValueError("Decode's expanded prompt does not match prefill")
    items = []
    for image in meta.images:
        if len(image.identity) != 32:
            raise ValueError("Invalid prefill media identity")
        start, end = image.span
        if any(token != image_token_id for token in ids[start : end + 1]):
            raise ValueError("Decode must supply expanded image placeholders")
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[image.span],
            model_specific_data={"image_grid_thw": [list(image.grid)]},
        )
        item.set_identity(image.identity)
        ids[start : end + 1] = item.padding_sequence(end - start + 1)
        items.append(item)
    mm = MultimodalInputs(
        mm_items=items,
        im_token_id=image_token_id,
        mrope_positions=positions,
        mrope_position_delta=delta,
    )
    req.origin_input_ids = array("q", ids)
    req.extend_image_inputs(mm)
    req.full_untruncated_fill_ids = req.origin_input_ids + array("q", req.output_ids)
