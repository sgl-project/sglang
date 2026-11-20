"""Rank-local Qwen-VL image preprocessing for a colocated TP vision tower."""

import hashlib
import logging
import math

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

logger = logging.getLogger(__name__)

DEFERRED_IMAGE_PATCH_WIDTH_KEY = "_qwen_vl_deferred_tp_image"
# Limit the batch by the size of its FP32 processor output.
MAX_FP32_PIXEL_BATCH_BYTES = 512 << 20


def prepare_deferred_image(image, processor, kwargs):
    """Build the prompt grid and CPU pixels from the same processor settings."""
    if not isinstance(image, Image.Image):
        return None
    settings = {
        key: kwargs.get(key, getattr(processor, key, None))
        for key in processor._valid_kwargs_names
    }
    settings.update(kwargs)
    settings = processor._standardize_kwargs(**settings)
    processor._validate_preprocess_kwargs(**settings)
    if image.mode != "RGB" and not settings.get("do_convert_rgb"):
        return None
    patch = settings["patch_size"]
    factor = patch * settings["merge_size"]
    height, width = image.height, image.width
    if settings["do_resize"]:
        size = settings["size"]
        height, width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
    if height % factor or width % factor:
        return None
    grid = (1, height // patch, width // patch)
    patch_width = 3 * settings["temporal_patch_size"] * patch**2
    expected_fp32_bytes = math.prod(grid) * patch_width * torch.float32.itemsize
    if (
        expected_fp32_bytes > MAX_FP32_PIXEL_BATCH_BYTES
        or image.width * image.height * 3 > expected_fp32_bytes
    ):
        return None
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()
    return pixels, torch.tensor([grid], dtype=torch.long), patch_width


def agree_on_image_misses(ordered_candidates, local_misses, group):
    """Return the union of rank-local misses in the shared candidate order."""
    digest = hashlib.sha256()
    for identity, token_count in ordered_candidates:
        digest.update(identity)
        digest.update(token_count.to_bytes(8, "little"))
    order_digest = digest.digest()
    local_miss_mask = tuple(key in local_misses for key in ordered_candidates)
    rank_plans = [None] * group.world_size
    dist.all_gather_object(
        rank_plans, (order_digest, local_miss_mask), group=group.cpu_group
    )
    if any(
        signature != order_digest or len(mask) != len(ordered_candidates)
        for signature, mask in rank_plans
    ):
        raise RuntimeError("Qwen-VL TP ranks disagree on the ordered image cache keys")
    rank_miss_masks = [mask for _, mask in rank_plans]
    return {
        key: ordered_candidates[key]
        for key, miss_by_rank in zip(ordered_candidates, zip(*rank_miss_masks))
        if any(miss_by_rank)
    }


def _assign_images_to_ranks(items, world_size):
    """Assign the largest images first, balancing source pixels by rank."""
    by_owner = [[] for _ in range(world_size)]
    loads = [0] * world_size
    first_owner = int.from_bytes(items[0].cache_key[:8], "little") % world_size
    ranks = tuple((first_owner + offset) % world_size for offset in range(world_size))
    areas = [item.feature.shape[-2] * item.feature.shape[-1] for item in items]
    for index in sorted(range(len(items)), key=areas.__getitem__, reverse=True):
        owner = min(ranks, key=loads.__getitem__)
        by_owner[owner].append(index)
        loads[owner] += areas[index]
    return by_owner


def _preprocess_and_broadcast_pixels(
    items, load_processor, image_kwargs, device, dtype, group
) -> tuple[torch.Tensor, list[int]]:
    """Return packed pixels and their image indices in owner order."""
    patch_rows = [math.prod(item.image_grid_thw.view(-1).tolist()) for item in items]
    patch_width = items[0].model_specific_data[DEFERRED_IMAGE_PATCH_WIDTH_KEY]
    images_by_rank = _assign_images_to_ranks(items, group.world_size)
    image_order = [index for indices in images_by_rank for index in indices]
    rows_by_rank = [
        sum(patch_rows[index] for index in indices) for indices in images_by_rank
    ]

    # Coordinate rank-local preprocessing errors before entering device collectives.
    failure = torch.zeros(1, dtype=torch.int32)
    error = None
    try:
        packed_pixels = torch.empty(
            (sum(patch_rows), patch_width), device=device, dtype=dtype
        )
        rank_spans = packed_pixels.split(rows_by_rank)
        local_indices = images_by_rank[group.rank_in_group]
        if local_indices:
            processor = load_processor()
            targets = rank_spans[group.rank_in_group].split(
                [patch_rows[index] for index in local_indices]
            )
            for index, target in zip(local_indices, targets):
                item = items[index]
                try:
                    output = processor(
                        images=[item.feature],
                        return_tensors="pt",
                        device=str(device),
                        **image_kwargs,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "Qwen-VL rank-local image preprocessing OOM; retrying on CPU"
                    )
                    output = processor(
                        images=[item.feature],
                        return_tensors="pt",
                        device="cpu",
                        **image_kwargs,
                    )
                pixels = output["pixel_values"]
                if (
                    pixels.shape != target.shape
                    or pixels.dtype != torch.float32
                    or not torch.equal(
                        output["image_grid_thw"].cpu(), item.image_grid_thw
                    )
                ):
                    raise ValueError(
                        "Qwen-VL TP image pixels disagree with the prompt grid"
                    )
                target.copy_(pixels)
                del output, pixels
    except Exception as exc:
        failure.fill_(1)
        error = exc
    dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group.cpu_group)
    if failure.item():
        raise RuntimeError("A TP rank could not prepare its Qwen-VL images") from error

    logger.debug(
        "Qwen-VL TP image preprocessing rank=%d local_images=%d batch_images=%d",
        group.rank_in_group,
        len(local_indices),
        len(items),
    )
    for owner, span in enumerate(rank_spans):
        if rows_by_rank[owner]:
            dist.broadcast(span, src=group.ranks[owner], group=group.device_group)
    return packed_pixels, image_order


def _encode_image_batch(
    items, encode, load_processor, image_kwargs, device, dtype, group
):
    packed_pixels, image_order = _preprocess_and_broadcast_pixels(
        items, load_processor, image_kwargs, device, dtype, group
    )
    ordered_items = [items[index] for index in image_order]
    grids = torch.cat([item.image_grid_thw for item in ordered_items], dim=0)
    embeddings = encode(packed_pixels, grid_thw=grids)

    # The tower returns embeddings in owner order. Restore the caller's image order.
    token_counts = [
        item.offsets[0][1] - item.offsets[0][0] + 1 for item in ordered_items
    ]
    embeddings_in_request_order = [None] * len(items)
    for index, embedding in zip(image_order, embeddings.split(token_counts, dim=0)):
        embeddings_in_request_order[index] = embedding
    return embeddings_in_request_order


def encode_tp_images(items, encode, load_processor, image_kwargs, device, dtype, group):
    """Bound the live packed batch and preserve image identity across groups."""
    kwargs = {
        key: value
        for key, value in image_kwargs.items()
        if key not in ("device", "return_tensors")
    }
    result = []
    start = batch_fp32_bytes = 0
    for index, item in enumerate(items):
        patch_width = item.model_specific_data[DEFERRED_IMAGE_PATCH_WIDTH_KEY]
        patch_rows = math.prod(item.image_grid_thw.view(-1).tolist())
        image_fp32_bytes = patch_rows * patch_width * torch.float32.itemsize
        if index > start and (
            batch_fp32_bytes + image_fp32_bytes > MAX_FP32_PIXEL_BATCH_BYTES
            or patch_width
            != items[start].model_specific_data[DEFERRED_IMAGE_PATCH_WIDTH_KEY]
        ):
            result.extend(
                _encode_image_batch(
                    items[start:index],
                    encode,
                    load_processor,
                    kwargs,
                    device,
                    dtype,
                    group,
                )
            )
            start, batch_fp32_bytes = index, 0
        batch_fp32_bytes += image_fp32_bytes
    if start < len(items):
        result.extend(
            _encode_image_batch(
                items[start:], encode, load_processor, kwargs, device, dtype, group
            )
        )
    return result
