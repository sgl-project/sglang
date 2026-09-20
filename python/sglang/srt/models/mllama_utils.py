"""Image layout and cross-attention visibility for Mllama."""

from typing import List, Sequence, Tuple

import torch


def mllama_image_layout(mm_input) -> Tuple[List[int], List[torch.Tensor]]:
    """Return image positions and tile masks in encoder KV order.

    Offsets are inclusive spans in decoder text coordinates, with one token
    per image.
    Aspect-ratio metadata survives release of the pixel tensors after prefill.
    """
    positions, tile_masks = [], []
    for item in mm_input.mm_items:
        item_positions = [
            pos for start, end in item.offsets for pos in range(start, end + 1)
        ]
        masks = torch.as_tensor(item.model_specific_data["aspect_ratio_mask"])
        masks = masks.reshape(-1, masks.shape[-1])
        positions.extend(item_positions)
        tile_masks.extend(masks.unbind(0))
    return positions, tile_masks


def build_mllama_cross_attention_mask(
    image_positions: Sequence[int],
    tile_masks: Sequence[torch.Tensor],
    num_patches: int,
    query_start: int,
    query_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean [text queries, encoder tokens] visibility mask.

    An image is visible from its marker through the text preceding the next
    image group. Consecutive markers share an end position, matching the
    Transformers Mllama processor. The final group remains visible during
    generation. Each tile contributes ``num_patches`` encoder tokens.
    """
    query_positions = torch.arange(
        query_start, query_start + query_length, device=device
    )
    ends = [*image_positions[1:], query_start + query_length]
    for i in range(len(ends) - 2, -1, -1):
        if image_positions[i] + 1 == image_positions[i + 1]:
            ends[i] = ends[i + 1]

    parts = []
    for start, end, tile_mask in zip(image_positions, ends, tile_masks):
        visible = (query_positions >= start) & (query_positions < end)
        parts.append(
            visible[:, None] & tile_mask.to(device=device, dtype=torch.bool)[None, :]
        )
    if not parts:
        return torch.empty(query_length, 0, dtype=torch.bool, device=device)
    return torch.cat(parts, dim=1).repeat_interleave(num_patches, dim=1)
