# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 hybrid causal/image-span attention masks."""

from __future__ import annotations

import torch


def build_u1_hybrid_allowed_matrix(
    t_indexes: torch.Tensor,
    image_token_tag: torch.Tensor,
) -> torch.Tensor:
    t_indexes = t_indexes.flatten()
    image_token_tag = image_token_tag.flatten().bool()
    if t_indexes.numel() != image_token_tag.numel():
        raise ValueError("image_token_tag length must match t_indexes length")

    positions = torch.arange(t_indexes.numel(), device=t_indexes.device)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    same_t = t_indexes.unsqueeze(1) == t_indexes.unsqueeze(0)
    same_image_span = (
        same_t & image_token_tag.unsqueeze(1) & image_token_tag.unsqueeze(0)
    )
    return causal | same_image_span


def build_u1_hybrid_backend_mask(
    indexes: torch.Tensor,
    image_token_tag: torch.Tensor,
    extend_seq_lens: list[int],
    extend_prefix_lens: list[int],
    *,
    force_custom_mask: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if indexes.ndim != 2 or indexes.shape[0] != 3:
        raise ValueError("indexes must have shape [3, total_extend_tokens]")
    if len(extend_seq_lens) != len(extend_prefix_lens):
        raise ValueError("extend and prefix length counts must match")

    image_token_tag = image_token_tag.flatten().bool()
    if image_token_tag.numel() != indexes.shape[1]:
        raise ValueError("image_token_tag length must match indexes")

    mask_parts = []
    mask_indptr = [0]
    token_offset = 0
    for extend_len, prefix_len in zip(
        extend_seq_lens,
        extend_prefix_lens,
        strict=True,
    ):
        request_indexes = indexes[:, token_offset : token_offset + extend_len]
        request_tag = image_token_tag[token_offset : token_offset + extend_len]
        current_mask = build_u1_hybrid_allowed_matrix(
            request_indexes[0],
            request_tag,
        )
        if prefix_len:
            prefix_mask = torch.ones(
                (extend_len, prefix_len),
                dtype=torch.bool,
                device=indexes.device,
            )
            current_mask = torch.cat([prefix_mask, current_mask], dim=1)
        flat_mask = current_mask.flatten()
        mask_parts.append(flat_mask)
        mask_indptr.append(mask_indptr[-1] + flat_mask.numel())
        token_offset += extend_len

    if token_offset != indexes.shape[1]:
        raise ValueError("extend lengths do not consume all indexes")

    indptr = torch.tensor(
        mask_indptr,
        dtype=torch.int64,
        device=indexes.device,
    )
    if not force_custom_mask and not image_token_tag.any():
        return None, indptr
    return torch.cat(mask_parts), indptr


__all__ = [
    "build_u1_hybrid_allowed_matrix",
    "build_u1_hybrid_backend_mask",
]
