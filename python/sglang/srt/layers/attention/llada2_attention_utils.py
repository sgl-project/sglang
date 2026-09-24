# SPDX-License-Identifier: Apache-2.0
"""Mask helpers shared by LLaDA2 attention backends."""

from __future__ import annotations

import torch


def build_llada_image_custom_mask(
    text_lens: list[int],
    sequence_lens: list[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Build the official one-pass text/query block mask in ragged layout."""
    if len(text_lens) != len(sequence_lens):
        raise RuntimeError("LLaDA-Image conditioning metadata batch mismatch")

    flattened_masks: list[torch.Tensor] = []
    for text_len, sequence_len in zip(text_lens, sequence_lens):
        text_len = int(text_len)
        sequence_len = int(sequence_len)
        if text_len <= 0 or text_len >= sequence_len:
            raise RuntimeError(
                "LLaDA-Image conditioning text length must leave query tokens"
            )
        attention_mask = torch.ones(
            (sequence_len, sequence_len), dtype=torch.bool, device=device
        )
        attention_mask[:text_len, text_len:] = False
        flattened_masks.append(attention_mask.flatten())
    return torch.cat(flattened_masks)
