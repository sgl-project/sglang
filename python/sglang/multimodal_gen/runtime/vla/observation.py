# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.managers.mm_utils import tensor_hash


@dataclass
class VLAObservationBatch:
    prompt: list[str]
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor | None
    noise: torch.Tensor | None
    tokens: torch.Tensor
    token_masks: torch.Tensor
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash tensor content with SRT's CPU/CUDA implementation."""

    shape = ",".join(str(dim) for dim in tensor.shape)
    return f"{tensor.dtype}:{shape}:{tensor_hash(tensor):016x}"


def collate_vla_observation_batches(
    observations: list[VLAObservationBatch],
) -> VLAObservationBatch:
    first = observations[0]
    camera_order = tuple(first.metadata.get("camera_order", ()))
    images = {
        name: torch.cat([obs.images[name] for obs in observations], dim=0)
        for name in camera_order
    }
    image_masks = {
        name: torch.cat([obs.image_masks[name] for obs in observations], dim=0)
        for name in camera_order
    }
    states = [obs.state for obs in observations]
    noises = [obs.noise for obs in observations]
    if any(item is None for item in states) and not all(
        item is None for item in states
    ):
        raise ValueError("Cannot collate mixed VLA state presence")
    if any(item is None for item in noises) and not all(
        item is None for item in noises
    ):
        raise ValueError("Cannot collate mixed VLA noise presence")
    state = (
        None
        if states[0] is None
        else torch.cat([item for item in states if item is not None], dim=0)
    )
    noise = (
        None
        if noises[0] is None
        else torch.cat([item for item in noises if item is not None], dim=0)
    )
    return VLAObservationBatch(
        prompt=[prompt for obs in observations for prompt in obs.prompt],
        images=images,
        image_masks=image_masks,
        state=state,
        noise=noise,
        tokens=torch.cat([obs.tokens for obs in observations], dim=0),
        token_masks=torch.cat([obs.token_masks for obs in observations], dim=0),
        batch_size=len(observations),
        metadata={"camera_order": camera_order},
    )
