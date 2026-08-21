# SPDX-License-Identifier: Apache-2.0
"""Per-model adapters for the ComfyUI DiT-forward contract.

The shared executor owns request construction and the ZMQ round-trip.
Each adapter only translates between ComfyUI's ``apply_model`` tensors and
the fields SGLang's ``Req`` expects. Design the interface around H3 (nested
latents, structured payload), not Flux's three-tensor case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

_ADAPTERS: dict[str, type["ComfyUIModelAdapter"]] = {}


@dataclass
class PackedForward:
    """One ComfyUI sampler step, already translated into SGLang tensors."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: list[torch.Tensor]
    height: int
    width: int
    guidance_scale: float = 1.0
    prompt_seq_lens: list[list[int]] | None = None
    pooled_embeds: list[torch.Tensor] | None = None
    extra_req: dict[str, Any] = field(default_factory=dict)
    unpack_ctx: dict[str, Any] = field(default_factory=dict)


class ComfyUIModelAdapter:
    """Model-specific pack / unpack / fill_req for one ComfyUI DiT family."""

    model_types: tuple[str, ...] = ()
    pipeline_class_name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for model_type in cls.model_types:
            _ADAPTERS[model_type] = cls

    def pack(self, x: torch.Tensor, timestep: torch.Tensor, context, **kwargs) -> PackedForward:
        raise NotImplementedError

    def unpack(self, noise_pred: torch.Tensor, packed: PackedForward, x: torch.Tensor) -> torch.Tensor:
        return noise_pred.to(x.device)

    def fill_req(self, req, packed: PackedForward) -> None:
        req.latents = packed.latents
        req.timesteps = packed.timesteps
        req.prompt_embeds = packed.prompt_embeds
        req.raw_latent_shape = torch.tensor(packed.latents.shape, dtype=torch.long)
        req.do_classifier_free_guidance = False
        if packed.prompt_seq_lens is not None:
            req.prompt_seq_lens = packed.prompt_seq_lens
        if packed.pooled_embeds is not None:
            req.pooled_embeds = packed.pooled_embeds
        for key, value in packed.extra_req.items():
            setattr(req, key, value)


def get_adapter_class(model_type: str) -> type[ComfyUIModelAdapter]:
    if model_type not in _ADAPTERS:
        raise ValueError(
            f"Unsupported ComfyUI model type {model_type!r}. "
            f"Registered: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[model_type]


def registered_model_types() -> list[str]:
    return sorted(_ADAPTERS)
