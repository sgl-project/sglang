# SPDX-License-Identifier: Apache-2.0
"""Build and whole-step dispatch for --enable-nvidia-megakernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.megakernels.spec import BuiltMegakernel

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def megakernel_enabled() -> bool:
    """Whether this server runs with --enable-nvidia-megakernel."""
    raise NotImplementedError


def install_megakernel(
    *,
    model: torch.nn.Module,
    hf_config: PretrainedConfig,
    capture_tail_hooks: list[Callable[..., None]],
) -> None:
    """Build the model's spec and install its forward; runs before KV cache sizing."""
    raise NotImplementedError


def built_megakernel(model: torch.nn.Module) -> Optional[BuiltMegakernel]:
    """The object the spec built for this model, or None."""
    raise NotImplementedError


def megakernel_draft_sampler(draft_model: torch.nn.Module) -> Optional[Callable]:
    """The built draft spec's draft_sampler factory, or None for SGLang's proposal."""
    raise NotImplementedError
