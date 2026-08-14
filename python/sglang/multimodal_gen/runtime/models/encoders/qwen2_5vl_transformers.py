# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Qwen2.5-VL loading for stages that require ``generate``."""

from __future__ import annotations

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LayerwiseOffloadableQwen2_5VLForConditionalGeneration(
    LayerwiseOffloadableModuleMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    """Transformers Qwen2.5-VL with SGLang layerwise residency support."""

    layerwise_offload_dit_group_enabled = False
    # Configure the larger language stack first to keep setup peak memory low.
    layer_names = ["model.language_model.layers", "model.visual.blocks"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Transformers selects legacy Qwen weight conversions by the concrete
        # class name. Load through the upstream class before adding this
        # stateless mixin so checkpoints keep their original conversion path.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(*args, **kwargs)
        model.__class__ = cls
        return model


def load_qwen2_5vl_generation_model(
    model_path: str,
    *,
    server_args: ServerArgs,
    dtype: torch.dtype,
) -> LayerwiseOffloadableQwen2_5VLForConditionalGeneration:
    """Load the Transformers implementation needed for autoregressive generation."""
    device = (
        torch.device("cpu")
        if server_args.should_start_component_on_cpu("text_encoder")
        else get_local_torch_device()
    )
    model = LayerwiseOffloadableQwen2_5VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        torch_dtype=dtype,
    )
    return model.to(device=device)
