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


class Qwen2_5VLGenerationEncoder(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    """Qwen2.5-VL encoder for stages that also use autoregressive generation."""

    layerwise_offload_dit_group_enabled = False
    # release the larger language stack before allocating vision offload buffers
    layer_names = [
        "transformers_model.model.language_model.layers",
        "transformers_model.model.visual.blocks",
    ]

    def __init__(self, model: Qwen2_5_VLForConditionalGeneration) -> None:
        super().__init__()
        self.transformers_model = model

    def forward(self, *args, **kwargs):
        return self.transformers_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.transformers_model.generate(*args, **kwargs)


def load_qwen2_5vl_generation_model(
    model_path: str,
    *,
    server_args: ServerArgs,
    dtype: torch.dtype,
) -> Qwen2_5VLGenerationEncoder:
    """Load the Transformers implementation needed for autoregressive generation."""
    device = (
        torch.device("cpu")
        if server_args.should_start_component_on_cpu("text_encoder")
        else get_local_torch_device()
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        torch_dtype=dtype,
    )
    return Qwen2_5VLGenerationEncoder(model).to(device=device)
