# SPDX-License-Identifier: Apache-2.0
"""Layerwise-offload adapter for the transformers Gemma-4-Unified encoder.

LTX-2.5's text encoder has no SGLang implementation and loads through
``transformers.AutoModel`` as ``Gemma4UnifiedModel``. Its decoder stack is a
plain ``ModuleList`` at ``language_model.layers``, which is all the layerwise
manager needs, so the loaded module's class is swapped for a subclass that also
carries the mixin. Forward, signature and weights are untouched.
"""

import torch
from torch import nn

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


class _Gemma4UnifiedLayerwise(LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # 48 decoder layers of ~454 MB in bf16: 21.8 of the 23.9 GB.
    layer_names = ["language_model.layers"]
    # The 2.0 GB vocab table is reached only through embed_tokens.__call__;
    # AutoModel builds no tied lm_head.
    host_resident_table_names = ["language_model.embed_tokens"]


class _HostGatherScaledEmbedding:
    """Apply Gemma's 0-dim ``embed_scale`` on whichever device ran the gather.

    Upstream multiplies by the buffer in place; under host residency the gather
    runs on the host while the buffer follows the module to the device.
    """

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = nn.Embedding.forward(self, input_ids)
        return embeds * self.embed_scale.to(device=embeds.device, dtype=embeds.dtype)


def _swap_class(module: nn.Module, *, extra: type, first: bool) -> None:
    cls = type(module)
    if issubclass(cls, extra):
        return
    bases = (extra, cls) if first else (cls, extra)
    module.__class__ = type(cls.__name__, bases, {"__module__": cls.__module__})


def make_gemma4_unified_layerwise_offloadable(model: nn.Module) -> nn.Module:
    _swap_class(model, extra=_Gemma4UnifiedLayerwise, first=False)
    _swap_class(
        model.language_model.embed_tokens,
        extra=_HostGatherScaledEmbedding,
        first=True,
    )
    return model


# transformers class name -> adapter applied after the native load
NATIVE_LAYERWISE_ADAPTERS = {
    "Gemma4UnifiedModel": make_gemma4_unified_layerwise_offloadable,
}
