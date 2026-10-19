# SPDX-License-Identifier: Apache-2.0
"""Shared layer discovery using the original CUDA DWDP selection rules."""


def collect_moe_layers(model):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    decoder = model.model if hasattr(model, "model") else model
    moe_layers = []
    for layer_idx, layer in enumerate(decoder.layers):
        experts = next((m for m in layer.modules() if isinstance(m, FusedMoE)), None)
        if experts is not None:
            moe_layers.append((layer_idx, experts))
    return moe_layers
