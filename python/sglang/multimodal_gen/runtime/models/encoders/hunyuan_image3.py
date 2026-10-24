# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from transformers import Siglip2VisionConfig

from sglang.srt.models.siglip2 import Siglip2Model


class _PerImageSDPA(torch.nn.Module):
    """Keep independent images separate without a quadratic packed-sequence mask."""

    def forward(self, q, k, v, cu_seqlens, **kwargs):
        boundaries = cu_seqlens.tolist()
        output = torch.empty_like(q)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output[start:end] = (
                F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1).unsqueeze(0),
                    k[start:end].transpose(0, 1).unsqueeze(0),
                    v[start:end].transpose(0, 1).unsqueeze(0),
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return output


class HunyuanImage3VisionModel(Siglip2Model):
    """Adapt padded HunyuanImage-3 conditioning inputs to SRT's packed SigLIP2."""

    def __init__(self, config: dict):
        config = dict(config, hidden_act="gelu")
        # The checkpoint stores a legacy read-only Transformers property.
        config.pop("use_return_dict", None)
        super().__init__(
            Siglip2VisionConfig(**config),
            qkv_backend="sdpa",
            use_data_parallel=True,
        )
        for layer in self.vision_model.encoder.layers:
            layer.self_attn.attn.qkv_backend = _PerImageSDPA()

    def forward(self, pixel_values, attention_mask, spatial_shapes):
        batch_size, max_patches, _ = pixel_values.shape
        mask = attention_mask.reshape(batch_size, max_patches).bool()
        # shape metadata stays on CPU and is shared by every encoder layer
        spatial_shapes = spatial_shapes.reshape(-1, 2).cpu()
        lengths = spatial_shapes.prod(dim=1).to(torch.int32)
        cu_seqlens = torch.nn.functional.pad(
            lengths.cumsum(0, dtype=torch.int32), (1, 0)
        )
        hidden_states = (
            super()
            .forward(
                pixel_values_packed=pixel_values[mask],
                spatial_shapes=spatial_shapes,
                cu_seqlens=cu_seqlens,
                max_seqlen=int(lengths.max()),
            )
            .squeeze(0)
        )
        output = hidden_states.new_zeros(
            batch_size, max_patches, hidden_states.shape[-1]
        )
        output[mask] = hidden_states
        return output

    def load_weights(self, weights):
        return super().load_weights(
            (f"vision_model.{name}", weight) for name, weight in weights
        )
