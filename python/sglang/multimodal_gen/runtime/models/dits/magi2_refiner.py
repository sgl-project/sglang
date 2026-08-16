# SPDX-License-Identifier: Apache-2.0
"""MAGI-2's 30-layer refiner DiT: one residual stream, dense GQA attention with
output gating, dense swiglu7 MLP, and block-window local attention in every layer.
"""

from __future__ import annotations

import msgspec
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2RefinerArchConfig,
    Magi2RefinerConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    SEQ_BUCKET,
    Magi2BlockGrid,
    block_scan_order,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    Magi2ModalityLinear,
    Magi2ModalityRMSNorm,
    Magi2PostAdapter,
    Magi2PreAdapter,
    Magi2SegmentLayout,
    apply_partial_rope,
    gather_packed_rows,
    pad_rows_to_multiple,
    shard_packed_rows,
    swiglu7_interleaved,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# magi2_refiner_condition_input is "none", so no reference-image kwargs exist.
MAGI2_REFINER_FORWARD_KWARGS = frozenset(
    {"video_latents", "audio_latents", "text_embeds", "layout", "coords", "timestep"}
)


class Magi2RefinerDiT(CachableDiT, LayerwiseOffloadableModuleMixin):
    """MAGI-2's refiner denoiser."""

    _fsdp_shard_conditions = [is_block]
    _compile_conditions = []
    param_names_mapping: dict = {}
    layer_names = ["blocks"]
    _fsdp_mixed_dtype_params = True
    # flex_attention drives the grid mask; this only picks the dense fallback.
    _supported_attention_backends = {AttentionBackendEnum.FA}

    def __init__(
        self,
        config: Magi2RefinerConfig,
        hf_config: dict | None = None,
        *,
        attention: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch: Magi2RefinerArchConfig = self.config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents

        self.pre_adapter = Magi2PreAdapter(arch)
        self.blocks = nn.ModuleList(
            Magi2RefinerBlock(arch, layer_idx=i, attention=attention)
            for i in range(arch.num_layers)
        )
        self.post_adapter = Magi2PostAdapter(arch)

        self.__post_init__()

    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        unexpected = set(kwargs) - MAGI2_REFINER_FORWARD_KWARGS
        if unexpected:
            raise TypeError(f"unexpected forward kwargs: {sorted(unexpected)}")

        layout: Magi2SegmentLayout = kwargs["layout"]
        rows, rope = self.pre_adapter(
            video=kwargs["video_latents"],
            audio=kwargs["audio_latents"],
            text=kwargs["text_embeds"],
            layout=layout,
            coords=kwargs["coords"],
            timestep=kwargs["timestep"],
        )

        # KNOWN DEVIATION: the reference keeps this residual fp32 for all 30 layers;
        # bf16 here trades accumulation error for footprint.
        rows = rows.to(self.blocks[0].attention.linear_qkv.weight.dtype)

        num_tail = layout.total_tokens - layout.video_index.numel()

        # Block-scan order is required by the mask, raster again after
        # (refiner_data_proxy.py:889-891, :921).
        order, restore = block_scan_order(
            grid=Magi2BlockGrid.from_arch_config(
                arch_config=self.config,
                latent_thw=layout.video_latent_thw,
                num_tail_tokens=num_tail,
            ),
            device=rows.device,
        )
        rows = rows.index_select(0, order)
        rope = rope.index_select(0, order)
        modality_ids = layout.modality_ids.index_select(0, order)

        # Rounded to SEQ_BUCKET because the mask compiles per length and keeps every
        # graph, so varying prompt lengths would recompile and leak memory per clip.
        (rows, rope, modality_ids), num_bucket_pad = pad_rows_to_multiple(
            rows, rope, modality_ids, multiple=SEQ_BUCKET
        )

        (rows, rope, modality_ids), plan = shard_packed_rows(rows, rope, modality_ids)
        local_layout = msgspec.structs.replace(layout, modality_ids=modality_ids)

        # Pad rows are declared separately from the tail: pad must not be attendable,
        # since the tail is globally visible and pad repeats a real row.
        grid = Magi2BlockGrid.from_arch_config(
            arch_config=self.config,
            latent_thw=layout.video_latent_thw,
            num_tail_tokens=num_tail,
            num_pad_tokens=num_bucket_pad + plan.num_pad,
        )

        for block in self.blocks:
            rows = block(rows, rope=rope, layout=local_layout, grid=grid)

        rows = gather_packed_rows(rows, plan=plan)
        return self.post_adapter(
            rows.narrow(0, 0, restore.numel()).index_select(0, restore), layout=layout
        )


class Magi2RefinerAttention(nn.Module):
    """Grouped-query attention with per-head output gating."""

    def __init__(
        self,
        config: Magi2RefinerArchConfig,
        *,
        num_modality: int,
        attention: nn.Module,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=num_modality)
        self.linear_qkv = Magi2ModalityLinear(
            hidden, qkv_out, num_modality=num_modality
        )
        self.q_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.k_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.linear_g = Magi2ModalityLinear(
            hidden, self.num_heads, num_modality=num_modality
        )
        self.linear_proj = Magi2ModalityLinear(
            self.num_heads * self.head_dim, hidden, num_modality=num_modality
        )
        self.attention = attention

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: torch.Tensor,
        layout: Magi2SegmentLayout,
        grid: Magi2BlockGrid,
    ) -> torch.Tensor:
        modality_ids = layout.modality_ids
        normed = self.pre_norm(x, modality_ids)
        qkv = self.linear_qkv(normed, modality_ids)
        gate = self.linear_g(normed, modality_ids)

        tokens = qkv.shape[0]
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_width, kv_width, kv_width], dim=-1)
        q = q.view(tokens, self.num_heads, self.head_dim)
        k = k.view(tokens, self.num_kv_heads, self.head_dim)
        v = v.view(tokens, self.num_kv_heads, self.head_dim)

        # KNOWN DEVIATION: the reference carries q and k in fp32 from the norm through
        # the rotary. The norms still reduce in fp32 internally.
        q = self.q_norm(q, modality_ids)
        k = self.k_norm(k, modality_ids)

        sin, cos = rope.tensor_split(2, -1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)

        out = self.attention(q, k, v, grid=grid)
        out = out * torch.sigmoid(gate).unsqueeze(-1)
        return self.linear_proj(out.reshape(tokens, q_width), modality_ids)


class Magi2RefinerMLP(nn.Module):
    """Dense per-modality swiglu7 MLP."""

    def __init__(self, config: Magi2RefinerArchConfig, *, num_modality: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.ffn_hidden_size
        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=num_modality)
        self.up_gate_proj = Magi2ModalityLinear(
            hidden, 2 * inter, num_modality=num_modality
        )
        self.down_proj = Magi2ModalityLinear(inter, hidden, num_modality=num_modality)

    def forward(self, x: torch.Tensor, *, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x, modality_ids)
        h = swiglu7_interleaved(self.up_gate_proj(h, modality_ids))
        return self.down_proj(h, modality_ids)


class Magi2RefinerBlock(nn.Module):
    """Pre-norm attention then MLP, both residual."""

    def __init__(
        self,
        config: Magi2RefinerArchConfig,
        *,
        layer_idx: int,
        attention: nn.Module,
    ) -> None:
        super().__init__()
        num_modality = config.num_modality if layer_idx in config.mm_layers else 1
        self.attention = Magi2RefinerAttention(
            config, num_modality=num_modality, attention=attention
        )
        self.mlp = Magi2RefinerMLP(config, num_modality=num_modality)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: torch.Tensor,
        layout: Magi2SegmentLayout,
        grid: Magi2BlockGrid,
    ) -> torch.Tensor:
        x = x + self.attention(x, rope=rope, layout=layout, grid=grid)
        return x + self.mlp(x, modality_ids=layout.modality_ids)
