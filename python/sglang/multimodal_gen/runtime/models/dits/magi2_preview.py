# SPDX-License-Identifier: Apache-2.0
"""MAGI-2-preview's 40-layer MoE DiT: video, audio and text share one packed
sequence over a 4-stream (mHC) residual; ``mm_layers`` are dense, the rest MoE.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2PreviewArchConfig,
    Magi2PreviewConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_block
from sglang.multimodal_gen.runtime.layers.attention.magi2_sink_attention import (
    Magi2SinkAttention,
)
from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadMoE
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
    shard_packed_rows,
    sharded_cu_seqlens,
    swiglu7_interleaved,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import Magi2MHC
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

MAGI2_PREVIEW_FORWARD_KWARGS = frozenset(
    {
        "video_latents",
        "audio_latents",
        "text_embeds",
        "layout",
        "coords",
        "timestep",
        "ref_patches",
        "ref_special",
    }
)


class Magi2PreviewDiT(CachableDiT, LayerwiseOffloadableModuleMixin):
    """MAGI-2-preview's denoiser.

    Keyword arguments only: ``DenoisingStage`` filters kwargs by signature, so a
    positional forward would silently drop the layout and coordinates.
    """

    _fsdp_shard_conditions = [is_block]
    _compile_conditions = []
    param_names_mapping: dict = {}
    # Adapters/mHC are fp32 and blocks bf16, so FSDP gathers per-parameter dtypes.
    _fsdp_mixed_dtype_params = True
    layer_names = ["blocks"]
    # Sinks and packed varlen are FA-only; other backends drop sink logits silently.
    _supported_attention_backends = {AttentionBackendEnum.FA}

    def __init__(
        self,
        config: Magi2PreviewConfig,
        hf_config: dict | None = None,
        *,
        ep_group=None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch: Magi2PreviewArchConfig = self.config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.num_stream = arch.num_stream

        self.pre_adapter = Magi2PreAdapter(arch)
        self.blocks = nn.ModuleList(
            Magi2PreviewBlock(arch, layer_idx=i, ep_group=ep_group)
            for i in range(arch.num_layers)
        )
        self.post_adapter = Magi2PostAdapter(arch)

        self.__post_init__()

    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        unexpected = set(kwargs) - MAGI2_PREVIEW_FORWARD_KWARGS
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
            ref_patches=kwargs.get("ref_patches"),
            ref_special=kwargs.get("ref_special"),
        )

        streams = rows.view(-1, self.num_stream, self.hidden_size)
        streams = streams.to(self.blocks[0].attention.linear_qkv.weight.dtype)

        (streams, rope, modality_ids), plan = shard_packed_rows(
            streams, rope, layout.modality_ids
        )
        # Not layout.cu_seqlens: pad rows from an uneven split get their own varlen
        # segment so real tokens cannot attend to them.
        cu_seqlens, max_seqlen = sharded_cu_seqlens(plan=plan, device=streams.device)

        for block in self.blocks:
            streams = block(
                streams,
                rope=rope,
                modality_ids=modality_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        rows = gather_packed_rows(streams.reshape(streams.shape[0], -1), plan=plan)
        return self.post_adapter(rows, layout=layout)


class Magi2PreviewBlock(nn.Module):
    """One transformer block: mHC-gated attention then mHC-gated MLP, over
    ``[T, num_stream, hidden]``."""

    def __init__(
        self,
        config: Magi2PreviewArchConfig,
        *,
        layer_idx: int,
        ep_group=None,
    ) -> None:
        super().__init__()
        self.num_stream = config.num_stream
        is_dense = layer_idx in config.mm_layers
        block_modality = config.num_modality if is_dense else 1

        self.mhc_norm = Magi2ModalityRMSNorm(
            config.num_stream * config.hidden_size, num_modality=block_modality
        )
        self.mhc_attn = Magi2MHC(
            num_stream=config.num_stream,
            hidden_size=config.hidden_size,
            alpha_init=config.mhc_alpha_init,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )
        self.mhc_mlp = Magi2MHC(
            num_stream=config.num_stream,
            hidden_size=config.hidden_size,
            alpha_init=config.mhc_alpha_init,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )

        self.attention = Magi2PreviewAttention(config, num_modality=block_modality)
        self.mlp = (
            Magi2DenseMLP(config)
            if is_dense
            else Magi2MoEMLP(config, ep_group=ep_group)
        )

    def forward(
        self,
        streams: torch.Tensor,
        *,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        flat = streams.reshape(streams.shape[0], -1)
        # fp32 out: project() upcasts anyway, so bf16 here would round twice.
        normed = self.mhc_norm(flat, modality_ids, out_dtype=torch.float32)

        h_pre, h_post, h_res = self.mhc_attn.project(normed)
        attn_out = self.attention(
            self.mhc_attn.mix_input(streams, h_pre),
            rope=rope,
            modality_ids=modality_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        streams = self.mhc_attn.mix_output(streams, attn_out, h_post, h_res)

        flat = streams.reshape(streams.shape[0], -1)
        normed = self.mhc_norm(flat, modality_ids, out_dtype=torch.float32)
        h_pre, h_post, h_res = self.mhc_mlp.project(normed)
        mlp_out = self.mlp(
            self.mhc_mlp.mix_input(streams, h_pre), modality_ids=modality_ids
        )
        return self.mhc_mlp.mix_output(streams, mlp_out, h_post, h_res)


class Magi2DenseMLP(nn.Module):
    """Per-modality dense MLP used by the ``mm_layers`` blocks."""

    def __init__(self, config: Magi2PreviewArchConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.dense_intermediate_size
        modality = config.num_modality

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=modality)
        self.up_gate_proj = Magi2ModalityLinear(
            hidden, 2 * inter, num_modality=modality
        )
        self.down_proj = Magi2ModalityLinear(inter, hidden, num_modality=modality)

    def forward(self, x: torch.Tensor, *, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x, modality_ids)
        h = swiglu7_interleaved(self.up_gate_proj(h, modality_ids))
        return self.down_proj(h, modality_ids)


class Magi2MoEMLP(nn.Module):
    """Multi-head MoE plus two always-on shared experts, one modality-agnostic and
    one per-modality."""

    def __init__(self, config: Magi2PreviewArchConfig, *, ep_group=None) -> None:
        super().__init__()
        hidden = config.hidden_size
        modality = config.num_modality
        shared = config.moe_shared_expert_intermediate_size
        modality_shared = config.moe_modality_expert_intermediate_size

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=modality)

        self.split_linear = Magi2ModalityLinear(hidden, hidden)
        self.merge_linear = Magi2ModalityLinear(hidden, hidden)
        self.moe_mlp = Magi2MultiHeadMoE(
            num_heads=config.moe_num_heads,
            num_experts=config.moe_num_experts,
            hidden_size=hidden,
            intermediate_size=config.moe_expert_intermediate_size,
            top_k=config.moe_top_k,
            route_scale=config.moe_route_scale,
            score_func=config.moe_score_func,
            route_norm=config.moe_route_norm,
            ep_group=ep_group,
        )

        self.shared_expert_fc1 = Magi2ModalityLinear(hidden, 2 * shared)
        self.shared_expert_fc2 = Magi2ModalityLinear(shared, hidden)
        self.modality_specific_shared_expert_fc1 = Magi2ModalityLinear(
            hidden, 2 * modality_shared, num_modality=modality
        )
        self.modality_specific_shared_expert_fc2 = Magi2ModalityLinear(
            modality_shared, hidden, num_modality=modality
        )

    def forward(self, x: torch.Tensor, *, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.pre_norm(x, modality_ids)

        routed = self.merge_linear(self.moe_mlp(self.split_linear(h)))

        agnostic = self.shared_expert_fc2(
            swiglu7_interleaved(self.shared_expert_fc1(h))
        )
        per_modality = self.modality_specific_shared_expert_fc2(
            swiglu7_interleaved(
                self.modality_specific_shared_expert_fc1(h, modality_ids)
            ),
            modality_ids,
        )
        return routed + agnostic + per_modality


class Magi2PreviewAttention(nn.Module):
    """Sink attention with QK-norm and per-head output gating."""

    def __init__(self, config: Magi2PreviewArchConfig, *, num_modality: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_heads = heads

        self.pre_norm = Magi2ModalityRMSNorm(hidden, num_modality=num_modality)
        self.linear_qkv = Magi2ModalityLinear(
            hidden, 3 * hidden, num_modality=num_modality
        )
        self.q_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.k_norm = Magi2ModalityRMSNorm(self.head_dim, num_modality=num_modality)
        self.linear_g = Magi2ModalityLinear(hidden, heads, num_modality=num_modality)
        self.linear_proj = Magi2ModalityLinear(
            hidden, hidden, num_modality=num_modality
        )

        self.attn = Magi2SinkAttention(
            num_heads=heads,
            head_dim=self.head_dim,
            sink_token_num=config.sink_token_num,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: torch.Tensor,
        modality_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        normed = self.pre_norm(x, modality_ids)
        qkv = self.linear_qkv(normed, modality_ids)
        gate = self.linear_g(normed, modality_ids)

        tokens = qkv.shape[0]
        qkv = qkv.view(tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # fp32 through the rope; the reference converts only at the attention boundary.
        q = self.q_norm(q, modality_ids, out_dtype=torch.float32)
        k = self.k_norm(k, modality_ids, out_dtype=torch.float32)

        sin, cos = rope.tensor_split(2, -1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)

        # FlashAttention requires q, k and v to share a dtype.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        out = self.attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out = out * torch.sigmoid(gate).unsqueeze(-1)
        out = out.reshape(tokens, self.num_heads * self.head_dim)
        return self.linear_proj(out, modality_ids)
