# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers/models/transformers/transformer_ovis_image.py,
# commit c6df88a511a98740646ee55577b590c9852650ce.
"""Native Ovis-Image transformer with tensor and sequence parallelism."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from torch import nn

from sglang.multimodal_gen.configs.models.dits.ovis_image import OvisImageConfig
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    SpShard,
    build_shard_plan,
    gather_seq,
    join_seqs,
    shard_like,
    should_shard_text,
    split_seqs,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    _apply_rotary_emb_complex,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.common import get_qkv_projections
from sglang.multimodal_gen.runtime.models.dits.flux import (
    FluxAttention,
    FluxPosEmbed,
    _rope_complex_freqs,
)


def _is_ovis_image_block(name, module):
    return is_module_list_entry_in(
        name, ("transformer_blocks", "single_transformer_blocks")
    )


class OvisImageAttention(FluxAttention):
    """Keep distributed FLUX projections with FP32 and small-head RoPE."""

    def __init__(self, *args, **kwargs):
        # Official nn.RMSNorm rounds the weighted norm to the activation dtype
        # before its FP32 RoPE. Keep that boundary in the fused CUDA path.
        kwargs["round_norm_before_rope"] = True
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        encoder_hidden_states=None,
        freqs_cis=None,
        complex_freqs=None,
        num_replicated_prefix=0,
        attn_mask=None,
        attn_mask_meta=None,
        skip_sequence_parallel_override=False,
    ):
        meta = attn_mask_meta or {}
        pad_start, pad_end = meta.get("pad_start"), meta.get("pad_end")
        replicated_text_with_padding = (
            num_replicated_prefix > 0
            and attn_mask is not None
            and get_ring_parallel_world_size() > 1
        )
        if (
            x.shape[0] > 1
            and get_ring_parallel_world_size() > 1
            and pad_start is not None
            and pad_end is not None
            and pad_end > pad_start
        ):
            # Shared Ring tail-pad attention accepts one batch row at a time.
            # Keep its real distributed kernel and local sequence layout while
            # splitting only this unsupported padded-batch edge. Do not mutate
            # the metadata shared by subsequent blocks or request captures.
            outputs = []
            for index in range(x.shape[0]):
                row_meta = dict(meta)
                row_meta["cu_seqlens_tail"] = torch.tensor(
                    [0, pad_start, pad_end], device=x.device, dtype=torch.int32
                )
                outputs.append(
                    self.forward(
                        x[index : index + 1],
                        encoder_hidden_states=(
                            encoder_hidden_states[index : index + 1]
                            if encoder_hidden_states is not None
                            else None
                        ),
                        freqs_cis=freqs_cis,
                        complex_freqs=complex_freqs,
                        num_replicated_prefix=num_replicated_prefix,
                        attn_mask=(
                            attn_mask[index : index + 1]
                            if attn_mask is not None
                            else None
                        ),
                        attn_mask_meta=row_meta,
                    )
                )
            if encoder_hidden_states is not None:
                return tuple(
                    torch.cat([output[stream] for output in outputs], dim=0)
                    for stream in range(2)
                )
            return torch.cat(outputs, dim=0)

        if (
            x.dtype != torch.float32
            and self.head_dim % 16 == 0
            and not replicated_text_with_padding
            and not skip_sequence_parallel_override
        ):
            return super().forward(
                x,
                encoder_hidden_states,
                freqs_cis,
                complex_freqs,
                num_replicated_prefix,
                attn_mask,
                attn_mask_meta,
            )

        # The shared CUDA RoPE kernel dispatches only FP16/BF16, and its small
        # head fallback uses 16-element vectors without tail guards. Complex
        # RoPE keeps FP32 activations and handles heads not divisible by 16.
        if complex_freqs is None:
            complex_freqs = _rope_complex_freqs(freqs_cis)
        query, key, value, text_query, text_key, text_value = get_qkv_projections(
            self, x, encoder_hidden_states
        )
        num_heads = self.local_heads if self.shard_qkv else self.heads
        query = self.norm_q(query.unflatten(-1, (num_heads, -1)))
        key = self.norm_k(key.unflatten(-1, (num_heads, -1)))
        value = value.unflatten(-1, (num_heads, -1))
        text_pad = (attn_mask_meta or {}).get("local_pad", 0)
        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            text_query = self.norm_added_q(text_query.unflatten(-1, (num_heads, -1)))
            text_key = self.norm_added_k(text_key.unflatten(-1, (num_heads, -1)))
            text_value = text_value.unflatten(-1, (num_heads, -1))
            if complex_freqs is not None:
                text_freqs = complex_freqs[:text_len].unsqueeze(-2)
                image_freqs = complex_freqs[text_len:].unsqueeze(-2)
                text_query = _apply_rotary_emb_complex(text_query, text_freqs)
                text_key = _apply_rotary_emb_complex(text_key, text_freqs)
                query = _apply_rotary_emb_complex(query, image_freqs)
                key = _apply_rotary_emb_complex(key, image_freqs)
            query = join_seqs(text_query, query, text_pad)
            key = join_seqs(text_key, key, text_pad)
            value = join_seqs(text_value, value, text_pad)
        elif complex_freqs is not None:
            freqs = complex_freqs.unsqueeze(-2)
            query = _apply_rotary_emb_complex(query, freqs)
            key = _apply_rotary_emb_complex(key, freqs)

        if replicated_text_with_padding:
            # Very short text can remain replicated because its padding would
            # span multiple SP shards. Ring does not support a replicated
            # prefix together with an image key mask: gather image KV once,
            # retain a single text prefix, and attend with the original local
            # queries. TP heads and output sequence shards remain unchanged.
            prefix = num_replicated_prefix
            full_key = torch.cat(
                [
                    key[:, :prefix],
                    sequence_model_parallel_all_gather(
                        key[:, prefix:].contiguous(), dim=1
                    ),
                ],
                dim=1,
            )
            full_value = torch.cat(
                [
                    value[:, :prefix],
                    sequence_model_parallel_all_gather(
                        value[:, prefix:].contiguous(), dim=1
                    ),
                ],
                dim=1,
            )
            full_mask = torch.cat(
                [
                    attn_mask[:, :prefix],
                    sequence_model_parallel_all_gather(
                        attn_mask[:, prefix:].contiguous(), dim=1
                    ),
                ],
                dim=1,
            )
            output = self.attn(
                query,
                full_key,
                full_value,
                attn_mask=full_mask,
                skip_sequence_parallel_override=True,
            )
        else:
            output = self.attn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                attn_mask_meta=attn_mask_meta,
                num_replicated_prefix=num_replicated_prefix,
                skip_sequence_parallel_override=skip_sequence_parallel_override,
            )
        output = output.flatten(2, 3)
        output = output.to(query.dtype)
        if encoder_hidden_states is not None:
            text_output, output = split_seqs(output, text_len, text_pad)
            text_output = self.to_add_out(text_output)[0]
        if not self.pre_only:
            output = self.to_out[0](output)[0]
            if len(self.to_out) == 2:
                output = self.to_out[1](output)
        if encoder_hidden_states is not None:
            return output, text_output
        return output


class OvisImageSwiGLU(nn.Module):
    def __init__(self, dim, inner_dim, prefix):
        super().__init__()
        self.proj = MergedColumnParallelLinear(
            dim,
            [inner_dim, inner_dim],
            bias=True,
            gather_output=False,
            prefix=f"{prefix}.proj",
        )

    def forward(self, hidden_states):
        hidden_states, _ = self.proj(hidden_states)
        value, gate = hidden_states.chunk(2, dim=-1)
        return value * F.silu(gate)


class OvisImageFeedForward(nn.Module):
    def __init__(self, dim, prefix):
        super().__init__()
        inner_dim = 4 * dim
        self.net = nn.ModuleList(
            [
                OvisImageSwiGLU(dim, inner_dim, f"{prefix}.net.0"),
                nn.Dropout(0.0),
                RowParallelLinear(
                    inner_dim,
                    dim,
                    bias=True,
                    input_is_parallel=True,
                    prefix=f"{prefix}.net.2",
                ),
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        return self.net[2](hidden_states)[0]


class OvisImageTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, prefix):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = OvisImageAttention(
            query_dim=dim,
            num_heads=num_heads,
            dim_head=head_dim,
            added_kv_proj_dim=dim,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=1e-6,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = OvisImageFeedForward(dim, f"{prefix}.ff")
        self.ff_context = OvisImageFeedForward(dim, f"{prefix}.ff_context")

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        freqs_cis,
        num_replicated_prefix=0,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        attention_kwargs = joint_attention_kwargs or {}
        attn_output, context_attn_output = self.attn(
            x=norm_hidden_states,
            encoder_hidden_states=norm_context,
            freqs_cis=freqs_cis,
            num_replicated_prefix=num_replicated_prefix,
            **attention_kwargs,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(
            norm_hidden_states
        )
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        )
        norm_context = self.norm2_context(encoder_hidden_states)
        norm_context = norm_context * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(
            1
        ) * self.ff_context(norm_context)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class OvisImageSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, prefix):
        super().__init__()
        self.mlp_hidden_dim = 4 * dim
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = MergedColumnParallelLinear(
            dim,
            [self.mlp_hidden_dim, self.mlp_hidden_dim],
            bias=True,
            gather_output=False,
            prefix=f"{prefix}.proj_mlp",
        )
        self.attn = OvisImageAttention(
            query_dim=dim,
            num_heads=num_heads,
            dim_head=head_dim,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            prefix=f"{prefix}.attn",
        )
        self.proj_out = RowParallelLinear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            prefix=f"{prefix}.proj_out",
        )
        self._patch_proj_out_weight_loader(dim)

    def _patch_proj_out_weight_loader(self, dim):
        projection = self.proj_out
        local_dim = dim // projection.tp_size
        local_mlp_dim = self.mlp_hidden_dim // projection.tp_size

        def weight_loader(param, loaded_weight):
            input_dim = getattr(param, "input_dim", None)
            if input_dim is not None:
                # Checkpoint columns are [attn_full, mlp_full]; each TP rank
                # consumes the corresponding slice of BOTH segments.
                attn = loaded_weight.narrow(
                    input_dim, projection.tp_rank * local_dim, local_dim
                )
                mlp = loaded_weight.narrow(
                    input_dim,
                    dim + projection.tp_rank * local_mlp_dim,
                    local_mlp_dim,
                )
                loaded_weight = torch.cat([attn, mlp], dim=input_dim)
            with torch.no_grad():
                param.copy_(loaded_weight)

        projection.weight_loader = weight_loader
        projection.weight.weight_loader = weight_loader
        projection.bias.weight_loader = weight_loader

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        freqs_cis,
        num_replicated_prefix=0,
        joint_attention_kwargs=None,
    ):
        text_seq_len = encoder_hidden_states.shape[1]
        attention_kwargs = joint_attention_kwargs or {}
        text_pad = (attention_kwargs.get("attn_mask_meta") or {}).get("local_pad", 0)
        hidden_states = join_seqs(encoder_hidden_states, hidden_states, text_pad)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp, _ = self.proj_mlp(norm_hidden_states)
        value, mlp_gate = mlp.chunk(2, dim=-1)
        mlp = value * F.silu(mlp_gate)
        attention = self.attn(
            x=norm_hidden_states,
            freqs_cis=freqs_cis,
            num_replicated_prefix=num_replicated_prefix,
            **attention_kwargs,
        )
        update = self.proj_out(torch.cat([attention, mlp], dim=-1))[0]
        hidden_states = residual + gate.unsqueeze(1) * update
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return split_seqs(hidden_states, text_seq_len, text_pad)


class OvisImageTransformer2DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = [_is_ovis_image_block]
    _compile_conditions = [_is_ovis_image_block]
    layer_names = ["transformer_blocks", "single_transformer_blocks"]
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    def __init__(self, config: OvisImageConfig, hf_config, quant_config=None):
        super().__init__(config=config, hf_config=hf_config)
        if quant_config is not None:
            raise ValueError(
                "Native Ovis-Image currently supports unquantized checkpoints"
            )
        arch = self.config
        if arch.num_attention_heads % get_tp_world_size():
            raise ValueError("Ovis-Image attention heads must be divisible by TP size")
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels // 4
        self.rotary_emb = FluxPosEmbed(theta=10000, axes_dim=arch.axes_dims_rope)
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=arch.hidden_size
        )
        self.context_embedder_norm = nn.RMSNorm(arch.joint_attention_dim, eps=1e-6)
        self.context_embedder = ReplicatedLinear(
            arch.joint_attention_dim, arch.hidden_size, bias=True
        )
        self.x_embedder = ReplicatedLinear(
            arch.in_channels, arch.hidden_size, bias=True
        )
        self.transformer_blocks = nn.ModuleList(
            [
                OvisImageTransformerBlock(
                    arch.hidden_size,
                    arch.num_attention_heads,
                    arch.attention_head_dim,
                    f"transformer_blocks.{index}",
                )
                for index in range(arch.num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                OvisImageSingleTransformerBlock(
                    arch.hidden_size,
                    arch.num_attention_heads,
                    arch.attention_head_dim,
                    f"single_transformer_blocks.{index}",
                )
                for index in range(arch.num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            arch.hidden_size, arch.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ReplicatedLinear(
            arch.hidden_size,
            arch.patch_size * arch.patch_size * arch.out_channels,
            bias=True,
        )
        self.__post_init__()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            parameter = parameters[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, weight)
            loaded.add(name)
        return loaded

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        freqs_cis,
        joint_attention_kwargs=None,
    ):
        image_len = hidden_states.shape[1]
        text_len = encoder_hidden_states.shape[1]
        hidden_states = self.x_embedder(hidden_states)[0]
        # The native denoising stage supplies scheduler timesteps in [0, 1000].
        # Diffusers divides by 1000 in its pipeline and multiplies back in DiT.
        # The public interface accepts scheduler units [0, 1000]. Diffusers
        # normalizes in the latent dtype before its DiT converts back; retain
        # that BF16/FP16 rounding rather than skipping the dtype roundtrip.
        normalized_timestep = timestep.to(hidden_states.dtype) / 1000
        projected_timestep = self.time_proj(normalized_timestep * 1000)
        temb = self.timestep_embedder(projected_timestep.to(hidden_states.dtype))
        encoder_hidden_states = self.context_embedder(
            self.context_embedder_norm(encoder_hidden_states)
        )[0]
        image_shard = build_shard_plan(image_len)
        # Tail padding must fit in the final SP shard. Extremely small legal
        # images can violate that invariant: retain full sequences on each SP
        # rank and bypass only SP attention instead of masking real text rows.
        # The existing TP projections and reductions remain active.
        skip_sp = image_shard.num_pad > image_shard.local_len
        if not skip_sp:
            hidden_states = shard_like(hidden_states, image_shard)
        cos, sin = freqs_cis
        image_cos = (
            cos[text_len:]
            if skip_sp
            else shard_like(cos[text_len:], image_shard, dim=0)
        )
        image_sin = (
            sin[text_len:]
            if skip_sp
            else shard_like(sin[text_len:], image_shard, dim=0)
        )
        attention_kwargs = dict(joint_attention_kwargs or {})
        if skip_sp:
            attention_kwargs["skip_sequence_parallel_override"] = True
        replicated_text = 0 if skip_sp else text_len
        single_freqs_cis = None
        if not skip_sp and should_shard_text(text_len):
            text_shard = build_shard_plan(text_len)
            encoder_hidden_states = shard_like(encoder_hidden_states, text_shard)
            text_cos = shard_like(cos[:text_len], text_shard, dim=0)
            text_sin = shard_like(sin[:text_len], text_shard, dim=0)
            joint_shard = SpShard(
                orig_len=text_len + image_len,
                local_len=text_shard.local_len + image_shard.local_len,
                num_pad=text_shard.num_pad + image_shard.num_pad,
                sp_size=image_shard.sp_size,
                sp_rank=image_shard.sp_rank,
            )
            meta = tail_attn_meta(
                joint_shard, hidden_states.shape[0], hidden_states.device
            )
            if meta is not None:
                # Flux attention relocates only TEXT padding behind the image.
                # Both inserted padding segments then form the joint tail.
                meta["local_pad"] = text_shard.local_pad
                attention_kwargs["attn_mask_meta"] = meta
            single_freqs_cis = (
                join_seqs(text_cos, image_cos, text_shard.local_pad, dim=0),
                join_seqs(text_sin, image_sin, text_shard.local_pad, dim=0),
            )
            replicated_text = 0
        else:
            text_cos, text_sin = cos[:text_len], sin[:text_len]
            if not skip_sp and image_shard.num_pad:
                mask = torch.ones(
                    (hidden_states.shape[0], text_len + image_shard.local_len),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                if image_shard.local_pad:
                    mask[:, -image_shard.local_pad :] = False
                attention_kwargs["attn_mask"] = mask
        freqs_cis = (
            torch.cat([text_cos, image_cos], dim=0),
            torch.cat([text_sin, image_sin], dim=0),
        )
        if single_freqs_cis is None:
            single_freqs_cis = freqs_cis
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                freqs_cis,
                replicated_text,
                attention_kwargs,
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                single_freqs_cis,
                replicated_text,
                attention_kwargs,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)[0]
        return output if skip_sp else gather_seq(output, image_len)


EntryClass = OvisImageTransformer2DModel
