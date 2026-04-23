"""PyTorch Moss-VL model for SGLang - Qwen3VL Vision + Text with Cross Attention."""

from __future__ import annotations

import logging
from functools import partial
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    MRotaryEmbedding,
    get_rope,
)
from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# ==================== Vision Components ====================


class MossVLVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor):
        x_fc1, _ = self.linear_fc1(x)
        mlp_output, _ = self.linear_fc2(self.act(x_fc1))
        return mlp_output


class MossVLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class MossVLVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        hidden_act: str = "silu",
        norm_layer=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = MossVLVisionMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x = x + mlp
        return x


class MossVLVisionPatchMerger(nn.Module):
    """Merges spatial patches and concatenates deepstack features.

    Unlike Qwen3VL which uses separate merger modules per deepstack layer,
    Moss-VL concatenates all features and processes them through a single MLP.
    """

    def __init__(
        self,
        config,
        num_deepstack_features: int = 0,
        norm_layer=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        base_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.input_hidden_size = base_hidden_size * (1 + num_deepstack_features)
        self.hidden_size = config.hidden_size

        num_features = 1 + num_deepstack_features
        self.norms = nn.ModuleList(
            [norm_layer(config.hidden_size) for _ in range(num_features)]
        )

        self.linear_fc1 = ColumnParallelLinear(
            self.input_hidden_size,
            self.input_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.input_hidden_size,
            config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        deepstack_features: List[torch.Tensor],
    ) -> torch.Tensor:
        all_inputs = [last_hidden_state] + deepstack_features
        outs = []
        for i, feat in enumerate(all_inputs):
            outs.append(self.norms[i](feat))
        x = torch.cat(outs, dim=-1)
        x = x.view(-1, self.input_hidden_size)
        x, _ = self.linear_fc1(x)
        x = self.act_fn(x)
        x, _ = self.linear_fc2(x)
        return x


class MossVLVisionModel(nn.Module):
    """Moss-VL Vision Encoder (same architecture as Qwen3VL vision)."""

    def __init__(
        self,
        config,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_position_embeddings = config.num_position_embeddings
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = config.temporal_patch_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self.patch_embed = MossVLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                MossVLVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_dim=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(config.depth)
            ]
        )

        num_deepstack = len(self.deepstack_visual_indexes)
        self.merger = MossVLVisionPatchMerger(
            config=config,
            num_deepstack_features=num_deepstack,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        num_grid_per_side = int(self.num_position_embeddings**0.5)
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        idx_parts = [[] for _ in range(4)]
        weight_parts = [[] for _ in range(4)]

        for _, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_int, w_int = int(h.item()), int(w.item())
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h_int, device=device)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w_int, device=device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_parts[i].append(indices[i])
                weight_parts[i].append(weights[i])

        idx_tensor = torch.stack([torch.cat(parts) for parts in idx_parts]).to(
            dtype=torch.long
        )
        weight_tensor = torch.stack([torch.cat(parts) for parts in weight_parts]).to(
            dtype=dtype
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int((h * w).item()) for h, w in zip(grid_hs, grid_ws)]
        )

        m_size = self.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = (
                pos_embed.repeat(t, 1)
                .view(t, h // m_size, m_size, w // m_size, m_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        x = x + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = x.size()
        rotary_pos_emb = rotary_pos_emb.to(x.device)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
                cu_seqlens.to(torch.int32),
            ]
        )

        x = x.unsqueeze(1)

        deepstack_features = []
        for layer_idx, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(x)

        # Merger: concatenate last hidden state + deepstack features, then project
        x = self.merger(x, deepstack_features)
        return x

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        stacked_params_mapping = [
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ==================== Cross-Attention Components ====================


class MossVLTextCrossAttention(nn.Module):
    """Cross attention layer for Moss-VL: text queries attend to vision keys/values.

    Key differences from Mllama cross attention:
    - Uses separate q/k/v projections (q from text hidden states, k/v from vision states)
    - Applies RoPE to both query (text positions) and key (vision positions)
    - Uses QKVParallelLinear for the query projection (reusing text hidden_size)
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // self.model_parallel_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_local_key_value_heads = (
            self.num_key_value_heads // self.model_parallel_size
        )
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_heads
        )
        self.layer_id = layer_id
        self.q_local_size = self.num_local_heads * self.head_dim
        self.kv_local_size = self.num_local_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Query projection from text hidden states
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        # Key/Value projections from vision cross_attention_states
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope_theta = getattr(config, "rope_theta", 1000000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            self.num_local_key_value_heads,
            layer_id=layer_id,
            is_cross_attention=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_cross_attn_rotary(
        self, positions: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """Apply MRoPE to a single tensor (q or k) for cross-attention.

        Since q and k have different sequence lengths in cross-attention,
        we cannot use rotary_emb(positions, q, k) which assumes matching lengths.
        """
        rotary_emb = self.rotary_emb
        num_tokens = positions.shape[-1]
        cos_sin = rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        if positions.ndim == 2 and isinstance(rotary_emb, MRotaryEmbedding):
            if rotary_emb.mrope_section:
                if rotary_emb.mrope_interleaved:
                    cos = apply_interleaved_rope(cos, rotary_emb.mrope_section)
                    sin = apply_interleaved_rope(sin, rotary_emb.mrope_section)
                else:
                    cos = torch.cat(
                        [
                            m[i]
                            for i, m in enumerate(
                                cos.split(rotary_emb.mrope_section, dim=-1)
                            )
                        ],
                        dim=-1,
                    )
                    sin = torch.cat(
                        [
                            m[i]
                            for i, m in enumerate(
                                sin.split(rotary_emb.mrope_section, dim=-1)
                            )
                        ],
                        dim=-1,
                    )

        states_shape = states.shape
        states = states.view(num_tokens, -1, rotary_emb.head_size)
        states_rot = states[..., : rotary_emb.rotary_dim]
        states_pass = states[..., rotary_emb.rotary_dim :]
        states_rot = apply_rotary_emb(states_rot, cos, sin, rotary_emb.is_neox_style)
        states = torch.cat((states_rot, states_pass), dim=-1).reshape(states_shape)
        return states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        vision_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Query from text
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)

        if cross_attention_states is not None:
            # Key/Value from vision
            k, _ = self.k_proj(cross_attention_states)
            v, _ = self.v_proj(cross_attention_states)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)

        # Apply RoPE: text positions for query, vision positions for key
        q = self._apply_cross_attn_rotary(positions, q)
        if cross_attention_states is not None and vision_position_ids is not None:
            k = self._apply_cross_attn_rotary(vision_position_ids, k)

        if cross_attention_states is None:
            k = None
            v = None

        output = self.attn(q, k, v, forward_batch)
        out, _ = self.o_proj(output)
        return out


class MossVLCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.cross_attn = MossVLTextCrossAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("cross_attn", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))

        self.mlp = MossVLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.is_first_cross_attention_layer = (
            bool(config.cross_attention_layers)
            and layer_id == config.cross_attention_layers[0]
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor],
        full_text_row_masked_out_mask: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        positions: torch.Tensor = None,
        vision_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            forward_batch=forward_batch,
            positions=positions,
            vision_position_ids=vision_position_ids,
        )
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        return hidden_states


class MossVLTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for MossVLTextMLP."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ==================== Self-Attention Decoder Layer ====================


class MossVLSelfAttention(nn.Module):
    """Self-attention for Moss-VL text model (same structure as Qwen3Attention)."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config, "rope_theta", 1000000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MossVLSelfAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = MossVLSelfAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = MossVLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # MLP
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


# ==================== Text Model ====================


class MossVLTextModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.cross_attention_layers = config.cross_attention_layers

        layers = []
        for layer_id in range(config.num_hidden_layers):
            if layer_id in self.cross_attention_layers:
                layers.append(
                    MossVLCrossAttentionDecoderLayer(
                        config,
                        layer_id,
                        quant_config=quant_config,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                )
            else:
                layers.append(
                    MossVLSelfAttentionDecoderLayer(
                        config,
                        layer_id,
                        quant_config=quant_config,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        cross_attention_states: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor],
        full_text_row_masked_out_mask: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
        vision_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for decoder_layer in self.layers:
            if isinstance(decoder_layer, MossVLCrossAttentionDecoderLayer):
                if not skip_cross_attention:
                    # Fuse residual before cross-attention
                    if residual is not None:
                        hidden_states = hidden_states + residual
                        residual = None
                    hidden_states = decoder_layer(
                        hidden_states=hidden_states,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=cross_attention_mask,
                        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                        forward_batch=forward_batch,
                        positions=positions,
                        vision_position_ids=vision_position_ids,
                    )
            elif isinstance(decoder_layer, MossVLSelfAttentionDecoderLayer):
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                )
            else:
                raise ValueError(f"Unknown decoder layer type {type(decoder_layer)}")

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class MossVLForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.model = MossVLTextModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        cross_attention_states: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor],
        full_text_row_masked_out_mask: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        skip_cross_attention: bool,
        vision_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
            vision_position_ids=vision_position_ids,
        )
        return hidden_states


# ==================== Main Model ====================


class MossVLForConditionalGeneration(nn.Module):

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix

        vision_config = config.vision_config
        text_config = config.text_config

        self.spatial_merge_size = max(
            1, int(getattr(vision_config, "spatial_merge_size", 2))
        )
        self.vision_seq_pad_multiple = 1

        self.visual = MossVLVisionModel(
            vision_config,
            quant_config=quant_config,
            prefix=add_prefix("model.visual", prefix),
        )

        self.language_model = MossVLForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )

        # Learnable separator token
        self.separator_token = nn.Parameter(torch.zeros(vision_config.out_hidden_size))

        self.is_mrope_enabled = (
            hasattr(text_config, "rope_scaling")
            and text_config.rope_scaling is not None
            and "mrope_section" in text_config.rope_scaling
        )

        self.logits_processor = LogitsProcessor(text_config)

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    # ---- pad_input_ids (called at request scheduling time) ----

    def _get_encoder_len(self, mm_inputs: MultimodalInputs) -> int:
        if not mm_inputs.mm_items:
            return 0

        grid_thw = getattr(mm_inputs.mm_items[0], "grid_thw", None)
        if grid_thw is None:
            return 0

        grid_thw = torch.as_tensor(grid_thw, dtype=torch.int64)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.numel() == 0:
            return 0

        merge_square = self.spatial_merge_size**2
        tokens_per_media = torch.prod(grid_thw, dim=1) // merge_square
        num_frames_per_media = grid_thw[:, 0]
        # Each frame contributes tokens_per_frame vision tokens + 1 separator
        total_len = int((tokens_per_media + num_frames_per_media).sum().item())

        pad_multiple = self.vision_seq_pad_multiple
        if total_len % pad_multiple != 0:
            total_len = ((total_len + pad_multiple - 1) // pad_multiple) * pad_multiple

        return total_len

    def _build_encoder_prefix_pad_ids(self, mm_inputs: MultimodalInputs) -> List[int]:
        encoder_len = self._get_encoder_len(mm_inputs)
        if encoder_len == 0 or not mm_inputs.mm_items:
            return []

        pad_value = mm_inputs.mm_items[0].pad_value
        return [pad_value] * encoder_len

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        encoder_len = self._get_encoder_len(mm_inputs)
        mm_inputs.num_image_tokens = encoder_len
        if encoder_len == 0:
            return input_ids

        return self._build_encoder_prefix_pad_ids(mm_inputs) + input_ids

    # ---- Collect and encode vision inputs ----

    def _collect_mm_data(self, forward_batch: ForwardBatch):
        """Collect pixel_values, grid_thw, and vision_position_ids from uncached requests."""
        if forward_batch.forward_mode.is_decode() or all(forward_batch.encoder_cached):
            return None, None, None, None

        pixel_values_list = []
        grid_thw_list = []
        encoder_lens_need = []
        vision_pos_ids_list = []

        for i, mm_input in enumerate(forward_batch.mm_inputs):
            if forward_batch.encoder_cached[i] or mm_input is None:
                continue
            if not mm_input.mm_items:
                continue

            item = mm_input.mm_items[0]
            pixel_values_list.append(item.feature)
            grid_thw = getattr(item, "grid_thw", None)
            if grid_thw is not None:
                grid_thw_list.append(torch.as_tensor(grid_thw, dtype=torch.long))
            encoder_len = forward_batch.encoder_lens_cpu[i]
            encoder_lens_need.append(encoder_len)

            vp = mm_input.vision_position_ids
            if vp is not None:
                vision_pos_ids_list.append(vp[:, :encoder_len])

        if not pixel_values_list:
            return None, None, None, None

        pixel_values = torch.cat(pixel_values_list, dim=0)
        grid_thw = torch.cat(grid_thw_list, dim=0) if grid_thw_list else None
        packed_vision_pos_ids = (
            torch.cat(vision_pos_ids_list, dim=1) if vision_pos_ids_list else None
        )

        return pixel_values, grid_thw, encoder_lens_need, packed_vision_pos_ids

    def _get_vision_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run ViT encoder and insert separator tokens."""
        hidden_states = self.visual(pixel_values, grid_thw=grid_thw)
        # hidden_states is packed: (total_vision_tokens, hidden_size)
        return hidden_states

    def _insert_separator_tokens(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Insert separator token after each frame's vision tokens.

        Input: packed vision tokens from ViT (no separators)
        Output: packed vision tokens with separator tokens inserted after each frame
        """
        merge_square = self.spatial_merge_size**2
        tokens_per_media = (
            grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        ) // merge_square

        hidden_size = hidden_states.shape[-1]
        separator = self.separator_token.to(hidden_states.dtype)

        output_parts = []
        src_offset = 0
        for i in range(grid_thw.shape[0]):
            num_tokens = tokens_per_media[i].item()
            num_frames = grid_thw[i, 0].item()
            tokens_per_frame = num_tokens // num_frames
            media_hidden_states = hidden_states[
                src_offset : src_offset + num_tokens
            ].view(num_frames, tokens_per_frame, hidden_size)
            separators = separator.view(1, 1, hidden_size).expand(
                num_frames, 1, hidden_size
            )
            output_parts.append(
                torch.cat([media_hidden_states, separators], dim=1).flatten(0, 1)
            )
            src_offset += num_tokens

        return torch.cat(output_parts, dim=0)

    def flat_encoder_result(
        self,
        cross_attention_states: torch.Tensor,
        encoder_lens_need: List[int],
    ) -> torch.Tensor:
        """Copy vision states into a flat packed tensor, trimmed to encoder_lens."""
        total_encoder_len = sum(encoder_lens_need)
        head_dim = cross_attention_states.shape[-1]

        if cross_attention_states.dim() == 1:
            return cross_attention_states

        # cross_attention_states is already packed (total_tokens, hidden_size)
        # We need to split it according to encoder_lens_need
        result = torch.zeros(
            total_encoder_len,
            head_dim,
            device=cross_attention_states.device,
            dtype=cross_attention_states.dtype,
        )

        src_offset = 0
        dst_offset = 0
        for encoder_len in encoder_lens_need:
            if encoder_len > 0:
                if src_offset + encoder_len > cross_attention_states.shape[0]:
                    raise RuntimeError(
                        "Encoder length mismatch: expected "
                        f"{encoder_len} tokens, but only "
                        f"{cross_attention_states.shape[0] - src_offset} remaining."
                    )
                result[dst_offset : dst_offset + encoder_len] = cross_attention_states[
                    src_offset : src_offset + encoder_len
                ]
            src_offset += encoder_len
            dst_offset += encoder_len

        if src_offset != cross_attention_states.shape[0]:
            raise RuntimeError(
                "Encoder length mismatch: produced "
                f"{cross_attention_states.shape[0]} tokens, expected {src_offset}."
            )

        return result

    # ---- prepare_forward_batch (called before attn backend init) ----

    def prepare_forward_batch(self, forward_batch: ForwardBatch):
        """Build cross-attention custom mask before attn backend init.

        This hook is called by model_runner before init_forward_metadata so
        that the packed 1D mask is ready when FlashInfer plans cross-attention.
        Decode does not use a custom mask: newly generated tokens can attend
        to all encoder vision tokens.
        """
        forward_batch.cross_attention_custom_mask = None
        if forward_batch.forward_mode.is_decode():
            return
        if forward_batch.encoder_lens is None or forward_batch.encoder_lens.max() == 0:
            return

        custom_mask = self._build_cross_attention_custom_mask(forward_batch)
        if custom_mask is not None:
            forward_batch.cross_attention_custom_mask = custom_mask

    def _build_cross_attention_custom_mask(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        """Build packed 1D extend-stage cross-attention custom mask.

        The mask controls frame-level causal visibility: which vision frames
        each extend-stage text token can attend to during cross-attention.

        IMPORTANT: by the time ForwardBatch reaches the model,
        prepare_encoder_info_extend() has already stripped the encoder prefix
        from input_ids / seq_lens / extend_lens / prefix_lens.  So the extend
        segment is purely decoder text — no encoder-prefix placeholder tokens.
        extend_prefix_len is the number of *cached text tokens*, and
        extend_seq_len is the number of *new text tokens* in this extend.

        Returns:
            1D uint8 tensor of shape (sum_i(q_len_i * kv_len_i),) in
            FlashInfer packed row-major format, or None when no frame-level
            mask is needed.
        """
        merge_square = self.spatial_merge_size**2
        device = forward_batch.seq_lens.device

        mask_parts = []
        need_mask = False

        for i in range(forward_batch.batch_size):
            encoder_len = forward_batch.encoder_lens_cpu[i]
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            extend_prefix_len = forward_batch.extend_prefix_lens_cpu[i]

            q_len = extend_seq_len
            kv_len = encoder_len

            if kv_len == 0 or q_len == 0:
                continue

            mm_input = forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None
            if mm_input is None:
                mask_parts.append(
                    torch.ones(q_len * kv_len, dtype=torch.uint8, device=device)
                )
                continue

            visible_frame_counts = mm_input.visible_frame_counts
            if visible_frame_counts is None:
                mask_parts.append(
                    torch.ones(q_len * kv_len, dtype=torch.uint8, device=device)
                )
                continue

            item = mm_input.mm_items[0] if mm_input.mm_items else None
            grid_thw = getattr(item, "grid_thw", None) if item else None
            if grid_thw is None:
                mask_parts.append(
                    torch.ones(q_len * kv_len, dtype=torch.uint8, device=device)
                )
                continue

            need_mask = True
            grid_thw_t = torch.as_tensor(grid_thw, dtype=torch.long)
            if grid_thw_t.ndim == 1:
                grid_thw_t = grid_thw_t.unsqueeze(0)

            # Build frame_ranges: each frame's [start, end) in the encoder
            # token sequence (vision tokens + separator per frame).
            frame_ranges: List[Tuple[int, int]] = []
            cursor = 0
            for row_idx in range(grid_thw_t.shape[0]):
                t = grid_thw_t[row_idx, 0].item()
                h = grid_thw_t[row_idx, 1].item()
                w = grid_thw_t[row_idx, 2].item()
                span = (h * w) // merge_square + 1
                for _ in range(t):
                    frame_ranges.append((cursor, cursor + span))
                    cursor += span

            # The extend segment is purely text (encoder prefix already
            # stripped by prepare_encoder_info_extend).  extend_prefix_len
            # is the cached-text offset into the full text sequence.
            text_offset = extend_prefix_len

            vis_counts = visible_frame_counts[text_offset : text_offset + q_len].to(
                device
            )

            mask = torch.zeros(q_len, kv_len, dtype=torch.uint8, device=device)

            for f, (start, end) in enumerate(frame_ranges):
                clamped_end = min(end, kv_len)
                if start >= kv_len:
                    break
                visible_rows = vis_counts > f
                if visible_rows.any():
                    mask[visible_rows, start:clamped_end] = 1

            mask_parts.append(mask.flatten())

        if not need_mask or not mask_parts:
            return None

        return torch.cat(mask_parts)

    # ---- full_text_row_masked_out_mask ----

    def get_full_text_row_masked_out_mask(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Create per-token mask that zeros cross-attn output for tokens
        that cannot see any vision token.

        HF semantics: a text token's cross-attn + cross-attn-MLP residuals
        are zeroed when that token has zero visible vision tokens.  This is
        derived from the token-level cross_attention_mask, not just from
        whether the request has vision.

        For decode, HF copies the previous token's cross_attention_mask row to
        the new token. Since the processor's frame-level mask is prefix-causal,
        this reduces to copying the last prefill token's visibility.

        NOTE: prepare_encoder_info_extend() already strips encoder prefix
        tokens, so extend_seq_len / extend_prefix_len are purely text.
        extend_prefix_len is the cached-text offset into visible_frame_counts.
        """
        encoder_lens_cpu = forward_batch.encoder_lens_cpu

        if forward_batch.forward_mode.is_decode():
            device = forward_batch.encoder_lens.device
            full_text_row_masked_out_mask = forward_batch.encoder_lens != 0

            if not forward_batch.mm_inputs:
                return full_text_row_masked_out_mask.reshape(-1, 1)

            bs = forward_batch.batch_size
            for i in range(bs):
                if not full_text_row_masked_out_mask[i]:
                    continue

                mm_input = forward_batch.mm_inputs[i]
                visible_frame_counts = (
                    mm_input.visible_frame_counts if mm_input else None
                )
                if visible_frame_counts is None:
                    # Fall back to request-level gating only when frame-level
                    # visibility metadata is unavailable. The request-level
                    # encoder_lens signal already marks this row as visible.
                    continue

                full_text_row_masked_out_mask[i] = visible_frame_counts[-1] > 0
        else:
            device = forward_batch.seq_lens.device
            total_extend_len = int(forward_batch.extend_seq_lens.sum().item())
            full_text_row_masked_out_mask = torch.zeros(
                total_extend_len, dtype=torch.bool, device=device
            )

            offset = 0
            for i in range(forward_batch.batch_size):
                encoder_len = encoder_lens_cpu[i]
                extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
                extend_prefix_len = forward_batch.extend_prefix_lens_cpu[i]

                if extend_seq_len == 0:
                    continue

                if encoder_len == 0:
                    offset += extend_seq_len
                    continue

                mm_input = (
                    forward_batch.mm_inputs[i] if forward_batch.mm_inputs else None
                )
                visible_frame_counts = (
                    mm_input.visible_frame_counts if mm_input else None
                )

                if visible_frame_counts is None:
                    full_text_row_masked_out_mask[offset : offset + extend_seq_len] = (
                        True
                    )
                    offset += extend_seq_len
                    continue

                # The extend is purely text; extend_prefix_len is the
                # cached-text offset into the full text sequence.
                text_offset = extend_prefix_len

                vis_counts = visible_frame_counts[
                    text_offset : text_offset + extend_seq_len
                ].to(device)
                full_text_row_masked_out_mask[offset : offset + extend_seq_len] = (
                    vis_counts > 0
                )

                # Last prefill chunk for this request: decode will only need
                # visible_frame_counts[-1], so shrink the tensor to that single
                # element and drop the rest. .clone() detaches the view from
                # the original storage so the large tensor can be freed.
                if text_offset + extend_seq_len >= visible_frame_counts.shape[0]:
                    mm_input.visible_frame_counts = visible_frame_counts[-1:].clone()

                offset += extend_seq_len

        return full_text_row_masked_out_mask.reshape(-1, 1)

    # ---- Forward ----

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # 1. Collect vision inputs for uncached requests
        pixel_values, grid_thw, encoder_lens_need, vision_position_ids = (
            self._collect_mm_data(forward_batch)
        )

        cross_attention_mask = None
        cross_attention_states = None

        if get_is_capture_mode():
            skip_cross_attention = False
        else:
            assert len(forward_batch.encoder_lens) == len(forward_batch.seq_lens)
            skip_cross_attention = forward_batch.encoder_lens.max() == 0

        # 2. Build full_text_row_masked_out_mask
        if not skip_cross_attention:
            full_text_row_masked_out_mask = self.get_full_text_row_masked_out_mask(
                forward_batch
            )
        else:
            full_text_row_masked_out_mask = None

        # 3. Encode vision if needed
        if pixel_values is not None and grid_thw is not None:
            # Run ViT
            vision_hidden_states = self._get_vision_features(pixel_values, grid_thw)
            # Insert separator tokens after each frame
            vision_with_sep = self._insert_separator_tokens(
                vision_hidden_states, grid_thw
            )
            # Flatten to match encoder_lens
            cross_attention_states = self.flat_encoder_result(
                vision_with_sep, encoder_lens_need
            )
            # Drop heavy per-request vision tensors now that the encoder KV
            # has been produced and will be cached. Otherwise pixel_values and
            # vision_position_ids stay pinned on req.multimodal_inputs across
            # the entire decode phase. (visible_frame_counts is shrunk to a
            # single scalar element at the end of the last prefill chunk in
            # get_full_text_row_masked_out_mask, so decode still works.)
            # Note: the local `vision_position_ids` is still needed by the LM
            # cross-attention below, so we keep it; but we drop the per-request
            # copy on mm_input, which we won't read again.
            del pixel_values, vision_hidden_states, vision_with_sep
            for i, mm_input in enumerate(forward_batch.mm_inputs):
                if forward_batch.encoder_cached[i] or mm_input is None:
                    continue
                mm_input.release_features()
                mm_input.vision_position_ids = None

        # 4. Run language model with cross attention
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            forward_batch=forward_batch,
            skip_cross_attention=skip_cross_attention,
            vision_position_ids=vision_position_ids,
        )

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.language_model.lm_head,
            forward_batch,
        )

    # ---- Weight Loading ----

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            original_name = name

            # Map HF names to local module names.
            if name == "lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith("model.visual."):
                name = name[len("model.") :]
            elif name.startswith("model.separator_token"):
                name = name[len("model.") :]

            # VisionAttention stores fused QKV weights under qkv_proj in SGLang.
            if "visual." in name:
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

            handled = False
            if "visual." not in name and ".cross_attn." not in name:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if handled:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug(f"Skipping weight: {original_name} -> {name}")


EntryClass = MossVLForConditionalGeneration
