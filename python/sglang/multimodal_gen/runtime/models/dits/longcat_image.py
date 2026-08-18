# Copied and adapted from: https://github.com/huggingface/diffusers
# main/src/diffusers/models/transformers/transformer_longcat_image.py
"""LongCat-Image Transformer2D model for SGLang.

Implements the LongCatImageTransformer2DModel architecture from diffusers
adapted for SGLang's inference pipeline.

Key adaptation: the transformer accepts raw scheduler timesteps (in [0, 1000])
and feeds them directly to the timestep embedder. The diffusers pipeline passes
``timestep / 1000`` to its transformer (which then multiplies by 1000 internally);
SGLang's DenoisingStage passes the raw timestep instead, so the value reaching
the embedder is identical and no division is needed here.

Attention alignment: uses USPAttention (FA3/FA4 on Hopper/Blackwell) with
SGLang fused RMSNorm (apply_qk_norm). RoPE is applied separately via
diffusers apply_rotary_emb because LongCat's axes_dims_rope=[16,56,56]
sums to head_dim=128 (full rotation), which is incompatible with flashinfer's
cos_sin_cache format that requires rotary_dim <= head_dim.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)

from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm, apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


class _LongCatFFN(nn.Module):
    """TP-parallel FFN matching diffusers FeedForward(activation_fn="gelu-approximate").

    Weight names mirror the diffusers checkpoint layout so loading works without remapping:
      net.0.proj  ->  ColumnParallelLinear  (project in, dim -> inner_dim)
      net.2       ->  RowParallelLinear     (project out, inner_dim -> dim)
    """

    def __init__(self, dim: int, inner_dim: int, bias: bool = True, prefix: str = ""):
        super().__init__()
        self.net = nn.ModuleList(
            [
                # net.0: GELU activation wrapper — only the inner proj is a parameter
                nn.ModuleDict(
                    {
                        "proj": ColumnParallelLinear(
                            dim,
                            inner_dim,
                            bias=bias,
                            gather_output=False,
                            prefix=f"{prefix}.net.0.proj",
                        )
                    }
                ),
                nn.Dropout(
                    0.0
                ),  # net.1: dummy dropout, matches diffusers checkpoint layout
                RowParallelLinear(
                    inner_dim,
                    dim,
                    bias=bias,
                    input_is_parallel=True,
                    prefix=f"{prefix}.net.2",
                ),
            ]
        )
        self.act = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.net[0]["proj"](hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.net[2](hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _LongCatJointAttention(nn.Module):
    """Double-stream (joint) attention for _TransformerBlock.

    img and txt tokens are projected separately, QK-norm applied via SGLang
    fused kernel, RoPE applied via diffusers apply_rotary_emb (supports full
    head_dim rotation), then concatenated (txt first) before USPAttention.

    TP: Q/K/V and add_q/k/v use ColumnParallelLinear (heads sharded across TP ranks).
    Output projections use RowParallelLinear (all-reduce after matmul).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        bias: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        self.num_local_heads = num_attention_heads // tp_size
        assert (
            num_attention_heads % tp_size == 0
        ), f"num_attention_heads ({num_attention_heads}) must be divisible by tp_size ({tp_size})"
        self.head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.norm_q = RMSNorm(attention_head_dim, eps=eps)
        self.norm_k = RMSNorm(attention_head_dim, eps=eps)
        self.norm_added_q = RMSNorm(attention_head_dim, eps=eps)
        self.norm_added_k = RMSNorm(attention_head_dim, eps=eps)

        self.to_q = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_v"
        )
        self.add_q_proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            gather_output=False,
            prefix=f"{prefix}.add_q_proj",
        )
        self.add_k_proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            gather_output=False,
            prefix=f"{prefix}.add_k_proj",
        )
        self.add_v_proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            gather_output=False,
            prefix=f"{prefix}.add_v_proj",
        )

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    inner_dim,
                    dim,
                    bias=bias,
                    input_is_parallel=True,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.to_add_out = RowParallelLinear(
            inner_dim,
            dim,
            bias=bias,
            input_is_parallel=True,
            prefix=f"{prefix}.to_add_out",
        )

        self.attn = USPAttention(
            num_heads=self.num_local_heads,
            head_size=attention_head_dim,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        txt_seq_len = encoder_hidden_states.shape[1]

        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        q = q.unflatten(-1, (self.num_local_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_local_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_local_heads, self.head_dim))

        eq, _ = self.add_q_proj(encoder_hidden_states)
        ek, _ = self.add_k_proj(encoder_hidden_states)
        ev, _ = self.add_v_proj(encoder_hidden_states)
        eq = eq.unflatten(-1, (self.num_local_heads, self.head_dim))
        ek = ek.unflatten(-1, (self.num_local_heads, self.head_dim))
        ev = ev.unflatten(-1, (self.num_local_heads, self.head_dim))

        # SGLang fused QK-norm
        q, k = apply_qk_norm(q, k, self.norm_q, self.norm_k, self.head_dim)
        eq, ek = apply_qk_norm(
            eq, ek, self.norm_added_q, self.norm_added_k, self.head_dim
        )

        # Concatenate: txt first, then img (matches diffusers convention)
        q = torch.cat([eq, q], dim=1)
        k = torch.cat([ek, k], dim=1)
        v = torch.cat([ev, v], dim=1)

        # RoPE applied after concat, over the full [txt+img] sequence.
        # image_rotary_emb shape: [txt_len+img_len, head_dim] — matches q/k dim=1.
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        x = self.attn(q, k, v, num_replicated_prefix=txt_seq_len)
        x = x.flatten(2, 3).to(q.dtype)

        encoder_out, hidden_out = x.split_with_sizes(
            [txt_seq_len, x.shape[1] - txt_seq_len], dim=1
        )
        hidden_out, _ = self.to_out[0](hidden_out)
        encoder_out, _ = self.to_add_out(encoder_out)
        return hidden_out, encoder_out


class _LongCatSingleAttention(nn.Module):
    """Single-stream attention for _SingleTransformerBlock.

    txt and img are already concatenated by the block before calling here.
    No output projection — the block handles proj_out itself.

    TP: Q/K/V use ColumnParallelLinear with gather_output=False (head-sharded).
    USPAttention receives [B, S, H_local, D] directly; no all-gather needed here.
    proj_mlp/proj_out in _SingleTransformerBlock must also use gather_output=False
    so the concat [attn_output, mlp_hidden_states] is uniformly sharded.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        bias: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        self.num_local_heads = num_attention_heads // tp_size
        assert (
            num_attention_heads % tp_size == 0
        ), f"num_attention_heads ({num_attention_heads}) must be divisible by tp_size ({tp_size})"
        self.head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.norm_q = RMSNorm(attention_head_dim, eps=eps)
        self.norm_k = RMSNorm(attention_head_dim, eps=eps)

        self.to_q = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, inner_dim, bias=bias, gather_output=False, prefix=f"{prefix}.to_v"
        )

        self.attn = USPAttention(
            num_heads=self.num_local_heads,
            head_size=attention_head_dim,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q, _ = self.to_q(hidden_states)
        k, _ = self.to_k(hidden_states)
        v, _ = self.to_v(hidden_states)
        q = q.unflatten(-1, (self.num_local_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_local_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_local_heads, self.head_dim))

        # SGLang fused QK-norm
        q, k = apply_qk_norm(q, k, self.norm_q, self.norm_k, self.head_dim)

        # RoPE via diffusers (supports full head_dim rotation, sequence_dim=1)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        x = self.attn(q, k, v)
        return x.flatten(2, 3).to(q.dtype)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class _SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = AdaLayerNormZeroSingle(dim)
        # proj_mlp: ColumnParallelLinear with gather_output=False keeps output
        # head-sharded, consistent with attn_output from _LongCatSingleAttention.
        self.proj_mlp = ColumnParallelLinear(
            dim,
            self.mlp_hidden_dim,
            bias=True,
            gather_output=False,
            prefix=f"{prefix}.proj_mlp",
        )
        self.act_mlp = nn.GELU(approximate="tanh")
        # proj_out: RowParallelLinear reduces sharded [attn | mlp] concat via
        # all-reduce, matching Flux2SingleTransformerBlockAttention.to_out.
        self.proj_out = RowParallelLinear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            prefix=f"{prefix}.proj_out",
        )
        self.attn = _LongCatSingleAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            bias=True,
            eps=1e-6,
            prefix=f"{prefix}.attn",
        )
        tp_size = get_tp_world_size()
        if tp_size > 1:
            self._patch_proj_out_weight_loader(
                inner_dim=num_attention_heads * attention_head_dim,
                mlp_dim=self.mlp_hidden_dim,
                tp_size=tp_size,
            )

    def _patch_proj_out_weight_loader(
        self, inner_dim: int, mlp_dim: int, tp_size: int
    ) -> None:
        # proj_out input is [attn_shard | mlp_shard] where the two shards come
        # from non-contiguous column ranges in the checkpoint weight matrix.
        # Default RowParallelLinear.weight_loader slices contiguously, which is
        # wrong here; override it to pick the correct columns per rank.
        proj_out = self.proj_out
        tp_rank = proj_out.tp_rank

        def _loader(param, loaded_weight):
            input_dim = getattr(param, "input_dim", None)
            if input_dim is not None:
                a = inner_dim // tp_size
                m = mlp_dim // tp_size
                attn_cols = loaded_weight.narrow(input_dim, tp_rank * a, a)
                mlp_cols = loaded_weight.narrow(input_dim, inner_dim + tp_rank * m, m)
                param.data.copy_(torch.cat([attn_cols, mlp_cols], dim=input_dim))
            else:
                param.data.copy_(loaded_weight)

        proj_out.weight_loader = _loader
        proj_out.weight.weight_loader = _loader

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        **kwargs,
    ):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states, _ = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = _LongCatJointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            bias=True,
            eps=qk_norm_eps,
            prefix=f"{prefix}.attn",
        )
        # norm2/norm2_context use eps=1e-6 matching diffusers (not configurable)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = _LongCatFFN(dim=dim, inner_dim=dim * 4, prefix=f"{prefix}.ff")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = _LongCatFFN(
            dim=dim, inner_dim=dim * 4, prefix=f"{prefix}.ff_context"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        **kwargs,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Position embedding
# ---------------------------------------------------------------------------


class _LongCatPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (freqs_cos, freqs_sin) each of shape [S, head_dim]."""
        cos_out, sin_out = [], []
        pos = ids.float()
        freqs_dtype = (
            torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        )
        for i, dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


class _TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep: torch.Tensor, hidden_dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class LongCatImageTransformer2DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    """SGLang implementation of the LongCat-Image transformer.

    Accepts raw scheduler timesteps (in [0, 1000]) and feeds them directly to
    the timestep embedder (no /1000). The diffusers pipeline divides timesteps
    by 1000 before calling its transformer, which multiplies them back by 1000
    internally — so the embedder receives the same value either way.
    """

    _aliases = ["LongCatImageTransformer2DModel"]

    param_names_mapping = {}
    _fsdp_shard_conditions = []
    _compile_conditions = []

    def __init__(self, config, hf_config: dict = None, quant_config=None):
        super().__init__(config=config, hf_config=hf_config or {})
        arch = config.arch_config

        patch_size = getattr(arch, "patch_size", 1)
        in_channels = getattr(arch, "in_channels", 64)
        num_layers = getattr(arch, "num_layers", 19)
        num_single_layers = getattr(arch, "num_single_layers", 38)
        attention_head_dim = getattr(arch, "attention_head_dim", 128)
        num_attention_heads = getattr(arch, "num_attention_heads", 24)
        joint_attention_dim = getattr(arch, "joint_attention_dim", 3584)
        axes_dims_rope = getattr(arch, "axes_dims_rope", [16, 56, 56])

        self.config = config
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Required by BaseDiT.__post_init__
        self.hidden_size = self.inner_dim
        self.num_attention_heads = num_attention_heads
        self.num_channels_latents = 16  # unpacked latent channels

        self.pos_embed = _LongCatPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.time_embed = _TimestepEmbeddings(embedding_dim=self.inner_dim)
        self.context_embedder = ColumnParallelLinear(
            joint_attention_dim,
            self.inner_dim,
            bias=True,
            gather_output=True,
            prefix="context_embedder",
        )
        self.x_embedder = ColumnParallelLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            gather_output=True,
            prefix="x_embedder",
        )

        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                _SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    prefix=f"single_transformer_blocks.{i}",
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = ColumnParallelLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            gather_output=True,
            prefix="proj_out",
        )

        self.layer_names = ["transformer_blocks", "single_transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = False,
        **kwargs,
    ):
        hidden_states, _ = self.x_embedder(hidden_states)

        # Raw scheduler timestep in [0, 1000] — fed directly (see module docstring).
        temb = self.time_embed(timestep.to(hidden_states.dtype), hidden_states.dtype)

        encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

        # RoPE is computed on the fly from txt_ids + img_ids, matching diffusers'
        # transformer (which calls self.pos_embed internally). image_rotary_emb is
        # only passed in for standalone model testing.
        image_rotary_emb = kwargs.get("image_rotary_emb") or self.pos_embed(
            torch.cat((txt_ids, img_ids), dim=0)
        )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output, _ = self.proj_out(hidden_states)

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return output


EntryClass = LongCatImageTransformer2DModel
