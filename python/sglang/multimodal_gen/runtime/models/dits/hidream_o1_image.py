# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image: a pixel-level unified transformer (UiT).

The checkpoint is a Qwen3-VL text backbone plus a flow-matching pixel head
(``x_embedder`` / ``t_embedder1`` / ``final_layer2``). There is no VAE and no
separate text encoder: the prompt is embedded by the backbone itself and the
image is a sequence of 32x32 raw pixel patches appended to the prompt tokens.
"""

import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    tensors_equal,
)
from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageArchConfig,
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    qwen3_apply_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

logger = init_logger(__name__)
_HIDREAM_ROPE = BitExactFusionGate("HiDream fused RoPE")

# Checkpoint prefixes that carry no parameter of this model.
UNUSED_CHECKPOINT_PREFIXES = ("lm_head.",)


def is_hidream_o1_image_weight(name: str) -> bool:
    return not name.startswith(UNUSED_CHECKPOINT_PREFIXES)


def timestep_embedding(
    timestep: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
        / half
    )
    args = timestep[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class HiDreamO1RotaryEmbedding(nn.Module):
    # Interleaved mrope: the three position axes are woven into every third
    # rotary channel rather than laid out in contiguous sections, so neither
    # get_rope nor the Qwen2-VL section-concat form can express it.
    def __init__(
        self, head_dim: int, rope_theta: float, mrope_section: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        # 64 floats; recomputing per step is cheaper than carrying a buffer that
        # meta-device init would leave unmaterialized.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )
        # Autocast would run the matmul below in bf16 and visibly quantize the
        # position grid.
        with torch.autocast(device_type=device.type, enabled=False):
            inv_freq_expanded = inv_freq[None, None, :, None].expand(
                3, position_ids.shape[1], -1, 1
            )
            positions = position_ids[:, :, None, :].float()
            freqs = (inv_freq_expanded @ positions).transpose(2, 3)
            freqs_t = freqs[0]
            for axis, offset in enumerate((1, 2), start=1):
                length = self.mrope_section[axis] * 3
                freqs_t[..., offset:length:3] = freqs[axis, ..., offset:length:3]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)


class HiDreamO1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"HiDream-O1-Image expects a SwiGLU MLP; got hidden_act={hidden_act!r}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class HiDreamO1Attention(nn.Module):
    def __init__(
        self,
        config: HiDreamO1ImageArchConfig,
        supported_attention_backends: set[AttentionBackendEnum],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} must be divisible by "
                f"tp_size={tp_size}"
            )
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={self.total_num_kv_heads} must be divisible by "
                f"tp_size={tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # The mask mixes causal text spans with a bidirectional image span, so
        # it is always supplied explicitly rather than via causal=True.
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(
            k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # cos/sin are [B, S, D]; broadcast over the head axis of [B, S, H, D].
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        if (
            q.is_cuda
            and q.dtype is torch.bfloat16
            and q.is_contiguous()
            and k.is_contiguous()
        ):
            # Triton-backed bit-exact RoPE is CUDA-only; import lazily so the
            # module still imports on Ascend NPU (which only ships ascend-triton).
            from sglang.kernels.ops.diffusion import fused_rope_rotate_half_bitexact

            # Keep the BF16 rounding after each product in the eager RoPE.
            q, k = _HIDREAM_ROPE.run(
                lambda: (
                    fused_rope_rotate_half_bitexact(q, cos, sin),
                    fused_rope_rotate_half_bitexact(k, cos, sin),
                ),
                lambda: qwen3_apply_rotary_pos_emb(q, k, cos, sin),
                equal=tensors_equal,
                logger=logger,
            )
        else:
            q, k = qwen3_apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self.attn(q, k, v, attn_mask=attention_mask)
        output, _ = self.o_proj(attn_output.reshape(batch_size, seq_len, -1))
        return output


class HiDreamO1DecoderLayer(nn.Module):
    def __init__(
        self,
        config: HiDreamO1ImageArchConfig,
        supported_attention_backends: set[AttentionBackendEnum],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = HiDreamO1Attention(
            config=config,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = HiDreamO1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HiDreamO1TextModel(nn.Module):
    def __init__(
        self,
        config: HiDreamO1ImageArchConfig,
        supported_attention_backends: set[AttentionBackendEnum],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                HiDreamO1DecoderLayer(
                    config=config,
                    supported_attention_backends=supported_attention_backends,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{index}",
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HiDreamO1RotaryEmbedding(
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add visual embeds at the masked positions (TI2I deepstack injection).

        Only the Dev variant's two-pass path supplies these; text-to-image
        calls leave both arguments None and never enter this branch.
        """
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, cos, sin, attention_mask
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # Sequential purely to reproduce the checkpoint's mlp.0 / mlp.2 names;
        # the parallel linears return (output, bias) so it cannot be called.
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                input_size=frequency_embedding_size,
                output_size=hidden_size,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
            ),
            nn.SiLU(),
            RowParallelLinear(
                input_size=hidden_size,
                output_size=hidden_size,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
            ),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # The flow timestep arrives in [0, 1] and the frequency grid is trained
        # on the 0-1000 scale.
        embedding = timestep_embedding(timestep * 1000.0, self.frequency_embedding_size)
        hidden, _ = self.mlp[0](embedding.to(self.mlp[0].weight.dtype))
        hidden = self.mlp[1](hidden)
        hidden, _ = self.mlp[2](hidden)
        return hidden


class BottleneckPatchEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        in_channels: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        bottleneck_dim = hidden_size // 4
        self.proj1 = ColumnParallelLinear(
            input_size=patch_size * patch_size * in_channels,
            output_size=bottleneck_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj1",
        )
        self.proj2 = RowParallelLinear(
            input_size=bottleneck_dim,
            output_size=hidden_size,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No activation between the projections; the bottleneck is linear.
        x, _ = self.proj1(x)
        x, _ = self.proj2(x)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=patch_size * patch_size * out_channels,
            bias=True,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear(x)
        return x


def _normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Shape mRoPE position ids to [3, batch, seq] for the rotary embedding."""
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    return position_ids


def _flash_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    idx_ar: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_embed: Optional[torch.Tensor],
    deepstack_process: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """One layer of the two-pass SDPA loop (causal AR + full; AR rows overwrite).

    A free function so the cache-dit adapter runs the same body. ``pad_mask``
    ([B, seq], True at pads) excludes padded keys in both passes; None keeps the
    mask-free is_causal path bit-identical to single-row inference.
    """
    attn = layer.self_attn
    head_dim = attn.head_dim
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    qkv, _ = attn.qkv_proj(normed)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(k.view(hidden_shape)).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    softmax_scale = attn.scaling

    if idx_ar.numel() > 0:
        q_ar = q[:, :, idx_ar].contiguous()
        k_ar = k[:, :, idx_ar].contiguous()
        v_ar = v[:, :, idx_ar].contiguous()
        if pad_mask is None:
            out_ar = F.scaled_dot_product_attention(
                q_ar,
                k_ar,
                v_ar,
                is_causal=True,
                scale=softmax_scale,
            )
        else:
            # Column-vs-row views give a real [L, L] causal matrix; two row
            # views degenerate to an all-True [1, L] vector (non-causal, and
            # the resulting [B, 1, 1, L] mask is rejected by torch-npu's flash
            # attention kernel).
            slot_q = idx_ar.view(-1, 1)
            slot_k = idx_ar.view(1, -1)
            keep = (slot_q >= slot_k).unsqueeze(0) & ~pad_mask[:, idx_ar].unsqueeze(1)
            out_ar = F.scaled_dot_product_attention(
                q_ar,
                k_ar,
                v_ar,
                attn_mask=keep.unsqueeze(1),
                scale=softmax_scale,
            )
    else:
        out_ar = None

    if pad_mask is None:
        out_full = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=softmax_scale,
        )
    else:
        # Expand the [B, 1, 1, Skv] key mask to [B, 1, Sq, Skv]: torch-npu's
        # flash attention kernel rejects the broadcast form (only [B, N, Sq,
        # Skv] / [B, 1, Sq, Skv] / [1, 1, Sq, Skv] / [Sq, Skv] tile).
        batch, seq = pad_mask.shape
        key_keep = (~pad_mask)[:, None, None, :].expand(batch, 1, seq, seq).contiguous()
        out_full = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_keep,
            scale=softmax_scale,
        )

    if out_ar is not None:
        out_full = out_full.clone()
        out_full[:, :, idx_ar] = out_ar

    attn_output = out_full.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_out, _ = attn.o_proj(attn_output)
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    if deepstack_embed is not None and visual_pos_masks is not None:
        hidden_states = deepstack_process(
            hidden_states,
            visual_pos_masks,
            deepstack_embed,
        )

    return hidden_states


class _HiO1CacheBlock(nn.Module):
    """Pattern_3 view of one decoder layer for cache-dit.

    cache-dit drives blocks as ``(hidden_states, *conds) -> hidden_states``
    (ForwardPattern.Pattern_3); the flash two-pass path is a pure
    ``hidden -> hidden`` map per layer. Held by plain reference
    (object.__setattr__) so parameters stay registered only under
    ``HiDreamO1TextModel.layers``. ``visual_pos_masks`` / ``deepstack_embed``
    (TI2I reference features) are per-call: the mask shifts with the prompt and
    differs between cond/uncond, so it cannot be baked at construction.
    """

    def __init__(
        self,
        layer: nn.Module,
        deepstack_process: Callable[..., torch.Tensor],
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_deepstack_process", deepstack_process)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        idx_ar: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_embed: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Drops the residual (the flash path keeps the pure hidden->hidden map).
        # T2I blocks carry None deepstack; TI2I injects the reference features.
        return _flash_decoder_layer(
            self._layer,
            hidden_states,
            cos,
            sin,
            idx_ar,
            pad_mask,
            visual_pos_masks,
            deepstack_embed,
            self._deepstack_process,
        )


class HiO1CacheBlockAdapter(nn.Module):
    """cache-dit-wrappable view of the flash decoder block loop (Pattern_3).

    Threads ``cos`` / ``sin`` / ``idx_ar`` (and the per-call TI2I reference
    features ``visual_pos_masks`` / ``deepstack_visual_embeds``, plus the
    per-batch ``pad_mask`` for multi-prompt padding) through every block
    unchanged, then leaves the final RMSNorm to
    ``forward_post(apply_norm=True)``. ``pad_mask`` is ``None`` for
    single-prompt batches and step-invariant otherwise, so it is safe to thread
    as a cache-dit condition across denoising steps. The name must NOT start
    with ``HiDream``: cache-dit's ``BlockAdapterRegister.is_supported``
    prefix-matches class names and would mis-resolve this to its built-in
    (dual-stream) HiDream adapter instead of the custom Pattern_3 spec.
    """

    def __init__(self, model: "HiDreamO1ImageModel") -> None:
        super().__init__()
        text_model = model.language_model
        self.blocks = nn.ModuleList(
            [
                _HiO1CacheBlock(layer, text_model._deepstack_process)
                for layer in text_model.layers
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        idx_ar: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_deepstack = (
            len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0
        )
        for layer_idx, block in enumerate(self.blocks):
            deepstack_embed = (
                deepstack_visual_embeds[layer_idx] if layer_idx < n_deepstack else None
            )
            hidden_states = block(
                hidden_states,
                cos,
                sin,
                idx_ar,
                visual_pos_masks,
                deepstack_embed,
                pad_mask,
            )
        return hidden_states.contiguous()


class HiDreamO1ImageModel(nn.Module):
    def __init__(
        self,
        config: HiDreamO1ImageArchConfig,
        supported_attention_backends: set[AttentionBackendEnum],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tms_token_id = config.tms_token_id
        self.language_model = HiDreamO1TextModel(
            config=config,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.language_model",
        )
        self.t_embedder1 = TimestepEmbedder(
            hidden_size=config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.t_embedder1",
        )
        self.x_embedder = BottleneckPatchEmbed(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            quant_config=quant_config,
            prefix=f"{prefix}.x_embedder",
        )
        self.final_layer2 = FinalLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=config.in_channels,
            quant_config=quant_config,
            prefix=f"{prefix}.final_layer2",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        patchified_latents: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        import torch.distributed as dist
        _r = dist.get_rank() if dist.is_initialized() else -1
        print(
            f"[DBG r{_r}] ids{tuple(input_ids.shape)} "
            f"pos{tuple(position_ids.shape)} mask{tuple(attention_mask.shape)} "
            f"lat{tuple(patchified_latents.shape)}",
            flush=True,
        )
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        # <|tms_token|> slots carry the flow timestep instead of a token embedding.
        timestep_embeds = self.t_embedder1(timestep).to(inputs_embeds.dtype)
        timestep_slots = (input_ids == self.tms_token_id).unsqueeze(-1)
        inputs_embeds = torch.where(
            timestep_slots,
            timestep_embeds.unsqueeze(1).expand_as(inputs_embeds),
            inputs_embeds,
        )
        image_embeds = self.x_embedder(patchified_latents).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, image_embeds], dim=1)

        hidden_states = self.language_model(inputs_embeds, position_ids, attention_mask)
        return self.final_layer2(hidden_states)


class HiDreamO1ImageTransformer(BaseDiT):
    _fsdp_shard_conditions = [lambda n, m: isinstance(m, HiDreamO1DecoderLayer)]
    _compile_conditions = [lambda n, m: isinstance(m, HiDreamO1DecoderLayer)]
    # Set by the pipeline when the FSDP loader actually wrapped the model: the
    # TI2I two-pass loop drives decoder submodules directly, which bypasses
    # FSDP2's per-layer unshard hooks, so FSDP inference must take the dense
    # 4D-mask branch (mathematically equivalent, verified by the parity suite).
    _fsdp_wrapped = False
    param_names_mapping = HiDreamO1ImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping: dict = {}
    # The mixed causal/bidirectional mask is passed to attention as a dense
    # [B, 1, S, S] additive tensor, which only the SDPA path consumes.
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(
        self,
        config: HiDreamO1ImageDitConfig,
        hf_config: dict[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
        with_vision_tower: bool = False,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {})
        arch: HiDreamO1ImageArchConfig = self.config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.patch_size = arch.patch_size
        self.in_channels = arch.in_channels
        self.tms_token_id = arch.tms_token_id
        # Special token ids the TI2I conditioning path reads.
        self.image_token_id = arch.image_token_id
        self.video_token_id = arch.video_token_id
        self.vision_start_token_id = arch.vision_start_token_id

        self.model = HiDreamO1ImageModel(
            config=arch,
            supported_attention_backends=self._supported_attention_backends,
            quant_config=quant_config,
            prefix="model",
        )
        if with_vision_tower and hf_config and "vision_config" in hf_config:
            self.model.visual = Qwen3VLMoeVisionModel(
                Qwen3VLVisionConfig(**hf_config["vision_config"]),
                prefix="model.visual",
            )
        else:
            self.model.visual = None

    # Access un-nest the checkpoint's ``model.*`` prefix for the TI2I path,
    # whose methods predate the self.model grouping.
    @property
    def language_model(self) -> HiDreamO1TextModel:
        return self.model.language_model

    @property
    def t_embedder1(self):
        return self.model.t_embedder1

    @property
    def x_embedder(self):
        return self.model.x_embedder

    @property
    def final_layer2(self):
        return self.model.final_layer2

    @property
    def visual(self):
        return self.model.visual

    def post_load_weights(self) -> None:
        """Rebuild the vision tower's RoPE cache, which the checkpoint does not carry.

        The Qwen3-VL vision tower creates its rotary embedding while the model is
        being built on the meta device. ``cos_sin_cache`` is a non-persistent
        buffer, so the weight loader never materializes it and it stays on meta.
        Recompute it here on the real device, mirroring
        ``RotaryEmbedding._compute_cos_sin_cache``.
        """
        visual = self.model.visual
        if visual is None:
            return
        rotary_emb = getattr(visual, "rotary_pos_emb", None)
        if rotary_emb is None:
            return
        cache = getattr(rotary_emb, "cos_sin_cache", None)
        if cache is None or not cache.is_meta:
            return

        device = next(visual.parameters()).device
        # Recompute exactly what RotaryEmbedding._compute_cos_sin_cache would
        # have produced had the model not been built on the meta device.
        inv_freq = 1.0 / (
            rotary_emb.base
            ** (
                torch.arange(
                    0, rotary_emb.rotary_dim, 2, dtype=torch.float, device=device
                )
                / rotary_emb.rotary_dim
            )
        )
        t = torch.arange(
            rotary_emb.position_start,
            rotary_emb.position_start + rotary_emb.max_position_embeddings,
            dtype=torch.float,
            device=device,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        new_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        # Match RotaryEmbedding.__init__: the cache stays in FP32 on CUDA-like
        # devices for numerical stability, and is cast to the model dtype
        # elsewhere.
        if device.type not in ("cuda", "hip", "xpu"):
            new_cache = new_cache.to(rotary_emb.dtype)
        rotary_emb.register_buffer("cos_sin_cache", new_cache, persistent=False)

    # ------------------------------------------------------------------
    # TI2I / two-pass SDPA interface (Dev variants)
    # ------------------------------------------------------------------

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.visual is None:
            raise ValueError(
                "TI2I conditioning requires the vision tower, which is only "
                "built when the model is loaded with with_vision_tower=True."
            )
        pixel_values = pixel_values.to(self.visual.patch_embed.proj.weight.dtype)
        visual_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(visual_out[..., : self.hidden_size], split_sizes)
        deepstack_image_embeds = list(
            visual_out[..., self.hidden_size :].chunk(
                len(self.visual.deepstack_visual_indexes), dim=-1
            )
        )
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        special_image_mask = input_ids == self.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask_3d = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if image_features is not None:
            total_feat = sum(
                f.numel()
                for f in (
                    image_features
                    if isinstance(image_features, (list, tuple))
                    else [image_features]
                )
            )
            if inputs_embeds[special_image_mask_3d].numel() != total_feat:
                raise ValueError(
                    f"Image features and tokens mismatch: tokens={n_image_tokens}, features={total_feat}"
                )
        return special_image_mask, None

    def build_decoder_conds(
        self,
        position_ids: torch.Tensor,
        token_types: torch.Tensor,
    ) -> dict[str, Any]:
        position_ids = _normalize_position_ids(position_ids)
        # fp32 cos/sin: the flash path's rope precision (the dense path casts
        # to the embed dtype instead).
        cos, sin = self.language_model.rotary_emb(position_ids, torch.float32)
        is_gen = token_types[0].bool()
        idx_ar = torch.nonzero(~is_gen, as_tuple=False).squeeze(-1)
        return {
            "cos": cos,
            "sin": sin,
            "idx_ar": idx_ar,
            "position_ids": position_ids,
            "token_types": token_types,
            "visual_pos_masks": None,
            "deepstack_visual_embeds": None,
        }

    def _run_decoder_sdpa(
        self,
        inputs_embeds: torch.Tensor,
        conds: dict[str, Any],
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Two-pass SDPA decoder loop; see :func:`_flash_decoder_layer`."""
        text_model = self.language_model
        deepstack_embeds = conds["deepstack_visual_embeds"]
        n_deepstack = len(deepstack_embeds) if deepstack_embeds is not None else 0
        visual_pos_masks = conds["visual_pos_masks"]
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(text_model.layers):
            deepstack_embed = (
                deepstack_embeds[layer_idx] if layer_idx < n_deepstack else None
            )
            hidden_states = _flash_decoder_layer(
                decoder_layer,
                hidden_states,
                conds["cos"],
                conds["sin"],
                conds["idx_ar"],
                pad_mask,
                visual_pos_masks,
                deepstack_embed,
                text_model._deepstack_process,
            )

        return text_model.norm(hidden_states)

    def forward_pre(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        precomputed_image_embeds: Optional[torch.Tensor] = None,
        precomputed_deepstack_image_embeds: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any], tuple]:
        """Embed tokens, refs and patches; depends only on request + timestep, so the
        stage calls it once and reuses ``conds`` per cache-dit step. Returns
        (inputs_embeds, conds, (image_embeds, deepstack_image_embeds)) for caching."""
        inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        image_embeds = None
        deepstack_image_embeds = None

        if pixel_values is not None:
            if (
                precomputed_image_embeds is not None
                and precomputed_deepstack_image_embeds is not None
            ):
                image_embeds = precomputed_image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                deepstack_image_embeds = [
                    e.to(inputs_embeds.device, inputs_embeds.dtype)
                    for e in precomputed_deepstack_image_embeds
                ]
            else:
                image_embeds_list, deepstack_image_embeds = self.get_image_features(
                    pixel_values, image_grid_thw
                )
                image_embeds = torch.cat(image_embeds_list, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )

        visual_pos_masks = None
        if image_mask is not None:
            visual_pos_masks = (
                image_mask[..., 0] if image_mask.dim() == 3 else image_mask
            )

        if isinstance(timestep, list):
            timestep = torch.cat(timestep, dim=0)
        timestep = timestep.to(inputs_embeds.device)
        t_emb = self.t_embedder1(timestep)
        tms_mask = input_ids == self.tms_token_id
        tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(inputs_embeds)
        t_emb_expanded = t_emb.unsqueeze(1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)

        if isinstance(vinputs, list):
            vinputs = torch.cat(vinputs, dim=0)
        vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, _, _ = inputs_embeds.shape

        if visual_pos_masks is not None:
            vinputs_seq_len = vinputs_embedded.shape[1]
            if visual_pos_masks.shape[0] != batch_size:
                visual_pos_masks = visual_pos_masks.expand(batch_size, -1)
            pad = torch.zeros(
                visual_pos_masks.shape[0],
                vinputs_seq_len,
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        if isinstance(token_types, list):
            token_types = torch.cat(token_types, dim=0)
        token_types = token_types.to(inputs_embeds.device)
        if token_types.dim() == 1:
            token_types = token_types.unsqueeze(0)
        elif token_types.dim() == 2 and token_types.shape[-1] == 1:
            token_types = token_types.squeeze(-1).unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)

        conds = self.build_decoder_conds(position_ids, token_types)
        conds["visual_pos_masks"] = visual_pos_masks
        conds["deepstack_visual_embeds"] = deepstack_image_embeds

        return inputs_embeds, conds, (image_embeds, deepstack_image_embeds)

    def forward_post(
        self, hidden_states: torch.Tensor, apply_norm: bool = False
    ) -> dict[str, torch.Tensor]:
        """Project decoder output to patch predictions.
        apply_norm=True for cache-dit callers; final RMSNorm is outside the blocks."""
        if apply_norm:
            hidden_states = self.language_model.norm(hidden_states)
        return {"x_pred": self.final_layer2(hidden_states)}

    def _forward_generation(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False,
        pad_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for one denoising step; returns {'x_pred': patches}."""
        precomputed_image_embeds = kwargs.pop("precomputed_image_embeds", None)
        precomputed_deepstack = kwargs.pop("precomputed_deepstack_image_embeds", None)

        inputs_embeds, conds, embeds_pair = self.forward_pre(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=vinputs,
            timestep=timestep,
            token_types=token_types,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            precomputed_image_embeds=precomputed_image_embeds,
            precomputed_deepstack_image_embeds=precomputed_deepstack,
        )
        image_embeds, deepstack_image_embeds = embeds_pair
        token_types = conds["token_types"]
        batch_size, total_seq_len, _ = inputs_embeds.shape

        if use_flash_attn and not self._fsdp_wrapped:
            hidden_states = self._run_decoder_sdpa(
                inputs_embeds,
                conds,
                pad_mask=pad_mask,
            )
        else:
            dtype = inputs_embeds.dtype
            min_val = torch.finfo(dtype).min
            attn_masks = []
            for b in range(batch_size):
                causal = torch.full(
                    (total_seq_len, total_seq_len),
                    min_val,
                    device=inputs_embeds.device,
                    dtype=dtype,
                )
                causal = torch.triu(causal, diagonal=1)
                gen_positions = token_types[b].bool()
                causal[gen_positions, :] = 0
                if pad_mask is not None:
                    # exclude padded keys for every query, incl. generation rows
                    causal[:, pad_mask[b]] = min_val
                attn_masks.append(causal)
            attention_mask_4d = torch.stack(attn_masks, dim=0).unsqueeze(1)

            hidden_states = self.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask_4d,
                visual_pos_masks=conds["visual_pos_masks"],
                deepstack_visual_embeds=deepstack_image_embeds,
            )

        out = self.forward_post(hidden_states)
        if pixel_values is not None:
            # vision embeds cached across denoising steps
            out["cond_image_embeds"] = image_embeds
            out["cond_deepstack_image_embeds"] = deepstack_image_embeds
        return out

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        vinputs: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        use_flash_attn: bool = False,
        pad_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        if vinputs is not None:
            # Dev variants' two-pass / dense generation path.
            return self._forward_generation(
                input_ids=input_ids,
                position_ids=position_ids,
                vinputs=vinputs,
                timestep=timestep,
                token_types=token_types,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_flash_attn=use_flash_attn,
                pad_mask=pad_mask,
                **kwargs,
            )
        return self._forward_dense(
            hidden_states=hidden_states,
            timestep=timestep,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Text conditioning arrives as input_ids because the backbone embeds the
        # prompt itself, so encoder_hidden_states is unused.
        seq_len = position_ids.shape[-1]
        # Dim 0 may be 1: every request in a batch shares one prompt, so SDPA
        # broadcasts a single [1, 1, S, S] mask across the batch.
        if tuple(attention_mask.shape) not in (
            (1, 1, seq_len, seq_len),
            (hidden_states.shape[0], 1, seq_len, seq_len),
        ):
            raise ValueError(
                f"attention_mask must be [1 or {hidden_states.shape[0]}, 1, "
                f"{seq_len}, {seq_len}]; got {tuple(attention_mask.shape)}"
            )
        x_pred = self.model(
            input_ids=input_ids,
            patchified_latents=hidden_states,
            timestep=timestep,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        # The patch tokens are appended after the prompt, so the image span is
        # exactly the trailing rows; text positions carry no velocity. fp32 so
        # the CFG mix and the velocity conversion in the denoising stage run at
        # the reference implementation's precision.
        return x_pred[:, -hidden_states.shape[1] :].float()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens


EntryClass = HiDreamO1ImageTransformer
