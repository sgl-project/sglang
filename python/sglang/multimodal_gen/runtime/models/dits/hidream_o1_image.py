# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image: a pixel-level unified transformer (UiT) for text-to-image.

The checkpoint is a Qwen3-VL text backbone plus a flow-matching pixel head
(``x_embedder`` / ``t_embedder1`` / ``final_layer2``). There is no VAE and no
separate text encoder: the prompt is embedded by the backbone itself and the
image is a sequence of 32x32 raw pixel patches appended to the prompt tokens.
``model.visual.*`` and ``lm_head.weight`` ship in the checkpoint but are unused
for text-to-image, so the loader filters them out.
"""

import math
from typing import Any

import torch
from torch import nn

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
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# Checkpoint prefixes that carry no parameter of this model.
UNUSED_CHECKPOINT_PREFIXES = ("model.visual.", "lm_head.")


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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

        # Rope wants [B, H, S, D]; LocalAttention wants [B, S, H, D].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        attn_output = self.attn(
            q.transpose(1, 2), k.transpose(1, 2), v, attn_mask=attention_mask
        )
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

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, cos, sin, attention_mask
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
    ) -> None:
        super().__init__(config=config, hf_config=hf_config or {})
        arch: HiDreamO1ImageArchConfig = self.config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.patch_size = arch.patch_size
        self.in_channels = arch.in_channels
        self.model = HiDreamO1ImageModel(
            config=arch,
            supported_attention_backends=self._supported_attention_backends,
            quant_config=quant_config,
            prefix="model",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
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


EntryClass = HiDreamO1ImageTransformer
