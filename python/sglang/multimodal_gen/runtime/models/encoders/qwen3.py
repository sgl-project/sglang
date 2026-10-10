from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm as MMGenRMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    load_llm_encoder_weights,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    TextEncoder,
    get_attention_head_partition,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm


class Qwen3HfRowParallelLinear(RowParallelLinear):
    """Preserve HF's full GEMM while retaining checkpoint-backed TP shards."""

    def forward(self, input_) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            self.tp_size == 1
            or self.quant_config is not None
            or not self.reduce_results
            or input_.dtype not in (torch.float16, torch.bfloat16)
        ):
            return super().forward(input_)

        if self.input_is_parallel:
            full_input = tensor_model_parallel_all_gather(
                input_.contiguous(), dim=-1, tp_group=self.tp_group
            )
        else:
            full_input = input_
        full_weight = tensor_model_parallel_all_gather(
            self.weight, dim=1, tp_group=self.tp_group
        )

        # Even FP32 split-K partials change the reduction order enough to
        # accumulate beyond Ovis conditioning tolerances. Reconstruct only
        # this projection for the exact full GEMM, then release its gathered
        # weight. Do not cache it: offload and weight updates own the shards.
        bias = None if self.skip_bias_add else self.bias
        with torch.autocast(device_type=input_.device.type, enabled=False):
            output = F.linear(full_input, full_weight, bias)
        return output, self.bias if self.skip_bias_add else None


class Qwen3MLP(nn.Module):
    """Qwen3 MLP with SwiGLU activation and tensor parallelism."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        preserve_hf_numerics: bool = False,
    ) -> None:
        super().__init__()
        self.preserve_hf_numerics = preserve_hf_numerics
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        row_parallel_cls = (
            Qwen3HfRowParallelLinear if preserve_hf_numerics else RowParallelLinear
        )
        self.down_proj = row_parallel_cls(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        if self.preserve_hf_numerics:
            gate, up = x.chunk(2, dim=-1)
            x = F.silu(gate) * up
        else:
            x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    """Qwen3 attention with QK-Norm and tensor parallelism.

    Key difference from LLaMA: RMSNorm is applied to Q and K before attention.
    """

    def __init__(
        self,
        config: Qwen3TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 40960,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.preserve_hf_numerics = config.preserve_hf_numerics
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads, self.num_kv_heads = get_attention_head_partition(
            num_heads, num_kv_heads, tp_size
        )

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.rotary_dim = self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.rope_scaling_factor = (
            float(rope_scaling.get("factor", 1.0))
            if rope_scaling
            and rope_scaling.get("rope_type", rope_scaling.get("type")) == "linear"
            else 1.0
        )
        self.max_position_embeddings = max_position_embeddings

        # QKV projection with tensor parallelism
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        row_parallel_cls = (
            Qwen3HfRowParallelLinear if self.preserve_hf_numerics else RowParallelLinear
        )
        self.o_proj = row_parallel_cls(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-Norm: Key difference from LLaMA
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        # Keep the small-hidden one-pass kernel used by diffusion QK norm.
        if self.preserve_hf_numerics:
            self.q_norm = RMSNorm(
                self.head_dim,
                eps=rms_norm_eps,
                cast_x_before_out_mul=True,
                force_native=True,
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                eps=rms_norm_eps,
                cast_x_before_out_mul=True,
                force_native=True,
            )
        else:
            self.q_norm = MMGenRMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = MMGenRMSNorm(self.head_dim, eps=rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        # Attention with FlashAttention/SageAttn support
        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
            supported_attention_backends=config._supported_attention_backends,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_lengths: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for QK-norm
        batch_size, seq_len = q.shape[0], q.shape[1]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK-Norm (key difference from LLaMA)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape back for rotary embeddings
        q = q.reshape(batch_size, seq_len, -1)
        k = k.reshape(batch_size, seq_len, -1)

        # Apply rotary embeddings
        if self.preserve_hf_numerics:
            q, k = self._apply_hf_rope(positions, q, k)
        else:
            q, k = self.rotary_emb(positions, q, k)

        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Attention
        attn_output = self._masked_causal_attention(q, k, v, attention_lengths)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output

    def _apply_hf_rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # HF casts cos/sin and rounds each product before their sum.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=query.device
                )
                / self.rotary_dim
            )
        )
        if self.rope_scaling_factor != 1.0:
            inv_freq = inv_freq / self.rope_scaling_factor
        frequencies = positions.float().unsqueeze(-1) * inv_freq
        cos = torch.cat([frequencies.cos()] * 2, dim=-1).to(query.dtype).unsqueeze(-2)
        sin = torch.cat([frequencies.sin()] * 2, dim=-1).to(query.dtype).unsqueeze(-2)

        def rotate(value):
            shape = value.shape
            value = value.reshape(*shape[:2], -1, self.head_dim)
            first, second = value.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            return (value * cos + rotated * sin).reshape(shape)

        return rotate(query), rotate(key)

    def _hf_masked_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_lengths: tuple[int, ...] | None,
    ) -> torch.Tensor:
        # A full padding mask also preserves HF's SDPA kernel selection.
        batch_size, seq_len = q.shape[:2]
        if attention_lengths is None or all(
            length == seq_len for length in attention_lengths
        ):
            # HF omits the mask for an unpadded causal prefill and enables GQA.
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=seq_len > 1,
                scale=self.scaling,
                enable_gqa=self.num_heads != self.num_kv_heads,
            ).transpose(1, 2)
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
        mask = mask[None, None].expand(batch_size, 1, seq_len, seq_len)
        if attention_lengths is not None:
            valid = (
                torch.arange(seq_len, device=q.device)[None, :]
                < torch.tensor(attention_lengths, device=q.device)[:, None]
            )
            mask = mask & valid[:, None, None, :]
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        ).transpose(1, 2)

    def _masked_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_lengths: tuple[int, ...] | None,
    ) -> torch.Tensor:
        if self.preserve_hf_numerics:
            return self._hf_masked_causal_attention(q, k, v, attention_lengths)
        if attention_lengths is None:
            return self.attn(q, k, v)

        seq_len = q.shape[1]
        if all(valid_len == seq_len for valid_len in attention_lengths):
            return self.attn(q, k, v)

        outputs: list[torch.Tensor] = []
        for batch_index, valid_len in enumerate(attention_lengths):
            q_item = q[batch_index : batch_index + 1]
            k_item = k[batch_index : batch_index + 1]
            v_item = v[batch_index : batch_index + 1]

            if valid_len == 0:
                outputs.append(torch.zeros_like(q_item))
                continue

            real_output = self.attn(
                q_item[:, :valid_len],
                k_item[:, :valid_len],
                v_item[:, :valid_len],
            )
            if valid_len == seq_len:
                outputs.append(real_output)
                continue

            pad_q = q_item[:, valid_len:].transpose(1, 2)
            real_k = k_item[:, :valid_len].transpose(1, 2)
            real_v = v_item[:, :valid_len].transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                real_k = real_k.repeat_interleave(repeat_factor, dim=1)
                real_v = real_v.repeat_interleave(repeat_factor, dim=1)
            pad_output = torch.nn.functional.scaled_dot_product_attention(
                pad_q,
                real_k,
                real_v,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            ).transpose(1, 2)
            outputs.append(torch.cat([real_output, pad_output], dim=1))

        return torch.cat(outputs, dim=0)


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer decoder layer."""

    def __init__(
        self,
        config: Qwen3TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.preserve_hf_numerics = config.preserve_hf_numerics
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        attention_bias = getattr(config, "attention_bias", False)

        self.self_attn = Qwen3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
            preserve_hf_numerics=self.preserve_hf_numerics,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cast_x_before_out_mul=self.preserve_hf_numerics,
            force_native=self.preserve_hf_numerics,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cast_x_before_out_mul=self.preserve_hf_numerics,
            force_native=self.preserve_hf_numerics,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_lengths: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if self.preserve_hf_numerics:
            if residual is not None:
                hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attention_lengths=attention_lengths,
        )

        # MLP
        if self.preserve_hf_numerics:
            residual = hidden_states + residual
            hidden_states = self.post_attention_layernorm(residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3ForCausalLM(TextEncoder):
    """Qwen3 causal language model for text encoding in diffusion models.

    Features:
    - Tensor parallelism support
    - FlashAttention/SageAttn/SDPA support via LocalAttention
    - QK-Norm for better training stability
    - FSDP sharding for CPU offload
    """

    _aliases = ["Qwen3Model"]

    def __init__(self, config: Qwen3TextConfig) -> None:
        super().__init__(config)

        self.config = config
        self.quant_config = config.quant_config
        self.preserve_hf_numerics = config.preserve_hf_numerics

        # Embedding layer with tensor parallelism
        if config.lora_config is not None:
            max_loras = getattr(config.lora_config, "max_loras", 1)
            lora_vocab_size = getattr(config.lora_config, "lora_extra_vocab_size", 1)
            lora_vocab = lora_vocab_size * max_loras
        else:
            lora_vocab = 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=config.quant_config,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config=config,
                    quant_config=config.quant_config,
                    prefix=f"{config.prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cast_x_before_out_mul=self.preserve_hf_numerics,
            force_native=self.preserve_hf_numerics,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        residual = None

        if position_ids is None:
            position_ids = (
                torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
            )

        attention_lengths = None
        if attention_mask is not None:
            attention_lengths = tuple(
                int(valid_len)
                for valid_len in attention_mask.sum(dim=-1).detach().cpu().tolist()
            )

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )
            hidden_states, residual = layer(
                position_ids, hidden_states, residual, attention_lengths
            )

        if self.preserve_hf_numerics:
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        # Add hidden states from the last decoder layer
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with support for tensor parallelism and weight remapping."""
        return load_llm_encoder_weights(
            weights,
            dict(self.named_parameters()),
            self.config.arch_config.stacked_params_mapping,
            strip_prefix="model.",
        )


EntryClass = Qwen3ForCausalLM
