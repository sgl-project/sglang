from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
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
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder


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
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
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
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.rotary_dim = self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
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
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-Norm: Key difference from LLaMA
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

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
        q, k = self.rotary_emb(positions, q, k)

        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Attention
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


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
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
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
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
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

    def __init__(self, config: Qwen3TextConfig) -> None:
        super().__init__(config)

        self.config = config
        self.quant_config = config.quant_config

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
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )
            hidden_states, residual = layer(position_ids, hidden_states, residual)

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
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Strip 'model.' prefix from HuggingFace Qwen3 weights
            if name.startswith("model."):
                name = name[6:]  # len("model.") == 6

            # Skip rotary embedding weights
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Handle KV scale remapping
            if "scale" in name:
                kv_scale_name: str | None = maybe_remap_kv_scale_name(name, params_dict)
                if kv_scale_name is None:
                    continue
                else:
                    name = kv_scale_name

            # Handle stacked params mapping (qkv_proj, gate_up_proj)
            for (
                param_name,
                weight_name,
                shard_id,
            ) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


EntryClass = Qwen3ForCausalLM
