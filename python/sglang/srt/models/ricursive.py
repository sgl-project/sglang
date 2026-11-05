"""Inference-only Ricursive (Ripple) model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.ricursive import SLIDING_ATTENTION, RicursiveConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import add_prefix, is_cuda, make_layers

_is_cuda = is_cuda()


class RicursiveMLP(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if self.clamp_threshold:
            gate, up = gate_up.chunk(2, dim=-1)
            gate.clamp_(max=self.clamp_threshold)
            up.clamp_(min=-self.clamp_threshold, max=self.clamp_threshold)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class RicursiveAttention(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        tp_size = get_parallel().tp_size
        hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.alt_stream = alt_stream
        total_q_size = config.num_attention_heads * self.head_dim
        total_kv_size = config.num_key_value_heads * self.head_dim
        if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
            raise ValueError(
                f"Ricursive needs --tp-size to divide {config.num_attention_heads} "
                f"Q heads and {config.num_key_value_heads} KV heads, got {tp_size}."
            )

        def column_linear(name: str, output_size: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                hidden_size,
                output_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix(name, prefix),
            )

        self.q_proj = column_linear("q_proj", total_q_size)
        self.k_proj = column_linear("k_proj", total_kv_size)
        self.v_proj = column_linear("v_proj", total_kv_size)
        self.g_proj = (
            column_linear("g_proj", total_q_size)
            if config.use_gated_attention
            else None
        )
        self.o_proj = RowParallelLinear(
            total_q_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.value_res_lambda = (
            nn.Parameter(torch.zeros(1), requires_grad=False)
            if config.use_valres
            else None
        )

        # Sliding layers use RoPE and full layers have no position embedding.
        is_sliding = config.layer_types[layer_id] == SLIDING_ATTENTION
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
            )
            if is_sliding
            else None
        )
        self.attn = RadixAttention(
            config.num_attention_heads // tp_size,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=config.num_key_value_heads // tp_size,
            layer_id=layer_id,
            # HF's window counts the current token, SGLang's does not.
            sliding_window_size=config.sliding_window - 1 if is_sliding else -1,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        value_input = hidden_states
        if self.value_res_lambda is not None:
            value_input = torch.lerp(
                value_residual, hidden_states, torch.sigmoid(self.value_res_lambda)
            )
        v, _ = self.v_proj(value_input)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate.float()).to(
                attn_output.dtype
            )
        output, _ = self.o_proj(attn_output)
        return output


class RicursiveDecoderLayer(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.self_attn = RicursiveAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = RicursiveMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output = self.self_attn(
            positions,
            self.input_layernorm(hidden_states),
            forward_batch,
            value_residual,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_output)
        mlp_output = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states + self.post_feedforward_layernorm(mlp_output)


class RicursiveModel(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_scale = config.embed_scale
        self.use_valres = config.use_valres
        alt_stream = get_stream("alt") if _is_cuda else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: RicursiveDecoderLayer(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if self.embed_scale is not None:
            hidden_states = hidden_states * self.embed_scale
        value_residual = hidden_states if self.use_valres else None
        for layer in self.layers:
            hidden_states = layer(
                positions, hidden_states, forward_batch, value_residual
            )
        return self.norm(hidden_states)


class RicursiveForCausalLM(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = RicursiveModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self) -> int:
        return self.config.sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for weight_name, shard_id in (("gate_proj", 0), ("up_proj", 1)):
                if weight_name in name:
                    param = params_dict[name.replace(weight_name, "gate_up_proj")]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = RicursiveForCausalLM
