from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import einops
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    JetBlockAttnBackend,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.utils import add_prefix

from .dynamic_conv import DynamicShortConvolution


def init_linear_conv1d(
    weight: torch.Tensor, std: float, bias: torch.Tensor | None = None
) -> None:
    weight.data.normal_(mean=0.0, std=std)
    if bias is not None:
        if not getattr(bias, "_no_reinit", False):
            nn.init.zeros_(bias)


class JetBlock(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        jet_block_config = JetBlockConfig(
            **self.config.efficient_attention_config[self.config.layer_types[layer_id]]
        )

        hidden_size = self.config.hidden_size
        num_heads = jet_block_config.num_heads
        head_k_dim = jet_block_config.head_dim
        total_k_dim = num_heads * head_k_dim
        head_v_dim = int(head_k_dim * jet_block_config.expand_v)
        total_v_dim = num_heads * head_v_dim

        self.head_v_dim = head_v_dim
        self.num_heads = num_heads

        # Submodules.
        self.q_proj = nn.Linear(hidden_size, total_k_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_v_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, total_v_dim, bias=False)
        self.o_proj = nn.Linear(total_v_dim, hidden_size, bias=False)

        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))

        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=total_v_dim,
            kernel_size=jet_block_config.conv_size,
            generator_input_size=hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=lambda x: init_linear_conv1d(
                x, std=self.config.initializer_range
            ),
            implementation=jet_block_config.dconv_implementation,
        )

        self.o_norm = RMSNormGated(
            self.head_v_dim,
            eps=flaot(jet_block_config.norm_eps),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = nn.functional.silu(q)
        q = einops.rearrange(q, "... (h d) -> ... h d", h=self.num_heads)

        k = self.k_proj(hidden_states)
        k = nn.functional.silu(k)
        k = einops.rearrange(k, "... (h d) -> ... h d", h=self.num_heads)

        v = self.v_proj(hidden_states)
        v = einops.rearrange(v, "... (h d) -> ... h d", h=self.num_heads)

        a = self.a_proj(hidden_states)

        b = self.b_proj(hidden_states)
        b = nn.functional.sigmoid(b)

        assert isinstance(forward_batch.attn_backend, HybridLinearAttnBackend)
        assert isinstance(
            forward_batch.attn_backend.linear_attn_backend, JetBlockAttnBackend
        )

        attn_backend_kwargs: dict[str, Any] = dict(
            q=q,
            k=k,
            v=v,
            layer=self,
            forward_batch=forward_batch,
        )

        o = forward_batch.attn_backend.linear_attn_backend.forward(
            **attn_backend_kwargs
        )

        g = self.g_proj(hidden_states)
        g = einops.rearrange(g, "... (h d) -> ... h d", h=self.num_heads)

        o = self.o_norm(o, g)

        o = einops.rearrange(o, "... h d -> ... (h d)")
        o = self.o_proj(o)

        return o


class JetNemotronAttention(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.q_size = self.config.num_attention_heads * self.head_dim
        self.kv_size = self.config.num_key_value_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.config.hidden_size,
            self.head_dim,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.config.num_attention_heads * self.head_dim,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.config.max_position_embeddings,
            base=int(self.config.rope_theta),
            rope_scaling=self.config.rope_scaling,
        )

        match self.config.layer_types[layer_id]:
            case "attn":
                sliding_window_size = -1

            case "swa":
                sliding_window_size = self.config.efficient_attention_config["swa"][
                    "window_size"
                ]

            case _:
                raise NotImplementedError

        self.attn = RadixAttention(
            self.config.num_attention_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            quant_config=quant_config,
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
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class JetNemotronDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()

        match config.layer_types[layer_id]:
            case "attn" | "swa":
                self.self_attn = JetNemotronAttention(
                    config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                )

            case "jet":
                self.self_attn = JetBlock(
                    config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                )

            case _:
                raise NotImplementedError

        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class JetNemotronForCausalLM(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.model = Qwen2Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=JetNemotronDecoderLayer,
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput | LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            # (param_name, shard_weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for weight_name, loaded_weight in weights:
            # Handle stacked parameters first.
            for (
                param_name_part,
                shard_weight_name_part,
                shard_id,
            ) in stacked_params_mapping:
                if shard_weight_name_part not in weight_name.split("."):
                    continue

                param_name = weight_name.replace(
                    shard_weight_name_part, param_name_part
                )

                if param_name not in params_dict:
                    # Fall back to direct match if no such stacked parameter.
                    continue

                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break

            else:
                param_name = weight_name

                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = JetNemotronForCausalLM
EntryClass = JetNemotronForCausalLM
