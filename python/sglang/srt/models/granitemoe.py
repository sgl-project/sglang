"""Inference-only GraniteMoe model."""

from typing import Iterable, Iterator, Optional

import torch
from torch import nn
from transformers import GraniteConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models import mixtral
from sglang.srt.models.granite import build_attention_sinks, granite_layer_attn_params
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

SHARED_MOE_MODEL_TYPES = frozenset({"granitemoeshared", "granitemoe_swa"})


def granitemoe_split_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Split the packed expert tensors into per-expert w1/w2/w3."""
    for name, weight in weights:
        if name.endswith(".experts.gate_up_proj"):
            for e in range(weight.size(0)):
                expert = name.replace(".experts.gate_up_proj", f".experts.{e}")
                w1, w3 = weight[e].chunk(2, dim=0)
                yield f"{expert}.w1.weight", w1
                yield f"{expert}.w3.weight", w3
        elif name.endswith(".experts.down_proj"):
            for e in range(weight.size(0)):
                expert = name.replace(".experts.down_proj", f".experts.{e}")
                yield f"{expert}.w2.weight", weight[e]
        else:
            yield name, weight


class GraniteMoeMoE(nn.Module):
    """A tensor-parallel MoE implementation for GraniteMoe that shards each
    expert across all ranks.
    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.topk = TopK(
            top_k=top_k,
            renormalize=True,
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        return final_hidden_states.view(orig_shape)


class GraniteMoeSharedMLP(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.input_linear = MergedColumnParallelLinear(
            input_size=self.input_size,
            output_sizes=[self.hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("input_linear", prefix),
        )
        self.output_linear = RowParallelLinear(
            self.hidden_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("output_linear", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.input_linear(hidden_states)
        x = self.act_fn(gate_up)
        x, _ = self.output_linear(x)
        return x


class GraniteMoeAttention(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        attention_multiplier: Optional[float] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = (
            attention_multiplier
            if attention_multiplier is not None
            else self.head_dim**-1
        )
        sliding_window_size, self.rope_theta, has_sink = granite_layer_attn_params(
            config, layer_id
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )
            if self.rope_theta
            else None
        )
        self.sinks = build_attention_sinks(self.num_heads) if has_sink else None
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            sliding_window_size=sliding_window_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.sinks is None:
            attn_output = self.attn(q, k, v, forward_batch)
        else:
            attn_output = self.attn(q, k, v, forward_batch, sinks=self.sinks)
        output, _ = self.o_proj(attn_output)
        return output


class GraniteMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GraniteMoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attention_multiplier=config.attention_multiplier,
        )
        self.block_sparse_moe = GraniteMoeMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe",
        )
        shared_intermediate_size = (
            config.shared_intermediate_size
            if config.model_type in SHARED_MOE_MODEL_TYPES
            else 0
        )
        self.shared_mlp = (
            None
            if shared_intermediate_size == 0
            else GraniteMoeSharedMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
            )
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            # Routed experts consume `hidden_states`, so compute shared first
            shared_output = self.shared_mlp(hidden_states)
            hidden_states = self.block_sparse_moe(hidden_states) + shared_output
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


class GraniteMoeModel(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.embedding_multiplier = config.embedding_multiplier

        self.layers = nn.ModuleList(
            [
                GraniteMoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        hidden_states *= self.embedding_multiplier

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                forward_batch,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GraniteMoeForCausalLM(nn.Module):
    # Legacy and current HF expert / router names with otherwise shared layout
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_suffix={
            ".block_sparse_moe.input_linear.weight": ".block_sparse_moe.experts.gate_up_proj",
            ".block_sparse_moe.output_linear.weight": ".block_sparse_moe.experts.down_proj",
            ".block_sparse_moe.router.layer.weight": ".block_sparse_moe.gate.weight",
            ".block_sparse_moe.router.weight": ".block_sparse_moe.gate.weight",
        }
    )

    def __init__(
        self,
        config: GraniteConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = GraniteMoeModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # Granite logit scaling factors are applied via division, but
        # LogitsProcessor expects a multiplicative factor.
        if hasattr(config, "logits_scaling"):
            logit_scale = 1.0 / config.logits_scaling
        else:
            logit_scale = None
        self.logits_processor = LogitsProcessor(config, logit_scale=logit_scale)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if not get_embedding:
            logits_processor_output: LogitsProcessorOutput = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
            return logits_processor_output
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = granitemoe_split_expert_weights(
            self.hf_to_sglang_mapper.apply(weights)
        )
        mixtral.MixtralForCausalLM.load_weights(self, weights)


class GraniteMoeSharedForCausalLM(GraniteMoeForCausalLM):
    pass


class GraniteMoeSWAForCausalLM(GraniteMoeForCausalLM):
    pass


EntryClass = [
    GraniteMoeForCausalLM,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSWAForCausalLM,
]
