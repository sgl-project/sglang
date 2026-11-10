"""Opt-in, topology-stable Qwen3-MoE arithmetic for the fixed SID fixture.

Native SGLang still owns checkpoint sharding, routing, scheduling, KV, beams,
catalogs and collectives. The reference uses Torch expert GEMMs and combines
separate top-k slots before rounding; this adds memory/communication and is
not the production Triton MoE performance path.
"""

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import append_stages
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.runtime_context import get_parallel

from .model import SIDLogitsProcessor, SIDQwen3Attention, SIDRMSNorm


def combine_expert_slots(local_slots, group, hidden_size):
    """Every token/route slot has one owning rank; reductions add only zeros."""
    if group.world_size > 1:
        local_slots = group.all_reduce(local_slots)
    full = local_slots.sum(dim=1).to(torch.bfloat16)
    if group.world_size == 1:
        return full
    # The ordinary sparse-FFN stage boundary still performs its all-reduce.
    # Return disjoint output rows so that it assembles rather than re-sums.
    width = hidden_size // group.world_size
    start = group.rank_in_group * width
    result = torch.zeros_like(full)
    result[:, start : start + width] = full[:, start : start + width]
    return result


class SIDReferenceExperts(FusedMoE):
    """Keep native expert weight ownership/loading; evaluate full local experts."""

    def forward(self, hidden_states, topk_output, **kwargs):
        if self.moe_tp_size != 1 or not isinstance(topk_output, StandardTopKOutput):
            raise ValueError("MoE reference requires whole experts and standard top-k")
        ids, weights = topk_output.topk_ids, topk_output.topk_weights
        slots = torch.zeros(
            (*ids.shape, self.hidden_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        first = self.moe_ep_rank * self.num_local_experts
        for local_id in range(self.num_local_experts):
            token, slot = torch.where(ids == first + local_id)
            if token.numel() == 0:
                continue
            gate, up = F.linear(hidden_states[token], self.w13_weight[local_id]).chunk(
                2, dim=-1
            )
            intermediate = F.silu(gate) * up
            value = F.linear(intermediate, self.w2_weight[local_id])
            slots[token, slot] = value.float() * weights[token, slot, None]
        return combine_expert_slots(slots, get_parallel().tp_group, self.hidden_size)


class SIDQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config,
        layer_id,
        start_layer=0,
        quant_config=None,
        prefix="",
        alt_stream=None,
    ):
        nn.Module.__init__(self)
        if quant_config is not None:
            raise ValueError("MoE reference requires unquantized BF16 weights")
        self.config, self.hidden_size, self.layer_id = (
            config,
            config.hidden_size,
            layer_id,
        )
        self.is_layer_sparse = True
        rope = getattr(config, "rope_parameters", None)
        theta = (rope or {}).get("rope_theta", getattr(config, "rope_theta", 1000000))
        scaling = rope or getattr(config, "rope_scaling", None)
        if scaling and scaling.get("rope_type", "default") != "default":
            raise ValueError("MoE reference requires ordinary Qwen3 RoPE")
        self.self_attn = SIDQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=theta,
            rope_scaling=scaling,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3MoeSparseMoeBlock(
            layer_id, config, prefix=add_prefix("mlp", prefix)
        )
        self.mlp.experts = SIDReferenceExperts(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=add_prefix("mlp.experts", prefix),
        )
        self.input_layernorm = SIDRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SIDRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        attn, ffn = self.stage_facts(config, layer_id)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm), (ffn, self.post_attention_layernorm)
        )


class SIDQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        parallel = get_parallel()
        if (
            parallel.tp_size not in (1, 2, 4)
            or parallel.moe_ep_size != parallel.tp_size
            or parallel.moe_tp_size != 1
        ):
            raise ValueError("MoE reference requires TP=EP in {1,2,4}")
        geometry = tuple(
            getattr(config, k, None)
            for k in (
                "hidden_size",
                "moe_intermediate_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "num_experts",
                "num_experts_per_tok",
            )
        )
        if geometry != (1024, 1024, 256, 16, 4, 64, 8, 2):
            raise ValueError(
                "MoE reference supports only the qualified dummy fixture geometry"
            )
        if (
            quant_config is not None
            or config.attention_bias
            or not config.norm_topk_prob
        ):
            raise ValueError(
                "MoE reference requires bias-free BF16 and normalized routing"
            )
        super().__init__(config, quant_config, prefix)
        self.model = Qwen3MoeModel(
            config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=SIDQwen3MoeDecoderLayer,
        )
        self.model.norm = SIDRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logits_processor = SIDLogitsProcessor(config)


EntryClass = SIDQwen3MoeForCausalLM
