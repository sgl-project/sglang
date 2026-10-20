# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#57963 (938301b11a751912fea631f19ef45975cc60e28c).
"""Inference-only AliceAI: KDA, gated full attention, block AttnRes and MoE.

Supports ordinary generation and checkpoint MTP through NEXTN/EAGLE.
Tensor and expert parallelism use the standard all-reduce MoE path.
Expert placement is fixed; EPLB and redundant experts are not supported.
Pipeline parallelism, DP attention and all-to-all MoE backends are not supported.
CPU weight offloading via --cpu-offload-gb is not supported.
The companion AliceAI MTP model loads the checkpoint's mtp.* weights.
"""

from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.alice_ai import AliceAIConfig
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attn_residual import aggregate_stream, aggregate_stream_torch
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream
from sglang.srt.utils import add_prefix, is_cuda, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config


def _attention_residual(
    residual_bank: torch.Tensor,
    num_completed_sources: int,
    partial: torch.Tensor | None,
    projection: ReplicatedLinear,
    norm: RMSNorm,
) -> torch.Tensor:
    if partial is None:
        prefix = residual_bank[:, num_completed_sources - 1]
        num_blocks = num_completed_sources - 1
    else:
        prefix = partial
        num_blocks = num_completed_sources
    if prefix.shape[0] == 0:
        return prefix
    # The current fused AttnRes kernel has 16 score slots. The published
    # 48-layer / block-size-4 model fits; keep other valid layouts correct too.
    aggregate = aggregate_stream if num_blocks < 16 else aggregate_stream_torch
    return aggregate(prefix, residual_bank, num_blocks, projection, norm)


class AliceAIAttentionResidualFinal(nn.Module):
    def __init__(self, config: AliceAIConfig, prefix: str):
        super().__init__()
        self.res_proj = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=f"{prefix}.res_proj"
        )
        self.res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, residual_bank, num_completed_sources, partial):
        return _attention_residual(
            residual_bank, num_completed_sources, partial, self.res_proj, self.res_norm
        )


class AliceAIFullAttention(nn.Module):
    """Qwen3-Next-style gated GQA, using SGLang's attention backend."""

    def __init__(self, config: AliceAIConfig, layer_id: int, quant_config, prefix: str):
        super().__init__()
        parallel = get_parallel()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // parallel.attn_tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // parallel.attn_tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=2 * config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=(rope_scaling or {}).get(
                "partial_rotary_factor", config.partial_rotary_factor
            ),
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states, positions, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        if (
            qkv.is_cuda
            and is_cuda()
            and qkv.dtype in (torch.float16, torch.bfloat16)
            and qkv.shape[0] > 0
            and positions.ndim == 1
            and positions.stride(0) == 1
            and type(self.rotary_emb) is RotaryEmbedding
            and not self.rotary_emb._force_native
            and self.rotary_emb.is_neox_style
            and 0 < self.rotary_emb.rotary_dim <= self.head_dim
            and self.rotary_emb.rotary_dim % 2 == 0
        ):
            from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
                fused_qk_gemma_rmsnorm_rope_gate,
            )

            # Preserve the BF16/FP16 norm boundary before RoPE, using SGLang's
            # native FP32 cos/sin cache and the checkpoint's per-head Q/gate layout.
            q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
        else:
            # The checkpoint interleaves query and gate within each attention head.
            q, gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim).chunk(
                2, dim=-1
            )
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_size)
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        output = output * gate.reshape(-1, self.q_size).sigmoid()
        return self.o_proj(output)[0]


class AliceAISparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: AliceAIConfig,
        layer_id: int,
        quant_config,
        prefix: str,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.alt_stream = alt_stream
        # Per-forward parameter replacement leaves TopK's cached bias on CPU.
        if get_exec().offload.cpu_offload_gb > 0:
            raise NotImplementedError(
                "AliceAI does not support --cpu-offload-gb; set it to 0."
            )
        if not get_moe_a2a_backend().is_none():
            raise NotImplementedError(
                "AliceAI requires MoE A2A backend 'none' for both target and draft"
            )
        moe_config = get_exec().moe
        # TopK returns logical IDs, so weights must keep the original expert order.
        if (
            moe_config.enable_eplb
            or moe_config.ep_num_redundant_experts
            or moe_config.init_expert_location != "trivial"
        ):
            raise NotImplementedError(
                "AliceAI requires --init-expert-location trivial without EPLB "
                "or redundant experts"
            )
        self.tp_size = get_parallel().tp_size
        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False, prefix=f"{prefix}.gate"
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=False,
            # FlashInfer names Alice's normalized sigmoid+bias routing MiniMax2.
            routing_method_type=RoutingMethodType.MiniMax2,
            routed_scaling_factor=1.0,
            # Shared and routed experts can read the input concurrently.
            inplace=config.shared_expert_intermediate_size == 0,
            prefix=f"{prefix}.experts",
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
        )
        self.shared_expert = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = LlamaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_expert",
            )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=f"{prefix}.shared_expert_gate"
        )

    def _forward_shared_expert(self, hidden_states):
        shared_output = self.shared_expert(hidden_states)
        gate = self.shared_expert_gate(hidden_states)[0].sigmoid()
        return gate * shared_output

    def _forward_router_experts(self, hidden_states):
        router_logits = self.gate(hidden_states)[0]
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward(self, hidden_states):
        shared_output = None
        if hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            output = self.experts(hidden_states, topk_output)
        elif (
            self.alt_stream is not None
            and self.shared_expert is not None
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            shared_output = self._forward_shared_expert(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                output = self._forward_router_experts(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            if self.shared_expert is not None:
                shared_output = self._forward_shared_expert(hidden_states)
            output = self._forward_router_experts(hidden_states)
        if shared_output is not None:
            output = output + shared_output
        # Routed experts use EP x MoE TP; the shared expert uses the full TP group.
        # Both leave partial outputs for this single reduction.
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class AliceAIDecoderBlock(nn.Module):
    def __init__(
        self,
        config: AliceAIConfig,
        layer_id: int,
        quant_config,
        prefix: str,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_id
        self.layer_type = config.layer_types[layer_id]
        self.attn_res_block_size = config.block_attn_res_block_size
        if self.layer_type == "linear_attention":
            self.linear_attn = KimiDeltaAttention(
                layer_idx=layer_id,
                hidden_size=config.hidden_size,
                config=config,
                rms_norm_eps=config.rms_norm_eps,
                quant_config=quant_config,
                shard_on_attn_tp=True,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = AliceAIFullAttention(
                config, layer_id, quant_config, prefix=f"{prefix}.self_attn"
            )
        self.mlp = AliceAISparseMoeBlock(
            config,
            layer_id,
            quant_config,
            prefix=f"{prefix}.mlp",
            alt_stream=alt_stream,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if layer_id > 0:
            self.attn_res_proj = ReplicatedLinear(
                config.hidden_size, 1, bias=False, prefix=f"{prefix}.attn_res_proj"
            )
            self.attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=f"{prefix}.mlp_res_proj"
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions,
        residual_bank,
        num_completed_sources,
        partial,
        forward_batch,
    ):
        if self.layer_idx > 0 and self.layer_idx % self.attn_res_block_size == 0:
            residual_bank[:, num_completed_sources].copy_(partial)
            num_completed_sources += 1
            partial = None

        hidden_states = (
            residual_bank[:, 0]
            if self.layer_idx == 0
            else _attention_residual(
                residual_bank,
                num_completed_sources,
                partial,
                self.attn_res_proj,
                self.attn_res_norm,
            )
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.input_layernorm(hidden_states)
            if self.layer_type == "linear_attention":
                attention_output = self.linear_attn(
                    hidden_states, positions, forward_batch, zero_allocator=None
                )
            else:
                attention_output = self.self_attn(
                    hidden_states, positions, forward_batch
                )
            partial = (
                attention_output if partial is None else partial + attention_output
            )
        elif partial is None:
            partial = hidden_states

        hidden_states = _attention_residual(
            residual_bank,
            num_completed_sources,
            partial,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.post_attention_layernorm(hidden_states)
        return num_completed_sources, partial + self.mlp(hidden_states)


class AliceAIModel(nn.Module):
    def __init__(self, config: AliceAIConfig, quant_config, prefix: str):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        alt_stream = get_stream("alt") if is_cuda() else None
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: AliceAIDecoderBlock(
                config, idx, quant_config, prefix, alt_stream=alt_stream
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attnres_final = AliceAIAttentionResidualFinal(
            config, f"{prefix}.attnres_final"
        )

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        num_blocks = (
            self.config.num_hidden_layers + self.config.block_attn_res_block_size - 1
        ) // self.config.block_attn_res_block_size
        residual_bank = hidden_states.new_empty(
            (hidden_states.shape[0], num_blocks, hidden_states.shape[-1])
        )
        residual_bank[:, 0].copy_(hidden_states)
        num_completed_sources, partial = 1, None
        recorder = get_global_expert_distribution_recorder()
        for i, layer in enumerate(self.layers):
            with recorder.with_current_layer(i):
                num_completed_sources, partial = layer(
                    positions,
                    residual_bank,
                    num_completed_sources,
                    partial,
                    forward_batch,
                )
        hidden_states = self.attnres_final(
            residual_bank, num_completed_sources, partial
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states)
        return hidden_states


class AliceAIForCausalLM(nn.Module):
    def __init__(
        self,
        config: AliceAIConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        parallel = get_parallel()
        if parallel.pp_size != 1:
            raise NotImplementedError(
                "AliceAI currently requires pipeline parallel size 1"
            )
        if (
            parallel.attn_tp_size != parallel.tp_size
            or parallel.moe_dp_size != 1
            or parallel.dwdp_size != 1
        ):
            raise NotImplementedError(
                "AliceAI does not yet support data/sequence parallel attention or MoE"
            )
        if (
            config.num_attention_heads % parallel.attn_tp_size
            or config.linear_num_key_heads % parallel.attn_tp_size
        ):
            raise ValueError(
                "AliceAI attention heads must be divisible by tensor parallel size"
            )
        self.config = config
        self.model = AliceAIModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=1,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], *, is_mtp: bool = False
    ) -> set[str]:
        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name.startswith("mtp.") or "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            name = name.replace(".attn_res_norm_weight", ".attn_res_norm.weight")
            name = name.replace(".mlp_res_norm_weight", ".mlp_res_norm.weight")
            name = name.replace(".res_norm_weight", ".res_norm.weight")
            if name.endswith(".linear_attn.a_log_bias"):
                name = name.replace(".a_log_bias", ".A_log")
                weight = weight.reshape(1, 1, -1, 1)

            if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
                if weight.ndim != 3 or weight.shape[0] != self.config.num_experts:
                    raise ValueError(f"Invalid AliceAI packed expert weight: {name}")
                if name.endswith(".gate_up_proj"):
                    name = name.replace(".experts.gate_up_proj", ".experts.w13_weight")
                    shards = zip(("w1", "w3"), weight.chunk(2, dim=-2))
                else:
                    name = name.replace(".experts.down_proj", ".experts.w2_weight")
                    shards = (("w2", weight),)
                param = params[name]
                for shard_id, shard in shards:
                    for expert_id in range(self.config.num_experts):
                        param.weight_loader(
                            param,
                            shard[expert_id],
                            name,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                loaded.add(name)
                continue

            if ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                        param = params[name]
                        param.weight_loader(
                            param, weight, name, expert_id=expert_id, shard_id=shard_id
                        )
                        break
                else:
                    raise ValueError(f"Unrecognized AliceAI expert weight: {name}")
            else:
                for param_name, weight_name, shard_id in stacked_mapping:
                    mapped = name.replace(weight_name, param_name)
                    if weight_name in name and mapped in params:
                        name = mapped
                        param = params[name]
                        param.weight_loader(param, weight, shard_id)
                        break
                else:
                    if name not in params:
                        raise ValueError(f"Unrecognized AliceAI weight: {name}")
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )
            loaded.add(name)
        missing = params.keys() - loaded
        if is_mtp:
            # The draft worker shares these tensors with the target model.
            missing -= {"model.embed_tokens.weight", "lm_head.weight"}
        if missing:
            raise ValueError(
                "AliceAI checkpoint is missing weights: " + ", ".join(sorted(missing))
            )
        return loaded


EntryClass = AliceAIForCausalLM
