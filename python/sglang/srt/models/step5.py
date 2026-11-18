# SPDX-License-Identifier: Apache-2.0
"""Step5 text backbone and the Step5 VL checkpoint wrapper."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from sglang.kernels.ops.attention.step5.sparse_attention import (
    step5_head_gate,
    step5_qknorm_rope,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.step5 import Step5Config, parse_step5_attention_profile
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.step5.indexer import Step5Indexer
from sglang.srt.layers.attention.step5.norm import Step5RMSNorm
from sglang.srt.layers.attention.step5_backend import Step5RadixAttention
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_model, get_parallel


class Step5MLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            reduce_results=False,
            prefix=add_prefix("down_proj", prefix),
        )
        self.limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.float().chunk(2, dim=-1)
        activation = F.silu(gate).clamp(max=self.limit) * up.clamp(
            min=-self.limit, max=self.limit
        )
        output, _ = self.down_proj(activation.to(hidden_states.dtype))
        return output


class Step5MoE(nn.Module):
    def __init__(
        self,
        *,
        config: Step5Config,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.moe_num_experts,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.router_bias = nn.Parameter(
            torch.zeros(config.moe_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.topk = TopK(
            top_k=config.moe_top_k,
            renormalize=True,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            correction_bias=self.router_bias,
            apply_routed_scaling_factor_on_output=False,
            output_format=TopKOutputFormat.STANDARD,
            layer_id=layer_id,
        )
        keep_bf16 = quant_config is None or is_layer_skipped(
            f"model.layers.{layer_id}.moe.experts.w13_weight",
            quant_config.ignored_layers,
        )
        self.experts = FusedMoE(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            inplace=False,
            quant_config=quant_config,
            quant_method=UnquantizedFusedMoEMethod() if keep_bf16 else None,
            gemm1_clamp_limit=config.swiglu_limits[layer_id],
            gate_up_interleaved=False,
            prefix=add_prefix("experts", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = torch.mm(
            hidden_states, self.gate.weight.t(), out_dtype=torch.float32
        )
        topk_output = self.topk(hidden_states, router_logits)
        topk_output = StandardTopKOutput(
            topk_weights=topk_output.topk_weights * self.routed_scaling_factor,
            topk_ids=topk_output.topk_ids,
            router_logits=topk_output.router_logits,
        )
        return self.experts(hidden_states, topk_output)


class Step5Attention(nn.Module):
    def __init__(
        self,
        *,
        config: Step5Config,
        layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        profile = parse_step5_attention_profile(config)
        parallel = get_parallel()
        self.is_dsa = config.layer_types[layer_id] == "full_attention"
        self.head_dim = profile.head_dim
        self.num_heads = profile.num_heads // parallel.attn_tp_size
        self.eps = config.rms_norm_eps
        self.rotary_emb = get_rope(
            head_size=profile.head_dim,
            rotary_dim=profile.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta[layer_id],
            is_neox_style=True,
            rope_scaling=config.rope_scaling if self.is_dsa else None,
            dtype=torch.bfloat16,
            partial_rotary_factor=config.partial_rotary_factors[layer_id],
        )
        self.qkv_proj = QKVParallelLinear(
            profile.hidden_size,
            profile.head_dim,
            profile.num_heads,
            profile.num_kv_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.g_proj = ColumnParallelLinear(
            profile.hidden_size,
            profile.num_heads,
            bias=False,
            gather_output=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("g_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            profile.num_heads * profile.head_dim,
            profile.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = Step5RMSNorm(
            profile.head_dim, eps=self.eps, weight_dtype=torch.float32
        )
        self.k_norm = Step5RMSNorm(
            profile.head_dim, eps=self.eps, weight_dtype=torch.float32
        )
        self.attn = Step5RadixAttention(
            num_heads=self.num_heads,
            head_dim=profile.head_dim,
            scaling=profile.head_dim**-0.5,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=-1 if self.is_dsa else config.sliding_window - 1,
        )
        if self.is_dsa:
            self.indexer = Step5Indexer(
                layer_id=layer_id,
                hidden_size=profile.hidden_size,
                index_head_dim=profile.index_head_dim,
                num_index_heads=profile.index_num_heads,
                index_tp_size=profile.provider_groups,
                tp_rank=parallel.attn_tp_rank,
                tp_size=parallel.attn_tp_size,
                rotary_dim=profile.index_rope_dim,
                rope_theta=config.rope_theta[layer_id],
                rope_scaling=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
                eps=self.eps,
                prefix=prefix,
            )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        gate, _ = self.g_proj(hidden_states)
        q, k, v = step5_qknorm_rope(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            positions,
            self.rotary_emb.cos_sin_cache_as(torch.bfloat16),
            eps=self.eps,
        )
        if is_in_breakable_cuda_graph():
            output = q.new_empty((q.shape[0], self.num_heads * self.head_dim))
            _breakable_step5_attention_core(
                q, k, v, hidden_states, positions, output, self
            )
        elif self.is_dsa:
            output = q.new_empty((q.shape[0], self.num_heads * self.head_dim))
            _step5_attention_core(
                q, k, v, hidden_states, positions, output, self, forward_batch
            )
        else:
            output = self.attn(q, k, v, forward_batch)
        output = step5_head_gate(
            output.reshape(-1, self.num_heads * self.head_dim),
            gate,
            head_dim=self.head_dim,
        )
        output, _ = self.o_proj(output)
        return output


def _step5_attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    owner: Step5Attention,
    forward_batch: ForwardBatch | None = None,
) -> None:
    if forward_batch is None:
        forward_batch = get_forward_batch()
    backend = get_attn_backend()
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    kwargs = {}
    if owner.is_dsa:
        metadata = backend.get_indexer_metadata(owner.attn.layer_id, forward_batch)
        if metadata is None:
            output.zero_()
            return
        num_tokens = metadata.num_input_tokens
        kwargs["index_selection"] = owner.indexer(
            hidden_states[:num_tokens], positions[:num_tokens], metadata
        )
    batch_view = copy.copy(forward_batch)
    batch_view.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]
    result = backend.forward(
        q[:num_tokens],
        k[:num_tokens],
        v[:num_tokens],
        owner.attn,
        batch_view,
        save_kv_cache=True,
        **kwargs,
    )
    output[:num_tokens].copy_(result.reshape(num_tokens, -1))
    output[num_tokens:].zero_()


_breakable_step5_attention_core = eager_on_graph(_step5_attention_core)


class Step5DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config: Step5Config,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Step5Attention(
            config=config, layer_id=layer_id, prefix=add_prefix("self_attn", prefix)
        )
        self.use_moe = layer_id in config.moe_layer_list
        limit_shared = config.swiglu_limits_shared[layer_id]
        if self.use_moe:
            self.moe = Step5MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("moe", prefix),
            )
            self.share_expert = Step5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.share_expert_dim,
                swiglu_limit=limit_shared,
                prefix=add_prefix("share_expert", prefix),
            )
        else:
            self.mlp = Step5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                swiglu_limit=limit_shared,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = Step5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.bfloat16
        )
        self.post_attention_layernorm = Step5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.bfloat16
        )
        attn, ffn = self.stage_facts(config, layer_id)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: Step5Config, layer_id: int):
        return (
            declare_attn(),
            declare_ffn(
                sparse=layer_id in config.moe_layer_list,
                next_layer_sparse=layer_id + 1 in config.moe_layer_list,
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if self.use_moe:
            shared_output = self.share_expert(hidden_states)
            hidden_states = self.moe(hidden_states) + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class Step5Model(nn.Module):
    def __init__(
        self,
        config: Step5Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Step5DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=partial(Step5DecoderLayer.stage_facts, config),
        )
        self.norm = Step5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states, residual = residual_batch.stream_of(forward_batch).export(
            hidden_states
        )
        forward_batch.residual_stream = None
        return self.norm(hidden_states + residual, backend=KernelBackend.TORCH)


def _duplicated_owners(model_path: str | None) -> dict[str, str]:
    index_path = Path(model_path or "") / "model.safetensors.index.json"
    if not model_path or not index_path.is_file():
        return {}
    owners = json.loads(index_path.read_text())["weight_map"]
    seen, duplicated = set(), set()
    for filename in sorted(set(owners.values())):
        with safe_open(Path(model_path) / filename, framework="pt") as shard:
            names = set(shard.keys())
        duplicated |= names & seen
        seen |= names
    return {name: owners[name] for name in duplicated}


def _deduplicated_weights(
    weights: Iterable[tuple[str, torch.Tensor]], model_path: str | None
) -> Iterator[tuple[str, torch.Tensor]]:
    owners = _duplicated_owners(model_path)
    emitted = set()
    for name, weight in weights:
        if name not in owners:
            yield name, weight
            continue
        if name in emitted:
            continue
        emitted.add(name)
        with safe_open(
            Path(model_path) / owners[name], framework="pt", device="cpu"
        ) as shard:
            yield name, shard.get_tensor(name)


class Step5ForCausalLM(nn.Module):
    _stacked_params = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )
    _expert_params = (
        ("w13_weight", "gate_proj.weight", "w1"),
        ("w13_weight", "up_proj.weight", "w3"),
        ("w2_weight", "down_proj.weight", "w2"),
        ("w13_weight_scale_inv", "gate_proj.weight_scale_inv", "w1"),
        ("w13_weight_scale_inv", "up_proj.weight_scale_inv", "w3"),
        ("w2_weight_scale_inv", "down_proj.weight_scale_inv", "w2"),
    )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.moe_num_experts,
        )

    def __init__(
        self,
        config: Step5Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Step5Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _target_name(self, name: str) -> tuple[str, object]:
        if ".self_attn.sparse_indexer_" in name:
            return name.replace(".self_attn.", ".self_attn.indexer."), None
        if ".moe." not in name:
            for param, source, shard in self._stacked_params:
                if f".{source}." in name:
                    return name.replace(f".{source}.", f".{param}."), shard
        return name, None

    def _load_experts(self, params, name: str, weight: torch.Tensor) -> str | None:
        head, marker, tail = name.rpartition(".moe.")
        if not marker:
            return None
        for param_name, source, shard in self._expert_params:
            if tail != source:
                continue
            target = f"{head}.moe.experts.{param_name}"
            param = params[target]
            for expert_id in range(self.config.moe_num_experts):
                param.weight_loader(
                    param,
                    weight[expert_id],
                    target,
                    shard_id=shard,
                    expert_id=expert_id,
                )
            return target
        return None

    def _is_skipped(self, name: str) -> bool:
        if name.startswith(("vision_model.", "vit_large_projector.")):
            return True
        if name.endswith(".self_attn.ssmax_s"):
            return True
        if not name.startswith("model.layers."):
            return False
        return int(name.split(".")[2]) >= self.config.num_hidden_layers

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self._load_language_weights(
            _deduplicated_weights(weights, get_model().model_path)
        )

    def _load_language_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if self._is_skipped(name):
                continue
            expert_target = self._load_experts(params, name, weight)
            if expert_target is not None:
                loaded.add(expert_target)
                continue
            target, shard = self._target_name(name)
            if target not in params:
                raise ValueError(f"Unknown Step5 weight: {name}")
            param = params[target]
            if shard is None:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
            else:
                param.weight_loader(param, weight, shard)
            loaded.add(target)
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Step5 weights not loaded: {sorted(missing)[:16]}")
        return loaded


class Step5ForConditionalGeneration(Step5ForCausalLM):
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return super().get_model_config_for_expert_location(config.text_config)

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        if not config.language_model_only:
            raise ValueError("Step5 does not support image inputs yet")
        super().__init__(config.text_config, quant_config=quant_config, prefix=prefix)


EntryClass = [Step5ForCausalLM, Step5ForConditionalGeneration]
