from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.kernels.ops.attention.clover_gdn_triton import (
    clover_local_memory_triton,
    supports_clover_local_memory_triton,
)
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.elementwise.elementwise import (
    fused_gate_sigmoid_mul_add,
    fused_sigmoid_mul,
)
from sglang.kernels.ops.mamba.clover_conv_triton import (
    clover_local_memory_conv,
    supports_clover_local_memory_conv,
)
from sglang.kernels.ops.moe.clover_coreset_triton import (
    clover_block_coreset_triton,
    supports_clover_block_coreset_triton,
)
from sglang.srt.configs.clover_moe import CloverMoeConfig
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READOUT,
    Fp8Input,
    NormQuantReadout,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedParallelGroup,
    RowParallelLinear,
    resolve_linear_parallel_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend, reduce_moe_output
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    _fp8_tuple_input,
    _layer_fusions,
    _linear_accepts_fp8_tuple,
    _refreshes_rope_cos_sin,
)
from sglang.srt.runtime_context import get_parallel, linear_attn_parallel_group
from sglang.srt.utils import add_prefix, set_weight_attrs
from sglang.srt.utils.hf_transformers.common import get_rope_config


def block_coreset(logits, block_size, top_k, top_p):
    # Preserve the reference's boundary behavior (including top_p=1).
    if 0.0 < top_p < 1.0 and supports_clover_block_coreset_triton(
        logits, top_k=top_k, block_size=block_size
    ):
        return clover_block_coreset_triton(logits, block_size, top_k, top_p)
    return _block_coreset_reference(logits, block_size, top_k, top_p)


def _block_coreset_reference(logits, block_size, top_k, top_p):
    scores = logits.view(-1, block_size, logits.shape[-1]).float()
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :top_k]
    mass = scores.softmax(-1).gather(-1, ids)
    totals = torch.zeros_like(scores[:, 0])
    totals.scatter_add_(1, ids.flatten(1), mass.flatten(1))
    probs = totals / totals.sum(-1, keepdim=True).clamp_min(1e-9)
    order = probs.argsort(dim=-1, descending=True, stable=True)
    ordered = probs.gather(-1, order)
    keep = ordered.cumsum(-1) - ordered < top_p
    keep[:, 0] = True
    allowed = torch.zeros_like(keep).scatter(1, order, keep)
    return (
        logits.view_as(scores)
        .masked_fill(~allowed[:, None], -float("inf"))
        .view_as(logits)
    )


# Adapted from qwen3_5.py::_qwen3_5_stage_facts.
# Clover declare every FFN sparse without Qwen model_type detection.
def _clover_stage_facts(config, layer_id, *, attn_read=NORM_QUANT_READOUT):
    return declare_attn(read=attn_read), declare_ffn(
        sparse=True, next_layer_sparse=True
    )


class CloverSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.clover_coreset = config.moe_block_coreset_enabled
        self.clover_top_k = config.num_experts_per_tok
        self.clover_top_p = config.moe_block_coreset_top_p

    @property
    def supports_deferred_finalize(self) -> bool:
        # Clover's layer boundaries do not install a deferred-finalize consumer.
        return False

    # Reuse Qwen routing and shared-expert fusion, with coreset applied only
    # to routed experts before top-k during diffusion.
    def _forward_router_experts(
        self,
        hidden_states: torch.Tensor,
        *,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        router_logits, _ = self.gate(hidden_states)
        if (
            self.clover_coreset
            and forward_batch is not None
            and forward_batch.forward_mode.is_dllm_extend()
        ):
            router_logits = block_coreset(
                router_logits,
                hidden_states.shape[0] // forward_batch.batch_size,
                self.clover_top_k,
                self.clover_top_p,
            )
        topk_output = self.topk(hidden_states, router_logits)
        if self.enable_shared_expert_fusion:
            if not TopKOutputChecker.format_is_standard(topk_output):
                raise RuntimeError("Clover shared-expert fusion requires standard TopK")
            topk_output = self._append_shared_to_topk_output(topk_output, hidden_states)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        use_fused_gate: bool = False,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = (
            self._forward_shared_experts(
                hidden_states.clone(), apply_gate=not use_fused_gate
            )
            if self.shared_expert is not None
            else None
        )

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(
                hidden_states, forward_batch=forward_batch
            )
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        defer_finalize: bool = False,
    ) -> torch.Tensor:
        # Clover's layer boundaries do not install a deferred-finalize consumer.
        if defer_finalize:
            raise RuntimeError("Clover does not use deferred MoE finalization")
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self._use_mega_moe:
            return self._forward_mega_moe(hidden_states, forward_batch)
        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_deepep_v2()
            or get_moe_a2a_backend().is_mori()
        ):
            return self._forward_deepep(hidden_states, forward_batch)

        use_fused_gate = self.shared_expert_gate is not None
        if hidden_states.shape[0] == 0:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)
        elif (
            self.alt_stream is not None
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states,
                use_fused_gate=use_fused_gate,
                forward_batch=forward_batch,
            )
        else:
            shared_output = self._forward_shared_experts(
                hidden_states, apply_gate=not use_fused_gate
            )
            final_hidden_states = self._forward_router_experts(
                hidden_states, forward_batch=forward_batch
            )

        if shared_output is not None:
            if use_fused_gate:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states += shared_output
        if self.reduce_results:
            final_hidden_states = reduce_moe_output(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class CloverRadixLinearAttention(RadixLinearAttention):
    def forward(self, forward_batch, mixed_qkv, a, b):
        # Use conventional GDN for causal context encoding.
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().forward(forward_batch, mixed_qkv, a, b)

        backend = get_attn_backend().linear_attn_backend
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        cache_indices = backend.forward_metadata.mamba_cache_indices
        cache = get_req_to_token_pool().mamba2_layer_cache(self.layer_id)
        conv_states = cache.conv[0]
        ssm_states = cache.temporal
        valid_mask = torch.ones(seq_len, dtype=torch.bool, device=mixed_qkv.device)
        block_size = seq_len // forward_batch.batch_size
        if not supports_clover_local_memory_conv(
            mixed_qkv,
            conv_weights=self.conv_weights,
            block_size=block_size,
            activation=self.activation,
        ):
            raise RuntimeError(
                "BD-GDN production preview has no compatible convolution "
                "kernel for this device, dtype, or checkpoint shape."
            )
        mixed_qkv = clover_local_memory_conv(
            mixed_qkv,
            clean_conv_states=conv_states,
            cache_indices=cache_indices,
            conv_weights=self.conv_weights,
            bias=self.bias,
            activation=self.activation,
            block_size=block_size,
            active_valid_mask=valid_mask,
            conv_mode="independent_symmetric",
            conv_right_weight=self.conv_right_weight,
        )
        query, key, value = torch.split(
            mixed_qkv, [self.q_dim, self.k_dim, self.v_dim], dim=-1
        )
        query = query.view(1, seq_len, self.num_q_heads, self.head_q_dim)
        key = key.view(1, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(1, seq_len, self.num_v_heads, self.head_v_dim)
        if not supports_clover_local_memory_triton(
            query, key, value, block_size=block_size
        ):
            raise RuntimeError(
                "Clover currently has no compatible local-memory "
                "kernel for this device, dtype, or checkpoint shape."
            )
        g, _ = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        return clover_local_memory_triton(
            query,
            key,
            value,
            g,
            clean_ssm_states=ssm_states,
            cache_indices=cache_indices,
            active_valid_mask=valid_mask,
            scale=self.head_k_dim**-0.5,
            softmax_logit_scale=self.bd_softmax_logit_scale,
            softmax_history_bias=self.bd_softmax_history_bias,
            softmax_history_c=self.bd_softmax_history_c,
            softmax_reference_block_size=32.0,
        )


# Clover adds parameters for diffusion and replaces only the internal attention module.
class CloverGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config, layer_id, quant_config=None, alt_stream=None, prefix=""):
        super().__init__(config, layer_id, quant_config, alt_stream, prefix)
        causal_attn = self.attn
        self.attn = CloverRadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=causal_attn.num_q_heads,
            num_k_heads=causal_attn.num_k_heads,
            num_v_heads=causal_attn.num_v_heads,
            head_q_dim=causal_attn.head_q_dim,
            head_k_dim=causal_attn.head_k_dim,
            head_v_dim=causal_attn.head_v_dim,
            conv_weights=causal_attn.conv_weights,
            bias=causal_attn.bias,
            activation=causal_attn.activation,
            A_log=causal_attn.A_log,
            dt_bias=causal_attn.dt_bias,
        )
        parallel_group = linear_attn_parallel_group()
        self.conv_right_weight = nn.Parameter(
            torch.empty(
                self.conv_dim // self.attn_tp_size, 1, self.conv_kernel_size - 1
            )
        )
        self._override_weight_loader(
            self.conv_right_weight, self.conv1d.weight.weight_loader
        )
        for name, value in (
            ("bd_softmax_logit_scale", 10.0),
            ("bd_softmax_history_bias", 5.0),
            ("bd_softmax_history_c", 1.0),
        ):
            param = nn.Parameter(
                torch.full((self.num_v_heads // self.attn_tp_size,), value)
            )
            set_weight_attrs(
                param,
                {
                    "weight_loader": sharded_weight_loader(
                        0, parallel_group=parallel_group
                    )
                },
            )
            setattr(self, name, param)
            # The parent owns these parameters, like conv_weights above.
            object.__setattr__(self.attn, name, param)
        object.__setattr__(self.attn, "conv_right_weight", self.conv_right_weight)


class CloverLinearDecoderLayer(Qwen3_5LinearDecoderLayer):
    stage_facts = staticmethod(_clover_stage_facts)

    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,  # Accepted for the inherited backbone constructor.
        build_stages=True,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.layer_id = layer_id
        self.linear_attn = CloverGatedDeltaNet(
            config, layer_id, quant_config, alt_stream, prefix
        )
        self.mlp = CloverSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            # Follow Qwen3.5: let the parent gate fusion without CUDA opt-in.
            support_shared_expert_fusion=True,
            reduce_results=False,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Qwen's optional CuTe DSL boundary fusion is currently not supported for Clover.
        if build_stages:
            attn, ffn = self.stage_facts(config, layer_id)
            self.attn_boundary, self.ffn_boundary = append_stages(
                (attn, self.input_layernorm, {"fusions": None}),
                (ffn, self.post_attention_layernorm, {"fusions": None}),
            )


class CloverAttentionDecoderLayer(Qwen3_5AttentionDecoderLayer):
    stage_facts = staticmethod(_clover_stage_facts)

    def __init__(
        self,
        config: CloverMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,  # Accepted for the inherited backbone constructor.
        build_stages: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.refreshes_rope_cos_sin = _refreshes_rope_cos_sin(config, layer_id, False)
        dcp_size = get_parallel().attn_dcp_size
        kv_parallel_group = ReplicatedParallelGroup("attn_tp", dcp_size)
        self.kv_tp_rank, self.kv_tp_size = resolve_linear_parallel_group(
            kv_parallel_group
        )
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.kv_tp_size:
            assert self.total_num_kv_heads % self.kv_tp_size == 0
        else:
            assert self.kv_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.kv_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        # If rope_scaling doesn't specify a scaling type, treat as no scaling
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        # Q stays sharded across attention TP ranks; K/V are replicated within
        # each DCP group.
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            kv_parallel_group=kv_parallel_group,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            parallel_group="attn_tp",
            prefix=add_prefix("o_proj", prefix),
        )

        # Clover change: causal encoding and noncausal preview share projections.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
        )

        self.attn_preview = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
        )

        self.mlp = CloverSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            support_shared_expert_fusion=True,
            reduce_results=False,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Standard attention layers benefit from a fused quant epilogue only
        # when qkv_proj can consume the returned quantized tuple.
        accepts_fp8_input = _linear_accepts_fp8_tuple(self.qkv_proj)
        boundary_fusions = _layer_fusions(config, False)
        # A subclass that brings its own residual builds the stages itself.
        if build_stages:
            attn, ffn = self.stage_facts(
                config,
                layer_id,
                attn_read=NormQuantReadout(
                    fp8_input=Fp8Input.TUPLE if accepts_fp8_input else None
                ),
            )
            self.attn_boundary, self.ffn_boundary = append_stages(
                (attn, self.input_layernorm, {"fusions": boundary_fusions}),
                (ffn, self.post_attention_layernorm, {"fusions": boundary_fusions}),
            )

        self.alt_stream = alt_stream

    # Adapted from Qwen3_5AttentionDecoderLayer.self_attention in qwen3_5.py.
    # Clover selects attn_preview for diffusion; QKV preparation and output
    # gate/projection behavior follow Qwen, including its FP8 input handling.
    def self_attention(self, positions, hidden_states, forward_batch):
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().self_attention(positions, hidden_states, forward_batch)

        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        attn_output = self.attn_preview(q, k, v, forward_batch)
        # Output gate and projection input handling.
        if self.attn_output_gate:
            if _fp8_tuple_input(self.o_proj, attn_output.shape[0]):
                attn_output = fused_sigmoid_mul(attn_output, gate, quant=True)
            else:
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
        output, _ = self.o_proj(attn_output)
        return output


class CloverMoeModel(Qwen3_5MoeForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep loader metadata consistent with the constructed MoE layers.
        self.num_fused_shared_experts = self._get_num_fused_shared_experts()
        self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

    decoder_layer_types = {
        "linear_attention": CloverLinearDecoderLayer,
        "attention": CloverAttentionDecoderLayer,
    }

    def get_hidden_dim(self, module_name, layer_idx):
        if module_name == "gate_up_proj":
            return (
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size * 2,
            )
        if module_name == "down_proj":
            return self.config.shared_expert_intermediate_size, self.config.hidden_size
        return super().get_hidden_dim(module_name, layer_idx)


class CloverMoeModelLM(nn.Module):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
    supported_lora_modules = Qwen3_5MoeForConditionalGeneration.supported_lora_modules

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_config = config
        self.config = text_config
        self.pp_group = get_parallel().pp_group
        if not (self.pp_group.is_first_rank and self.pp_group.is_last_rank):
            raise ValueError("Clover requires pipeline parallel size 1.")

        self.model = CloverMoeModel(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=text_config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(text_config, return_full_logits=True)
        self.num_fused_shared_experts = self.model.num_fused_shared_experts
        self.enable_shared_expert_fusion = self.model.enable_shared_expert_fusion

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        result = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        return result

    # Clover-owned loader. Projection mappings and expert shard conventions are
    # adapted from qwen3_5.py::Qwen3_5MoeForCausalLM.load_weights.
    # Clover changes: explicit supported checkpoint paths
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_mapping = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        )
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts + self.num_fused_shared_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        loaded_shards = {}

        def load_expert(name, weight, expert_id, shard_id):
            param = params[name]
            param.weight_loader(
                param, weight, name, shard_id=shard_id, expert_id=expert_id
            )
            loaded.add(name)
            loaded_shards.setdefault(name, set()).add((expert_id, shard_id))

        for source_name, weight in weights:
            # Rotary frequencies are derived from the configuration.
            if source_name.endswith(".rotary_emb.inv_freq"):
                continue
            name = source_name.replace("model.language_model.", "model.")
            name = name.replace(".self_attn.", ".")
            if self.enable_shared_expert_fusion and "mlp.shared_expert." in name:
                # Match Qwen: append the shared expert after routed experts.
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{self.config.num_experts}.",
                )
            if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                if weight.shape[0] != self.config.num_experts:
                    raise ValueError(
                        f"Wrong expert count in Clover weight {source_name}"
                    )
                if name.endswith(".gate_up_proj"):
                    name = name.replace("experts.gate_up_proj", "experts.w13_weight")
                    gate, up = weight.chunk(2, dim=-2)
                    shards = (("w1", gate), ("w3", up))
                else:
                    name = name.replace("experts.down_proj", "experts.w2_weight")
                    shards = (("w2", weight),)
                for shard_id, shard in shards:
                    for expert_id in range(self.config.num_experts):
                        load_expert(name, shard[expert_id], expert_id, shard_id)
                continue

            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                        load_expert(name, weight, expert_id, shard_id)
                        break
                else:
                    raise ValueError(f"Unexpected Clover expert weight: {source_name}")
                continue

            for param_name, weight_name, shard_id in stacked_mapping:
                if f".{weight_name}." in name:
                    name = name.replace(f".{weight_name}.", f".{param_name}.")
                    param = params[name]
                    param.weight_loader(param, weight, shard_id)
                    loaded.add(name)
                    shards = shard_id if isinstance(shard_id, tuple) else (shard_id,)
                    loaded_shards.setdefault(name, set()).update(shards)
                    break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected Clover weight: {source_name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)

        missing = sorted(params.keys() - loaded)
        for name, shards in loaded_shards.items():
            if name.endswith(".experts.w13_weight"):
                expected = {
                    (i, s)
                    for i in range(
                        self.config.num_experts + self.num_fused_shared_experts
                    )
                    for s in ("w1", "w3")
                }
            elif name.endswith(".experts.w2_weight"):
                expected = {
                    (i, "w2")
                    for i in range(
                        self.config.num_experts + self.num_fused_shared_experts
                    )
                }
            else:
                projection = name.rsplit(".", 2)[-2]
                expected = {
                    "qkv_proj": {"q", "k", "v"},
                    "gate_up_proj": {0, 1},
                    "in_proj_qkvz": {0, 1, 2, 3},
                    "in_proj_ba": {0, 1},
                }[projection]
            if shards != expected:
                missing.append(f"{name} (missing shards: {expected - shards})")
        if missing:
            raise ValueError("Missing Clover weights: " + ", ".join(missing))

        for module in self.modules():
            if isinstance(module, CloverGatedDeltaNet):
                module.finalize_fused_in_proj()
        return loaded

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # Clover requires PP=1, so both modules are always local.
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return Qwen3_5MoeForConditionalGeneration.get_model_config_for_expert_location(
            config
        )


EntryClass = CloverMoeModelLM
