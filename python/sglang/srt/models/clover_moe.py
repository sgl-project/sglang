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
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    ReplicatedParallelGroup,
    RowParallelLinear,
    resolve_linear_parallel_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
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
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    _fp8_tuple_input,
)
from sglang.srt.runtime_context import get_parallel, linear_attn_parallel_group
from sglang.srt.utils import (
    add_prefix,
    get_bool_env_var,
    is_cuda,
    is_hip,
    set_weight_attrs,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if _is_cuda:
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )


def block_coreset(logits, block_size, top_k, top_p):
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
    allowed = torch.zeros_like(keep).scatter(1, order, keep)
    return (
        logits.view_as(scores)
        .masked_fill(~allowed[:, None], -float("inf"))
        .view_as(logits)
    )


# Adapted from qwen3_5.py::_qwen3_5_stage_facts.
def _clover_stage_facts(config, layer_id, *, attn_read=NORM_QUANT_READOUT):
    return declare_attn(read=attn_read), declare_ffn(
        sparse=True, next_layer_sparse=True
    )


class CloverRouterGate(ReplicatedLinear):
    """Router gate that applies the block coreset while a preview forward runs."""

    def __init__(self, *args, top_k, top_p, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.top_p = top_p
        # Set by CloverSparseMoeBlock.forward for the duration of one preview call.
        self.preview_block_size = None

    def forward(self, x):
        logits, bias = super().forward(x)
        if self.preview_block_size is not None:
            logits = block_coreset(
                logits, self.preview_block_size, self.top_k, self.top_p
            )
        return logits, bias


class CloverSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    """Qwen2-MoE block with the coreset applied through the router gate, so every
    routing path that reads self.gate (normal, dual stream, DeepEP, MegaMoE) sees it."""

    def __init__(self, *, layer_id, config, quant_config=None, prefix="", **kwargs):
        coreset = config.moe_block_coreset_enabled
        super().__init__(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            # The AITER fused router computes top-k without self.gate and would
            # skip the coreset.
            support_shared_expert_fusion=(
                not is_shared_experts_fusion_disabled() and not (_use_aiter and coreset)
            ),
            **kwargs,
        )
        self.clover_coreset = coreset
        if coreset:
            self.gate = CloverRouterGate(
                config.hidden_size,
                config.num_experts,
                bias=False,
                quant_config=None,
                prefix=add_prefix("gate", prefix),
                top_k=config.num_experts_per_tok,
                top_p=config.moe_block_coreset_top_p,
            )

    @property
    def supports_deferred_finalize(self) -> bool:
        # Clover's layer boundaries do not install a deferred-finalize consumer.
        return False

    def forward(self, hidden_states, forward_batch=None, defer_finalize=False):
        # The coreset restricts routed experts only, before top-k, and only in preview.
        if not (
            self.clover_coreset
            and forward_batch is not None
            and forward_batch.forward_mode.is_dllm_extend()
        ):
            return super().forward(hidden_states, forward_batch, defer_finalize)
        self.gate.preview_block_size = (
            hidden_states.shape[0] // forward_batch.batch_size
        )
        try:
            return super().forward(hidden_states, forward_batch, defer_finalize)
        finally:
            self.gate.preview_block_size = None


class CloverRadixLinearAttention(RadixLinearAttention):
    def forward(self, forward_batch, mixed_qkv, a, b):
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


def _build_moe(config, layer_id, quant_config, prefix, alt_stream):
    return CloverSparseMoeBlock(
        layer_id=layer_id,
        config=config,
        quant_config=quant_config,
        # Follow Qwen3.5: HIP overlaps the shared expert only without fusion.
        alt_stream=(
            alt_stream if (_is_cuda or is_shared_experts_fusion_disabled()) else None
        ),
        prefix=add_prefix("mlp", prefix),
        # The stage boundary completes this stage's sum.
        reduce_results=False,
    )


def _build_stages(config, layer_id, input_layernorm, post_attention_layernorm):
    attn, ffn = _clover_stage_facts(config, layer_id)
    # Qwen's optional CuTe DSL boundary fusion is not supported for Clover.
    return append_stages(
        (attn, input_layernorm, {"fusions": None}),
        (ffn, post_attention_layernorm, {"fusions": None}),
    )


class CloverLinearDecoderLayer(nn.Module):
    """GatedDeltaNet mixer plus sparse MoE, built from shared Qwen3.5 components."""

    stage_facts = staticmethod(_clover_stage_facts)

    def __init__(
        self,
        config: CloverMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.linear_attn = CloverGatedDeltaNet(
            config, layer_id, quant_config, alt_stream, prefix
        )
        self.mlp = _build_moe(
            config,
            layer_id,
            quant_config,
            prefix.replace(".linear_attn", ""),
            alt_stream,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_boundary, self.ffn_boundary = _build_stages(
            config, layer_id, self.input_layernorm, self.post_attention_layernorm
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs=None,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=captured_last_layer_outputs
        )
        if not forward_batch.forward_mode.is_idle() and hidden_states.shape[0] > 0:
            hidden_states = self.linear_attn(hidden_states, forward_batch)
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class CloverAttentionDecoderLayer(nn.Module):
    """Gated full attention plus sparse MoE. Causal encoding and the noncausal
    preview share projections; the layout matches Qwen3.5 checkpoints."""

    stage_facts = staticmethod(_clover_stage_facts)

    def __init__(
        self,
        config: CloverMoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        attn_tp_size = get_parallel().attn_tp_size
        kv_parallel_group = ReplicatedParallelGroup(
            "attn_tp", get_parallel().attn_dcp_size
        )
        _, kv_tp_size = resolve_linear_parallel_group(kv_parallel_group)
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        assert total_num_heads % attn_tp_size == 0
        self.num_heads = total_num_heads // attn_tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // kv_tp_size)
        self.head_dim = config.head_dim or (config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = config.attn_output_gate

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            total_num_heads * (1 + self.attn_output_gate),
            total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            kv_parallel_group=kv_parallel_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            parallel_group="attn_tp",
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        attn_kwargs = dict(
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            attn_type=AttentionType.DECODER,
            **attn_kwargs,
        )
        self.attn_preview = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            attn_type=AttentionType.ENCODER_ONLY,
            **attn_kwargs,
        )

        self.mlp = _build_moe(
            config,
            layer_id,
            quant_config,
            prefix.replace(".self_attn", ""),
            alt_stream,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_boundary, self.ffn_boundary = _build_stages(
            config, layer_id, self.input_layernorm, self.post_attention_layernorm
        )

    def _prepare_qkv_gate(self, *, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q_width = self.q_size * (1 + self.attn_output_gate)
        q_gate, k, v = qkv.split([q_width, self.kv_size, self.kv_size], dim=-1)
        if _is_cuda:
            q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
                mrope_axis_map=(
                    self.rotary_emb.axis_map if positions.dim() == 2 else None
                ),
            )
            seq_len = hidden_states.shape[0]
            q, k = q.view(seq_len, -1), k.view(seq_len, -1)
            gate = gate.view(seq_len, -1) if gate is not None else None
            return q, k, v, gate

        gate = None
        if self.attn_output_gate:
            q_gate = q_gate.view(*q_gate.shape[:-1], self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*q.shape[:-2], -1)
        else:
            q = q_gate
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def self_attention(self, positions, hidden_states, forward_batch):
        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions, hidden_states=hidden_states
        )
        attn = (
            self.attn_preview
            if forward_batch.forward_mode.is_dllm_extend()
            else self.attn
        )
        attn_output = attn(q, k, v, forward_batch)
        if self.attn_output_gate:
            if _fp8_tuple_input(self.o_proj, attn_output.shape[0]):
                attn_output = fused_sigmoid_mul(attn_output, gate, quant=True)
            else:
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs=None,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=captured_last_layer_outputs
        )
        if not forward_batch.forward_mode.is_idle() and hidden_states.shape[0] > 0:
            hidden_states = self.self_attention(positions, hidden_states, forward_batch)
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


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


class CloverMoeModelLM(nn.Module):
    fall_back_to_pt_during_load = False

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

    # Adapted from qwen3_5.py::Qwen3_5MoeForCausalLM.load_weights.
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
