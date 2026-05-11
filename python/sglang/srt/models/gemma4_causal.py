# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import (
    Gemma4TextConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.gemma4_fused_ops import (
    gemma_dual_rmsnorm_residual_scalar,
    gemma_qkv_rmsnorm,
    gemma_rmsnorm_residual_scalar,
)
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3MLP, Gemma3TextScaledWordEmbedding
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


Gemma4MLP = Gemma3MLP
Gemma4TextScaledWordEmbedding = Gemma3TextScaledWordEmbedding


class Gemma4Router(nn.Module):
    """Router for Gemma4 MoE that preprocesses input before projection.

    Applies RMSNorm (no learned weight), root_size scaling
    (hidden_size^{-0.5}), then a learned per-dimension scale before
    projecting to expert logits.

    This preprocessing is applied ONLY to the router's input, not to
    the expert MLPs' input.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # RMSNorm without learned weight — pure normalization only
        self.norm = Gemma4RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, with_scale=False
        )
        # Per-dimension learned scale, applied after norm + root_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        # Constant 1/sqrt(hidden_size) scaling factor
        self.register_buffer(
            "root_size",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        # Project to expert logits; replicated across TP for consistent routing
        self.proj = ReplicatedLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("proj", prefix),
        )
        self._fused_scale: Optional[torch.Tensor] = None

    def fuse_scale(self):
        """Pre-compute scale * root_size. Call after weights are loaded."""
        self._fused_scale = (self.scale * self.root_size).to(self.scale.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw router logits [T, E]."""
        x = self.norm(x)
        if self._fused_scale is None:
            self.fuse_scale()
        x = x * self._fused_scale.to(x.dtype)
        router_logits, _ = self.proj(x)
        return router_logits


class Gemma4MoE(nn.Module):
    """Mixture of Experts for Gemma4.

    Wraps MoE implementation with custom routing. The router projection is
    external (Gemma4Router) — this class only handles expert dispatch.

    Gemma4 routing: softmax over ALL experts → top-k → renormalize.
    per_expert_scale is folded into routing weights for mathematical
    correctness with MoE's fused kernel.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_id: int,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.tp_size = get_tensor_model_parallel_world_size()

        # Per-expert output scale folded into routing weights so that
        # MoE's fused kernel computes: Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        # Capture param directly to avoid closing over self in the routing closure.
        per_expert_scale = self.per_expert_scale

        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,  # always True for Gemma4; softmax identity only holds when renormalizing
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # softmax(all)[topk] / sum(softmax(all)[topk]) = softmax(topk_logits),
            # so we softmax only the top-k logits (fewer kernel launches).
            topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
            topk_weights = torch.nn.functional.softmax(topk_logits, dim=-1)

            # Fold per_expert_scale into routing weights
            topk_weights = topk_weights * per_expert_scale[topk_ids].to(
                topk_weights.dtype
            )

            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        self.topk = TopK(
            top_k=config.top_k_experts,
            layer_id=layer_id,
            custom_routing_function=routing_function,
        )

        experts_type = get_moe_impl_class(quant_config)

        self.experts = experts_type(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            top_k=config.top_k_experts,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="gelu",
            reduce_results=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        topk_output = self.topk(hidden_states, router_logits)
        hidden_states = self.experts(hidden_states, topk_output)
        return hidden_states.view(num_tokens, hidden_dim)


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma4TextConfig,
        head_dim: int,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        layer_type = config.layer_types[layer_id]
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if layer_type == "sliding_attention":
            self.total_num_kv_heads = getattr(
                config, "swa_num_key_value_heads", config.num_key_value_heads
            )
        else:
            self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False
        )

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(
                rope_type="default",
                rope_theta=10000.0,
            )

        # KV sharing logic
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
        self.is_kv_shared_layer = (
            layer_id >= first_kv_shared_layer_idx and num_kv_shared_layers > 0
        )

        self.kv_shared_layer_index = None
        if num_kv_shared_layers > 0 and self.layer_id >= first_kv_shared_layer_idx:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            current_layer_type = config.layer_types[self.layer_id]
            if current_layer_type not in prev_layers:
                raise ValueError(
                    f"KV sharing layer {self.layer_id} has type '{current_layer_type}' "
                    f"but no matching type found in layers 0..{first_kv_shared_layer_idx - 1}. "
                    f"Available types: {set(prev_layers)}"
                )
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},
            partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            1,  # scaling factor
            num_kv_heads=self.num_kv_heads,
            layer_id=(
                self.kv_shared_layer_index if self.is_kv_shared_layer else self.layer_id
            ),
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Fused Q/K/V RMSNorm: replaces three separate norm kernels with one.
        # Preconditions for the fused path: tensors on CUDA, q_norm/k_norm use
        # the standard norm*weight (scale_shift==0) and v_norm has weight=ones
        # (with_scale=False) — the canonical Gemma4 attention configuration.
        is_kv_shared = (
            self.is_kv_shared_layer and self.kv_shared_layer_index is not None
        )
        can_fuse_qkv_norm = (
            q.is_cuda
            and self.q_norm.scale_shift == 0.0
            and self.k_norm.scale_shift == 0.0
            and not self.v_norm.with_scale
        )
        if can_fuse_qkv_norm:
            if is_kv_shared:
                gemma_qkv_rmsnorm(
                    q,
                    None,
                    None,
                    self.q_norm.weight.data,
                    None,
                    num_q_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    eps=self.q_norm.eps,
                )
                k = None
                v = None
            else:
                gemma_qkv_rmsnorm(
                    q,
                    k,
                    v,
                    self.q_norm.weight.data,
                    self.k_norm.weight.data,
                    num_q_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    eps=self.q_norm.eps,
                )
                # Match the original norm path's output shapes: q stays 2D,
                # k/v become 3D so the subsequent `.flatten(-2, -1)` works.
                # Use reshape (not view) since k/v are strided slice views of
                # the qkv buffer and may not satisfy view's contiguity rules.
                k = k.reshape(-1, self.num_kv_heads, self.head_dim)
                v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        else:
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            q = self.q_norm(q)
            q = q.flatten(-2, -1)
            if is_kv_shared:
                k = None
                v = None
            else:
                k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
                k = self.k_norm(k)
                v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
                v = self.v_norm(v)

        # Apply rotary embedding
        if k is not None:
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        else:
            # Rotary embedding requires a key input; use zeros since KV is shared from another layer
            dummy_k = torch.zeros_like(q[:, : self.kv_size])
            q, _ = self.rotary_emb(positions, q, dummy_k)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=not self.is_kv_shared_layer,
        )
        if attn_output.dim() == 3:
            attn_output = attn_output.flatten(-2, -1)
        output, _ = self.o_proj(attn_output)

        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = (
            getattr(config, "hidden_size_per_layer_input", None) or 0
        )

        self.layer_id = layer_id

        # Gemma 4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_id]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = config.head_dim  # following sglang naming
        else:
            head_dim = getattr(config, "swa_head_dim", config.head_dim)

        self.self_attn = Gemma4Attention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=head_dim,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = self.layer_id >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        layer_intermediate_size = config.intermediate_size * (
            2 if use_double_wide_mlp else 1
        )

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-Layer Embedding (PLE) components — present in each decoder layer
        if self.hidden_size_per_layer_input > 0:
            # Gate: projects hidden_states → per-layer dim for gating
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_input_gate", prefix),
            )
            # Projection: projects gated per-layer input back → hidden size
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_projection", prefix),
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Parallel MoE
        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=add_prefix("router", prefix),
            )
            self.moe = Gemma4MoE(
                hidden_size=self.hidden_size,
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("moe", prefix),
            )

            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.router = None
            self.moe = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.has_ple = self.hidden_size_per_layer_input > 0
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # Gemma4 residual pattern following JAX implementation:
        # 1. input_norm(x) -> attn -> post_attn_norm -> ADD residual
        # 2. pre_ff_norm -> mlp -> post_ff_norm -> ADD residual
        #
        # Optimization: fuse "post_attn_norm(h) + residual; pre_ff_norm(...)"
        # into "post_attn_norm(h); pre_ff_norm(h, residual)" using
        # gemma_fused_add_rmsnorm which computes:
        #   residual = h + residual (in-place)
        #   h = gemma_norm(residual)
        residual = hidden_states

        # Apply input layernorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.enable_moe_block:
            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states
            # Also need raw (unfused) residual for router and pre_ff_norm_2
            hidden_states, residual = self.pre_feedforward_layernorm(
                hidden_states, residual
            )
            # For MoE: router and pre_ff_norm_2 need the unfused residual
            # (which is now updated to post_attn_out + old_residual)
            moe_input = residual

            # Dense MLP branch
            hidden_states_1 = self.mlp(hidden_states)

            # MoE branch: router sees residual (= post_attn_out + old_residual)
            router_logits = self.router(moe_input)
            hidden_states_2 = self.pre_feedforward_layernorm_2(moe_input)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)

            # Fused: (rmsnorm(rmsnorm(h1,w1) + rmsnorm(h2,w2), w3) + residual) * scalar
            if (
                not self.has_ple
                and hidden_states_1.is_cuda
                and hidden_states_1.dim() == 2
            ):
                norm1 = self.post_feedforward_layernorm_1
                norm2 = self.post_feedforward_layernorm_2
                norm3 = self.post_feedforward_layernorm
                hidden_states = gemma_dual_rmsnorm_residual_scalar(
                    hidden_states_1,
                    norm1.weight.data,
                    hidden_states_2,
                    norm2.weight.data,
                    norm3.weight.data,
                    residual,
                    self.layer_scalar,
                    norm1.variance_epsilon,
                    norm2.variance_epsilon,
                    norm3.variance_epsilon,
                )
                return hidden_states, None

            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine branches
            hidden_states = hidden_states_1 + hidden_states_2
        else:
            # Fuse: hidden_states + residual -> residual; pre_ff_norm(residual) -> hidden_states
            hidden_states, residual = self.pre_feedforward_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)

        if not self.has_ple and hidden_states.is_cuda and hidden_states.dim() == 2:
            # Fused: (post_ff_norm(h) + residual) * layer_scalar in one kernel
            norm = self.post_feedforward_layernorm
            hidden_states = gemma_rmsnorm_residual_scalar(
                hidden_states,
                norm.weight.data,
                residual,
                self.layer_scalar,
                norm.variance_epsilon,
            )
        else:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = hidden_states + residual

            if self.has_ple and per_layer_input is not None:
                gate, _ = self.per_layer_input_gate(hidden_states)
                gate = torch.nn.functional.gelu(gate, approximate="tanh")
                gated_per_layer = gate * per_layer_input
                per_layer_contribution, _ = self.per_layer_projection(gated_per_layer)
                per_layer_contribution = self.post_per_layer_input_norm(
                    per_layer_contribution
                )
                hidden_states = hidden_states + per_layer_contribution

            hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4TextModel(PreTrainedModel):
    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,  # embedded normalizer
        )

        # Per-layer input embeddings
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = (
            getattr(config, "hidden_size_per_layer_input", None) or 0
        )
        self.vocab_size_per_layer_input = (
            getattr(config, "vocab_size_per_layer_input", None) or config.vocab_size
        )

        if self.hidden_size_per_layer_input and self.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                self.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )

            self.per_layer_model_projection = ReplicatedLinear(
                self.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_model_projection", prefix),
            )

            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
            )
            self.per_layer_input_scale = torch.rsqrt(torch.tensor(2.0))
            self.per_layer_projection_scale = torch.tensor(
                config.hidden_size**-0.5,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.embed_tokens_per_layer is None:
            return None

        # Handle out-of-vocab tokens for PLE (vocab_size_per_layer_input may
        # be smaller than the main vocab_size). Following Gemma3n pattern.
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )

        # Get packed per-layer embeddings: (num_tokens, total_ple_dim)
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)

        # Apply embed_scale (sqrt of per-layer hidden dim)
        # Already done in embedding layer
        # per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return per_layer_embeds

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project inputs_embeds and combine with per_layer_inputs.

        Following HF/Gemma3n reference:
        1. Project inputs_embeds: hidden_size → total_ple_dim
        2. Scale by hidden_size^{-0.5} (Gemma4ScaledLinear w_scale)
        3. Reshape to (num_tokens, num_layers, per_layer_dim)
        4. Normalize with per_layer_projection_norm
        5. Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        """
        if self.per_layer_model_projection is None:
            return None

        # Project from hidden_size to total_ple_dim
        per_layer_projection, _ = self.per_layer_model_projection(inputs_embeds)

        # Apply w_scale (HF: Gemma4ScaledLinear with w_scale=hidden_size^{-0.5})
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        # Normalize
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        # Combine: (projection + per_layer_inputs) * scale
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:
            input_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)

        hidden_states = input_embeds

        aux_hidden_states = []
        num_layers = len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)

            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, layer_idx, :]
            else:
                per_layer_input = None
            layer_outputs = layer(
                positions=positions,
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                forward_batch=forward_batch,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            residual = layer_outputs[1] if len(layer_outputs) > 1 else None

        # Capture the output of the last layer if requested.
        # layers_to_capture uses +1 offset, so num_layers means
        # "output of the last layer" which is only available after the loop.
        if num_layers in self.layers_to_capture:
            aux_hidden_states.append(hidden_states)

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Gemma4ForCausalLM(PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "language_model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = False

    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma4TextModel(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.capture_aux_hidden_states = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            per_layer_inputs,
            **kwargs,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def _get_k_eq_v_layers(self) -> set:
        """Return set of layer indices where attention_k_eq_v applies (full-attention layers)."""
        if not getattr(self.config, "attention_k_eq_v", False):
            return set()
        return {
            i for i, lt in enumerate(self.config.layer_types) if lt == "full_attention"
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = [
            # (param_name, ckpt_weight_name, shard_ids)
            # gate_up_proj is fused [E, 2*I, H] — chunk into w1 (gate) + w3 (up)
            ("experts.w13_weight", "experts.gate_up_proj", ("w1", "w3")),
            ("experts.w2_weight", "experts.down_proj", ("w2",)),
        ]
        num_experts = self.config.num_experts

        k_eq_v_layers = self._get_k_eq_v_layers()

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        non_persistent_buffers: Set[str] = set()
        for mod_name, mod in self.named_modules():
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name
                non_persistent_buffers.add(full)

        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")

            # HF has router.per_expert_scale and experts.* on the decoder layer;
            # remap into our moe.* subtree since Gemma4MoE owns both.
            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")
            if ".experts." in name and ".moe.experts." not in name:
                name = name.replace(".experts.", ".moe.experts.")

            # attention_k_eq_v: full-attention layers have no v_proj in the
            # checkpoint (K and V share weights).  When we see a k_proj weight
            # for one of these layers, load it into both the "k" and "v" shards
            # of the fused QKV so the forward produces v_raw == k_raw.
            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )

            # MoE expert weights checked first (gate_up_proj contains "up_proj"
            # which would false-match the stacked dense MLP mapping).
            orig_name = name
            for param_name, weight_name, shard_ids in expert_params_mapping:
                name = orig_name
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                for i in range(num_experts):
                    chunks = loaded_weight[i].chunk(len(shard_ids), dim=0)
                    for chunk, sid in zip(chunks, shard_ids):
                        weight_loader(param, chunk, name, sid, i)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    name = orig_name
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    if should_dup_k_to_v:
                        weight_loader(param, loaded_weight, "v")
                    break
                else:
                    name = orig_name
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            param_names = set(dict(self.named_parameters()).keys())
            buckets = {
                logging.WARNING: (
                    "Some weights are not initialized from checkpoints",
                    lambda p: p in param_names,
                ),
                logging.INFO: (
                    "Persistent buffers not in checkpoint (using default init)",
                    lambda p: p not in param_names and p not in non_persistent_buffers,
                ),
                logging.DEBUG: (
                    "Non-persistent buffers not in checkpoint (expected)",
                    lambda p: p in non_persistent_buffers,
                ),
            }
            for level, (msg, pred) in buckets.items():
                names = sorted(p for p in unloaded_params if pred(p))
                if names:
                    logger.log(level, "%s: %s", msg, names)
        return loaded_params

    def _shard_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Shard a full embedding/lm_head weight along vocab dim for the current TP rank.

        Gemma4 uses nn.Embedding (unsharded) but the Eagle3 draft model uses
        VocabParallelEmbedding (sharded). This method extracts the correct
        shard so the weights can be shared.
        """
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return weight
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = (weight.shape[0] + tp_size - 1) // tp_size
        return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

    def get_embed(self):
        return self._shard_weight(self.model.embed_tokens.weight)

    def get_embed_and_head(self):
        embed = self._shard_weight(self.model.embed_tokens.weight)
        head = self._shard_weight(self.lm_head.weight)
        return embed, head

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Gemma4ForCausalLM
