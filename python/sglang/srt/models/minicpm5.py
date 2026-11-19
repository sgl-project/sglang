# Copyright 2025-2026 SGLang Team
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

"""Inference-only MiniCPM5 MoE model compatible with HuggingFace weights.

The architecture follows GLM-4.5 MoE, with three differences: gated attention
(q_proj also produces a per-head sigmoid gate on the attention output), full
rotary embedding without q/k norm, and a router computed in the model dtype.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.hf_transformers import get_rope_config
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layer_boundary import append_stages, is_dense_ffn_fully_dp
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_num_fused_shared_experts,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeModel,
    Glm4MoeSparseMoeBlock,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


class MiniCPM5Attention(Glm4MoeAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        attention_bias: bool = False,
        use_gated_attention: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        attn_tp_size = get_parallel().attn_tp_size

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_gated_attention = use_gated_attention

        # With gated attention q_proj holds [q, gate] interleaved per head.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads * (2 if use_gated_attention else 1),
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        gate = None
        if self.use_gated_attention:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q, gate = q_gate.view(*q_gate.shape[:-1], self.num_heads, -1).chunk(
                2, dim=-1
            )
            q = q.reshape(*q_gate.shape[:-1], -1)
            gate = gate.reshape(*q_gate.shape[:-1], -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return None, forward_batch, (q, k, v, forward_batch, gate)

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        *attn_inputs, gate = inner_state
        attn_output = self.attn(*attn_inputs)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM5Gate(nn.Module):
    """Router in the model dtype; only the correction bias stays in fp32."""

    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight, None)


class MiniCPM5SparseMoeBlock(Glm4MoeSparseMoeBlock):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.tp_size = get_parallel().tp_size
        self.moe_ep_size = get_parallel().moe_ep_size
        self.routed_scaling_factor = config.routed_scaling_factor
        # Match the training combine's single rounding boundary: reduce the
        # routed experts unscaled, then add shared + routed * scale in one op.
        self._defer_routed_scaling = (
            get_exec().deterministic.enable_deterministic_inference
        )
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = get_num_fused_shared_experts()

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MiniCPM5Gate(config=config, prefix=add_prefix("gate", prefix))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=(
                None if self._defer_routed_scaling else self.routed_scaling_factor
            ),
            routing_method_type=RoutingMethodType.DeepSeekV3,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=self.top_k,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                not self._defer_routed_scaling
                and getattr(
                    self.experts, "should_fuse_routed_scaling_factor_in_topk", False
                )
            ),
            fused_shared_experts_scaling_factor=1,
            scoring_func="sigmoid",
        )

        self.topk.bind_runner_backend(
            self.experts.runner_backend,
            routing_method_type=self.experts.routing_method_type,
        )

        a2a = get_moe_a2a_backend()
        self._enable_a2a_moe = (
            a2a.is_deepep()
            or a2a.is_mooncake()
            or a2a.is_nixl()
            or a2a.is_mori()
            or a2a.is_ascend_fuseep()
            or a2a.is_flashinfer()
        )
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                parallel_group=(
                    "replicated"
                    if self._enable_a2a_moe
                    or a2a.is_flashinfer_megamoe()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else "tp"
                ),
            )

        if self._enable_a2a_moe and not a2a.is_flashinfer():
            self.ep_size = get_parallel().moe_ep_size
            self.num_experts = (
                config.n_routed_experts + get_exec().moe.ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = self.gate.e_score_correction_bias.data

        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def forward_normal_dual_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._defer_routed_scaling:
            return super().forward_normal_dual_stream(hidden_states)
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states)
        with torch.cuda.stream(self.alt_stream):
            topk_output = self.topk(hidden_states, self.gate(hidden_states))
            routed_output = self.experts(hidden_states, topk_output)
        current_stream.wait_stream(self.alt_stream)
        return self._add_scaled_routed_output(routed_output, shared_output)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._defer_routed_scaling:
            return super().forward_normal(hidden_states)
        shared_output = self._forward_shared_experts(hidden_states)
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, self.gate(hidden_states))
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        routed_output = self.experts(hidden_states, topk_output)
        return self._add_scaled_routed_output(routed_output, shared_output)

    def _add_scaled_routed_output(self, routed_output, shared_output):
        if shared_output is None:
            output = routed_output.mul_(self.routed_scaling_factor)
        else:
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                output = torch.empty_like(routed_output)
            torch.add(
                shared_output,
                routed_output,
                alpha=self.routed_scaling_factor,
                out=output,
            )
        return output


class MiniCPM5DecoderLayer(Glm4MoeDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id
        rope_theta, rope_scaling = get_rope_config(config)

        self.self_attn = MiniCPM5Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            attention_bias=config.attention_bias,
            use_gated_attention=config.use_gated_attention,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        attn, ffn = self.stage_facts(config, layer_id)
        self.is_layer_sparse = ffn.sparse

        if self.is_layer_sparse:
            self.mlp = MiniCPM5SparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
            )
        else:
            mlp_parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                parallel_group=mlp_parallel_group,
                reduce_results=False,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )
        # MiniCPM5Attention.forward_prepare takes a plain tensor, so the
        # aiter fused RMSNorm+FP8 input stays off.
        self.attn_quant_format = ""


class MiniCPM5Model(Glm4MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.embed_dim = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = get_stream("alt") if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: MiniCPM5DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: MiniCPM5DecoderLayer.stage_facts(config, idx),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []


class MiniCPM5ForCausalLM(Glm4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.num_fused_shared_experts = get_num_fused_shared_experts()
        self.model = MiniCPM5Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        if get_exec().deterministic.enable_deterministic_inference:
            return (
                "Deterministic inference requires separate shared experts: "
                "fusion sums the shared expert inside the routed MoE, which "
                "training adds separately, so the output rounding differs."
            )
        return super().shared_experts_fusion_disable_reason(hf_config, quant_config)

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        # +1 as in set_eagle3_layers_to_capture: layer i receives the output of
        # layer i-1. SpecForge trains DFlash drafts on the same capture points.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = [MiniCPM5ForCausalLM]
