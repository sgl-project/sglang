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

"""Inference-only MiniCPM5 model with gated GQA and routed and shared experts."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
from sglang.srt.batch_overlap.two_batch_overlap import model_forward_stages
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.hf_transformers import get_rope_config
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    LinearParallelGroup,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.topk import TopK, is_routed_scaling_factor_fused_in_topk
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = _is_hip and get_bool_env_var("SGLANG_USE_AITER")

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import (
        process_shared_expert,
        wait_share_stream,
    )


class MiniCPM5Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        attention_bias: bool = False,
        use_gated_attention: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
        self.head_dim = head_dim
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)
        gate = None
        if self.use_gated_attention:
            q_gate, k, v = qkv.split(
                (2 * self.q_size, self.kv_size, self.kv_size), dim=-1
            )
            q, gate = q_gate.unflatten(-1, (self.num_heads, 2 * self.head_dim)).chunk(
                2, dim=-1
            )
            q, gate = q.flatten(-2), gate.flatten(-2)
        else:
            q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        if gate is not None:
            output = output * torch.sigmoid(gate)
        output, _ = self.o_proj(output)
        return output


class MiniCPM5MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        parallel_group: LinearParallelGroup,
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"MiniCPM5 only supports SiLU, got {config.hidden_act}.")
        self.is_replicated = parallel_group == "replicated"
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            parallel_group=parallel_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            parallel_group=parallel_group,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor, forward_batch=None) -> torch.Tensor:
        if self.is_replicated and hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, _ = self.gate_up_proj(hidden_states)
        output, _ = self.down_proj(self.act(gate_up))
        return output


class MiniCPM5Gate(nn.Module):
    """Compute router logits in FP32 with weights stored in the model dtype."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states):
        return F.linear(hidden_states.float(), self.weight.float())


class MiniCPM5Moe(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_fused_shared_experts = get_num_fused_shared_experts()

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

        self.gate = MiniCPM5Gate(config)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
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
            apply_routed_scaling_factor_on_output=getattr(
                self.experts, "should_fuse_routed_scaling_factor_in_topk", False
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
        self.shared_experts = None
        if config.n_shared_experts and not self.num_fused_shared_experts:
            self.shared_experts = MiniCPM5MLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                parallel_group=(
                    "replicated"
                    if self._enable_a2a_moe
                    or a2a.is_flashinfer_megamoe()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else "tp"
                ),
            )

        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def get_moe_weights(self):
        return [
            weight.data
            for name, weight in self.experts.named_parameters()
            if name != "correction_bias"
            and filter_moe_weight_param_global_expert(
                name, weight, self.experts.num_local_experts
            )
        ]

    def _forward_routed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            topk_output = self.topk(hidden_states, self.gate(hidden_states))
        return self.experts(hidden_states, topk_output)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self.shared_experts is None or hidden_states.shape[0] == 0:
            return self._combine(self._forward_routed(hidden_states), None)

        if (
            self.alt_stream is not None
            and not self._enable_a2a_moe
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            shared_output = self.shared_experts(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                routed_output = self._forward_routed(hidden_states)
            current_stream.wait_stream(self.alt_stream)
            if isinstance(self.experts.quant_method, KTEPWrapperMethod):
                routed_output.mul_(self.routed_scaling_factor)
        elif (
            _is_npu
            and self._enable_a2a_moe
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and (
                forward_batch.forward_mode.is_extend()
                or forward_batch.forward_mode.is_target_verify()
            )
        ):
            shared_output = process_shared_expert(hidden_states, self.shared_experts)
            routed_output = self._forward_routed(hidden_states)
            wait_share_stream()
        else:
            shared_output = self.shared_experts(hidden_states)
            routed_output = self._forward_routed(hidden_states)

        return self._combine(routed_output, shared_output)

    def _combine(self, routed_output, shared_output):
        scale = 1.0
        if self._enable_a2a_moe:
            if not is_routed_scaling_factor_fused_in_topk(self.topk.topk_config):
                scale = self.routed_scaling_factor
        elif not _is_cuda and not _use_aiter:
            routed_output.mul_(self.routed_scaling_factor)

        if shared_output is None:
            return routed_output if scale == 1.0 else routed_output.mul_(scale)
        if self._enable_a2a_moe:
            return shared_output.add_(routed_output, alpha=scale)

        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            output = torch.empty_like(routed_output)
        torch.add(shared_output, routed_output, alpha=scale, out=output)
        return output


class MiniCPM5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
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
            self.mlp = MiniCPM5Moe(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
            )
        else:
            mlp_parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"
            self.mlp = MiniCPM5MLP(
                config=config,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                parallel_group=mlp_parallel_group,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: PretrainedConfig, layer_id: int):
        has_experts = config.n_routed_experts is not None
        first_sparse_layer = config.first_k_dense_replace
        return declare_attn(), declare_ffn(
            sparse=has_experts and layer_id >= first_sparse_layer,
            next_layer_sparse=has_experts and layer_id + 1 >= first_sparse_layer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        capture_output=None,
    ) -> torch.Tensor:
        attention_input = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture=capture_output
        )
        attention_output = self.self_attn(positions, attention_input, forward_batch)
        hidden_states = self.attn_boundary.finish(attention_output, forward_batch)

        ffn_input = self.ffn_boundary.prepare(hidden_states, forward_batch)
        ffn_output = self.mlp(ffn_input, forward_batch)
        return self.ffn_boundary.finish(ffn_output, forward_batch)


class MiniCPM5Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
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

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        aux_hidden_states = AuxHiddenStateList()
        tbo_start = self.end_layer
        if forward_batch.can_run_tbo and not self.layers_to_capture:
            tbo_start = min(
                self.end_layer, max(self.start_layer, self.first_k_dense_replace)
            )

        for layer_id in range(self.start_layer, tbo_start):
            capture = (
                aux_hidden_states.capture
                if layer_id in self.layers_to_capture
                else None
            )
            with get_global_expert_distribution_recorder().with_current_layer(layer_id):
                hidden_states = self.layers[layer_id](
                    positions, hidden_states, forward_batch, capture_output=capture
                )

        if tbo_start < self.end_layer:
            hidden_states = model_forward_stages(
                layers=self.layers[tbo_start : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
            )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)

        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.norm
            )
        if len(aux_hidden_states):
            return hidden_states, aux_hidden_states
        return hidden_states


class MiniCPM5ForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
    def num_fusable_shared_experts(cls, hf_config) -> int:
        return int(hf_config.n_shared_experts == 1)

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        cuda_supported = _is_cuda and torch.cuda.get_device_capability("cuda") >= (8, 0)
        hip_supported = _is_hip and torch.cuda.get_device_capability("cuda") >= (9, 4)
        if not (cuda_supported or hip_supported):
            return "MiniCPM5 shared-expert fusion requires CUDA SM80 or ROCm gfx942."
        if get_parallel().moe_ep_size > 1 and not hip_supported:
            return "MiniCPM5 shared-expert fusion under EP requires ROCm gfx942."
        a2a = get_moe_a2a_backend()
        if a2a.is_deepep() or a2a.is_mori():
            return "MiniCPM5 uses a separate shared expert with DeepEP or MORI."
        if quant_config and quant_config.get_name() == "w4afp8":
            return (
                "MiniCPM5 W4AFP8 uses distinct routed and shared expert quantization."
            )
        return None

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

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
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_projections = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        expert_mapping = {
            checkpoint_prefix: (fused_prefix, expert_id, shard)
            for fused_prefix, checkpoint_prefix, expert_id, shard in FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=(self.config.n_routed_experts or 0)
                + self.num_fused_shared_experts,
            )
        }
        params = dict(self.named_parameters())
        for name, weight in weights:
            if name.endswith(".rotary_emb.inv_freq"):
                continue
            if self.num_fused_shared_experts:
                # The shared expert follows the routed experts in the fused tensor.
                name = name.replace(
                    ".mlp.shared_experts.",
                    f".mlp.experts.{self.config.n_routed_experts}.",
                )

            expert_id = shard_id = None
            prefix, separator, expert = name.partition(".mlp.experts.")
            if separator:
                projection, _, tensor_name = expert.rpartition(".")
                mapped = expert_mapping.get(f"experts.{projection}.")
                if mapped is not None:
                    fused_prefix, expert_id, shard_id = mapped
                    name = f"{prefix}.mlp.{fused_prefix}{tensor_name}"
            else:
                for source, destination, shard in stacked_projections:
                    if source in name:
                        name = name.replace(source, destination)
                        shard_id = shard
                        break

            param = params.get(name)
            if param is None:
                continue
            if expert_id is not None:
                param.weight_loader(
                    param, weight, name, shard_id=shard_id, expert_id=expert_id
                )
            elif shard_id is not None:
                param.weight_loader(param, weight, shard_id)
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
        self.set_dflash_layers_to_capture(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        # Capture the post-layer residual at the next attention boundary.
        self.model.layers_to_capture = [layer_id + 1 for layer_id in layer_ids]


EntryClass = MiniCPM5ForCausalLM
