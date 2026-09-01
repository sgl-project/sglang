# Copyright 2023-2024 SGLang Team
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

# Adapted from vLLM's DeepSeek V2 implementation:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference implementation for dots.note.omni."""

import concurrent.futures
import logging
import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.configs.dots3 import Dots3Config
from sglang.srt.distributed import (
    get_pp_group,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_indexer_wk,
)
from sglang.srt.models.dots3_common.fp8 import per_token_group_quant_einsum_fp8
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_parallel,
)
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    bind_or_assign,
    ceil_align,
    ceil_div,
    get_bool_env_var,
    get_device_sm,
    is_cuda,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
)

_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_device_sm = get_device_sm()

# Import-time CUDA kernels would block processor imports on CPU CI.
if _is_cuda:
    from sgl_kernel import merge_state_v2

    from sglang.kernels.ops.gemm.dsv3_router_gemm import dsv3_router_gemm
else:
    merge_state_v2 = None
    dsv3_router_gemm = None


def _require_cuda() -> None:
    if not _is_cuda:
        raise RuntimeError("Dots3 model only supports CUDA backend.")


logger = logging.getLogger(__name__)


@runtime_checkable
class _SupportsWeightLoader(Protocol):
    weight_loader: Callable[..., Any]


def _get_scale_block_n(
    quant_config: Optional[QuantizationConfig],
) -> int:
    if quant_config is not None and quant_config.weight_block_size is not None:
        return quant_config.weight_block_size[0]
    return 1


def _get_param_weight_loader(param: nn.Parameter):
    return (
        param.weight_loader
        if isinstance(param, _SupportsWeightLoader)
        else default_weight_loader
    )


class Dots3AttnForwardMethod(IntEnum):
    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use dense MHA for short DSA prefills selected by the DSA backend.

    SWA_MHA = auto()


class Dots3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class Dots3MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None

    def forward(self, hidden_states):
        # Use the fused router only for its tuned shapes.
        if (
            hidden_states.shape[0] <= 16
            and hidden_states.shape[1] == 7168
            and self.weight.shape[0] == 256
            and _device_sm >= 90
        ):
            # router gemm output float32
            logits = dsv3_router_gemm(hidden_states, self.weight)
        else:
            logits = F.linear(hidden_states, self.weight, None)

        return logits


class Dots3MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else config.n_shared_experts
        )
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

        self.gate = Dots3MoEGate(config)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_exec().moe.ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        )

        self.shared_experts = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
            self.shared_experts = Dots3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            is_packed_weight = quant_config is not None and (
                quant_config.get_name()
                in {
                    "awq",
                    "awq_marlin",
                    "moe_wna16",
                }
            )
            shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            if shared_experts_is_fp8:
                assert (
                    self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                    == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                )

        self.top_k = config.num_experts_per_tok

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_parallel().moe_ep_size
            self.num_experts = (
                config.n_routed_experts + get_exec().moe.ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.n_routed_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True,
                return_recv_hook=True,
            )

        self._enable_deepep_moe = get_moe_a2a_backend().is_deepep()

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if not self._enable_deepep_moe:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024
            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
            ):
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)

        current_stream.wait_stream(self.alt_stream)
        with use_symmetric_memory(parallel_state.get_tp_group()) as sm:
            final_hidden_states_out = torch.empty_like(final_hidden_states)

        torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
        final_hidden_states = final_hidden_states_out
        sm.tag(final_hidden_states)
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            shared_output = self._forward_shared_experts(hidden_states)
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)
        if shared_output is not None:
            with use_symmetric_memory(parallel_state.get_tp_group()) as sm:
                final_hidden_states_out = torch.empty_like(final_hidden_states)
            torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
            final_hidden_states = final_hidden_states_out
            sm.tag(final_hidden_states)
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        elif not self.experts.should_fuse_routed_scaling_factor_in_topk:
            final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states):
        if self.num_fused_shared_experts == 0:
            return self.shared_experts(hidden_states)
        else:
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_weights_local, state.topk_idx_local, _ = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.hidden_states_experts_output = self.experts.moe_impl(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.dispatch_output.topk_idx,
                topk_weights=state.dispatch_output.topk_weights,
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = (
                self.experts.deepep_dispatcher.combine_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


# Aligned with HF's implementation, using sliding window inclusive with the last token.
# SGLang assumes exclusive.
def get_attention_sliding_window_size(config):
    return config.sliding_window_size - 1


class Dots3AttentionMLA(nn.Module):
    @dataclass
    class Dots3AttentionMLAConfig:
        attention_gate_type: str
        kv_lora_rank: int
        q_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        num_attention_heads: int
        num_key_value_heads: int
        v_head_dim: int
        rope_theta: float

        @classmethod
        def from_config(
            cls,
            config: Dots3Config,
            layer_type: str,
        ):
            if layer_type == "sliding_attention":
                return cls(
                    attention_gate_type=config.swa_attention_gate_type,
                    kv_lora_rank=config.swa_kv_lora_rank,
                    q_lora_rank=config.swa_q_lora_rank,
                    qk_nope_head_dim=config.swa_qk_nope_head_dim,
                    qk_rope_head_dim=config.swa_qk_rope_head_dim,
                    num_attention_heads=config.swa_num_attention_heads,
                    num_key_value_heads=config.swa_num_key_value_heads,
                    v_head_dim=config.swa_v_head_dim,
                    rope_theta=config.swa_rope_theta,
                )
            else:
                return cls(
                    attention_gate_type=config.attention_gate_type,
                    kv_lora_rank=config.kv_lora_rank,
                    q_lora_rank=config.q_lora_rank,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    v_head_dim=config.v_head_dim,
                    rope_theta=config.rope_theta,
                )

    def __init__(
        self,
        config: Dots3Config,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = False,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # Determine the layer type and sliding window size
        layer_type = config.layer_types[layer_id]
        assert layer_type in {"sliding_attention", "full_attention"}
        use_sliding_window = layer_type == "sliding_attention"
        self.use_swa = use_sliding_window
        self.sliding_window_size = (
            get_attention_sliding_window_size(config) if use_sliding_window else -1
        )

        # Get Attention config based on layer type.
        attn_config = self.Dots3AttentionMLAConfig.from_config(config, layer_type)
        self.qk_nope_head_dim = attn_config.qk_nope_head_dim
        self.qk_rope_head_dim = attn_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = attn_config.v_head_dim
        self.q_lora_rank = attn_config.q_lora_rank
        self.kv_lora_rank = attn_config.kv_lora_rank
        self.apply_mla_qkv_lora_rescale = config.apply_mla_qkv_lora_rescale
        self.num_heads = attn_config.num_attention_heads
        assert self.num_heads % attn_tp_size == 0
        self.num_local_heads = self.num_heads // attn_tp_size
        assert (
            attn_config.num_attention_heads == attn_config.num_key_value_heads
        ), "Dots3 Only supports equal number of query and key value heads."
        self.attention_gate_type = attn_config.attention_gate_type
        assert self.attention_gate_type in {
            "headwise",
            "elementwise",
        }, f"Unsupported attention_gate_type: {self.attention_gate_type}. Expected 'headwise' or 'elementwise'."
        self.g_proj_local_dim = self.num_local_heads * (
            1 if self.attention_gate_type == "headwise" else self.v_head_dim
        )
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = attn_config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        # For tensor parallel attention
        assert self.q_lora_rank is not None, "Dots3AttentionMLA requires q_lora_rank."
        scale_block_n = _get_scale_block_n(quant_config)
        self.qk_rope_head_dim_padded = ceil_align(self.qk_rope_head_dim, scale_block_n)
        # NOTE(xiaozhi): For the sake of DeepGEMM kernel, we align the g_proj_local_dim to 8.
        self.g_proj_local_dim_padded = ceil_align(
            self.g_proj_local_dim, max(8, scale_block_n)
        )
        self.use_nsa = (
            config.index_n_heads is not None
            and config.index_head_dim is not None
            and config.index_topk is not None
            and layer_type == "full_attention"
        )
        fused_qkv_out_size = (
            self.q_lora_rank
            + self.kv_lora_rank
            + self.qk_rope_head_dim_padded
            + self.g_proj_local_dim_padded
        )
        self.fused_qkv_a_g_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            fused_qkv_out_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fused_qkv_a_g_proj_with_mqa", prefix),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_out_size = self.num_heads * self.qk_head_dim
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.q_b_out_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.k_rope_only_layernorm = RMSNorm(
            self.qk_rope_head_dim, eps=config.rms_norm_eps
        )

        # Transformers v5 represents the default RoPE configuration as a
        # non-empty ``{"rope_type": "default", ...}`` dictionary.  Pass it
        # through to the shared RoPE factory, which handles both that form and
        # actual scaled-RoPE configurations.
        self.rope_scaling = config.rope_scaling

        self.rotary_emb = get_rope_wrapper(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
            device=get_device().device,
        )

        # Optional NSA (Native Sparse Attention) indexer.
        if self.use_nsa:
            assert (
                self.q_lora_rank is not None
            ), "Dots3 NSA requires q_lora_rank to be set in the config."
            self.indexer = Indexer(
                hidden_size=self.hidden_size,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                rope_head_dim=self.qk_rope_head_dim,
                index_topk=config.index_topk,
                q_lora_rank=self.q_lora_rank,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=self.rope_scaling,
                is_neox_style=False,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=self.layer_id,
                alt_stream=alt_stream,
            )

        self.attn_mqa = RadixAttention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scaling=self.scaling,
            num_kv_heads=1,
            layer_id=self.layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
            sliding_window_size=self.sliding_window_size,
        )

        self.attn_mha = RadixAttention(
            num_heads=self.num_local_heads,
            head_dim=self.qk_head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=self.layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
            sliding_window_size=self.sliding_window_size,
        )

        self.alt_stream = alt_stream

        self.w_kc = None
        self.w_vc = None

        self.w_scale_k = None
        self.w_scale_v = None

        # Attention backend used by current forward batch
        self.current_attention_backend = None

    def _split_fused_qkv_a_g_proj_out(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_offset = 0
        kv_offset = q_offset + self.q_lora_rank
        g_offset = kv_offset + self.kv_lora_rank + self.qk_rope_head_dim_padded
        q = x[..., q_offset : q_offset + self.q_lora_rank]
        latent_cache = x[
            ..., kv_offset : kv_offset + self.kv_lora_rank + self.qk_rope_head_dim
        ]
        g = x[..., g_offset : g_offset + self.g_proj_local_dim]
        return q, latent_cache, g

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> Dots3AttnForwardMethod:
        if (
            self.use_swa
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            raise NotImplementedError(
                "Dots3 SWA attention does not support context-parallel prefill yet. "
                "Please disable --enable-prefill-cp."
            )

        backend = get_attn_backend()
        from sglang.srt.layers.attention.dots_hybrid_backend import (
            DotsHybridAttnBackend,
            DotsSWAMLAAttnBackend,
        )
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        if isinstance(backend, DotsHybridAttnBackend) and self.use_swa:
            backend = backend.selected_swa_backend(forward_batch)
        elif isinstance(backend, HybridAttnBackend):
            backend = backend._select_backend(forward_batch.forward_mode)
        if isinstance(backend, DotsSWAMLAAttnBackend):
            backend = backend.selected_backend(forward_batch)
        backend_name = type(backend).__name__.lower()
        if "flashattention" in backend_name:
            attention_backend = "fa3"
        elif "flashmla" in backend_name:
            attention_backend = "flashmla"
        elif "triton" in backend_name:
            attention_backend = "triton"
        elif "sparse" in backend_name or self.use_nsa:
            attention_backend = "nsa"
        else:
            raise NotImplementedError(
                f"Unsupported Dots3 attention backend: {type(backend).__name__}"
            )
        self.current_attention_backend = attention_backend

        if attention_backend == "fa3":
            if self.use_swa:
                # Expanded SWA-MHA is prefill-only; other modes use paged MLA.
                return (
                    Dots3AttnForwardMethod.MLA
                    if forward_batch.forward_mode.is_decode_or_idle()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend_v2()
                    else Dots3AttnForwardMethod.SWA_MHA
                )
            if forward_batch.forward_mode.is_extend_without_speculative():
                return Dots3AttnForwardMethod.MHA_CHUNKED_KV
            else:
                return Dots3AttnForwardMethod.MLA
        elif attention_backend in ("flashmla", "triton", "nsa"):
            return Dots3AttnForwardMethod.MLA
        else:
            raise NotImplementedError(
                f"Dots3 only supports CUDA attention backends (fa3, triton, flashmla, nsa), got: {attention_backend}"
            )

    def _apply_attention_gate(self, attn_output: torch.Tensor, g: torch.Tensor):
        if attn_output.ndim != 3:
            attn_output = attn_output.reshape(-1, self.num_local_heads, self.v_head_dim)

        g = torch.nn.functional.sigmoid(g)
        return attn_output * (
            g.unsqueeze(-1)
            if self.attention_gate_type == "headwise"
            else g.reshape(-1, self.num_local_heads, self.v_head_dim)
        )

    # TODO(xiaozhi): Fuse this to post_load_weights.
    def _maybe_apply_lora_rescale(self, lora_tensor: torch.Tensor, lora_dim):
        if not self.apply_mla_qkv_lora_rescale:
            return lora_tensor
        return lora_tensor * math.sqrt(self.hidden_size / lora_dim)

    @staticmethod
    def _absorbed_bmm(
        lhs: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weight.dtype == torch.float8_e4m3fn:
            import deep_gemm

            assert weight_scale is not None
            lhs_fp8 = per_token_group_quant_einsum_fp8(lhs)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                lhs_fp8,
                (weight, weight_scale),
                out,
            )
        else:
            torch.bmm(
                lhs.transpose(0, 1),
                weight,
                out=out.transpose(0, 1),
            )
        return out

    def _get_kv_write_loc(self, forward_batch):
        swa_loc = None
        if self.use_swa:
            backend = get_attn_backend()
            from sglang.srt.layers.attention.dots_hybrid_backend import (
                DotsHybridAttnBackend,
                DotsSWAMLAAttnBackend,
            )

            if isinstance(backend, (DotsHybridAttnBackend, DotsSWAMLAAttnBackend)):
                backend.maybe_rebuild_metadata_after_dp_padding(forward_batch)
            if isinstance(backend, DotsHybridAttnBackend):
                backend = backend.selected_swa_backend(forward_batch)
            metadata = backend.forward_metadata
            swa_loc = metadata.swa_out_cache_loc
            if (
                swa_loc is None
                or swa_loc.shape[0] != forward_batch.out_cache_loc.shape[0]
            ):
                if isinstance(backend, DotsSWAMLAAttnBackend):
                    out_cache_loc = backend.select_draft_step_out_cache_loc(
                        forward_batch
                    )
                else:
                    out_cache_loc = forward_batch.out_cache_loc
                swa_loc = get_token_to_kv_pool().translate_loc_from_full_to_swa(
                    out_cache_loc
                )
        return KVWriteLoc(forward_batch.out_cache_loc, swa_loc=swa_loc)

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        return self.forward_core(s)

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, None, forward_batch, None

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)

        if attn_forward_method == Dots3AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == Dots3AttnForwardMethod.SWA_MHA:
            inner_state = self.forward_swa_mha_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == Dots3AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, _, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states

        if attn_forward_method == Dots3AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == Dots3AttnForwardMethod.SWA_MHA:
            return self.forward_swa_mha_core(*inner_state)
        elif attn_forward_method == Dots3AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        else:
            raise NotImplementedError

    def forward_normal_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        q, latent_cache, g = self._split_fused_qkv_a_g_proj_out(
            self.fused_qkv_a_g_proj_with_mqa(hidden_states)[0]
        )
        q = self.q_a_layernorm(q)
        q = self._maybe_apply_lora_rescale(q, self.q_lora_rank)
        q = self.q_b_proj(q)[0]
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv_a = self._maybe_apply_lora_rescale(kv_a, self.kv_lora_rank)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        k_pe = self.k_rope_only_layernorm(k_pe)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        get_token_to_kv_pool().set_kv_buffer(
            self.attn_mha,
            self._get_kv_write_loc(forward_batch),
            latent_cache,
            None,
        )

        return q, k, v, g, forward_batch

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        q, latent_cache, g = self._split_fused_qkv_a_g_proj_out(
            self.fused_qkv_a_g_proj_with_mqa(hidden_states)[0]
        )
        k_nope = latent_cache[..., : self.kv_lora_rank]

        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q = self.q_a_layernorm(q)
            q = self._maybe_apply_lora_rescale(q, self.q_lora_rank)
            with torch.cuda.stream(self.alt_stream):
                k_nope = self.kv_a_layernorm(k_nope)
                k_nope = self._maybe_apply_lora_rescale(k_nope, self.kv_lora_rank)
            current_stream.wait_stream(self.alt_stream)
        else:
            q = self.q_a_layernorm(q)
            q = self._maybe_apply_lora_rescale(q, self.q_lora_rank)
            k_nope = self.kv_a_layernorm(k_nope)
            k_nope = self._maybe_apply_lora_rescale(k_nope, self.kv_lora_rank)

        # NSA indexer owns its projections and uses the standard SGLang layer.
        topk_indices = None
        if self.use_nsa:
            topk_indices = self.indexer(
                x=hidden_states,
                q_lora=q,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )
        k_nope = k_nope.unsqueeze(1)
        q = self.q_b_proj(q)[0]
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        k_pe = self.k_rope_only_layernorm(k_pe)

        q_nope_out = q_nope.new_empty(
            (q_nope.shape[0], self.num_local_heads, self.kv_lora_rank)
        )
        self._absorbed_bmm(q_nope, self.w_kc, q_nope_out, self.w_scale_k)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            g,
            forward_batch,
            zero_allocator,
            topk_indices,
        )

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        g,
        forward_batch,
        zero_allocator,
        topk_indices=None,
    ):
        attn_kwargs = {}
        if topk_indices is not None:
            attn_kwargs["topk_indices"] = topk_indices

        from sglang.srt.layers.attention.dots_hybrid_backend import (
            DotsHybridAttnBackend,
            DotsSWAMLAAttnBackend,
        )

        backend = get_attn_backend()
        if (
            self.use_swa
            and (
                isinstance(backend, DotsHybridAttnBackend)
                or (
                    isinstance(backend, DotsSWAMLAAttnBackend)
                    and backend.uses_flash_attention(forward_batch)
                )
            )
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
        ):
            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mqa,
                self._get_kv_write_loc(forward_batch),
                k_nope,
                k_pe,
            )
            q_input = torch.cat([q_nope_out, q_pe], dim=-1)
            attn_output = backend.forward_swa_mla_absorbed(
                q_input, self.attn_mqa, forward_batch
            )
        elif self.current_attention_backend in ("fa3", "nsa") or self.use_nsa:
            assert (
                not self.use_nsa
                or topk_indices is not None
                or forward_batch.forward_mode.is_idle()
            ), "NSA attention requires topk_indices from indexer."
            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mqa,
                self._get_kv_write_loc(forward_batch),
                k_nope,
                k_pe,
            )
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                save_kv_cache=False,
                **attn_kwargs,
            )
        elif self.current_attention_backend in {"triton", "flashmla"}:
            q_input = torch.cat([q_nope_out, q_pe], dim=-1)
            k_input = torch.cat([k_nope, k_pe], dim=-1)
            v_input = k_nope.contiguous()

            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mqa,
                self._get_kv_write_loc(forward_batch),
                k_nope,
                k_pe,
            )
            attn_output = self.attn_mqa(
                q_input, k_input, v_input, forward_batch, save_kv_cache=False
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = attn_output.new_empty(
            (attn_output.shape[0], self.num_local_heads, self.v_head_dim),
        )
        self._absorbed_bmm(attn_output, self.w_vc, attn_bmm_output, self.w_scale_v)
        attn_bmm_output = self._apply_attention_gate(attn_bmm_output, g)
        output, _ = self.o_proj(
            attn_bmm_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = get_token_to_kv_pool().get_key_buffer(
                self.attn_mha.layer_id
            )

            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]

            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # Materialize long prefixes in chunks to limit expanded K/V memory.
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(self, q, k, v, g, forward_batch):
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            backend = get_attn_backend()
            from sglang.srt.layers.attention.dots_hybrid_backend import (
                DotsHybridAttnBackend,
                DotsSWAMLAAttnBackend,
            )

            if isinstance(backend, (DotsHybridAttnBackend, DotsSWAMLAAttnBackend)):
                backend.init_mha_chunk_metadata(forward_batch)

        # Zero padded V rows because FA3 tiles may read past cu_seqlens_q[-1].
        real_num_tokens = forward_batch.extend_num_tokens
        if real_num_tokens is not None and v.shape[0] > real_num_tokens:
            v[real_num_tokens:] = 0

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = self._apply_attention_gate(attn_output, g)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_swa_mha_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        q_lora, latent_cache, g = self._split_fused_qkv_a_g_proj_out(
            self.fused_qkv_a_g_proj_with_mqa(hidden_states)[0]
        )
        q_lora = self.q_a_layernorm(q_lora)
        q_lora = self._maybe_apply_lora_rescale(q_lora, self.q_lora_rank)
        q = self.q_b_proj(q_lora)[0]
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)

        kv_a = latent_cache[..., : self.kv_lora_rank]
        kv_a = self.kv_a_layernorm(kv_a)
        kv_a = self._maybe_apply_lora_rescale(kv_a, self.kv_lora_rank)

        q_pe = q[..., -self.qk_rope_head_dim :]
        latent_cache = latent_cache.unsqueeze(1)
        k_pe = latent_cache[..., -self.qk_rope_head_dim :]
        k_pe = self.k_rope_only_layernorm(k_pe)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        latent_cache[..., : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe

        # Save latent cache
        get_token_to_kv_pool().set_kv_buffer(
            self.attn_mha,
            self._get_kv_write_loc(forward_batch),
            latent_cache,
            None,
        )

        # Load full latent cache
        backend = get_attn_backend()
        from sglang.srt.layers.attention.dots_hybrid_backend import (
            DotsHybridAttnBackend,
            DotsSWAMLAAttnBackend,
        )

        assert isinstance(backend, (DotsHybridAttnBackend, DotsSWAMLAAttnBackend))
        latent_cache = backend.get_swa_mla_prefill_latent_cache(
            forward_batch, self.attn_mha.layer_id
        )
        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # `split` keeps the 576-wide latent-cache stride while this projection
        # consumes only the 512-wide LoRA slice. Block-FP8 activation
        # quantization requires a contiguous input.
        kv = self.kv_b_proj(kv_a.contiguous())[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        kv_len_sum = latent_cache.shape[0]
        k = torch.empty(
            (
                kv_len_sum,
                self.num_local_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            ),
            dtype=q.dtype,
            device=q.device,
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        return q, k, v, g, forward_batch

    def forward_swa_mha_core(self, q, k, v, g, forward_batch):
        backend = get_attn_backend()
        from sglang.srt.layers.attention.dots_hybrid_backend import (
            DotsHybridAttnBackend,
            DotsSWAMLAAttnBackend,
        )

        assert isinstance(backend, (DotsHybridAttnBackend, DotsSWAMLAAttnBackend))
        attn_output = backend.forward_swa_mla_expanded(
            q, k, v, self.attn_mha, forward_batch
        )
        attn_output = self._apply_attention_gate(attn_output, g)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        return self.o_proj(attn_output)[0]


class Dots3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        _require_cuda()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id
        self.self_attn = Dots3AttentionMLA(
            config=config,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Dots3MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Dots3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
        )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        # The MTP block uses a dense MLP.
        if is_nextn:
            return False
        return (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class Dots3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _require_cuda()
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Dots3DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0

        for i in range(normal_start_layer, normal_end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual, zero_allocator
                )

        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Dots3LanguageModelForCausalLM(nn.Module):
    # for quark model load
    packed_modules_mapping = {}
    fused_shared_experts_architecture = "Dots3NoteForCausalLM"

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Always fuse q_a_proj/kv_a_proj_with_mqa/g_proj when loading Dots3.
        self.fuse_qkv_a_g_proj = True
        assert (
            config.q_lora_rank is not None
        ), "Dots3 requires q_lora_rank to enable fused_qkv_a_g_proj_with_mqa loading."
        if self.fuse_qkv_a_g_proj:
            self.packed_modules_mapping["fused_qkv_a_g_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "g_proj",
            ]

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = Dots3Model(
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

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Dots3MoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        """Return why shared-expert fusion is unavailable, or ``None``."""
        if (
            not _is_cuda
            or torch.cuda.get_device_capability("cuda") < (8, 0)
            or hf_config.architectures[0] != cls.fused_shared_experts_architecture
            or hf_config.n_routed_experts != 256
            or hf_config.n_shared_experts != 1
        ):
            return "Shared-expert fusion is unsupported for this Dots3 configuration."
        if get_parallel().moe_ep_size > 1:
            return "Dots3 shared-expert fusion is unsupported with expert parallelism."
        if quant_config is not None and quant_config.get_name() == "w4afp8":
            return (
                "Dots3 W4AFP8 shared and routed experts use incompatible quantization."
            )
        return None

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else self.config.n_shared_experts
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def pad_input_ids(
        self,
        input_ids: List[int],
        mm_inputs: MultimodalInputs,
        **kwargs,
    ) -> List[int]:
        token_pairs = []
        if mm_inputs.im_start_id is not None and mm_inputs.im_end_id is not None:
            token_pairs.append((mm_inputs.im_start_id, mm_inputs.im_end_id))
        if mm_inputs.audio_start_id is not None and mm_inputs.audio_end_id is not None:
            token_pairs.append((mm_inputs.audio_start_id, mm_inputs.audio_end_id))
        return MultiModalityDataPaddingPatternTokenPairs(
            data_token_pairs=token_pairs or None
        ).pad_input_tokens(input_ids, mm_inputs)

    def _get_precomputed_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        embed_param = next(self.model.embed_tokens.parameters())
        features = []
        for item in items:
            feature = item.precomputed_embeddings
            if feature is None:
                feature = item.feature
            if isinstance(feature, list):
                features.extend(feature)
            else:
                features.append(feature)
        return torch.cat(
            [
                feature.to(
                    device=embed_param.device,
                    dtype=embed_param.dtype,
                    non_blocking=True,
                )
                for feature in features
            ],
            dim=0,
        )

    get_image_feature = _get_precomputed_feature
    get_audio_feature = _get_precomputed_feature

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank and input_embeds is None:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                data_embedding_funcs={
                    Modality.IMAGE: self.get_image_feature,
                    Modality.AUDIO: self.get_audio_feature,
                },
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def post_load_weights(self, is_nextn=False, weight_names=None):

        # Perform post-processing after loading weights
        if is_nextn:
            # K MTP heads — one self_attn per head module.
            num_nextn_layers = self.config.num_nextn_predict_layers
            head_attns = [
                self.model.heads[k].decoder.self_attn for k in range(num_nextn_layers)
            ]
        else:
            if weight_names is None:
                layer_ids = range(self.model.start_layer, self.model.end_layer)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if layer_id < self.config.num_hidden_layers:
                            layer_ids.add(layer_id)
            head_attns = None  # not used for non-nextn path

        # Iterate either over base-model layer ids OR over the K MTP head self_attns.
        iterator = head_attns if is_nextn else layer_ids

        for item in iterator:
            self_attn = item if is_nextn else self.model.layers[item].self_attn
            w = self_attn.kv_b_proj.weight
            # Only two kv_b_proj formats are supported:
            # 1) BF16 weights.
            # 2) FP8 block-wise weights with block_size=(128, 128), consumed by DeepGEMM grouped kernels.
            weight_block_size = None
            block_scale = None

            if w.dtype == torch.float8_e4m3fn:
                assert (
                    self.quant_config is not None
                    and self.quant_config.weight_block_size is not None
                ), "Dots3 MLA kv_b_proj only supports FP8 block quantization with weight_block_size=(128, 128)."
                weight_block_size = tuple(self.quant_config.weight_block_size)
                assert weight_block_size == (
                    128,
                    128,
                ), f"Dots3 MLA kv_b_proj only supports FP8 block_size=(128, 128), got {weight_block_size}."
                block_scale = self_attn.kv_b_proj.weight_scale_inv

                if not (
                    self_attn.qk_nope_head_dim % weight_block_size[0] == 0
                    and self_attn.v_head_dim % weight_block_size[0] == 0
                    and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                ):
                    # NOTE(xiaozhi): At this point, we do not requant but change to use BF16.
                    # The issue is that when multiplying w_kc the reduction dim can not be divided
                    # by 128. May modify the DeepGEMM kernel to support this.
                    w = block_quant_dequant(
                        w,
                        block_scale,
                        weight_block_size,
                        torch.bfloat16,
                    )
            else:
                assert (
                    w.dtype == torch.bfloat16
                ), f"Dots3 MLA kv_b_proj only supports BF16 or FP8(128x128), got dtype={w.dtype}."

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            if w.dtype == torch.bfloat16:
                self_attn.w_kc = bind_or_assign(self_attn.w_kc, w_kc)
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.transpose(1, 2))
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[0]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign(
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign(
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc)

        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and self.quant_config is not None
            and self.quant_config.weight_block_size is not None
        ):
            self._weight_requant_ue8m0(is_nextn)

    def _weight_requant_ue8m0(self, is_nextn=False):
        weight_block_size = self.quant_config.weight_block_size

        moe_layers = list(
            range(
                self.config.first_k_dense_replace,
                self.config.num_hidden_layers,
                self.config.moe_layer_freq,
            )
        )

        num_nextn_layers = self.config.num_nextn_predict_layers
        num_hidden_layers = (
            num_nextn_layers if is_nextn else self.config.num_hidden_layers
        )
        for layer_id in range(num_hidden_layers):
            if is_nextn:
                layer = self.model.heads[layer_id].decoder
            else:
                layer = self.model.layers[layer_id]

            for module in [
                layer.self_attn.fused_qkv_a_g_proj_with_mqa,
                layer.self_attn.q_b_proj,
                layer.self_attn.kv_b_proj,
                layer.self_attn.o_proj,
            ]:
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

            if layer_id in moe_layers or is_nextn:
                if (
                    isinstance(layer.mlp, Dots3MoE)
                    and layer.mlp.shared_experts is not None
                ):
                    shared_experts = layer.mlp.shared_experts
                    for module in [
                        shared_experts.gate_up_proj,
                        shared_experts.down_proj,
                    ]:
                        requant_weight_ue8m0_inplace(
                            module.weight, module.weight_scale_inv, weight_block_size
                        )

                experts = layer.mlp.experts
                if isinstance(experts, DeepEPMoE):
                    for w in [
                        experts.w13_weight_fp8,
                        experts.w2_weight_fp8,
                    ]:
                        requant_weight_ue8m0_inplace(w[0], w[1], weight_block_size)
            else:
                mlp = layer.mlp
                assert isinstance(mlp, Dots3MLP)
                for module in [
                    mlp.gate_up_proj,
                    mlp.down_proj,
                ]:
                    requant_weight_ue8m0_inplace(
                        module.weight, module.weight_scale_inv, weight_block_size
                    )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn=False,
        extra_params_mapping=None,
    ):

        if is_nextn:
            num_nextn_layers = self.config.num_nextn_predict_layers
            # compatible with old design: when the main model has only 1 layer,
            # the MTP layer is at id 0; otherwise it starts at num_hidden_layers
            # and spans num_nextn_layers consecutive layer ids.
            base_nextn_layer_id = (
                0
                if self.config.num_hidden_layers == 1
                else self.config.num_hidden_layers
            )
            # Set of source layer prefixes for K MTP heads.
            nextn_layer_ids = list(
                range(base_nextn_layer_id, base_nextn_layer_id + num_nextn_layers)
            )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        if extra_params_mapping is not None:
            stacked_params_mapping.extend(extra_params_mapping)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Map per-expert input scales used by mixed-precision checkpoints.
        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # Always fuse q_a_proj/kv_a_proj_with_mqa/g_proj when loading Dots3.
        fuse_qkv_a_g_proj = True
        assert (
            self.config.q_lora_rank is not None
        ), "Dots3 requires q_lora_rank to enable fused_qkv_a_g_proj_with_mqa loading."
        cached_a_proj = {} if fuse_qkv_a_g_proj else None
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        def shard_g_proj_for_attention_tp(
            weight: torch.Tensor, cat_dim: int, is_scale: bool
        ):
            assert (
                weight.ndim > cat_dim
            ), f"weight.ndim={weight.ndim}, cat_dim={cat_dim}"
            dim_size = weight.shape[cat_dim]
            if not is_scale:
                assert dim_size % attn_tp_size == 0, (
                    f"dim_size={dim_size} is not divisible by attn_tp_size="
                    f"{attn_tp_size}"
                )
            start_idx = (attn_tp_rank * dim_size) // attn_tp_size
            end_idx = ceil_div((attn_tp_rank + 1) * dim_size, attn_tp_size)
            return weight.narrow(cat_dim, start_idx, end_idx - start_idx)

        def align_tensor(
            weight: torch.Tensor,
            cat_dim: int,
            align_size: int,
        ) -> torch.Tensor:
            original_size = weight.shape[cat_dim]
            aligned_size = ceil_align(original_size, align_size)
            if aligned_size == original_size:
                return weight
            pad_shape = list(weight.shape)
            pad_shape[cat_dim] = aligned_size - original_size
            return torch.cat([weight, weight.new_zeros(pad_shape)], dim=cat_dim)

        def _try_load_fused_qkv(
            q_a_proj_name: str,
            kv_a_proj_name: str,
            g_proj_name: str,
            param_name: str,
            cat_dim: int,
        ) -> bool:
            if not (
                q_a_proj_name in cached_a_proj
                and kv_a_proj_name in cached_a_proj
                and g_proj_name in cached_a_proj
            ):
                return False
            if param_name not in params_dict:
                raise ValueError(f"{param_name} not found in params_dict.")
            param = params_dict[param_name]
            target_dim = param.shape[cat_dim] if param.dim() > cat_dim else None
            is_scale = param_name.endswith(".weight_scale_inv")
            q_a_proj_weight = cached_a_proj[q_a_proj_name]
            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
            g_proj_weight = cached_a_proj[g_proj_name]
            g_proj_shard = shard_g_proj_for_attention_tp(
                g_proj_weight, cat_dim, is_scale
            )
            scale_block_n = _get_scale_block_n(self.quant_config)
            kv_a_proj_weight_aligned = (
                align_tensor(kv_a_proj_weight, cat_dim, scale_block_n)
                if not is_scale
                else kv_a_proj_weight
            )
            g_proj_shard_aligned = (
                align_tensor(g_proj_shard, cat_dim, max(8, scale_block_n))
                if not is_scale
                else g_proj_shard
            )
            fused_parts = [
                q_a_proj_weight,
                kv_a_proj_weight_aligned,
                g_proj_shard_aligned,
            ]
            fused_weight = torch.cat(fused_parts, dim=cat_dim)
            fused_dim = fused_weight.shape[cat_dim]
            if target_dim is not None and fused_dim != target_dim:
                raise ValueError(
                    f"Cannot match fused_qkv_a_g_proj_with_mqa shape for {param_name}: "
                    f"target_dim={target_dim}, fused_dim={fused_dim}."
                )
            weight_loader = _get_param_weight_loader(param)
            futures.append(executor.submit(weight_loader, param, fused_weight))
            cached_a_proj.pop(q_a_proj_name)
            cached_a_proj.pop(kv_a_proj_name)
            cached_a_proj.pop(g_proj_name)
            return True

        if is_nextn:
            nextn_layer_prefixes = [f"model.layers.{lid}" for lid in nextn_layer_ids]
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

            # Map source layer id -> head index (0..num_nextn_layers-1).
            def _match_nextn_prefix(name: str):
                for k, p in enumerate(nextn_layer_prefixes):
                    # require dot-boundary: layer 30 vs 300
                    if name == p or name.startswith(p + "."):
                        return k, p
                return None, None

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            pending_indexer_wk = {}
            weight_names = []
            for name, loaded_weight in weights:
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and not is_nextn
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)

                if not is_nextn:
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
                    # The NextN draft model owns model.mtp.* weights.
                    if num_nextn_layers > 0 and name.startswith("model.mtp."):
                        continue
                else:
                    # Remap the MTP-specific embedding into the draft model.
                    if name == "model.mtp.embed_tokens.weight":
                        name = "model.embed_tokens.weight"
                    elif name == "model.embed_tokens.weight":
                        # The target model owns the main embedding.
                        continue
                    else:
                        head_idx, matched_prefix = _match_nextn_prefix(name)
                        if matched_prefix is None:
                            continue
                        # Remap an MTP head into the draft model's shared head.
                        if "shared_head.head" in name:
                            name = name.replace(
                                matched_prefix, "model.shared_head.head"
                            )
                            param = params_dict.get(name)
                            if param is None:
                                continue
                            weight_loader = _get_param_weight_loader(param)
                            futures.append(
                                executor.submit(weight_loader, param, loaded_weight)
                            )
                            continue

                        is_decoder = True
                        # nextn-specific adapters (enorm/hnorm/eh_proj/shared_head.norm)
                        for weight_name in nextn_spec_weight_names:
                            if weight_name in name:
                                name = name.replace(
                                    matched_prefix, f"model.heads.{head_idx}"
                                )
                                is_decoder = False
                                break
                        # MTP transformer block weights — go under
                        # model.heads.{k}.decoder.* (with shared block for "block"/"full"
                        # sharing the duplicate writes converge on the same parameter).
                        if is_decoder:
                            name = name.replace(
                                matched_prefix, f"model.heads.{head_idx}.decoder"
                            )

                if "rotary_emb.inv_freq" in name:
                    continue

                if (
                    ".indexer.wk." in name or ".indexer.weights_proj." in name
                ) and _load_fused_indexer_wk(
                    name,
                    loaded_weight,
                    params_dict,
                    pending_indexer_wk,
                    self.quant_config,
                ):
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight, shard_id)
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        futures.append(
                            executor.submit(
                                weight_loader,
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # Skip loading embed_tokens if not first rank in pipeline parallelism
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        # Skip loading norm if not last rank in pipeline parallelism
                        if ".norm." in name and not self.pp_group.is_last_rank:
                            continue
                        if (
                            fuse_qkv_a_g_proj
                            and ".self_attn." in name
                            and (
                                "q_a_proj" in name
                                or "kv_a_proj_with_mqa" in name
                                or "g_proj" in name
                            )
                        ):
                            cached_a_proj[name] = loaded_weight
                            if "q_a_proj" in name:
                                q_a_proj_name = name
                                kv_a_proj_name = name.replace(
                                    "q_a_proj", "kv_a_proj_with_mqa"
                                )
                                g_proj_name = name.replace("q_a_proj", "g_proj")
                                param_name = name.replace(
                                    "q_a_proj", "fused_qkv_a_g_proj_with_mqa"
                                )
                            elif "kv_a_proj_with_mqa" in name:
                                q_a_proj_name = name.replace(
                                    "kv_a_proj_with_mqa", "q_a_proj"
                                )
                                kv_a_proj_name = name
                                g_proj_name = name.replace(
                                    "kv_a_proj_with_mqa", "g_proj"
                                )
                                param_name = name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_g_proj_with_mqa"
                                )
                            else:
                                q_a_proj_name = name.replace("g_proj", "q_a_proj")
                                kv_a_proj_name = name.replace(
                                    "g_proj", "kv_a_proj_with_mqa"
                                )
                                g_proj_name = name
                                param_name = name.replace(
                                    "g_proj", "fused_qkv_a_g_proj_with_mqa"
                                )

                            cat_dim = 0
                            if self.quant_config is not None and (
                                self.quant_config.get_name() == "awq"
                                or self.quant_config.get_name() == "awq_marlin"
                                or self.quant_config.get_name() == "moe_wna16"
                            ):
                                cat_dim = 1
                            _try_load_fused_qkv(
                                q_a_proj_name,
                                kv_a_proj_name,
                                g_proj_name,
                                param_name,
                                cat_dim,
                            )
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = _get_param_weight_loader(param)
                            futures.append(
                                executor.submit(weight_loader, param, loaded_weight)
                            )

            if fuse_qkv_a_g_proj and cached_a_proj:
                unresolved = sorted(cached_a_proj.keys())
                preview = ", ".join(unresolved[:6])
                extra = f" (+{len(unresolved) - 6} more)" if len(unresolved) > 6 else ""
                raise ValueError(
                    "Unresolved fused q/kv/g projection weights while loading "
                    "fused_qkv_a_g_proj_with_mqa. Missing counterparts or unexpected "
                    f"names: {preview}{extra}"
                )
            if pending_indexer_wk:
                unresolved = ", ".join(sorted(pending_indexer_wk.keys())[:6])
                raise ValueError(
                    "Incomplete native DSA Indexer wk weights: " + unresolved
                )
            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        # Share target embeddings and output head with the draft model.
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


class DotsNoteOmniThinkerForConditionalGeneration(nn.Module):
    """Dots thinker with in-process audio, vision, and language submodels."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        from pathlib import Path

        from sglang.srt.models.dots3_common.dots_omni_towers import (
            DotsNoteOmniAudioEncoder,
            DotsNoteOmniVisionEncoder,
        )

        self.config = config
        self.pp_group = get_pp_group()
        model_dir = Path(config._name_or_path)
        self.language_model = Dots3LanguageModelForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        # Load multimodal towers only where embeddings are produced.
        if self.pp_group.is_first_rank and not config.language_only:
            self.audio_tower = DotsNoteOmniAudioEncoder(str(model_dir))
            self.visual = DotsNoteOmniVisionEncoder(str(model_dir))
        else:
            self.audio_tower = None
            self.visual = None

    @property
    def model(self):
        return self.language_model.model

    @property
    def lm_head(self):
        return self.language_model.lm_head

    @property
    def logits_processor(self):
        return self.language_model.logits_processor

    @property
    def routed_experts_weights_of_layer(self):
        return self.language_model.routed_experts_weights_of_layer

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pad_input_ids(self, input_ids, mm_inputs, **kwargs):
        return self.language_model.pad_input_ids(input_ids, mm_inputs, **kwargs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert self.visual is not None
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=self.visual.device,
            dtype=self.visual.dtype,
            non_blocking=True,
        )
        grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0).to(
            device=self.visual.device,
            non_blocking=True,
        )
        return self.visual(pixel_values, grid_thw=grid_thw)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert self.audio_tower is not None
        waveforms = [
            item.feature.to(
                device=self.audio_tower.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for item in items
        ]
        lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            dtype=torch.long,
        )
        features, token_lengths = self.audio_tower(waveforms, lengths)
        expected = sum(
            sum(end - start + 1 for start, end in item.offsets) for item in items
        )
        if features.shape[0] != expected or sum(token_lengths) != expected:
            raise RuntimeError(
                "Dots audio feature/token mismatch: "
                f"features={features.shape[0]}, tower_lengths={token_lengths}, "
                f"placeholders={expected}"
            )
        return features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank and input_embeds is None:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                language_model=self.language_model.model,
                multimodal_model=self,
                data_embedding_funcs={
                    Modality.IMAGE: self.get_image_feature,
                    Modality.AUDIO: self.get_audio_feature,
                },
                pp_proxy_tensors=pp_proxy_tensors,
            )
        else:
            hidden_states = self.language_model.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors,
            )

        if self.pp_group.is_last_rank:
            return self.language_model.logits_processor(
                input_ids,
                hidden_states,
                self.language_model.lm_head,
                forward_batch,
            )
        return hidden_states


class DotsNoteOmniForConditionalGeneration(nn.Module):
    """Native dots.note.omni conditional-generation model."""

    packed_modules_mapping = Dots3LanguageModelForCausalLM.packed_modules_mapping
    fall_back_to_pt_during_load = False

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return Dots3LanguageModelForCausalLM.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.thinker = DotsNoteOmniThinkerForConditionalGeneration(
            config,
            quant_config=quant_config,
            prefix=add_prefix("thinker", prefix),
        )
        language_model = self.thinker.language_model
        self.pp_group = language_model.pp_group
        self.tp_size = language_model.tp_size
        self.quant_config = language_model.quant_config
        self.num_fused_shared_experts = language_model.num_fused_shared_experts
        self.forward = self.thinker.forward
        self.pad_input_ids = self.thinker.pad_input_ids

    @property
    def model(self):
        return self.thinker.language_model.model

    @property
    def lm_head(self):
        return self.thinker.language_model.lm_head

    @property
    def logits_processor(self):
        return self.thinker.language_model.logits_processor

    @property
    def routed_experts_weights_of_layer(self):
        return self.thinker.language_model.routed_experts_weights_of_layer

    @property
    def start_layer(self):
        return self.thinker.language_model.start_layer

    @property
    def end_layer(self):
        return self.thinker.language_model.end_layer

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def load_weights(self, weights, *args, **kwargs):
        # Partition the flat checkpoint in one pass. This avoids reading the
        # large tower tensors once through the model loader and again through
        # safetensors.
        load_towers = self.thinker.visual is not None
        vision_state = {}
        audio_state = {}

        def language_weights():
            for name, weight in weights:
                if name.startswith("vision_encoder."):
                    if load_towers:
                        vision_state[name.removeprefix("vision_encoder.")] = weight
                    continue
                if name.startswith("audio_encoder."):
                    if load_towers:
                        audio_state[name.removeprefix("audio_encoder.")] = weight
                    continue
                yield name, weight

        self.thinker.language_model.load_weights(language_weights(), *args, **kwargs)
        if load_towers:
            self.thinker.visual.load_converted_state(vision_state)
            self.thinker.audio_tower.load_converted_state(audio_state)

    def post_load_weights(self, *args, **kwargs):
        return self.thinker.language_model.post_load_weights(*args, **kwargs)

    def get_embed_and_head(self):
        return self.thinker.language_model.get_embed_and_head()

    def set_embed_and_head(self, embed, head):
        return self.thinker.language_model.set_embed_and_head(embed, head)

    def get_attention_sliding_window_size(self):
        return self.thinker.language_model.get_attention_sliding_window_size()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return Dots3LanguageModelForCausalLM.get_model_config_for_expert_location(
            config
        )


class Dots3NoteForCausalLM(DotsNoteOmniForConditionalGeneration):
    """Canonical dots.note architecture exported by the flat checkpoint."""
