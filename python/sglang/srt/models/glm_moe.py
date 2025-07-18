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

# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only GLMMoe model compatible with THUDM weights."""
import logging
import os
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
from transformers import PretrainedConfig

from sglang.srt.configs import ChatGLMConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_reduce_scatter,
    dp_gather_partial,
    dp_scatter,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE, get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.expert_distribution import ExpertDistributionRecorder
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    add_prefix,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    log_info_on_rank0,
)

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2

    from sglang.srt.layers.quantization.deep_gemm import (
        grouped_gemm_nt_f8f8bf16_masked as deep_gemm_grouped_gemm_nt_f8f8bf16_masked,
    )
else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )

expert_distribution_recorder = ExpertDistributionRecorder()

logger = logging.getLogger(__name__)

LoraConfig = None


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        ffn_hidden_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear
        self.ffn_hidden_size = ffn_hidden_size
        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [self.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
            prefix=add_prefix("dense_h_to_4h", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            self.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense_4h_to_h", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states, forward_batch: Optional[ForwardBatch] = None):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.act_fn(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.num_moe_experts, config.hidden_size))
        )
        if config.moe_router_load_balancing_type in ("logit_bias", "score_bias"):
            self.router_bias = nn.Parameter(torch.zeros(config.num_moe_experts))
        else:
            self.router_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


def is_non_idle_and_non_empty(forward_mode, hidden_states):
    return (
        (forward_mode is not None)
        and not forward_mode.is_idle()
        and hidden_states.shape[0] > 0
    )


class GLMMoE(nn.Module):
    # referred to DeepseekV2MoE

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.moe_router_topk
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.moe_num_shared_experts
        self.n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]

        if self.tp_size > config.num_moe_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_moe_experts}."
            )

        self.router = MoEGate(config=config, prefix=add_prefix("router", prefix))

        self.experts = get_moe_impl_class()(
            num_experts=config.num_moe_experts + self.n_share_experts_fusion,
            top_k=config.moe_router_topk + min(self.n_share_experts_fusion, 1),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=getattr(config, "n_group", None) is not None,
            num_expert_group=getattr(config, "n_group", None),
            topk_group=getattr(config, "topk_group", None),
            correction_bias=self.router.router_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        self.use_shared_experts = (
            config.moe_num_shared_experts is not None
            and config.moe_num_shared_experts > 0
        )
        if self.use_shared_experts:
            intermediate_size = (
                config.moe_ffn_hidden_size * config.moe_num_shared_experts
            )
            self.shared_experts = GLMMLP(
                config=config,
                ffn_hidden_size=intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if global_server_args_dict["enable_deepep_moe"]
                    else {}
                ),
            )
        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = config.num_moe_experts
            self.renormalize = config.norm_topk_prob
            self.topk_group = getattr(config, "n_group", None)
            self.num_expert_group = getattr(config, "topk_group", None)
            self.correction_bias = (
                self.router.router_bias.data
                if self.router.router_bias is not None
                else None
            )

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=config.num_moe_experts,
                num_local_experts=config.num_moe_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    @property
    def _enable_deepep_moe(self):
        return global_server_args_dict["enable_deepep_moe"]

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        forward_mode = forward_batch.forward_mode
        if (not self._enable_deepep_moe) or is_non_idle_and_non_empty(
            forward_mode, hidden_states
        ):
            # router_logits: (num_tokens, n_experts)

            # TODO: Support fp32 router
            # hidden_states_dtype = hidden_states.dtype
            # hidden_states = hidden_states.to(torch.float32)
            # router_logits = self.router(hidden_states)
            # hidden_states = hidden_states.to(hidden_states_dtype)

            router_logits = self.router(hidden_states)
        else:
            router_logits = None

        if (self.n_share_experts_fusion == 0) and (
            (not self._enable_deepep_moe)
            or is_non_idle_and_non_empty(forward_mode, hidden_states)
        ):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        if self._enable_deepep_moe and (router_logits is not None):
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=self.num_expert_group is not None,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                num_token_non_padded=forward_batch.num_token_non_padded,
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

        if self._enable_deepep_moe and (self.ep_size > 1):
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=forward_mode,
            )

        if self._enable_deepep_moe:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr,
                masked_m=masked_m,
                expected_m=expected_m,
                num_recv_tokens_per_expert=num_recv_tokens_per_expert,
                forward_mode=forward_mode,
            )
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self._enable_deepep_moe and (self.ep_size > 1):
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_mode,
            )

        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if (not self._enable_deepep_moe) and (self.tp_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states


class GLMAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        num_heads = config.num_attention_heads
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.multi_query_attention = config.multi_query_attention
        self.num_kv_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        )
        if self.num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.num_kv_heads == 0
        self.num_local_kv_heads = max(1, self.num_kv_heads // attn_tp_size)
        self.head_dim = config.kv_channels
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.dense = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        self.use_qk_layernorm = getattr(config, "qk_layernorm", False)
        if self.use_qk_layernorm:
            self.q_layernorm = layer_norm_func(
                config.kv_channels, eps=config.layernorm_epsilon
            )
            self.k_layernorm = layer_norm_func(
                config.kv_channels, eps=config.layernorm_epsilon
            )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        rotary_percent = getattr(config, "rotary_percent", 0.5)
        rotary_dim = int(self.head_dim * rotary_percent)
        is_neox_style = not getattr(config, "original_rope", True)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            assert (
                not self.dense.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states

        return self.forward_normal(positions, hidden_states, forward_batch)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_layernorm:
            orgin_q_shape, origin_k_shape = q.shape, k.shape
            q = self.q_layernorm(q.contiguous().view(-1, self.head_dim)).view(
                orgin_q_shape
            )
            k = self.k_layernorm(k.contiguous().view(-1, self.head_dim)).view(
                origin_k_shape
            )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.dense(attn_output)
        return output


class _FFNInputMode(Enum):
    # The MLP sublayer requires 1/tp_size tokens as input
    SCATTERED = auto()
    # The MLP sublayer requires all tokens as input
    FULL = auto()


@dataclass
class _DecoderLayerInfo:
    is_sparse: bool
    ffn_input_mode: _FFNInputMode


class GLMDecoderLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

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
        assert (
            not config.apply_residual_connection_post_layernorm
        ), "apply_residual_connection_post_layernorm is not supported in GLMDecoderLayer."

        self.fp32_residual_connection = config.fp32_residual_connection

        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.layer_id = layer_id
        self.local_dp_size = get_local_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        # Self attention.
        self.self_attention = GLMAttention(
            config=config,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attention", prefix),
            alt_stream=alt_stream,
        )

        self.info = self._compute_info(config, layer_id=layer_id)
        previous_layer_info = self._compute_info(config, layer_id=layer_id - 1)

        if self.info.is_sparse:
            self.mlp = GLMMoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            # TODO (zilin): currently we hardcode the tp of dense layers 1 to run
            # multiple-node inference. This should be fixed when we upgrade sglang
            # to a version which supports --moe-dense-tp-size correctly.
            if self._enable_moe_dense_fully_dp() or True:
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = GLMMLP(
                config=config,
                ffn_hidden_size=config.ffn_hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_is_scattered = (
            layer_id > 0
            and previous_layer_info.ffn_input_mode == _FFNInputMode.SCATTERED
        )
        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        self.input_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon
        )

        self.use_post_self_attn_layernorm = getattr(
            config, "post_self_attn_layernorm", False
        )
        if self.use_post_self_attn_layernorm:
            self.post_self_attn_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon
            )
        self.use_post_mlp_layernorm = getattr(config, "post_mlp_layernorm", False)
        if self.use_post_mlp_layernorm:
            self.post_mlp_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon
            )

    @staticmethod
    def _enable_moe_dense_fully_dp():
        return global_server_args_dict["moe_dense_tp_size"] == 1

    @staticmethod
    def _compute_info(config: PretrainedConfig, layer_id: int):
        is_sparse = (
            config.num_moe_experts is not None
            and layer_id >= config.moe_num_first_dense_layers
            and layer_id % config.moe_layer_freq == 0
        )
        ffn_input_mode = (
            _FFNInputMode.SCATTERED
            if (global_server_args_dict["enable_deepep_moe"] and is_sparse)
            or (GLMDecoderLayer._enable_moe_dense_fully_dp() and not is_sparse)
            else _FFNInputMode.FULL
        )
        return _DecoderLayerInfo(is_sparse=is_sparse, ffn_input_mode=ffn_input_mode)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.info.ffn_input_mode == _FFNInputMode.SCATTERED:
            return self.forward_ffn_with_scattered_input(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        elif self.info.ffn_input_mode == _FFNInputMode.FULL:
            return self.forward_ffn_with_full_input(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        else:
            raise NotImplementedError

    def forward_ffn_with_full_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            assert not (
                self.attn_tp_size != 1 and self.input_is_scattered
            ), "moe_layer_freq > 1 is not supported when attn_tp_size > 1"

            # Self Attention
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            if self.use_post_self_attn_layernorm:
                hidden_states = self.post_self_attn_layernorm(hidden_states)

        # Gather
        if get_tensor_model_parallel_world_size() > 1:
            # all gather and all reduce
            if self.local_dp_size != 1:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                hidden_states, local_hidden_states = (
                    forward_batch.gathered_buffer,
                    hidden_states,
                )
                dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                dp_scatter(residual, hidden_states, forward_batch)
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        hidden_states = self.mlp(hidden_states, forward_batch)
        if hidden_states.shape[0] != 0 and self.use_post_mlp_layernorm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        # TODO(ch-wan): use reduce-scatter in MLP to avoid this scatter
        # Scatter
        if self.local_dp_size != 1:
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        return hidden_states, residual

    def forward_ffn_with_scattered_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.attn_tp_size != 1 and self.input_is_scattered:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        # Self Attention
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            if self.use_post_self_attn_layernorm:
                attention_output = self.post_self_attn_layernorm(attention_output)

        if self.attn_tp_size != 1:
            if self.input_is_scattered:
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                attn_tp_reduce_scatter(hidden_states, tensor_list)
                if hidden_states.shape[0] != 0:
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                if self.attn_tp_rank == 0:
                    hidden_states += residual
                tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                hidden_states = tensor_list[self.attn_tp_rank]
                attn_tp_reduce_scatter(hidden_states, tensor_list)
                residual = hidden_states
                if hidden_states.shape[0] != 0:
                    hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        if not (
            self._enable_moe_dense_fully_dp()
            and (not self.info.is_sparse)
            and hidden_states.shape[0] == 0
        ):
            hidden_states = self.mlp(hidden_states, forward_batch)
            if self.use_post_mlp_layernorm:
                hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.is_last_layer and self.attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(self.attn_tp_size)), local_hidden_states
            )

        return hidden_states, residual


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.post_layer_norm = config.post_layer_norm

        # Transformer layers.
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                GLMDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_layers)
            ]
        )

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon
            )

        self.dp_size = get_local_attention_dp_size()

    def forward(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            expert_distribution_recorder.set_current_layer(i)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        if not forward_batch.forward_mode.is_idle():
            if self.post_layer_norm:
                if residual is None:
                    hidden_states = self.final_layernorm(hidden_states)
                else:
                    hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class GLMMoeM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.embedding = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            prefix=add_prefix("embedding", prefix),
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )

        self.encoder = GLMTransformer(
            config, quant_config, add_prefix("encoder", prefix)
        )

        self.output_layer = ParallelLMHead(
            config.padded_vocab_size,
            config.hidden_size,
            prefix=add_prefix("output_layer", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids)

        hidden_states = self.encoder(positions, forward_batch, inputs_embeds)

        return hidden_states


class GLMMoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: ChatGLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config: ChatGLMConfig = config
        self.quant_config = quant_config
        self.max_position_embeddings = getattr(config, "max_sequence_length", 8192)
        self.transformer = GLMMoeM(
            config, quant_config, prefix=add_prefix("transformer", prefix)
        )
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(config)
        self.dp_size = get_local_attention_dp_size()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        hidden_states = self.transformer(input_ids, positions, forward_batch)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_moe_experts,
        )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if "rotary_pos_emb.inv_freq" in name:
                    continue
                if "word_embeddings" in name:
                    name = name.replace(".word_embeddings", "")

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


class GLMMoeModel(GLMMoeForCausalLM):
    pass


EntryClass = [GLMMoeModel]
