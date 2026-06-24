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
"""Radix attention."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

        if os.environ.get("SGLANG_GEMMA_KV_GEOMETRY") == "1":
            logger.info(
                "SGLANG_GEMMA_KV_GEOMETRY layer=%s heads=%s kv_heads=%s "
                "head_dim=%s v_head_dim=%s sliding_window=%s attn_type=%s",
                self.layer_id,
                self.tp_q_head_num,
                self.tp_k_head_num,
                self.qk_head_dim,
                self.v_head_dim,
                self.sliding_window_size,
                self.attn_type.value,
            )

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        if (
            forward_batch.forward_mode.is_extend()
            and get_tc_piecewise_forward_context() is not None
        ):
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                breakable_unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output
        else:
            return get_attn_backend().forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    # MLA / TRT-LLM / NSA paths pass these through RadixAttention.forward(**kwargs);
    # they must appear in the schema when --enforce-piecewise-cuda-graph is on.
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> None:
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]

    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:real_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks
    if cos_sin_cache is not None:
        kwargs["cos_sin_cache"] = cos_sin_cache
    if is_neox is not None:
        kwargs["is_neox"] = is_neox
    if llama_4_scaling is not None:
        kwargs["llama_4_scaling"] = llama_4_scaling
    if topk_indices is not None:
        kwargs["topk_indices"] = topk_indices[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    # Keep the original ForwardBatch object and only narrow cache locations for
    # this backend call so model/backend state is still written to the same batch.
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    # Store pre-allocated output for FA backend to write directly into.
    # Must slice to real_num_tokens to match the narrowed query shape —
    # the FA kernel validates out.size(0) == q.size(0).
    forward_batch._attn_output = output[:real_num_tokens]

    ret = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc

    if ret.data_ptr() != output.data_ptr():
        output[:real_num_tokens].view(ret.shape).copy_(ret)

    if _is_hip:
        # During PCG replay on AMD, varlen attention kernels only fill positions
        # 0..actual_tokens-1 and leave padded positions with uninitialized
        # garbage from torch.empty.  Zero these so garbage (NaN/Inf) does not
        # propagate through residual connections, MoE routing, and allreduce.
        # Use context.raw_num_tokens (pre-padding count from PCG runner)
        # instead of forward_batch.extend_num_tokens, because
        # extend_num_tokens is None for TARGET_VERIFY (EAGLE) batches.
        pcg_static_tokens = context.num_tokens
        actual_tokens = context.raw_num_tokens
        if (
            pcg_static_tokens is not None
            and actual_tokens is not None
            and pcg_static_tokens > actual_tokens
        ):
            first_dim = output.shape[0]
            elems_per_token = output.numel() // first_dim
            output.view(first_dim, elems_per_token)[actual_tokens:].zero_()
    return


breakable_unified_attention_with_output = eager_on_graph(True)(
    unified_attention_with_output
)
