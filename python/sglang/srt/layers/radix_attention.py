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

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.layers.attention.graph_utils import (
    _attention_on_real_rows,
    _zero_padded_tokens,
    allocate_attention_outputs,
    attention_input_scope,
    padded_extend_real_tokens,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    get_prefill_raw_num_tokens,
    is_in_full_prefill_graph,
)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """String-valued attention types, compatible with torch.compile."""

    DECODER = "decoder"
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
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
        use_prefill_attention_wrapper: bool = True,
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
        self.use_prefill_attention_wrapper = use_prefill_attention_wrapper
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.q_scale_float = None
        self.idx_q_scale_float = None
        self.idx_k_scale_float = None
        self.idx_v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        **kwargs,
    ):
        if k is not None:
            assert v is not None
            k = k.view(
                -1,
                self.tp_k_head_num,
                self.v_head_dim if "k_rope" in kwargs else self.qk_head_dim,
            )
            if "k_rope" not in kwargs:
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        breakable_cg = is_in_breakable_cuda_graph()
        full_cg = is_in_full_prefill_graph()
        use_eager_attention = (
            self.use_prefill_attention_wrapper
            and forward_batch.forward_mode.is_extend()
            and (breakable_cg or full_cg)
        )
        # Sparse BCG already owns its backend-specific capture path.
        if use_eager_attention and not (
            breakable_cg and kwargs.get("idx_q") is not None
        ):
            is_sparse = kwargs.get("idx_q") is not None
            output, idx_output = allocate_attention_outputs(
                self, q, v, kwargs.get("idx_q")
            )
            lse = self._eager_attention(
                q,
                k,
                v,
                output,
                save_kv_cache,
                key_value_num_tokens,
                idx_output,
                **kwargs,
            )
            if is_sparse:
                return idx_output, output
            if kwargs.get("return_lse") or forward_batch.mha_return_lse:
                return output.view(-1, self.tp_q_head_num, self.v_head_dim), lse
            return output
        real_num_tokens = padded_extend_real_tokens(q, forward_batch)
        if real_num_tokens is not None:
            return _attention_on_real_rows(
                self,
                real_num_tokens,
                q,
                k,
                v,
                forward_batch,
                save_kv_cache,
                key_value_num_tokens=key_value_num_tokens,
                **kwargs,
            )
        return get_attn_backend().forward(
            q, k, v, self, forward_batch, save_kv_cache, **kwargs
        )

    @eager_on_graph
    def _eager_attention(
        self,
        q,
        k,
        v,
        output,
        save_kv_cache: bool = True,
        key_value_num_tokens: Optional[int] = None,
        idx_output=None,
        **kwargs,
    ):
        """Run real tokens, retaining padded outputs for the next graph segment."""
        forward_batch = get_forward_batch()
        rows = output.shape[0]
        n = forward_batch.global_num_token_non_padded_cpu
        kv_n = n if key_value_num_tokens is None else key_value_num_tokens
        return_lse = bool(kwargs.get("return_lse") or forward_batch.mha_return_lse)
        lse = None
        if n == 0:
            output.zero_()
            if idx_output is not None:
                idx_output.zero_()
            if return_lse:
                lse = q.new_zeros((rows, q.shape[1]), dtype=torch.float32)
        else:
            with attention_input_scope(
                forward_batch, output, n, kv_n, kwargs
            ) as kwargs:
                result = get_attn_backend().forward(
                    q[:n],
                    k[:kv_n] if k is not None else None,
                    v[:kv_n] if v is not None else None,
                    self,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
            if idx_output is not None:
                idx_result, result = result
                if idx_result is not None:
                    idx_output[:n].view(idx_result.shape).copy_(idx_result)
            elif return_lse:
                result, lse, *_ = result
            if result.data_ptr() != output.data_ptr():
                output[:n].view(result.shape).copy_(result)
            raw = get_prefill_raw_num_tokens()
            _zero_padded_tokens(output, raw)
            if idx_output is not None:
                _zero_padded_tokens(idx_output, raw)
            if lse is not None and lse.shape[0] != rows:
                padded = lse.new_zeros((rows, *lse.shape[1:]))
                padded[:n].copy_(lse)
                lse = padded
        return lse
