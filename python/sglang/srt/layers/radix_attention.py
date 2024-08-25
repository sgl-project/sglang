"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Radix attention."""

from typing import Optional

import torch
from flashinfer.cascade import merge_state
from torch import nn

from sglang.global_config import global_config
from sglang.srt.layers.decode_attention import decode_attention_fwd
from sglang.srt.layers.extend_attention import extend_attention_fwd
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from sglang.srt.model_executor.model_runner import global_server_args_dict


class RadixAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        sliding_window_size: Optional[int] = None,
        logit_cap: int = -1,
        v_head_dim: int = -1,
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
        self.sliding_window_size = sliding_window_size if sliding_window_size else -1

        if (
            not global_server_args_dict.get("disable_flashinfer", False)
            and self.qk_head_dim == self.v_head_dim
        ):
            self.extend_forward = self.extend_forward_flashinfer
            self.decode_forward = self.decode_forward_flashinfer
        else:
            self.extend_forward = self.extend_forward_triton
            self.decode_forward = self.decode_forward_triton

        self.logit_cap = logit_cap if logit_cap is not None and logit_cap > 0 else 0

    def extend_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        if self.qk_head_dim != self.v_head_dim:
            o = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        else:
            o = torch.empty_like(q)

        self.store_kv_cache(k, v, input_metadata)
        extend_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, self.tp_q_head_num, self.v_head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.triton_start_loc,
            input_metadata.seq_lens,
            input_metadata.triton_prefix_lens,
            input_metadata.extend_start_loc,
            input_metadata.extend_seq_lens,
            input_metadata.triton_max_seq_len,
            input_metadata.triton_max_extend_len,
            sm_scale=self.scaling,
            logit_cap=self.logit_cap,
        )

        return o

    def decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        if self.qk_head_dim != self.v_head_dim:
            o = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        else:
            o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)

        decode_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.qk_head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.v_head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.triton_start_loc,
            input_metadata.seq_lens,
            input_metadata.triton_max_seq_len,
            input_metadata.total_num_tokens,
            sm_scale=self.scaling,
            logit_cap=self.logit_cap,
        )

        return o

    def extend_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        # using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        prefill_wrapper_paged = input_metadata.flashinfer_prefill_wrapper_paged
        if self.sliding_window_size != -1:
            prefill_wrapper_paged = prefill_wrapper_paged[0]
        else:
            if isinstance(prefill_wrapper_paged, list):
                prefill_wrapper_paged = prefill_wrapper_paged[1]

        if not input_metadata.flashinfer_use_ragged:
            if k is not None:
                assert v is not None
                self.store_kv_cache(k, v, input_metadata)

            o = prefill_wrapper_paged.forward(
                q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                causal=True,
                sm_scale=self.scaling,
                window_left=self.sliding_window_size,
                logits_soft_cap=self.logit_cap,
            )
        else:
            o1, s1 = (
                input_metadata.flashinfer_prefill_wrapper_ragged.forward_return_lse(
                    q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                    k.contiguous().view(-1, self.tp_k_head_num, self.head_dim),
                    v.contiguous().view(-1, self.tp_v_head_num, self.head_dim),
                    causal=True,
                    sm_scale=self.scaling,
                    logits_soft_cap=self.logit_cap,
                )
            )

            if input_metadata.extend_no_prefix:
                o = o1
            else:
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                    input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                    causal=False,
                    sm_scale=self.scaling,
                    logits_soft_cap=self.logit_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            self.store_kv_cache(k, v, input_metadata)

            if input_metadata.total_num_tokens >= global_config.layer_sync_threshold:
                torch.cuda.synchronize()

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def decode_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        decode_wrapper = input_metadata.flashinfer_decode_wrapper
        if self.sliding_window_size != -1:
            decode_wrapper = decode_wrapper[0]
        else:
            if isinstance(decode_wrapper, list):
                decode_wrapper = decode_wrapper[1]

        if k is not None:
            assert v is not None
            self.store_kv_cache(k, v, input_metadata)

        o = decode_wrapper.forward(
            q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
            sm_scale=self.scaling,
            logits_soft_cap=self.logit_cap,
        )

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def forward(self, q, k, v, input_metadata: InputMetadata):
        if k is not None:
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

        if input_metadata.forward_mode == ForwardMode.EXTEND:
            return self.extend_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.decode_forward(q, k, v, input_metadata)

    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        input_metadata.token_to_kv_pool.set_kv_buffer(
            self.layer_id, input_metadata.out_cache_loc, cache_k, cache_v
        )
