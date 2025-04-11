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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_hpu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_hpu = is_hpu()
if _is_hpu:
    import vllm_hpu_extension.ops as ops
    from vllm_hpu_extension.utils import Matmul, ModuleFusedSDPA, Softmax


class HPUAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA)
        self.matmul_qk = Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul()
        self.batch2block_matmul = Matmul()
        self.block2batch_matmul = Matmul()

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        query = q.view(1, -1, layer.tp_q_head_num, layer.qk_head_dim)
        key = k.view(1, -1, layer.tp_k_head_num, layer.qk_head_dim)
        value = v.view(1, -1, layer.tp_v_head_num, layer.v_head_dim)

        output = ops.prompt_attention(
            impl="fsdpa",
            query=query,
            key=key,
            value=value,
            attn_bias=forward_batch.attn_bias,
            is_causal=False,
            p=0.0,
            scale=layer.scaling,
            matmul_qk_op=self.matmul_qk,
            softmax_op=self.softmax,
            matmul_av_op=self.matmul_av,
            fsdpa_op=self.fused_scaled_dot_product_attention,
        )
        output = output.reshape(q.shape)

        return output

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # Get key and value caches
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        query = q.view(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
        key_cache = key_cache.view(
            -1, forward_batch.page_size, layer.tp_k_head_num, layer.qk_head_dim
        )
        value_cache = value_cache.view(
            -1, forward_batch.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        if forward_batch.use_contiguous_pa:

            def fetch_key_cache(cache, blocks):
                return cache[: blocks.size(0)]

            def fetch_value_cache(cache, blocks):
                return cache[: blocks.size(0)]

        else:

            def fetch_key_cache(cache, blocks):
                return cache.index_select(0, blocks)

            def fetch_value_cache(cache, blocks):
                return cache.index_select(0, blocks)

        # Run paged attention decode
        output = ops.flat_pa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_list=forward_batch.block_list,
            block_mapping=forward_batch.block_mapping,
            block_bias=forward_batch.attn_bias,
            block_groups=forward_batch.block_groups,
            scale=layer.scaling,
            matmul_qk_op=self.matmul_qk,
            matmul_av_op=self.matmul_av,
            batch2block_matmul_op=self.batch2block_matmul,
            block2batch_matmul_op=self.block2batch_matmul,
            keys_fetch_func=fetch_key_cache,
            values_fetch_func=fetch_value_cache,
        )

        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
