from __future__ import annotations

"""
Support attention backend for flashMLA.

"""


from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


class FlashMLABackend(FlashInferMLAAttnBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        print("test Flashmla backend")
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )

        mla_metadata, mla_splits = get_mla_metadata(
            forward_batch.seq_lens.to(torch.int32),
            1 * self.num_q_heads // self.num_kv_heads,
            self.num_kv_heads,
        )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        reshape_q = q.view(
            forward_batch.batch_size, -1, layer.tp_q_head_num, layer.head_dim
        )
        o, _ = flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(
                -1, 64, 1, 576
            ),  # TODO num_blocks x page_block_size x num_heads_k x head_size
            block_table=forward_batch.out_cache_loc.to(torch.int32).view(
                forward_batch.batch_size, -1
            ),
            cache_seqlens=forward_batch.seq_lens.to(torch.int32),
            head_dim_v=512,  # TODO Retrieve from config.
            tile_scheduler_metadata=mla_metadata,
            num_splits=mla_splits,
            softmax_scale=layer.scaling,
            causal=False,
        )
        """
        #TODO
        1. Hard-code the reshaping of kv cache.
        2. Modifying the indexing of kv cache is also needed, but I haven't done that yet.
        3. The consistency of the output shape hasn't been confirmed yet.
        """
        return o
