from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
# from sglang.srt.managers.schedule_batch import global_server_args_dict
# from sglang.srt.mem_cache.memory_pool import SWAKVPool
# from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class ForwardMetaData:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    page_table: Optional[torch.Tensor] = None

class BlackwellPrefillAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        from sglang.srt.layers.attention.cute_ops.prefill_attention import flash_attn_varlen_func

        super().__init__()
        self.flash_attn_func = flash_attn_varlen_func
        self.page_size = model_runner.page_size
        self.device = model_runner.device
        self.forward_metadata: Optional[ForwardMetaData] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_extend(), "Only support extend (i.e., prefill) batches."

        max_seqlen_k = forward_batch.seq_lens_cpu.max().item()
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), pad=(1, 0))
        page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :max_seqlen_k]

        if any(forward_batch.extend_prefix_lens_cpu):
            extend_seq_lens = forward_batch.extend_seq_lens
            cu_seqlens_q = F.pad(torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), pad=(1, 0))
        else:
            cu_seqlens_q = cu_seqlens_k

        self.forward_metadata = ForwardMetaData(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)

        metadata = self.forward_metadata
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k = k_cache[metadata.page_table, :, :, :].reshape(-1, layer.tp_k_head_num, layer.head_dim)
        v = v_cache[metadata.page_table, :, :, :].reshape(-1, layer.tp_v_head_num, layer.head_dim)

        out = self.flash_attn_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=metadata.cu_seqlens_q,
            page_table=metadata.page_table,
            softcap=layer.logit_cap,
            softmax_scale=layer.scaling,
            window_size=(layer.sliding_window_size, 0),
            causal=True,
            learnable_sink=sinks.to(torch.bfloat16) if sinks is not None else None,
        )[0]

        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        raise NotImplementedError("BlackwellPrefillAttentionBackend does not support forward_decode")

    forward = forward_extend

    def support_triton(self):
        return False
