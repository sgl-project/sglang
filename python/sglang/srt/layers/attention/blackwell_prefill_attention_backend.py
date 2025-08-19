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

    # Sliding window
    # window_kv_indptr: torch.Tensor
    # window_kv_indices: torch.Tensor
    # window_num_kv_splits: torch.Tensor

class BlackwellPrefillAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        from sglang.srt.layers.attention.cute_ops.prefill_attention import flash_attn_varlen_func

        super().__init__()
        self.flash_attn_func = flash_attn_varlen_func
        self.page_size = model_runner.page_size
        self.device = model_runner.device
        self.forward_metadata: Optional[ForwardMetaData] = None
        # self.sliding_window_size = model_runner.sliding_window_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_extend(), "Only support extend (i.e., prefill) batches."

        max_seqlen_k = forward_batch.seq_lens_cpu.max().item()
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), pad=(1, 0))
        page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :max_seqlen_k]

        extending = False
        if any(forward_batch.extend_prefix_lens_cpu):
            extend_seq_lens = forward_batch.extend_seq_lens
            max_seqlen_q = max(forward_batch.extend_seq_lens_cpu)
            cu_seqlens_q = F.pad(torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), pad=(1, 0))
        else:
            extending = True
            max_seqlen_q = max_seqlen_k
            cu_seqlens_q = cu_seqlens_k

        page_table = page_table.view(-1)
        page_table = page_table[page_table > 0]

        self.forward_metadata = ForwardMetaData(cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, page_table=page_table)

        # # Sliding window
        # if self.sliding_window_size is not None and self.sliding_window_size > 0:
        #     window_kv_indptr, window_kv_indices, _ = update_sliding_window_buffer(
        #         self.window_kv_indptr,
        #         self.req_to_token,
        #         self.sliding_window_size,
        #         forward_batch.extend_prefix_lens,
        #         forward_batch.req_pool_indices,
        #         bs,
        #         self.device,
        #         self.token_to_kv_pool_allocator,
        #     )

        # print("=" * 80)
        # print(f"{'EXTEND' if extending else 'NOT EXTEND'}")
        # print(f"  max_seqlen_q: {max_seqlen_q}")
        # print(f"  max_seqlen_k: {max_seqlen_k}")
        # print(f"  cu_seqlens_q: {cu_seqlens_q.shape=}", cu_seqlens_q)
        # print(f"  cu_seqlens_k: {cu_seqlens_k.shape=}", cu_seqlens_k)
        # print(f"  page_table: {page_table.shape=}", page_table)
        # print("-" * 40)
        # import dataclasses
        # for field in dataclasses.fields(forward_batch):
        #     value = getattr(forward_batch, field.name)
        #     if value is not None:
        #         print(f"{field.name}: {value}")
        # print("=" * 80, flush=True)

        return

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks=None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)

        metadata = self.forward_metadata
        k = k_cache[metadata.page_table, :, :, :].reshape(-1, layer.tp_k_head_num, layer.head_dim)
        v = v_cache[metadata.page_table, :, :, :].reshape(-1, layer.tp_v_head_num, layer.head_dim)

        print(f"{' forward_extend ':=^120}")
        print(f"{q.shape=}")
        print(f"{k_cache.shape=}")
        print(f"{v_cache.shape=}")
        print(f"{k.shape=}")
        print(f"{v.shape=}")
        print("=" * 120, flush=True)

        out = self.flash_attn_func(
            q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            k=k,
            v=v,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            causal=True,
        )[0]

        return out

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
