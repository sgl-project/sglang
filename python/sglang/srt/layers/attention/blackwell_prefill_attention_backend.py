from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

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
class _ForwardMetaData:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor

    # Sliding windowcute
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor


class BlackwellPrefillAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        from sglang.srt.layers.attention.cute_ops.prefill_attention import flash_attn_varlen_func

        super().__init__()
        self._flash_attn_varlen_func = flash_attn_varlen_func
        # self.sliding_window_size = model_runner.sliding_window_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        # kv_indptr = self.kv_indptr
        # window_kv_indptr = self.window_kv_indptr
        # window_kv_indices = None
        # window_num_kv_splits = None

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
        return q

        # o = torch.empty_like(q)
        # if save_kv_cache:
        #     forward_batch.token_to_kv_pool.set_kv_buffer(
        #         layer, forward_batch.out_cache_loc, k, v
        #     )
        # return o

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
