from re import U
import torch
from typing import Optional, List, TYPE_CHECKING
from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
from sglang.srt.layers.dp_attention import get_attention_tp_size


if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.attention import FlashInferAttnBackend

class SparseCacheConfig:
    pass


class SparseCacheUpdaterFlashInferBackend:
    def __init__(self, model_runner: "ModelRunner", attn_backend: "FlashInferAttnBackend"):
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged


    def update_decode(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, seq_lens_cpu:Optional[torch.Tensor],
                      seq_lens_sum: int, decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper], 
                      encoder_lens: Optional[torch.Tensor]):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        for decode_wrapper in decode_wrappers:
            self._update_decode_wrapper(
                decode_wrapper,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                self.kv_indptr[0],
                None,
                seq_lens_cpu,
            )
    
    def update_prefill(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, seq_lens_cpu:Optional[torch.Tensor],
                      seq_lens_sum: int, prefix_lens: torch.Tensor, prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper], 
                      use_ragged: bool, encoder_lens: Optional[torch.Tensor], extend_no_prefix: bool):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        for prefill_wrapper in prefill_wrappers:
            self._update_prefill_wrapper(
                prefill_wrapper,
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                None,
                self.kv_indptr[0],
                self.qo_indptr[0],
                use_ragged,
                extend_no_prefix
            )
    
    def _update_decode_wrapper(self, decode_wrapper: BatchDecodeWithPagedKVCacheWrapper, req_pool_indices: torch.Tensor, 
                               paged_kernel_lens: torch.Tensor, paged_kernel_lens_sum: int, kv_indptr: torch.Tensor, 
                               kv_start_idx: torch.Tensor, seq_lens_cpu: torch.Tensor):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]

        if decode_wrapper.is_cuda_graph_enabled:
            # Directly write to the cuda graph input buffer
            kv_indices = decode_wrapper._paged_kv_indices_buf
        else:
            kv_indices = torch.empty(
                paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
            )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )
        
        decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
        )

    
    def _update_prefill_wrapper(self, prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper, req_pool_indices: torch.Tensor, 
                                paged_kernel_lens: torch.Tensor, paged_kernel_lens_sum: int,  seq_lens: torch.Tensor, prefix_lens: torch.Tensor, 
                                kv_start_idx: torch.Tensor, kv_indptr: torch.Tensor, qo_indptr: torch.Tensor, use_ragged: bool, extend_no_prefix: bool):
        assert len(paged_kernel_lens) == len(req_pool_indices)
        bs = len(seq_lens)
        
        kv_indptr, qo_indptr, kv_indices = self.attn_info_updater.update_prefill(
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            kv_start_idx,
            kv_indptr,
            qo_indptr,
            use_ragged,
            extend_no_prefix
        )
        
        
        if use_ragged:
            self.prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
    
        # cached part
        prefill_wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=None,
            non_blocking=True,
        )
        

    def call_begin_forward_batch(self, *args, **kwargs):
        pass
    
    def call_end_forward_batch(self, *args, **kwargs):
        pass
    
    def call_begin_forward_attn(self, *args, **kwargs):
        pass
    
    def call_end_forward_attn(self, *args, **kwargs):
        pass
    