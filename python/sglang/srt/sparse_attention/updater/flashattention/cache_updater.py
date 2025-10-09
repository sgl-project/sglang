import torch
from typing import Optional, List, TYPE_CHECKING
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from sglang.srt.sparse_attention.cache_manager.cache_manager import CacheManager, ManagerConfig
from .custom_retriver import DenseRetriver, NaiveDecodeSparseRetriver

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.attention import FlashInferAttnBackend
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

class SparseCacheConfig:
    pass


class SparseCacheUpdaterFlashAttentionBackend:
    def __init__(self, manager_config: ManagerConfig):
        self.cache_manager = CacheManager(manager_config)

    def update_decode(self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata") -> "FlashAttentionMetadata":
        raise NotImplementedError()
    
    def update_extend(self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata") -> "FlashAttentionMetadata":
        raise NotImplementedError()
    
    def update_query(self, query: torch.Tensor, forward_batch: "ForwardBatch", layer_id: int):
        pass
    
    def call_begin_forward_attn_extend(self, *args, **kwargs):
        pass
    
    def call_begin_forward_attn_decode(self, query: torch.Tensor, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata", layer: "RadixAttention"):
        pass
    
    
class LServerUpdaterFlashAttentionBackend(SparseCacheUpdaterFlashAttentionBackend):
    def __init__(self, manager_config: ManagerConfig):
        super().__init__(manager_config)
        self.dense_retriver = DenseRetriver(self.cache_manager)
        self.naive_decode_sparse_retriver = NaiveDecodeSparseRetriver(self.cache_manager)
        
    def update_decode(self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata") -> "FlashAttentionMetadata":
        #return self.dense_retriver.retrive_decode(forward_batch, metadata)
        return self.naive_decode_sparse_retriver.build_stream(forward_batch, metadata)
    
    def update_extend(self, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata") -> "FlashAttentionMetadata":
        return self.dense_retriver.retrive_extend(forward_batch, metadata)
    
    def update_extend_proxy_k_tensor(self, forward_batch: "ForwardBatch"):
        self.naive_decode_sparse_retriver.update_extend(forward_batch)
    
    def call_begin_forward_attn_decode(self, query: torch.Tensor, forward_batch: "ForwardBatch", metadata: "FlashAttentionMetadata", layer: "RadixAttention"):
        self.naive_decode_sparse_retriver.retrive_decode(query, forward_batch, metadata, layer)
    
    