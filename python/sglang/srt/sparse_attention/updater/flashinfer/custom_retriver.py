import torch
from typing import Optional, List, TYPE_CHECKING, Tuple
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.attention import FlashInferAttnBackend

class DenseAttnInfoUpdater:
    def __init__(self, model_runner: "ModelRunner", attn_backend: "FlashInferAttnBackend"):
        self.model_runner = model_runner
        self.attn_backend = attn_backend

    def update_prefill(self, req_pool_indices: torch.Tensor, paged_kernel_lens: torch.Tensor, paged_kernel_lens_sum: int,  
                       seq_lens: torch.Tensor, prefix_lens: torch.Tensor, kv_start_idx: torch.Tensor, kv_indptr: torch.Tensor, 
                       qo_indptr: torch.Tensor, use_ragged: bool, extend_no_prefix: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = len(seq_lens)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum + 256,
            dtype=torch.int32,
            device=req_pool_indices.device,
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
        
        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]
        return kv_indptr, qo_indptr, kv_indices
    
    def update_decode(self, *args, **kwargs):
        pass