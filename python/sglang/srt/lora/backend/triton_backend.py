import torch
import triton
import triton.language as tl

from sglang.srt.lora.backend.base_backend import BaseLoraBackend
from sglang.srt.lora.lora import LoraBatchInfo



class TritonLoraBackend(BaseLoraBackend):
    
    def __init__(self, name: str, batch_info: LoraBatchInfo = None):
        super().__init__(name, batch_info)

    def run_sgemm(self, 
                  x: torch.Tensor, 
                  weights: torch.Tensor) -> torch.Tensor:


        # if self.gemm_expand is not None:
        #     for i in range(2):
        #         self.gemm_expand(
        #             base_output,
        #             lora_a_output,
        #             self.B_buffer[i],
        #             batch_size=self.batch_info.bs,
        #             seg_lens=self.batch_info.seg_lens,
        #             seg_start=self.batch_info.seg_indptr,
        #             weight_indices=self.batch_info.weight_indices,
        #             max_len=self.batch_info.max_len,
        #             input_slice_offset=self.lora_rank * i,
        #             output_slice_offset=output_dim * i,
        #             output_add=True,
        #             scaling=self.scaling,
        #         )
        #     return base_output
        
        raise NotImplementedError()
        
    
    def run_qkv_lora(self,
                     x: torch.Tensor,
                     qkv_lora_a: torch.Tensor,
                     q_lora_b: torch.Tensor,
                     kv_lora_b: torch.Tensor) -> torch.Tensor:
        
        # if self.gemm_expand is not None:
        #     for i in range(2):
        #         self.gemm_expand(
        #             base_output,
        #             lora_a_output,
        #             self.B_buffer_kv[i],
        #             batch_size=self.batch_info.bs,
        #             seg_lens=self.batch_info.seg_lens,
        #             seg_start=self.batch_info.seg_indptr,
        #             weight_indices=self.batch_info.weight_indices,
        #             max_len=self.batch_info.max_len,
        #             input_slice_offset=self.lora_rank * (i + 1),
        #             output_slice_offset=output_dim_q + output_dim_kv * i,
        #             output_add=True,
        #             scaling=self.scaling,
        #         )
        #     return base_output
        
        raise NotImplementedError()