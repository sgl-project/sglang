import torch
import triton
import triton.language as tl

from sglang.srt.lora.backend import BaseLoraBackend
from sglang.srt.lora.lora import LoraBatchInfo
from sglang.srt.lora.triton_ops import (
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)


class TritonLoraBackend(BaseLoraBackend):

    def __init__(self, name: str, batch_info: LoraBatchInfo = None):
        super().__init__(name, batch_info)

    def run_lora_a_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sgemm_lora_a_fwd(x, weights, self.batch_info)

    def run_lora_b_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return sgemm_lora_b_fwd(x, weights, self.batch_info)

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        q_lora_b: torch.Tensor,
        kv_lora_b: torch.Tensor,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # q_lora_b: (1, num_lora, output_dim_q, r)
        # kv_lora_b: (2, num_lora, output_dim_kv, r)

        lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, self.batch_info)
        lora_output = qkv_lora_b_fwd(
            lora_a_output, q_lora_b, kv_lora_b, self.batch_info
        )
        return lora_output
