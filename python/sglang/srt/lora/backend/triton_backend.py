import torch

from sglang.srt.lora.backend import BaseLoRABackend
from sglang.srt.lora.triton_ops import (
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo


class TritonLoRABackend(BaseLoRABackend):

    def __init__(self, name: str, batch_info: LoRABatchInfo = None):
        super().__init__(name, batch_info)

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return sgemm_lora_a_fwd(x, weights, self.batch_info)

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        scaling: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return sgemm_lora_b_fwd(x, weights, self.batch_info, base_output, scaling)

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        scaling: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, self.batch_info)
        lora_output = qkv_lora_b_fwd(
            lora_a_output,
            qkv_lora_b,
            self.batch_info,
            output_offset,
            max_qkv_out_dim,
            base_output,
            scaling,
        )
        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        base_output: torch.Tensor = None,
        scaling: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        assert isinstance(gate_up_lora_b, torch.Tensor)
        output_dim = gate_up_lora_b.shape[-2] // 2

        # lora_a_output: (s, 2 * r)
        lora_a_output = sgemm_lora_a_fwd(x, gate_up_lora_a, self.batch_info)
        lora_output = gate_up_lora_b_fwd(
            lora_a_output,
            gate_up_lora_b,
            self.batch_info,
            output_dim,
            base_output,
            scaling,
        )
        return lora_output
