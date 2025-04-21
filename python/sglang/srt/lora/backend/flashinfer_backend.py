from typing import Tuple

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    from flashinfer import SegmentGEMMWrapper


class FlashInferLoRABackend(BaseLoRABackend):

    def __init__(self, name: str, batch_info: LoRABatchInfo = None):
        super().__init__(name, batch_info)

        # Set up SGemm Wrapper from flashinfer
        # FIXME wait for flashinfer segment gemm update
        workspace_buffer = torch.empty(1 * 1024 * 1024, dtype=torch.int8, device="cuda")
        self.segment_gemm = SegmentGEMMWrapper(workspace_buffer)

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:

        return self.segment_gemm.run(
            x=x,
            weights=weights,
            batch_size=self.batch_info.bs,
            weight_column_major=True,
            seg_indptr=self.batch_info.seg_indptr,
            weight_indices=self.batch_info.weight_indices,
        )

    def run_lora_b_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:

        return (
            self.segment_gemm.run(
                x=x,
                weights=weights,
                batch_size=self.batch_info.bs,
                weight_column_major=True,
                seg_indptr=self.batch_info.seg_indptr,
                weight_indices=self.batch_info.weight_indices,
            )
            * self.batch_info.scalings[0]
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: Tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:

        assert isinstance(qkv_lora_b, tuple) and len(qkv_lora_b) == 2

        # Shape of lora_a_output: (s, 3 * r)
        lora_a_output = self.run_lora_a_sgemm(x=x, weights=qkv_lora_a)

        q_lora_b, kv_lora_b = qkv_lora_b
        lora_rank = kv_lora_b.shape[-1]
        output_dim_q = q_lora_b.shape[-2]
        output_dim_kv = kv_lora_b.shape[-2]
        lora_output = torch.empty(
            (x.shape[0], output_dim_q + 2 * output_dim_kv),
            device=x.device,
            dtype=x.dtype,
        )

        # q
        lora_output[:, :output_dim_q] = self.run_lora_b_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(), weights=q_lora_b[0]
        )

        # kv
        lora_output[:, output_dim_q : output_dim_q + output_dim_kv] = (
            self.run_lora_b_sgemm(
                x=lora_a_output[:, lora_rank : 2 * lora_rank].contiguous(),
                weights=kv_lora_b[0],
            )
        )

        lora_output[
            :, output_dim_q + output_dim_kv : output_dim_q + 2 * output_dim_kv
        ] = self.run_lora_b_sgemm(
            x=lora_a_output[:, 2 * lora_rank : 3 * lora_rank].contiguous(),
            weights=kv_lora_b[1],
        )

        return lora_output * self.batch_info.scalings[0]

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: Tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:

        assert isinstance(gate_up_lora_b, tuple) and len(gate_up_lora_b) == 2
        lora_rank = gate_up_lora_b[0].shape[-1]
        output_dim = gate_up_lora_b[0].shape[-2]

        # Shape of lora_a_output: (s, 2 * r)
        lora_a_output = self.run_lora_a_sgemm(x=x, weights=gate_up_lora_a)

        lora_output = torch.empty(
            (x.shape[0], 2 * output_dim),
            device=x.device,
            dtype=x.dtype,
        )

        # Compute lora for gate and up proj respectively
        lora_output[:, :output_dim] = self.run_lora_b_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(),
            weights=gate_up_lora_b[0],
        )

        lora_output[:, output_dim:] = self.run_lora_b_sgemm(
            x=lora_a_output[:, lora_rank:].contiguous(),
            weights=gate_up_lora_b[1],
        )

        return lora_output * self.batch_info.scalings[0]
