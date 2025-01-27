import torch
from flashinfer import SegmentGEMMWrapper

from sglang.srt.lora.backend.base_backend import BaseLoraBackend
from sglang.srt.lora.lora import LoraBatchInfo


class FlashInferLoraBackend(BaseLoraBackend):

    def __init__(self, name: str, batch_info: LoraBatchInfo = None):
        super().__init__(name, batch_info)

        # Set up SGemm Wrapper from flashinfer
        # FIXME wait for flashinfer segment gemm update
        workspace_buffer = torch.empty(1 * 1024 * 1024, dtype=torch.int8, device="cuda")
        self.segment_gemm = SegmentGEMMWrapper(workspace_buffer)

    def run_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        return self.segment_gemm.run(
            x=x,
            weights=weights,
            batch_size=self.batch_info.bs,
            weight_column_major=True,
            seg_indptr=self.batch_info.seg_indptr,
            weight_indices=self.batch_info.weight_indices,
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        q_lora_b: torch.Tensor,
        kv_lora_b: torch.Tensor,
    ) -> torch.Tensor:

        # Shape of lora_a_output: (s, 3 * r)
        lora_a_output = self.run_sgemm(x=x, weights=qkv_lora_a)

        lora_rank = kv_lora_b.shape[-1]
        output_dim_q = q_lora_b.shape[-2]
        output_dim_kv = kv_lora_b.shape[-2]
        lora_output = torch.empty(
            (x.shape[0], output_dim_q + 2 * output_dim_kv),
            device=x.device,
            dtype=x.dtype,
        )

        # FIXME parallelize qkv
        # q
        lora_output[:, :output_dim_q] = self.run_sgemm(
            x=lora_a_output[:, :lora_rank].contiguous(), weights=q_lora_b[0]
        )

        # kv
        lora_output[:, output_dim_q : output_dim_q + output_dim_kv] = self.run_sgemm(
            x=lora_a_output[:, lora_rank : 2 * lora_rank].contiguous(),
            weights=kv_lora_b[0],
        )

        lora_output[
            :, output_dim_q + output_dim_kv : output_dim_q + 2 * output_dim_kv
        ] = self.run_sgemm(
            x=lora_a_output[:, 2 * lora_rank : 3 * lora_rank].contiguous(),
            weights=kv_lora_b[1],
        )

        return lora_output
