import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.triton_ops import (
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo


class TritonLoRABackend(BaseLoRABackend):
    name = "triton"

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        super().__init__(max_loras_per_batch, device)

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return sgemm_lora_a_fwd(x, weights, self.batch_info)

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return sgemm_lora_b_fwd(x, weights, self.batch_info, base_output)

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, self.batch_info, stack_num=3)
        lora_output = qkv_lora_b_fwd(
            lora_a_output,
            qkv_lora_b,
            self.batch_info,
            output_offset,
            max_qkv_out_dim,
            base_output,
        )
        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # gate_up_lora_a: (num_lora, 2 * r, input_dim)
        # gate_up_lora_b: (num_lora, 2 * output_dim, r)
        assert isinstance(gate_up_lora_b, torch.Tensor)
        output_dim = gate_up_lora_b.shape[-2] // 2

        # lora_a_output: (s, 2 * r)
        lora_a_output = sgemm_lora_a_fwd(
            x, gate_up_lora_a, self.batch_info, stack_num=2
        )
        lora_output = gate_up_lora_b_fwd(
            lora_a_output,
            gate_up_lora_b,
            self.batch_info,
            output_dim,
            base_output,
        )
        return lora_output

    def prepare_lora_batch(self, forward_batch: ForwardBatch):

        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size

        def transfer_adapter_info(
            weight_indices_out: torch.Tensor,
            lora_ranks_out: torch.Tensor,
            scalings_out: torch.Tensor,
        ):
            """
            Transfer adapter metadata (weight indices, LoRA rank, scalings) from host
            to device (CUDA) asynchronously.
            """
            weight_indices = [0] * len(forward_batch.lora_ids)
            lora_ranks = [0] * self.max_loras_per_batch
            scalings = [0] * self.max_loras_per_batch
            for i, uid in enumerate(forward_batch.lora_ids):
                weight_indices[i] = self.memory_pool.get_buffer_id(uid)
                if uid is not None:
                    lora = self.loras[uid]
                    lora_ranks[weight_indices[i]] = lora.config.r
                    scalings[weight_indices[i]] = lora.scaling

            # Use pinned memory to avoid synchronizations during host-to-device transfer
            weight_indices_tensor = torch.tensor(
                weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            lora_ranks_tensor = torch.tensor(
                lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
            )
            scalings_tensor = torch.tensor(
                scalings, dtype=torch.float, pin_memory=True, device="cpu"
            )

            # Copy to device tensors asynchronously
            weight_indices_out[:bs].copy_(weight_indices_tensor, non_blocking=True)
            lora_ranks_out[: self.max_loras_per_batch].copy_(
                lora_ranks_tensor, non_blocking=True
            )
            scalings_out[: self.max_loras_per_batch].copy_(
                scalings_tensor, non_blocking=True
            )

        if (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        ):
            # Do in-place updates when CUDA graph is enabled and the batch forward mode
            # could use CUDA graph.

            transfer_adapter_info(
                self.cuda_graph_batch_info.weight_indices,
                self.cuda_graph_batch_info.lora_ranks,
                self.cuda_graph_batch_info.scalings,
            )

            self.cuda_graph_batch_info.bs = bs
            self.cuda_graph_batch_info.max_len = 1
            batch_info = self.cuda_graph_batch_info
        else:
            weight_indices = torch.empty((bs,), dtype=torch.int32, device=self.device)
            lora_ranks = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.int64, device=self.device
            )
            scalings = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.float, device=self.device
            )
            transfer_adapter_info(
                weight_indices,
                lora_ranks,
                scalings,
            )

            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode.is_extend()
                else torch.ones(bs, device=self.device)
            )

            max_len = (
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                max(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend()
                else 1
            )

            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

            batch_info = LoRABatchInfo(
                bs=bs,
                seg_lens=seg_lens,
                seg_indptr=seg_indptr,
                max_len=max_len,
                weight_indices=weight_indices,
                lora_ranks=lora_ranks,
                scalings=scalings,
            )

        self.lora_backend.set_batch_info(batch_info)
