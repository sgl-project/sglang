from typing import Optional
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.triton_ops import (
    chunked_sgmv_lora_shrink_forward,
    chunked_sgmv_lora_expand_forward,
)
from sglang.srt.lora.utils import LoRABatchInfo


class ChunkedSgmvLoRABackend(BaseLoRABackend):
    name = "csgmv"

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        super().__init__(max_loras_per_batch, device)
        self.segment_size = 16  # TODO (lifuhuang): make it configurable?

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return chunked_sgmv_lora_shrink_forward(
            x,
            weights,
            self.batch_info,
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        output_offset: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        # For simple lora B, we use slice offsets [0, output_dim]
        output_dim = weights.shape[-2]
        max_slice_size = output_dim
        return chunked_sgmv_lora_expand_forward(
            x=x,
            lora_weight_b=weights,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_slice_size,
            base_output=base_output,
        )

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

        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x,
            qkv_lora_a,
            self.batch_info,
            num_slices=3,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            lora_weight_b=qkv_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=max_qkv_out_dim,
            base_output=base_output,
        )
        return lora_output

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
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
        lora_a_output = chunked_sgmv_lora_shrink_forward(
            x,
            gate_up_lora_a,
            self.batch_info,
            num_slices=2,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            lora_weight_b=gate_up_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=output_dim,
            base_output=base_output,
        )
        return lora_output

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        batch_info: Optional[LoRABatchInfo] = None,
    ):
        permutation, weight_indices_reordered = ChunkedSgmvLoRABackend._get_permutation(
            weight_indices, forward_batch
        )

        seg_weight_indices, seg_indptr = self._get_segments_info(
            weight_indices_reordered
        )
        num_segments = len(seg_weight_indices)

        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        )

        if batch_info is None:
            max_len = (
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                max(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            batch_info = LoRABatchInfo(
                bs=forward_batch.batch_size,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=None, 
                # TODO (lifu): technically we do not need seg_lens in triton either, we can convenge this logic later.
                # seg_lens=torch.empty(
                #     (num_segments,), dtype=torch.int32, device=self.device
                # ),
                seg_indptr=torch.empty(
                    (num_segments + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (num_segments,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int64, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=torch.empty(
                    (len(permutation),), dtype=torch.int32, device=self.device
                ),
            )
        else:
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:num_segments].copy_(
            seg_weight_indices, non_blocking=True
        )
        batch_info.seg_indptr[: num_segments + 1].copy_(seg_indptr, non_blocking=True)
        batch_info.permutation[: len(permutation)].copy_(
            permutation, non_blocking=True
        )

        self.batch_info = batch_info

    @staticmethod
    def _get_permutation(seq_weight_indices, forward_batch: ForwardBatch):
        with torch.device("cpu"):
            seq_weight_indices = torch.tensor(seq_weight_indices, dtype=torch.int32)

            seg_lens_cpu = (
                torch.tensor(
                    forward_batch.extend_seq_lens_cpu,
                    dtype=torch.int32,
                )
                if forward_batch.forward_mode.is_extend()
                else torch.ones(forward_batch.batch_size, dtype=torch.int32)
            )

            row_weight_indices = torch.repeat_interleave(
                seq_weight_indices, seg_lens_cpu
            )
            permutation = torch.empty(
                (len(row_weight_indices),), dtype=torch.long, pin_memory=True
            )
            torch.argsort(row_weight_indices, stable=True, out=permutation)
            weights_reordered = row_weight_indices[permutation]

            return permutation, weights_reordered

    def _get_segments_info(self, weights_reordered: torch.Tensor):
        with torch.device("cpu"):
            unique_weights, counts = torch.unique_consecutive(
                weights_reordered, return_counts=True
            )

            weight_indices_list = []
            seg_lens_list = []

            for weight_idx, group_len in zip(unique_weights, counts):
                group_len = group_len.item()
                num_segs = (group_len + self.segment_size - 1) // self.segment_size

                weight_indices_list.extend([weight_idx.item()] * num_segs)
                seg_lens_list.extend([self.segment_size] * (num_segs - 1))
                seg_lens_list.append(group_len - (num_segs - 1) * self.segment_size)

            seg_lens = torch.tensor( seg_lens_list, dtype=torch.int32)

            weight_indices_list = torch.tensor(
                weight_indices_list, dtype=torch.int32, pin_memory=True)

            seg_indptr = torch.empty(
                (len(seg_lens) + 1,), dtype=torch.int32, pin_memory=True)
            seg_indptr[0] = 0
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

            return weight_indices_list, seg_indptr
