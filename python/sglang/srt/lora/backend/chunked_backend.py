from typing import Optional

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.triton_ops import (
    chunked_sgmv_lora_expand_forward,
    chunked_sgmv_lora_shrink_forward,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs

MIN_CHUNK_SIZE = 16


class ChunkedSgmvLoRABackend(BaseLoRABackend):
    """
    Chunked LoRA backend using segmented matrix-vector multiplication.

    This backend is largely based on the SGMV (Segmented Gather Matrix-Vector multiplication) algorithm
    introduced in the Punica paper (https://arxiv.org/pdf/2310.18547). One main variation made here is to
    segment the input sequences into fixed-size chunks, which reduces excessive kernel launches especially
    when the LoRA distribution is skewed.
    """

    name = "csgmv"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        server_args: ServerArgs,
    ):
        super().__init__(max_loras_per_batch, device)
        self.max_chunk_size = server_args.max_lora_chunk_size

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return chunked_sgmv_lora_shrink_forward(
            x=x,
            weights=weights,
            batch_info=self.batch_info,
            num_slices=1,
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
            weights=weights,
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
            x=x,
            weights=qkv_lora_a,
            batch_info=self.batch_info,
            num_slices=3,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=qkv_lora_b,
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
            x=x,
            weights=gate_up_lora_a,
            batch_info=self.batch_info,
            num_slices=2,
        )
        lora_output = chunked_sgmv_lora_expand_forward(
            x=lora_a_output,
            weights=gate_up_lora_b,
            batch_info=self.batch_info,
            slice_offsets=output_offset,
            max_slice_size=output_dim,
            base_output=base_output,
        )
        return lora_output

    def _determine_chunk_size(self, forward_batch: ForwardBatch) -> int:
        """
        Heuristically determine the chunk size based on token token number in a batch.

        Args:
            forward_batch (ForwardBatch): The batch information containing sequence lengths.

        Returns:
            The determined chunk size
        """

        if self.max_chunk_size <= MIN_CHUNK_SIZE:
            return MIN_CHUNK_SIZE

        num_tokens = (
            forward_batch.extend_num_tokens
            if forward_batch.forward_mode.is_extend()
            else forward_batch.batch_size
        )
        if num_tokens >= 256:
            chunk_size = 128
        elif num_tokens >= 64:
            chunk_size = 32
        else:  # num_tokens < 64
            chunk_size = 16
        return min(self.max_chunk_size, chunk_size)

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        batch_info: Optional[LoRABatchInfo] = None,
    ):
        chunk_size = self._determine_chunk_size(forward_batch)

        permutation, weight_indices_reordered = ChunkedSgmvLoRABackend._get_permutation(
            seq_weight_indices=weight_indices,
            forward_batch=forward_batch,
        )

        seg_weight_indices, seg_indptr = self._get_segments_info(
            weights_reordered=weight_indices_reordered,
            chunk_size=chunk_size,
        )
        num_segments = len(seg_weight_indices)

        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        )

        if batch_info is None:
            batch_info = LoRABatchInfo(
                bs=forward_batch.batch_size,
                num_segments=num_segments,
                max_len=chunk_size,
                use_cuda_graph=False,
                seg_indptr=torch.empty(
                    (num_segments + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (num_segments,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=torch.empty(
                    (len(permutation),), dtype=torch.int32, device=self.device
                ),
                # Not used in chunked kernels
                seg_lens=None,
            )
        else:
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments
            batch_info.max_len = chunk_size

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
        batch_info.permutation[: len(permutation)].copy_(permutation, non_blocking=True)

        self.batch_info = batch_info

    @staticmethod
    def _get_permutation(seq_weight_indices, forward_batch: ForwardBatch):
        """
        Computes permutation indices for reordering tokens by their LoRA adapter assignments.

        This function implements the "gather" step in Chunked Segmented Gather Matrix Vector
        multiplication by creating a permutation that groups tokens by their LoRA adapter.
        Tokens using the same LoRA adapter are placed together to enable efficient batched
        computation.

        Example:
            seq_weight_indices = [0, 1, 0]  # 3 sequences using adapters [0, 1, 0]
            extend_seq_lens = [2, 1, 3]     # sequence lengths [2, 1, 3 tokens]

            # Creates row_weight_indices: [0, 0, 1, 0, 0, 0] (6 tokens total)
            # Returns permutation: [0, 1, 3, 4, 5, 2] (groups adapter 0 tokens together)
            # weights_reordered: [0, 0, 0, 0, 0, 1] (sorted by adapter)

        Args:
            seq_weight_indices: List of LoRA adapter indices for each sequence
            forward_batch (ForwardBatch): Batch information containing sequence lengths

        Returns:
            tuple: (permutation, weights_reordered) where:
                - permutation: Token reordering indices to group by adapter
                - weights_reordered: Sorted adapter indices for each token
        """
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

    def _get_segments_info(self, weights_reordered: torch.Tensor, chunk_size: int):
        """
        Computes segment information for chunked SGMV operations.

        This function takes the reordered weight indices and creates segments of fixed size
        (self.segment_size) for efficient kernel execution. Each segment contains tokens
        that use the same LoRA adapter, enabling vectorized computation.

        The segmentation is necessary because:
        1. GPU kernels work efficiently on fixed-size blocks
        2. Large groups of tokens using the same adapter are split into manageable chunks
        3. Each segment can be processed independently in parallel

        Example:
            weights_reordered = [0, 0, 0, 0, 0, 1]  # 5 tokens with adapter 0, 1 with adapter 1
            segment_size = 3

            # Creates segments:
            # Segment 0: tokens 0-2 (adapter 0), length=3
            # Segment 1: tokens 3-4 (adapter 0), length=2
            # Segment 2: token 5 (adapter 1), length=1

            # Returns:
            # weight_indices_list: [0, 0, 1] (adapter for each segment)
            # seg_indptr: [0, 3, 5, 6] (cumulative segment boundaries)

        Args:
            weights_reordered (torch.Tensor): Sorted adapter indices for each token
            chunk_size (int): Fixed size for each segment

        Returns:
            tuple: (weight_indices_list, seg_indptr) where:
                - weight_indices_list: LoRA adapter index for each segment
                - seg_indptr: Cumulative segment boundaries (CSR-style indptr)
        """
        with torch.device("cpu"):
            unique_weights, counts = torch.unique_consecutive(
                weights_reordered, return_counts=True
            )

            weight_indices_list = []
            seg_lens_list = []

            for weight_idx, group_len in zip(unique_weights, counts):
                group_len = group_len.item()
                num_segs = (group_len + chunk_size - 1) // chunk_size

                weight_indices_list.extend([weight_idx.item()] * num_segs)
                seg_lens_list.extend([chunk_size] * (num_segs - 1))
                seg_lens_list.append(group_len - (num_segs - 1) * chunk_size)

            seg_lens = torch.tensor(seg_lens_list, dtype=torch.int32)

            weight_indices_list = torch.tensor(
                weight_indices_list, dtype=torch.int32, pin_memory=True
            )

            seg_indptr = torch.empty(
                (len(seg_lens) + 1,), dtype=torch.int32, pin_memory=True
            )
            seg_indptr[0] = 0
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

            return weight_indices_list, seg_indptr
