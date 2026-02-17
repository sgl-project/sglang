import dataclasses
from typing import List, Tuple

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.triton_ops import (
    embedding_lora_a_fwd,
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TritonLoRABackend(BaseLoRABackend):
    name = "triton"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_loras_per_batch, device)

    def run_lora_a_embedding(
        self,
        input_ids: torch.Tensor,
        weights: torch.Tensor,
        vocab_size: int,
        extra_embeddings: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run LoRA A embedding lookup using Triton kernel."""
        return embedding_lora_a_fwd(
            input_ids=input_ids,
            weights=weights,
            batch_info=self.batch_info,
            vocab_size=vocab_size,
            extra_embeddings=extra_embeddings,
        )

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        pruned_batch_info: LoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )
        return sgemm_lora_a_fwd(x, weights, batch_info)

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        pruned_batch_info: LoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_info = (
            pruned_batch_info if pruned_batch_info is not None else self.batch_info
        )
        return sgemm_lora_b_fwd(x, weights, batch_info, base_output)

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
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
        **kwargs,
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

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=max_bs_in_cuda_graph,
                use_cuda_graph=True,
                num_segments=None,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                max_len=num_tokens_per_bs,
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                permutation=None,
            )

            # Initialize seg_indptr for CUDA graph as they remain constant
            # across batches.
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
            )

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
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

        bs = forward_batch.batch_size

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = forward_batch.batch_size
        else:
            max_len = (
                # Calculate max_len from the CPU copy to avoid D2H transfer.
                max(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode.is_extend()
                else torch.ones(bs, dtype=torch.int32, device=self.device)
            )
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

            batch_info = LoRABatchInfo(
                bs=forward_batch.batch_size,
                num_segments=forward_batch.batch_size,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=seg_lens,
                seg_indptr=seg_indptr,
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int64, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=None,
            )

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True)

        self.batch_info = batch_info

        # Precompute lm_head_batch_info for pruned lm_head LoRA
        pruned_lens = get_lm_head_pruned_lens(forward_batch)

        if pruned_lens is not None:
            pruned_total = sum(pruned_lens)
            lm_head_segments = merge_and_chunk_segments(
                weight_indices, pruned_lens, chunk_size=pruned_total
            )
            self.lm_head_batch_info = self._build_lm_head_batch_info(
                lm_head_segments, batch_info, pruned_total
            )

            # Precompute per-pass batch_infos for logprobs chunking
            pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)
            if pass_segments is not None:
                self.lm_head_pass_batch_infos = []
                for seg_wi, seg_lens_list in pass_segments:
                    pass_total = sum(seg_lens_list)
                    merged_segments = merge_and_chunk_segments(
                        seg_wi, seg_lens_list, chunk_size=pass_total
                    )
                    self.lm_head_pass_batch_infos.append(
                        self._build_lm_head_batch_info(
                            merged_segments, batch_info, pass_total
                        )
                    )
            else:
                self.lm_head_pass_batch_infos = None
        else:
            self.lm_head_batch_info = None
            self.lm_head_pass_batch_infos = None

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: LoRABatchInfo,
        expected_tokens: int,
    ) -> LoRABatchInfo:
        """Build a LoRABatchInfo for pruned lm_head input."""
        seg_weight_indices_cpu, seg_lens_cpu = lm_head_segments
        num_segments = len(seg_weight_indices_cpu)

        seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=self.device)
        seg_indptr = torch.zeros(
            (num_segments + 1,), dtype=torch.int32, device=self.device
        )
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        return dataclasses.replace(
            batch_info,
            bs=num_segments,
            num_segments=num_segments,
            max_len=max(seg_lens_cpu),
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            weight_indices=torch.tensor(
                seg_weight_indices_cpu, dtype=torch.int32, device=self.device
            ),
            expected_tokens=expected_tokens,
        )
