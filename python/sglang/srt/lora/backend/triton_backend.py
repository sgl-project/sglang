import dataclasses
from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.gemm.embedding_lora_a import embedding_lora_a_fwd
from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.kernels.ops.gemm.qkv_lora_b import qkv_lora_b_fwd
from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Segment slots baked into kernel launch grids captured by the prefill CUDA
# graph. The triton backend uses one segment per request, so this bounds the
# request count a LoRA batch may have while replaying the prefill graph
# (larger batches fall back to eager prefill). Kept small because every
# captured LoRA kernel launches a (cdiv(max_len, 16) * cdiv(rank, 16),
# num_slots) grid and unused slots cost a no-op block each.
PREFILL_CUDA_GRAPH_LORA_SEGMENTS = 32


class TritonLoRABackend(BaseLoRABackend):
    name = "triton"
    supports_prefill_cuda_graph = True

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

    def _sgemm_info(self, pruned_batch_info=None):
        """Return the sgemm batch_info (merged segments when available)."""
        if pruned_batch_info is not None:
            return pruned_batch_info
        return getattr(self, "sgemm_batch_info", None) or self.batch_info

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        pruned_batch_info: LoRABatchInfo = None,
        stack_num: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return sgemm_lora_a_fwd(
            x, weights, self._sgemm_info(pruned_batch_info), stack_num=stack_num
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        pruned_batch_info: LoRABatchInfo = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return sgemm_lora_b_fwd(
            x, weights, self._sgemm_info(pruned_batch_info), base_output
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        n_slices: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, n_slices * r, input_dim)
        # qkv_lora_b: (num_lora, total_output_dim, r)
        assert isinstance(qkv_lora_b, torch.Tensor)

        sgemm_info = self._sgemm_info()
        lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, sgemm_info, stack_num=n_slices)
        lora_output = qkv_lora_b_fwd(
            lora_a_output,
            qkv_lora_b,
            sgemm_info,
            output_offset,
            max_qkv_out_dim,
            base_output,
            n_slices=n_slices,
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

        sgemm_info = self._sgemm_info()
        # lora_a_output: (s, 2 * r)
        lora_a_output = sgemm_lora_a_fwd(x, gate_up_lora_a, sgemm_info, stack_num=2)
        lora_output = gate_up_lora_b_fwd(
            lora_a_output,
            gate_up_lora_b,
            sgemm_info,
            output_dim,
            base_output,
        )
        return lora_output

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        max_tokens = max_bs_in_cuda_graph * num_tokens_per_bs
        mlpb = self.max_loras_per_batch
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
                lora_ranks=torch.zeros(mlpb, dtype=torch.int32),
                scalings=torch.zeros(mlpb, dtype=torch.float),
                permutation=None,
            )

            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
            )

            # Sgemm batch_info with segments merged by adapter.
            # Updated each batch by compute_sgemm_routing().
            self.cuda_graph_sgemm_batch_info = LoRABatchInfo(
                bs=mlpb,
                use_cuda_graph=True,
                num_segments=mlpb,
                seg_lens=torch.zeros(mlpb, dtype=torch.int32),
                seg_indptr=torch.zeros(mlpb + 1, dtype=torch.int32),
                max_len=max_tokens,
                weight_indices=torch.arange(mlpb, dtype=torch.int32),
                lora_ranks=torch.zeros(mlpb, dtype=torch.int32),
                scalings=torch.zeros(mlpb, dtype=torch.float),
                permutation=torch.zeros(max_tokens, dtype=torch.int32),
            )

    def init_prefill_cuda_graph_batch_info(self, max_num_tokens: int):
        num_slots = PREFILL_CUDA_GRAPH_LORA_SEGMENTS
        mlpb = self.max_loras_per_batch
        with torch.device(self.device):
            # bs is pinned at num_slots so the (.., bs) launch grids recorded
            # by the prefill graph cover any replay batch; slots beyond the
            # live batch size keep seg_lens == 0 and no-op in-kernel.
            self.prefill_cuda_graph_batch_info = LoRABatchInfo(
                bs=num_slots,
                use_cuda_graph=True,
                num_segments=num_slots,
                seg_lens=torch.zeros(num_slots, dtype=torch.int32),
                seg_indptr=torch.zeros(num_slots + 1, dtype=torch.int32),
                max_len=0,
                weight_indices=torch.zeros(num_slots, dtype=torch.int32),
                lora_ranks=torch.zeros(mlpb, dtype=torch.int32),
                scalings=torch.zeros(mlpb, dtype=torch.float),
                permutation=None,
            )
        self.prefill_cuda_graph_max_bs = num_slots
        self.prefill_cuda_graph_max_tokens = max_num_tokens

    def compute_sgemm_routing(self, use_cuda_graph: bool):
        """Sort tokens by adapter and build merged segments for sgemm LoRA."""
        bi = self.batch_info
        bs = bi.bs
        mlpb = self.max_loras_per_batch
        wi = bi.weight_indices[:bs]

        perm = torch.argsort(wi, stable=True).to(torch.int32)
        sorted_wi = wi[perm]
        adapter_ids = torch.arange(mlpb, device=wi.device, dtype=torch.int32)
        seg_starts = torch.searchsorted(sorted_wi, adapter_ids)
        seg_ends = torch.searchsorted(sorted_wi, adapter_ids, right=True)
        seg_lens = seg_ends - seg_starts

        if use_cuda_graph:
            sgemm = getattr(self, "cuda_graph_sgemm_batch_info", None)
            if sgemm is None:
                return
            sgemm.permutation[:bs] = perm
            sgemm.seg_lens[:] = seg_lens
            sgemm.seg_indptr[0:1].zero_()
            torch.cumsum(sgemm.seg_lens, dim=0, out=sgemm.seg_indptr[1:])
            sgemm.max_len = bs
            sgemm.lora_ranks[:mlpb] = bi.lora_ranks[:mlpb]
            sgemm.scalings[:mlpb] = bi.scalings[:mlpb]
        else:
            seg_indptr = torch.zeros(mlpb + 1, dtype=torch.int32, device=wi.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
            sgemm = LoRABatchInfo(
                bs=mlpb,
                use_cuda_graph=False,
                num_segments=mlpb,
                seg_lens=seg_lens,
                seg_indptr=seg_indptr,
                max_len=bs,
                weight_indices=adapter_ids,
                lora_ranks=bi.lora_ranks[:mlpb].clone(),
                scalings=bi.scalings[:mlpb].clone(),
                permutation=perm,
            )

        self.sgemm_batch_info = sgemm

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
        use_prefill_cuda_graph: bool = False,
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
        elif use_prefill_cuda_graph:
            batch_info = self.prefill_cuda_graph_batch_info
            # batch_info.bs stays pinned at the allocated slot count: the
            # launch grids recorded by the prefill graph iterate every slot,
            # and slots past the live batch no-op via seg_lens == 0. Ragged
            # extend lengths are refreshed in place each batch.
            batch_info.num_segments = bs
            batch_info.max_len = max(forward_batch.extend_seq_lens_cpu)
            batch_info.seg_lens[:bs].copy_(
                forward_batch.extend_seq_lens, non_blocking=True
            )
            batch_info.seg_lens[bs:].zero_()
            torch.cumsum(
                batch_info.seg_lens, dim=0, out=batch_info.seg_indptr[1:]
            )
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

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)
        self.batch_info = batch_info

        # Biggest win is in decode.
        is_decode = not forward_batch.forward_mode.is_extend()
        if is_decode:
            self.compute_sgemm_routing(use_cuda_graph)
        else:
            self.sgemm_batch_info = None

        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)
        )

    def _prepare_lm_head_batch_info(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        batch_info: LoRABatchInfo,
    ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:

        # Precompute lm_head_batch_info for pruned lm_head LoRA
        pruned_lens = get_lm_head_pruned_lens(forward_batch)
        lm_head_batch_info = None
        lm_head_pass_batch_infos = None

        if pruned_lens is not None:
            pruned_total = sum(pruned_lens)
            lm_head_segments = merge_and_chunk_segments(
                weight_indices, pruned_lens, chunk_size=pruned_total
            )
            lm_head_batch_info = self._build_lm_head_batch_info(
                lm_head_segments, batch_info, pruned_total
            )

            # Precompute per-pass batch_infos for logprobs chunking
            pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)
            if pass_segments is not None:
                lm_head_pass_batch_infos = []
                for seg_wi, seg_lens_list in pass_segments:
                    pass_total = sum(seg_lens_list)
                    merged_segments = merge_and_chunk_segments(
                        seg_wi, seg_lens_list, chunk_size=pass_total
                    )
                    lm_head_pass_batch_infos.append(
                        self._build_lm_head_batch_info(
                            merged_segments, batch_info, pass_total
                        )
                    )

        return lm_head_batch_info, lm_head_pass_batch_infos

    def _build_lm_head_batch_info(
        self,
        lm_head_segments: Tuple[List[int], List[int]],
        batch_info: LoRABatchInfo,
        expected_tokens: int,
    ) -> LoRABatchInfo:
        seg_weight_indices_cpu, seg_lens_cpu = lm_head_segments
        num_segments = len(seg_weight_indices_cpu)

        seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int32, device=self.device)
        seg_indptr = torch.zeros(
            (num_segments + 1,), dtype=torch.int32, device=self.device
        )
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

        return dataclasses.replace(
            batch_info,
            # lm_head LoRA runs in the eager tail outside any captured prefill
            # graph, on freshly allocated pruned metadata.
            use_cuda_graph=False,
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
