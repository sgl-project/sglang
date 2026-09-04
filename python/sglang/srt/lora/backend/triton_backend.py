import copy
import dataclasses
from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.gemm.embedding_lora_a import embedding_lora_a_fwd
from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.kernels.ops.gemm.qkv_lora_b import qkv_lora_b_fwd
from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    dp_gather_replicate,
    get_attention_dp_rank,
    get_attention_dp_size,
)
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    MoELoRABatchInfo,
    generate_sequence_lengths,
    get_batch_token_counts,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import LoRABatchLayout, get_forward

# Fixed segment slots (one per request) baked into the captured prefill LoRA
# kernel grids; batches with more requests fall back to eager prefill.
PREFILL_CUDA_GRAPH_LORA_SEGMENTS = 32


def build_token_lora_batch_info(
    template: LoRABatchInfo,
    weight_indices: torch.Tensor,
    *,
    use_cuda_graph: bool = False,
    is_moe_lora: bool = False,
) -> LoRABatchInfo:
    """Build routing metadata for an explicit token order."""
    num_tokens = weight_indices.shape[0]
    batch_info = dataclasses.replace(
        template,
        use_cuda_graph=use_cuda_graph,
        bs=num_tokens,
        num_segments=num_tokens,
        seg_lens=torch.ones_like(weight_indices),
        seg_indptr=torch.arange(
            num_tokens + 1, dtype=torch.int32, device=weight_indices.device
        ),
        max_len=1 if num_tokens else 0,
        weight_indices=weight_indices,
        permutation=None,
        expected_tokens=num_tokens,
        req_seg_indptr=None,
        req_weight_indices=None,
        moe_lora_info=None,
    )
    if is_moe_lora or template.moe_lora_info is not None:
        batch_info.moe_lora_info = MoELoRABatchInfo(
            seg_indptr=batch_info.seg_indptr,
            req_to_lora=weight_indices,
            adapter_enabled=(
                batch_info.lora_ranks
                if use_cuda_graph
                else (batch_info.lora_ranks > 0).to(torch.int32)
            ),
            token_lora_mapping=weight_indices,
        )
    return batch_info


def _gather_dp_attention_weight_indices(
    forward_batch: ForwardBatch,
    local_batch_info: LoRABatchInfo,
    no_lora_weight_index: int,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_num_tokens = forward_batch.global_num_tokens_cpu
    assert global_num_tokens is not None
    local_num_tokens = global_num_tokens[get_attention_dp_rank()]
    local_weight_indices = torch.repeat_interleave(
        local_batch_info.weight_indices[: local_batch_info.num_segments],
        local_batch_info.seg_lens[: local_batch_info.num_segments],
        output_size=local_num_tokens if local_batch_info.use_cuda_graph else None,
    )[:local_num_tokens]
    if local_weight_indices.shape[0] < local_num_tokens:
        local_weight_indices = torch.nn.functional.pad(
            local_weight_indices,
            (0, local_num_tokens - local_weight_indices.shape[0]),
            value=no_lora_weight_index,
        )
    num_global_tokens = sum(global_num_tokens)
    if output is None:
        output = torch.empty(
            num_global_tokens,
            dtype=torch.int32,
            device=local_batch_info.weight_indices.device,
        )
    else:
        assert num_global_tokens <= output.shape[0]
        output = output[:num_global_tokens]
    dp_gather_replicate(output, local_weight_indices.clone(), forward_batch)
    return local_weight_indices, output


def gather_dp_attention_lora_batch_info(
    forward_batch: ForwardBatch,
    local_batch_info: LoRABatchInfo,
    cuda_graph_global_batch_info: LoRABatchInfo | None,
    has_global_active_lora: bool,
    no_lora_weight_index: int,
    local_lm_head_batch_info: LoRABatchInfo | None,
) -> tuple[
    LoRABatchInfo,
    LoRABatchInfo | None,
    tuple[LoRABatchInfo, list[LoRABatchInfo] | None] | None,
]:
    """Gather DP-local routing for TP-global model sections."""
    global_num_tokens = forward_batch.global_num_tokens_cpu
    if global_num_tokens is None or len(global_num_tokens) <= 1:
        return local_batch_info, None, None

    graph_batch_info = (
        cuda_graph_global_batch_info if local_batch_info.use_cuda_graph else None
    )
    local_weight_indices, global_weight_indices = _gather_dp_attention_weight_indices(
        forward_batch,
        local_batch_info,
        no_lora_weight_index,
        None if graph_batch_info is None else graph_batch_info.weight_indices,
    )
    local_num_tokens = global_num_tokens[get_attention_dp_rank()]
    if (
        not local_batch_info.use_cuda_graph
        and local_num_tokens != local_batch_info.expected_tokens
    ):
        local_batch_info = build_token_lora_batch_info(
            local_batch_info, local_weight_indices
        )

    num_global_tokens = sum(global_num_tokens)
    if graph_batch_info is None:
        global_batch_info = build_token_lora_batch_info(
            local_batch_info, global_weight_indices
        )
    else:
        graph_batch_info.bs = num_global_tokens
        graph_batch_info.num_segments = num_global_tokens
        graph_batch_info.expected_tokens = num_global_tokens
        global_batch_info = graph_batch_info
    global_batch_info.has_active_lora = has_global_active_lora

    if local_batch_info.use_cuda_graph and not forward_batch.is_extend_in_batch:
        return local_batch_info, global_batch_info, None

    global_num_tokens = forward_batch.global_num_tokens_for_logprob_cpu
    if global_num_tokens is None or len(global_num_tokens) <= 1:
        return local_batch_info, global_batch_info, None

    assert forward_batch.global_num_tokens_for_logprob_gpu is not None
    routing_forward_batch = copy.copy(forward_batch)
    routing_forward_batch.global_num_tokens_cpu = global_num_tokens
    routing_forward_batch.global_num_tokens_gpu = (
        forward_batch.global_num_tokens_for_logprob_gpu
    )
    routing_forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    routing_forward_batch.dp_local_start_pos = None
    routing_forward_batch.dp_local_num_tokens = None
    _, global_weight_indices = _gather_dp_attention_weight_indices(
        routing_forward_batch,
        local_lm_head_batch_info or local_batch_info,
        no_lora_weight_index,
    )

    template = global_batch_info
    lm_head_batch_info = build_token_lora_batch_info(template, global_weight_indices)
    chunk_size = envs.SGLANG_LOGPROB_CHUNK_SIZE.get()
    if (
        envs.SGLANG_ENABLE_LOGPROB_CHUNK.get()
        and global_weight_indices.shape[0] > chunk_size
    ):
        lm_head_pass_batch_infos = [
            build_token_lora_batch_info(
                template, global_weight_indices[start : start + chunk_size]
            )
            for start in range(0, global_weight_indices.shape[0], chunk_size)
        ]
    else:
        lm_head_pass_batch_infos = None
    return (
        local_batch_info,
        global_batch_info,
        (lm_head_batch_info, lm_head_pass_batch_infos),
    )


class TritonLoRABackend(BaseLoRABackend):
    name = "triton"
    supports_dp_attention = True
    supports_prefill_cuda_graph = True

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_loras_per_batch, device)
        # Merged-segment variant of batch_info; set alongside it in
        # prepare_lora_batch and cleared together in reset_batch_state.
        self.sgemm_batch_info: Optional[LoRABatchInfo] = None
        self.global_batch_info: Optional[LoRABatchInfo] = None
        self.global_sgemm_batch_info: Optional[LoRABatchInfo] = None
        self.cuda_graph_global_batch_info: LoRABatchInfo | None = None
        self.cuda_graph_global_sgemm_batch_info: LoRABatchInfo | None = None
        self.has_global_active_lora = False
        self._no_lora_weight_index = 0

    def reset_batch_state(self):
        super().reset_batch_state()
        self.sgemm_batch_info = None
        self.global_batch_info = None
        self.global_sgemm_batch_info = None
        self.has_global_active_lora = False

    def get_batch_info(
        self, layout: LoRABatchLayout | None = None
    ) -> Optional[LoRABatchInfo]:
        if layout is None:
            layout = get_forward().lora_batch_layout
        if layout is LoRABatchLayout.TP_GLOBAL and self.global_batch_info is not None:
            return self.global_batch_info
        return self.batch_info

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
            batch_info=self.get_batch_info(LoRABatchLayout.DP_LOCAL),
            vocab_size=vocab_size,
            extra_embeddings=extra_embeddings,
        )

    def _sgemm_info(self, pruned_batch_info=None):
        """Return the sgemm batch_info (merged segments when available)."""
        if pruned_batch_info is not None:
            return pruned_batch_info
        batch_info = self.get_batch_info()
        assert batch_info is not None, (
            "LoRA kernel invoked with no prepared batch (DP-attention idle "
            "forward?). Gate the caller on lora_active, as in "
            "sglang/srt/lora/layers.py forwards."
        )
        if batch_info is self.global_batch_info:
            return self.global_sgemm_batch_info or batch_info
        return self.sgemm_batch_info or batch_info

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

    def _allocate_cuda_graph_sgemm_batch_info(self, max_tokens: int) -> LoRABatchInfo:
        mlpb = self.max_loras_per_batch
        return LoRABatchInfo(
            bs=mlpb,
            use_cuda_graph=True,
            num_segments=mlpb,
            seg_lens=torch.zeros(mlpb, dtype=torch.int32, device=self.device),
            seg_indptr=torch.zeros(mlpb + 1, dtype=torch.int32, device=self.device),
            max_len=max_tokens,
            weight_indices=torch.arange(mlpb, dtype=torch.int32, device=self.device),
            lora_ranks=torch.zeros(mlpb, dtype=torch.int32, device=self.device),
            scalings=torch.zeros(mlpb, dtype=torch.float, device=self.device),
            permutation=torch.zeros(max_tokens, dtype=torch.int32, device=self.device),
        )

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_req: int,
    ):
        max_tokens = max_bs_in_cuda_graph * num_tokens_per_req
        mlpb = self.max_loras_per_batch
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=max_bs_in_cuda_graph,
                use_cuda_graph=True,
                num_segments=None,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_req, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                max_len=num_tokens_per_req,
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
            self.cuda_graph_sgemm_batch_info = (
                self._allocate_cuda_graph_sgemm_batch_info(max_tokens)
            )

    def init_dp_attention_cuda_graph_batch_info(self, max_num_tokens: int) -> None:
        local_batch_info = getattr(self, "cuda_graph_batch_info", None)
        assert (
            local_batch_info is not None
        ), "init_cuda_graph_batch_info must run before DP-attention graph init"
        max_global_num_tokens = max_num_tokens * get_attention_dp_size()
        self.cuda_graph_global_batch_info = build_token_lora_batch_info(
            local_batch_info,
            torch.zeros(max_global_num_tokens, dtype=torch.int32, device=self.device),
            use_cuda_graph=True,
            is_moe_lora=self.is_moe_lora,
        )
        self.cuda_graph_global_sgemm_batch_info = (
            self._allocate_cuda_graph_sgemm_batch_info(max_global_num_tokens)
        )

    def init_prefill_cuda_graph_batch_info(self, max_num_tokens: int):
        num_slots = PREFILL_CUDA_GRAPH_LORA_SEGMENTS
        mlpb = self.max_loras_per_batch
        with torch.device(self.device):
            # bs pinned at num_slots so the captured grids cover any replay
            # batch; slots past the live batch keep seg_lens == 0 and no-op.
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

    def _build_sgemm_routing(
        self,
        batch_info: LoRABatchInfo,
        out: Optional[LoRABatchInfo] = None,
    ) -> LoRABatchInfo:
        """Sort tokens by adapter and build merged segments for sgemm LoRA."""
        bs = batch_info.bs
        mlpb = self.max_loras_per_batch
        wi = batch_info.weight_indices[:bs]

        perm = torch.argsort(wi, stable=True).to(torch.int32)
        sorted_wi = wi[perm]
        adapter_ids = torch.arange(mlpb, device=wi.device, dtype=torch.int32)
        seg_starts = torch.searchsorted(sorted_wi, adapter_ids)
        seg_ends = torch.searchsorted(sorted_wi, adapter_ids, right=True)
        seg_lens = seg_ends - seg_starts

        if out is not None:
            out.permutation[:bs] = perm
            out.seg_lens[:] = seg_lens
            out.seg_indptr[0:1].zero_()
            torch.cumsum(out.seg_lens, dim=0, out=out.seg_indptr[1:])
            out.max_len = bs
            out.lora_ranks[:mlpb] = batch_info.lora_ranks[:mlpb]
            out.scalings[:mlpb] = batch_info.scalings[:mlpb]
            return out

        seg_indptr = torch.zeros(mlpb + 1, dtype=torch.int32, device=wi.device)
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        return LoRABatchInfo(
            bs=mlpb,
            use_cuda_graph=False,
            num_segments=mlpb,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            max_len=bs,
            weight_indices=adapter_ids,
            lora_ranks=batch_info.lora_ranks[:mlpb].clone(),
            scalings=batch_info.scalings[:mlpb].clone(),
            permutation=perm,
        )

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
        self._no_lora_weight_index = next(
            (index for index, rank in enumerate(lora_ranks) if rank == 0),
            weight_indices[0] if weight_indices else 0,
        )
        self.has_global_active_lora = any(rank > 0 for rank in lora_ranks)

        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_target_verify():
            expected_tokens = bs * forward_batch.spec_info.draft_token_num
        elif forward_batch.forward_mode.is_extend():
            expected_tokens = sum(forward_batch.extend_seq_lens_cpu)
        else:
            expected_tokens = bs

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            if forward_batch.forward_mode.is_target_verify():
                # seg_lens were pre-filled at the captured per-request width
                # (stored as max_len); another width would silently
                # mis-segment adapters onto the wrong token rows.
                assert forward_batch.spec_info.draft_token_num == batch_info.max_len, (
                    "target-verify width "
                    f"{forward_batch.spec_info.draft_token_num} does not match "
                    f"the captured LoRA cuda-graph width {batch_info.max_len}"
                )
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = forward_batch.batch_size
        elif use_prefill_cuda_graph:
            batch_info = self.prefill_cuda_graph_batch_info
            # bs stays pinned at the allocated slot count; slots past the
            # live batch no-op via seg_lens == 0.
            batch_info.num_segments = bs
            batch_info.max_len = max(forward_batch.extend_seq_lens_cpu)
            batch_info.seg_lens[:bs].copy_(
                forward_batch.extend_seq_lens, non_blocking=True
            )
            batch_info.seg_lens[bs:].zero_()
            torch.cumsum(batch_info.seg_lens, dim=0, out=batch_info.seg_indptr[1:])
        else:
            if forward_batch.forward_mode.is_idle():
                # DP-attention idle ranks still participate in TP-global LoRA
                # routing, but have no local request segments.
                max_len = 1
                seg_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
            else:
                # max_len comes from the CPU-side counts to avoid a D2H transfer.
                _, max_len = get_batch_token_counts(forward_batch)
                seg_lens = generate_sequence_lengths(forward_batch, device=self.device)
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

        batch_info.expected_tokens = expected_tokens

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True)
        if use_cuda_graph and bs < batch_info.weight_indices.shape[0]:
            batch_info.weight_indices[bs:].fill_(self._no_lora_weight_index)

        batch_info = self._add_moe_lora_info(forward_batch, batch_info)
        self.batch_info = batch_info
        self.global_batch_info = None
        self.global_sgemm_batch_info = None

        # Biggest win is in decode.
        is_decode = not forward_batch.forward_mode.is_extend()
        if is_decode:
            self.sgemm_batch_info = self._build_sgemm_routing(
                batch_info,
                self.cuda_graph_sgemm_batch_info if use_cuda_graph else None,
            )
        else:
            self.sgemm_batch_info = None

        self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
            self._prepare_lm_head_batch_info(forward_batch, weight_indices, batch_info)
        )

    def prepare_global_lora_batch(self, forward_batch: ForwardBatch) -> None:
        assert self.batch_info is not None
        previous_local_batch_info = self.batch_info
        self.batch_info, self.global_batch_info, lm_head_batch_infos = (
            gather_dp_attention_lora_batch_info(
                forward_batch,
                self.batch_info,
                self.cuda_graph_global_batch_info,
                self.has_global_active_lora,
                self._no_lora_weight_index,
                self.lm_head_batch_info,
            )
        )
        if self.global_batch_info is None:
            self.global_sgemm_batch_info = None
            return
        if self.global_batch_info.use_cuda_graph:
            self.global_sgemm_batch_info = self._build_sgemm_routing(
                self.global_batch_info,
                self.cuda_graph_global_sgemm_batch_info,
            )
            return

        if lm_head_batch_infos is not None:
            self.lm_head_batch_info, self.lm_head_pass_batch_infos = lm_head_batch_infos

        is_decode = not forward_batch.is_extend_in_batch
        if is_decode:
            if self.batch_info is not previous_local_batch_info:
                self.sgemm_batch_info = self._build_sgemm_routing(self.batch_info)
            self.global_sgemm_batch_info = self._build_sgemm_routing(
                self.global_batch_info
            )
        else:
            self.global_sgemm_batch_info = None

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
