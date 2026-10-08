"""Dense LoRA execution with request metadata shared by Linear and MoE layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from sglang.kernels.ops.lora.dense.embedding_lora_a import embedding_lora_a_tokens_fwd
from sglang.srt.lora.backend.base_backend import (
    BaseLoRABackend,
    _compute_token_lora_mapping,
)
from sglang.srt.lora.dense.plan import DenseLoraKind
from sglang.srt.lora.dense.runner import DenseLoraRunner
from sglang.srt.lora.utils import (
    Phase,
    get_batch_token_counts,
    get_lm_head_pruned_lens,
    merge_and_chunk_segments,
)
from sglang.srt.lora.workspace import LoraWorkspace
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class _BatchInfo:
    packed: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    weight_indices: torch.Tensor
    seg_indptr: torch.Tensor
    token_slots: torch.Tensor
    use_cuda_graph: bool  # Static metadata for either graph family, including warmup.
    num_requests: int = 0
    num_tokens: int = 0
    tokens_per_request: int = 0
    has_active_lora: bool = False  # Inkling's shared sink skips inactive eager batches.
    is_prefill: bool = False  # MoE plan phase; target verification uses decode.


@dataclass
class _LmHeadBatchInfo:
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    expected_tokens: int
    max_len: int
    use_cuda_graph: ClassVar[bool] = False


class TritonV2LoRABackend(BaseLoRABackend):
    name = "triton_v2"
    supports_prefill_cuda_graph = True
    skip_inactive_dense_lora = True

    def __init__(self, max_loras_per_batch: int, device: torch.device, **kwargs):
        super().__init__(max_loras_per_batch, device)
        self.lora_workspace = LoraWorkspace()
        self.runner = DenseLoraRunner(
            self.lora_workspace, max_loras=max_loras_per_batch, device=device
        )
        self.lm_head_runner: DenseLoraRunner | None = None
        self.decode_cuda_graph_batch_info: _BatchInfo | None = None

    def validate_lora_targets(self, base_model, target_modules: set[str]) -> None:
        if "lm_head" in target_modules and self.lm_head_runner is None:
            # Pruned/chunked logits run outside the model graph on their own rows.
            self.lm_head_runner = DenseLoraRunner(
                LoraWorkspace(),
                max_loras=self.max_loras_per_batch,
                device=self.device,
                architecture=self.runner.architecture,
            )

    def reset_batch_state(self):
        super().reset_batch_state()
        self.runner.reset()
        if self.lm_head_runner is not None:
            self.lm_head_runner.reset()

    def reset_routing_cache(self) -> None:
        self.runner.reset_routes()
        if self.lm_head_runner is not None:
            self.lm_head_runner.reset_routes()

    def init_cuda_graph_moe_buffers(
        self, max_bs, max_loras, compute_dtype, moe_layer, *, prefill=False
    ):
        # V2 owns the metadata; the MoE workspace owns kernel scratch.
        pass

    def _new_batch(
        self, requests: int, tokens: int, *, graph: bool = False
    ) -> _BatchInfo:
        slots = self.max_loras_per_batch
        head = 2 * slots
        alloc = torch.zeros if graph else torch.empty
        packed = alloc(head + 2 * requests + 1, dtype=torch.int32, device=self.device)
        token_slots = torch.empty((tokens,), dtype=torch.int32, device=self.device)
        if graph:
            token_slots.fill_(-1)
        return _BatchInfo(
            packed=packed,
            lora_ranks=packed[:slots],
            scalings=packed[slots : 2 * slots].view(torch.float32),
            weight_indices=packed[head : head + requests],
            seg_indptr=packed[head + requests :],
            token_slots=token_slots,
            use_cuda_graph=graph,
        )

    def init_decode_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_req: int
    ):
        batch = self._new_batch(
            max_bs_in_cuda_graph,
            max_bs_in_cuda_graph * num_tokens_per_req,
            graph=True,
        )
        batch.tokens_per_request = num_tokens_per_req
        batch.seg_indptr.copy_(
            torch.arange(
                max_bs_in_cuda_graph + 1, dtype=torch.int32, device=self.device
            )
            * num_tokens_per_req
        )
        self.decode_cuda_graph_batch_info = batch

    def init_prefill_cuda_graph_batch_info(
        self, max_num_tokens: int, max_num_requests: int | None = None
    ):
        requests = max_num_tokens if max_num_requests is None else max_num_requests
        self.prefill_cuda_graph_batch_info = self._new_batch(
            requests, max_num_tokens, graph=True
        )
        self.prefill_cuda_graph_max_bs = requests
        self.prefill_cuda_graph_max_tokens = max_num_tokens

    def _fill_batch(
        self,
        batch: _BatchInfo,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        lengths: list[int] | None,
    ) -> None:
        slots = self.max_loras_per_batch
        requests = len(weight_indices)
        if requests > batch.weight_indices.numel():
            raise ValueError("LoRA requests exceed the graph metadata capacity")
        indptr = []
        if lengths is not None:
            indptr = [0]
            for length in lengths:
                indptr.append(indptr[-1] + length)
        head = 2 * slots + requests
        host = torch.empty(head + len(indptr), dtype=torch.int32, pin_memory=True)
        host[:slots] = torch.as_tensor(lora_ranks, dtype=torch.int32)
        host[slots : 2 * slots].view(torch.float32).copy_(
            torch.as_tensor(scalings, dtype=torch.float32)
        )
        host[2 * slots : head] = torch.as_tensor(weight_indices, dtype=torch.int32)
        batch.packed[:head].copy_(host[:head], non_blocking=True)
        if lengths is not None:
            host[head:] = torch.as_tensor(indptr, dtype=torch.int32)
            batch.seg_indptr[: requests + 1].copy_(host[head:], non_blocking=True)

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_decode_cuda_graph: bool,
        use_prefill_cuda_graph: bool = False,
    ):
        requests = forward_batch.batch_size
        num_tokens, max_len = get_batch_token_counts(forward_batch)
        if use_decode_cuda_graph:
            batch = self.decode_cuda_graph_batch_info
            if batch is None:
                raise RuntimeError("LoRA decode graph metadata is not initialized")
            if max_len != batch.tokens_per_request:
                raise ValueError(
                    f"LoRA request width {max_len} does not match captured width "
                    f"{batch.tokens_per_request}"
                )
            lengths = None
        else:
            if use_prefill_cuda_graph:
                batch = self.prefill_cuda_graph_batch_info
                if batch is None:
                    raise RuntimeError("LoRA prefill graph metadata is not initialized")
            else:
                batch = self._new_batch(requests, num_tokens)
            lengths = (
                list(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend_without_speculative()
                else [max_len] * requests
            )
        if len(weight_indices) != requests:
            raise ValueError("LoRA needs one adapter slot per request")
        self._fill_batch(batch, weight_indices, lora_ranks, scalings, lengths)

        request_indptr = batch.seg_indptr[: requests + 1]
        request_slots = batch.weight_indices[:requests]
        # Only live requests map tokens; the kernel also clears the padded tail.
        _compute_token_lora_mapping(
            num_tokens,
            request_indptr,
            batch.lora_ranks,
            request_slots,
            batch.token_slots,
            max_len=max_len,
            bucket_len=batch.token_slots.numel(),
        )
        batch.num_requests = requests
        batch.num_tokens = num_tokens
        self.batch_info = batch
        self.runner.begin_batch(
            token_slots=batch.token_slots,
            lora_ranks=batch.lora_ranks,
            scalings=batch.scalings,
            num_tokens=num_tokens,
            phase=(
                Phase.PREFILL
                if forward_batch.forward_mode.is_extend()
                else Phase.DECODE
            ),
            graph_mode=batch.use_cuda_graph,
            is_prefill_graph=use_prefill_cuda_graph,
        )
        if self.lm_head_runner is not None:
            self.lm_head_batch_info, self.lm_head_pass_batch_infos = (
                self._prepare_lm_head_batch_info(forward_batch, weight_indices)
            )
            self._lm_head_pass_idx = None

    def _prepare_lm_head_batch_info(self, forward_batch, weight_indices):
        pruned_lens = get_lm_head_pruned_lens(forward_batch)
        if pruned_lens is None:
            return None, None
        pruned_total = sum(pruned_lens)
        full_pruned_info = self._build_lm_head_batch_info(
            merge_and_chunk_segments(weight_indices, pruned_lens, pruned_total)
        )
        pass_segments = self._get_lm_head_pass_segments(weight_indices, pruned_lens)
        return full_pruned_info, (
            [self._build_lm_head_batch_info(segments) for segments in pass_segments]
            if pass_segments is not None
            else None
        )

    def _build_lm_head_batch_info(self, segments) -> _LmHeadBatchInfo:
        weight_indices, lengths = segments
        indptr = [0]
        for length in lengths:
            indptr.append(indptr[-1] + length)
        host = torch.tensor(indptr + weight_indices, dtype=torch.int32, pin_memory=True)
        device = host.to(self.device, non_blocking=True)
        return _LmHeadBatchInfo(
            seg_indptr=device[: len(indptr)],
            weight_indices=device[len(indptr) :],
            expected_tokens=indptr[-1],
            max_len=max(lengths, default=0),
        )

    def _runner_for(
        self, pruned_batch_info: _LmHeadBatchInfo | None
    ) -> DenseLoraRunner:
        if pruned_batch_info is None:
            return self.runner
        if self.lm_head_runner is None:
            raise RuntimeError("lm_head LoRA was not enabled in the target modules")
        batch = self.batch_info
        token_slots = _compute_token_lora_mapping(
            pruned_batch_info.expected_tokens,
            pruned_batch_info.seg_indptr,
            batch.lora_ranks,
            pruned_batch_info.weight_indices,
            None,
            max_len=pruned_batch_info.max_len,
        )
        self.lm_head_runner.begin_batch(
            token_slots=token_slots,
            lora_ranks=batch.lora_ranks,
            scalings=batch.scalings,
            num_tokens=pruned_batch_info.expected_tokens,
            phase=self.runner.phase,
            graph_mode=False,
        )
        return self.lm_head_runner

    def forward_with_base(
        self,
        layer,
        x,
        base_fn,
        lora_a,
        lora_b,
        output_offset,
        offsets,
        *,
        all_reduce=None,
        kind: DenseLoraKind = DenseLoraKind.LINEAR,
        pruned_batch_info: _LmHeadBatchInfo | None = None,
    ):
        runner = self._runner_for(pruned_batch_info)
        plan = runner.plan_for(
            kind,
            lora_b.shape[-1],
            0 if kind is DenseLoraKind.EMBEDDING else lora_a.shape[-1],
            offsets[-1],
            num_tokens=x.shape[0],
        )
        a_fn = None
        if kind is DenseLoraKind.EMBEDDING:

            def a_fn():
                return embedding_lora_a_tokens_fwd(
                    input_ids=x,
                    weights=lora_a,
                    token_lora_mapping=runner.token_slots,
                    lora_ranks=runner.lora_ranks,
                    vocab_size=layer.vocab_size,
                    extra_embeddings=layer.new_embeddings_buffer,
                )

        return runner.apply(
            x,
            base_fn,
            plan,
            a=lora_a,
            b=lora_b,
            offsets=offsets,
            all_reduce=all_reduce,
            a_fn=a_fn,
        )

    def run_qkv_lora(self, *args, **kwargs):
        raise NotImplementedError("triton_v2 linear layers must use forward_with_base")

    def run_gate_up_lora(self, *args, **kwargs):
        raise NotImplementedError("triton_v2 linear layers must use forward_with_base")
