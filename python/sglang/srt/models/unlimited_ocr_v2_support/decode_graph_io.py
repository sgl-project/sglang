"""Relay Unlimited-OCR decode inputs and outputs through CUDA graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.ops.attention.clamp_position import clamp_position

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.unlimited_ocr_v2_support.greedy_graph import UocrGreedyGraph


class UOCRDecodeGraphIO:
    """Manage fixed-address Unlimited-OCR decode graph inputs and outputs."""

    def __init__(
        self, *, num_rows: int, device: torch.device | str, captured_ids: torch.Tensor
    ):
        # Next decode input per req_to_token row.
        self.relay = torch.zeros(num_rows, dtype=torch.int64, device=device)
        self._captured_ids = captured_ids

    @classmethod
    def maybe_create(
        cls,
        *,
        attn_backend,
        greedy_graph: Optional[UocrGreedyGraph],
        num_rows: int,
        device: torch.device | str,
    ) -> Optional[UOCRDecodeGraphIO]:
        """Create graph I/O state when the loaded model supports it."""
        if greedy_graph is None:
            return None
        from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
            UnlimitedOCRV2AttnBackend,
        )

        if not isinstance(attn_backend, UnlimitedOCRV2AttnBackend):
            return None
        return cls(num_rows=num_rows, device=device, captured_ids=greedy_graph.ids)

    def can_defer(self, batch: ScheduleBatch) -> bool:
        """Return whether the CUDA graph can produce this batch's decode inputs."""
        sampling = batch.sampling_info
        penalizer = sampling.penalizer_orchestrator
        return bool(
            batch.enable_overlap
            and batch.forward_mode.is_decode()
            and batch.spec_algorithm.is_none()
            and batch.input_ids is None
            and batch.beam_tail is None
            and batch.return_logprob is False
            and sampling.is_all_greedy
            and sampling.logit_bias is None
            and sampling.grammar_mask is None
            and not sampling.has_custom_logit_processor
            and not sampling.has_watermark_candidates
            and (penalizer is None or not penalizer.is_required)
        )

    def seed(self, indices: torch.Tensor, tokens: torch.Tensor) -> None:
        """Seed relay slots from an eager decode result."""
        self.relay[indices] = tokens.to(self.relay.dtype)

    def record_prologue(self, forward_batch: ForwardBatch) -> None:
        """Record graph operations that load relayed decode inputs."""
        forward_batch.input_ids.copy_(self.relay[forward_batch.req_pool_indices])
        forward_batch.positions.copy_(clamp_position(forward_batch.seq_lens))

    def record_tail(
        self, runner, out, forward_batch: ForwardBatch, num_tokens: int
    ) -> None:
        """Record graph operations that publish sampled decode tokens."""
        if out is None:
            return
        rows = int(out.next_token_logits.shape[0])
        self.relay[forward_batch.req_pool_indices[:rows]] = self._captured_ids[:rows]

    def materialize_eager(self, forward_batch: ForwardBatch) -> None:
        """Load relayed inputs when the current decode runs eagerly."""
        forward_batch.input_ids = self.relay[forward_batch.req_pool_indices]
        forward_batch.positions = clamp_position(forward_batch.seq_lens)
        forward_batch.uocr_graph_io_deferred = False

    def graph_published(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> bool:
        """Return whether the graph already published the sampled tokens."""
        return result.can_run_cuda_graph and self.can_defer(batch)

    def relay_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        indices: torch.Tensor,
    ) -> None:
        """Seed the relay from a result the graph did not already publish."""
        if result.has_sampled_token_ids and not self.graph_published(batch, result):
            self.seed(indices, result.next_token_ids)


def resolve_uocr_decode_graph_io(decode_runner: Any) -> Optional[UOCRDecodeGraphIO]:
    """Graph I/O state of the current decode graph runner, if it relays."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    if isinstance(decode_runner, DecodeCudaGraphRunner):
        return decode_runner.uocr_graph_io
    return None
