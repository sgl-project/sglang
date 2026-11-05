"""Greedy argmax recorded inside Unlimited-OCR v2 decode CUDA graphs.

The decode graph I/O (``runner/uocr_decode_graph_io.py``) relays these ids to the
next step's inputs on device, so all-greedy decode skips the eager sampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.configs.model_config import is_unlimited_ocr_v2

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.watermarking.core import WatermarkState


class UocrGreedyGraph:
    """Own the fixed-address argmax output that decode graph capture records into."""

    def __init__(self, *, max_bs: int, device: torch.device | str):
        # int64 because torch.argmax(out=...) requires it.
        self.ids = torch.empty((max_bs,), dtype=torch.int64, device=device)
        # The batch whose forward replayed the graph; ids are stale for any other.
        self._replayed_batch: Optional[ForwardBatch] = None
        # Records the relay of these ids; replaced by each decode-graph capture.
        self._relay_tail: Optional[Callable[..., None]] = None

    @classmethod
    def maybe_create(
        cls,
        *,
        model_config: ModelConfig,
        max_running_requests: Optional[int],
        device: torch.device | str,
    ) -> Optional[UocrGreedyGraph]:
        """Recorder for Unlimited-OCR v2; None for every other model."""
        if not is_unlimited_ocr_v2(model_config.hf_config):
            return None
        max_bs = int(max_running_requests or 0)
        if max_bs <= 0:
            return None
        return cls(max_bs=max_bs, device=device)

    def capture_tail_hook(
        self, runner: Any, out: Any, forward_batch: ForwardBatch, num_tokens: int
    ) -> None:
        """Record the argmax of the captured logits into ``ids``."""
        if out is None:
            return
        logits = out.next_token_logits
        torch.argmax(logits, dim=-1, keepdim=False, out=self.ids[: logits.shape[0]])
        if self._relay_tail is not None:
            self._relay_tail(runner, out, forward_batch, num_tokens)

    def bind_relay_tail(self, hook: Optional[Callable[..., None]]) -> None:
        """Record ``hook`` after the argmax in later captures, replacing any earlier one."""
        self._relay_tail = hook

    def before_forward(
        self,
        forward_batch: ForwardBatch,
        *,
        replays_graph: bool,
        decode_graph_runner: Any,
    ) -> None:
        """Track graph replays; load relayed inputs for a deferred eager decode."""
        self._replayed_batch = forward_batch if replays_graph else None
        if not replays_graph and forward_batch.uocr_graph_io_deferred:
            decode_graph_runner.uocr_graph_io.materialize_eager(forward_batch)

    def recorded_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        *,
        watermark_state: Optional[WatermarkState],
    ) -> Optional[torch.Tensor]:
        """The graph-recorded greedy ids if they equal what the sampler would pick."""
        if forward_batch is not self._replayed_batch:
            return None
        sampling_info = forward_batch.sampling_info
        if watermark_state is not None and sampling_info.has_watermark_candidates:
            return None
        if (
            not forward_batch.forward_mode.is_decode()
            or not sampling_info.is_all_greedy
            or forward_batch.return_logprob
        ):
            return None
        # The argmax ran inside the graph, before _preprocess_logits and watermark
        # forcing, so any transform of the logits invalidates it.
        if (
            sampling_info.logit_bias is not None
            or sampling_info.grammar_mask is not None
            or sampling_info.has_custom_logit_processor
            or (
                sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required
            )
        ):
            return None
        rows = int(logits_output.next_token_logits.shape[0])
        if rows > self.ids.shape[0]:
            return None
        # The sampler's contract (and every consumer) is int32.
        return self.ids[:rows].to(torch.int32)
