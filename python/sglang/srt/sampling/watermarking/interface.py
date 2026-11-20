"""The algorithm-neutral contract between the engine and a watermark algorithm.

Shared engine code calls only these types and the entry points in ``runtime``;
everything a particular algorithm does lives behind them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from torch import Tensor

if TYPE_CHECKING:
    import torch

    from sglang.srt.arg_groups.fields.exec_ import ExecFeatures
    from sglang.srt.arg_groups.model_override_base import ResolvingConfig
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams


class WatermarkRows(Protocol):
    """One batch's per-row watermark policy.

    Exists only if at least one row is a candidate (enabled and top_k > 1); the
    batch carries None otherwise. Immutable: an overlapped forward may still read
    the old object, so filter and concat return new objects.
    """

    def filter(
        self, keep_indices: list[int], keep_indices_device: Tensor
    ) -> Optional[WatermarkRows]:
        """The kept rows, or None when no candidate row is kept."""
        ...

    def concat(
        self, other: Optional[WatermarkRows], self_len: int, other_len: int
    ) -> WatermarkRows:
        """These rows followed by ``other``'s, or by ``other_len`` off rows."""
        ...

    def prepend_off(self, num_rows: int) -> WatermarkRows: ...


def merge_watermark_rows(
    a: Optional[WatermarkRows],
    b: Optional[WatermarkRows],
    a_len: int,
    b_len: int,
) -> Optional[WatermarkRows]:
    if a is None:
        return None if b is None else b.prepend_off(a_len)
    return a.concat(b, a_len, b_len)


@dataclass(frozen=True)
class SampleStep:
    # What the stock sampler must sample with.
    sampling_info: SamplingBatchInfo
    # The pre-watermark logits, iff the caller asked to keep them for logprobs.
    logprob_logits: Optional[Tensor]
    # Set when the algorithm already sampled. The stock sampler is then skipped
    # whole: no logprobs, no sampling masks, no grammar or TP token all-reduce;
    # these become the algorithm's job. commit() still runs on these tokens,
    # so an algorithm must not also commit them inside apply().
    token_ids: Optional[Tensor] = None


class SampleWatermark(Protocol):
    """One ordinary sampling step's watermark, bound to its forward batch."""

    def apply(self, logits: Tensor, *, keep_logits: bool) -> SampleStep: ...

    def commit(self, token_ids: Tensor) -> None: ...


class VerifyWatermark(Protocol):
    """One speculative verify's watermark, bound to its batch rows."""

    def apply(
        self,
        logits: Tensor,
        *,
        sampling_info: SamplingBatchInfo,
        req_pool_indices: Tensor,
        draft_tokens: Tensor,
        positions: Tensor,
        draft_token_num: int,
        custom_mask: Optional[Tensor],
        keep_logits: bool,
    ) -> tuple[Tensor, SamplingBatchInfo]:
        """The logits to verify against and the sampling info to verify with.

        Callers accept with the returned sampling info, not the one passed in.
        The NV mega runtime path (SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION)
        builds the provider's ``VerifySamplingView`` from the sampling info
        before any apply (``make_post_verify_io``), and a provider forces
        through the bound object's ABI v2 methods instead of apply, so an
        algorithm that changes the sampling info must refuse that path in its
        ``check_server_args``.

        ``custom_mask`` None means a chain draft: ``draft_tokens`` and
        ``positions`` are [batch, draft_token_num].
        """
        ...

    def record(self, accept_index: Optional[Tensor], accept_lens: Tensor) -> None:
        """Record the accepted positions; ``accept_index`` None after a chain
        apply, where each row accepts a prefix of its draft positions."""
        ...


class WatermarkEngine(Protocol):
    """Per-server device state, owned by the target ModelRunner (``model_runner.watermark``)."""

    def seed_payload(self, batch: ScheduleBatch) -> Optional[Any]:
        """Host-side state the next sample seeds from (``ForwardBatch.watermark_seed``)."""
        ...

    def bind_sample(self, forward_batch: ForwardBatch) -> Optional[SampleWatermark]:
        """The batch's sampling watermark, or None when it has no watermark rows."""
        ...

    def bind_verify(
        self, rows: Optional[WatermarkRows], *, full_mask: bool
    ) -> Optional[VerifyWatermark]:
        """The verify's watermark, or None when the batch has no watermark rows."""
        ...

    def append_accept_tokens(
        self,
        rows: WatermarkRows,
        req_pool_indices: Tensor,
        accept_tokens: Tensor,
        accept_lens: Tensor,
    ) -> None: ...


# Provider ABI v2 (``VerifySamplingView.watermark_*``) fields of a batch without
# watermark rows.
NO_WATERMARK_VIEW_FIELDS: dict[str, Any] = {
    "watermark_keys": None,
    "watermark_context_windows": None,
    "watermark_enabled": None,
    "has_watermark_candidates": False,
}


class WatermarkAdapter(Protocol):
    """An algorithm's adapter module, as ``runtime`` dispatches to it."""

    def make_engine(
        self,
        features: ExecFeatures,
        *,
        max_num_reqs: int,
        max_contexts_per_req: int,
        vocab_size: int,
        device: str,
    ) -> WatermarkEngine: ...

    def build_rows(
        self,
        requests: Sequence[Req],
        *,
        features: ExecFeatures,
        device: torch.device | str,
    ) -> Optional[WatermarkRows]:
        """The batch's rows, or None when no request is a candidate."""
        ...

    def request_enabled(
        self, sampling_params: SamplingParams, *, features: ExecFeatures
    ) -> bool:
        """Whether the request is watermarked; raises ValueError for an invalid
        request. Runs for every request, watermarking on or off."""
        ...

    def check_server_args(self, cfg: ResolvingConfig) -> None:
        """Server-argument checks; runs only with --enable-watermark."""
        ...

    def watermark_view_fields(self, rows: WatermarkRows) -> dict[str, Any]:
        """The provider ABI v2 ``VerifySamplingView.watermark_*`` fields."""
        ...
