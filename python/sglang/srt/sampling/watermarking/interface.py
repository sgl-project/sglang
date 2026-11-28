"""The algorithm-neutral contract between the engine and a watermark algorithm.

Shared engine code calls only these types and the entry points in ``runtime``;
everything a particular algorithm does lives behind them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from torch import Tensor

if TYPE_CHECKING:
    from sglang.srt.arg_groups.fields.exec_ import ExecFeatures
    from sglang.srt.arg_groups.model_override_base import ResolvingConfig
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.sampling.verify_accept import VerifyAccept


class WatermarkRows(Protocol):
    """One step's per-row watermark policy, built by the scheduler at each
    launch from the step's rows of ``SamplingParamsPool``.

    Exists only if at least one row is a candidate (enabled and top_k > 1); the
    batch carries None otherwise. Only the algorithm's own engine reads its
    fields; other code only checks it for None.
    """


class SampleWatermark(Protocol):
    """One ordinary sampling step's watermark, bound to its forward batch.

    ``apply`` edits logits only and leaves the draw to the pipeline;
    ``fused_sample`` is a fast path that runs the stage and the draw in one kernel.
    """

    # Batch rows whose request top-p and min-p the stage already applied, which
    # the stock sampler draws with top_p=1 and min_p=0; None when there are none.
    request_filtered_rows: Optional[Tensor]

    def fused_sample(self, logits: Tensor, *, keep_logits: bool) -> Optional[Tensor]:
        """The step's token ids, from the stage and the draw in one kernel; None
        when the algorithm has no such kernel for this step, and the pipeline
        runs ``apply`` and the stock sampler instead. With token ids the stock
        sampler is skipped whole: no logprobs, no sampling masks, no grammar or
        TP token all-reduce; ``commit`` still runs on them, so this must not
        commit them."""
        ...

    def apply(self, logits: Tensor, *, keep_logits: bool) -> Optional[Tensor]:
        """Watermark ``logits`` in place; returns the pre-watermark logits iff
        ``keep_logits``, for the logprobs."""
        ...

    def commit(self, token_ids: Tensor) -> None: ...


class VerifyWatermark(Protocol):
    """One speculative verify's watermark, bound to its batch rows."""

    # As in ``SampleWatermark``; the accept draws these rows with top_p=1 and
    # min_p=0.
    request_filtered_rows: Optional[Tensor]

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
    ) -> Tensor:
        """The logits to verify against: ``logits`` watermarked in place, or a
        watermarked copy when ``keep_logits``.

        The NV mega runtime path (SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION)
        builds the provider's ``VerifySamplingView`` from the sampling info
        before any apply (``make_post_verify_io``), and a provider forces
        through the bound object's ABI v2 methods instead of apply, so an
        algorithm with request-filtered rows must refuse that path in its
        ``check_server_args``.

        ``custom_mask`` None means a chain draft: ``draft_tokens`` and
        ``positions`` are [batch, draft_token_num].
        """
        ...

    def commit(self, accept: VerifyAccept) -> None:
        """Commit the applied verify's accepted tokens to the watermark state."""
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

    # How many int32 fields of `SamplingParamsTable` the algorithm keeps per request.
    NUM_POOL_FIELDS: int

    def pool_values(
        self, request: Req, *, features: ExecFeatures
    ) -> tuple[bool, Sequence[int]]:
        """Whether the request is a candidate (enabled and top_k > 1), and its
        NUM_POOL_FIELDS int32 values, written to the pool at admission."""
        ...

    def build_rows(self, values: Tensor, candidates_host: list[bool]) -> WatermarkRows:
        """The batch's rows from its gathered [NUM_POOL_FIELDS, bs] values;
        called only when some row is a candidate."""
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
