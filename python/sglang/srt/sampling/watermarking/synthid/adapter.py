"""SynthID Text behind the engine's watermark interface: request policy,
per-batch rows, server checks, and the engine that binds them to the SynthID
sampler.

A watermarked row's slot is seeded at every extend that prefills it: a fresh
request starts from a zero context, a resumed one replays its generated
tokens. Since the sampler's commit has no mask, the token a non-final chunk of
a chunked prefill commits is undone by the reseed at the next chunk.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.async_probe import sanitize_nan_logits
from sglang.srt.utils.device import is_cuda

from ..batch_layout import num_prefill_rows
from ..config import SYNTHID_CONFIG_FIELDS, WatermarkRequestConfig
from ..interface import NO_WATERMARK_VIEW_FIELDS, SampleStep
from .algorithm import SynthIDConfig
from .sampler import FilterProbs, SynthIDSampler, top_k_top_p_min_p_filter_probs

if TYPE_CHECKING:
    from sglang.srt.arg_groups.fields.exec_ import ExecFeatures
    from sglang.srt.arg_groups.model_override_base import ResolvingConfig
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams


class SynthIDRows(msgspec.Struct, frozen=True):
    # Batch indices of the watermarked rows, int64 on the batch device.
    indices: torch.Tensor
    on_host: list[bool]
    all_on: bool

    def __repr__(self) -> str:
        return f"SynthIDRows(rows={len(self.on_host)!r})"

    def filter(
        self, keep_indices: list[int], keep_indices_device: torch.Tensor
    ) -> Optional[SynthIDRows]:
        return _rows(
            [self.on_host[index] for index in keep_indices], self.indices.device
        )

    def concat(
        self, other: Optional[SynthIDRows], self_len: int, other_len: int
    ) -> SynthIDRows:
        other_on = other.on_host if other is not None else [False] * other_len
        return _rows(self.on_host + other_on, self.indices.device)

    def prepend_off(self, num_rows: int) -> SynthIDRows:
        return _rows([False] * num_rows + self.on_host, self.indices.device)

    def take(self, tensor: torch.Tensor) -> torch.Tensor:
        """The watermarked rows of a per-row tensor."""
        return tensor if self.all_on else tensor[self.indices]

    @property
    def partial_indices(self) -> Optional[torch.Tensor]:
        """The watermarked row indices, or None when every row is watermarked:
        the row selector ``SynthIDSampler`` takes."""
        return None if self.all_on else self.indices


def _rows(on_host: list[bool], device: torch.device | str) -> Optional[SynthIDRows]:
    indices = [index for index, on in enumerate(on_host) if on]
    if not indices:
        return None
    return SynthIDRows(
        indices=torch.tensor(indices, dtype=torch.int64, device=device),
        on_host=on_host,
        all_on=len(indices) == len(on_host),
    )


def _resolve_enabled(
    config: Optional[WatermarkRequestConfig],
    *,
    server_enabled: bool,
    default_enabled: bool,
    enforce_all: bool,
) -> bool:
    if config is None:
        enabled = default_enabled or enforce_all
    else:
        if config.key is not None or config.context_window is not None:
            raise ValueError(
                "request watermark key and context_window are not supported "
                "with --watermark-algorithm synthid"
            )
        if config.enabled is None:
            raise ValueError("request watermark must set enabled")
        if not config.enabled and enforce_all:
            raise ValueError(
                "request watermark enabled=false is not allowed when the "
                "server sets watermark enforce_all"
            )
        enabled = config.enabled
    if enabled and not server_enabled:
        raise ValueError(
            "request watermarking requires the server to enable --enable-watermark"
        )
    return enabled


def build_rows(
    requests: Sequence[Req],
    *,
    features: ExecFeatures,
    device: torch.device | str,
) -> Optional[SynthIDRows]:
    return _rows(
        [
            _resolve_enabled(
                request.sampling_params.watermark,
                server_enabled=True,
                default_enabled=features.watermark_default_enabled,
                enforce_all=features.watermark_enforce_all,
            )
            and request.sampling_params.top_k > 1
            for request in requests
        ],
        device,
    )


def request_enabled(sampling_params: SamplingParams, *, features: ExecFeatures) -> bool:
    return _resolve_enabled(
        sampling_params.watermark,
        server_enabled=features.enable_watermark,
        default_enabled=features.watermark_default_enabled,
        enforce_all=features.watermark_enforce_all,
    )


def _synthid_config(cfg: ExecFeatures | ResolvingConfig) -> SynthIDConfig:
    """The SynthID config of resolved server args; unset fields keep the
    SynthIDConfig defaults."""
    optional = {
        name: getattr(cfg, f"watermark_{name}")
        for name in SYNTHID_CONFIG_FIELDS
        if name != "keys" and getattr(cfg, f"watermark_{name}") is not None
    }
    return SynthIDConfig(keys=tuple(cfg.watermark_keys), **optional)


def make_engine(
    features: ExecFeatures,
    *,
    max_num_reqs: int,
    max_contexts_per_req: int,
    vocab_size: int,
    device: str,
) -> SynthIDEngine:
    return SynthIDEngine(
        SynthIDSampler(_synthid_config(features), max_num_reqs, device),
        # At TP > 1 the stock sampler all-reduces token ids under
        # SGLANG_SYNC_TOKEN_IDS_ACROSS_TP or with grammars; the fused sample
        # bypasses the stock sampler.
        allow_fused_sample=get_parallel().tp_size == 1,
    )


class SynthIDEngine:
    def __init__(self, sampler: SynthIDSampler, *, allow_fused_sample: bool) -> None:
        self.sampler = sampler
        self.allow_fused_sample = allow_fused_sample

    def seed_payload(
        self, batch: ScheduleBatch
    ) -> Optional[tuple[list[int], list[list[int]]]]:
        """The watermarked prefill rows of an extend batch, and the generated
        tokens each one replays (none for a fresh request)."""
        rows = batch.sampling_info.watermark
        if rows is None or not batch.forward_mode.is_extend_without_speculative():
            return None
        seed_rows = [
            index for index in range(num_prefill_rows(batch)) if rows.on_host[index]
        ]
        if not seed_rows:
            return None
        return seed_rows, [list(batch.reqs[index].output_ids) for index in seed_rows]

    def bind_sample(self, forward_batch: ForwardBatch) -> Optional[SynthIDSample]:
        rows = forward_batch.sampling_info.watermark
        if rows is None:
            return None
        return SynthIDSample(
            self.sampler,
            rows,
            forward_batch,
            allow_fused_sample=self.allow_fused_sample,
        )

    def bind_verify(
        self, rows: Optional[SynthIDRows], *, full_mask: bool
    ) -> Optional[SynthIDVerify]:
        if rows is None:
            return None
        return SynthIDVerify(self.sampler, rows)

    def append_accept_tokens(
        self,
        rows: SynthIDRows,
        req_pool_indices: torch.Tensor,
        accept_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        self.sampler.state.commit(
            rows.take(req_pool_indices).to(torch.int64),
            rows.take(accept_tokens),
            rows.take(accept_lens),
        )


def _request_filter(
    sampling_info: SamplingBatchInfo, rows: SynthIDRows, *, width: int = 1
) -> Optional[FilterProbs]:
    """The watermarked rows' request filters, each repeated ``width`` times."""
    if not (
        sampling_info.need_top_k_sampling
        or sampling_info.need_top_p_sampling
        or sampling_info.need_min_p_sampling
    ):
        return None

    def per_row(values: torch.Tensor) -> torch.Tensor:
        values = rows.take(values)
        return values if width == 1 else values.repeat_interleave(width)

    return functools.partial(
        top_k_top_p_min_p_filter_probs,
        top_ks=per_row(sampling_info.top_ks),
        top_ps=per_row(sampling_info.top_ps),
        min_ps=(
            per_row(sampling_info.min_ps) if sampling_info.need_min_p_sampling else None
        ),
    )


def _without_request_filters(
    sampling_info: SamplingBatchInfo, rows: SynthIDRows
) -> SamplingBatchInfo:
    """The sampling info the stock sampler draws watermarked rows with: their
    top-p and min-p already ran inside the transform. The ``need_*`` flags
    stay, so the other rows keep their sampling path."""
    return dataclasses.replace(
        sampling_info,
        top_ps=sampling_info.top_ps.index_fill(0, rows.indices, 1.0),
        min_ps=sampling_info.min_ps.index_fill(0, rows.indices, 0.0),
    )


class SynthIDSample:
    def __init__(
        self,
        sampler: SynthIDSampler,
        rows: SynthIDRows,
        forward_batch: ForwardBatch,
        *,
        allow_fused_sample: bool,
    ) -> None:
        self.sampler = sampler
        self.rows = rows
        self.forward_batch = forward_batch
        self.allow_fused_sample = allow_fused_sample
        self._slots: Optional[torch.Tensor] = None

    def apply(self, logits: torch.Tensor, *, keep_logits: bool) -> SampleStep:
        forward_batch = self.forward_batch
        sampling_info = forward_batch.sampling_info
        rows = self.rows
        logprob_logits = logits.clone() if keep_logits else None
        req_pool_indices = forward_batch.req_pool_indices[: logits.shape[0]]
        if forward_batch.watermark_seed is not None:
            seed_rows, replay_token_ids = forward_batch.watermark_seed
            seed_rows = torch.tensor(seed_rows, device=req_pool_indices.device)
            self.sampler.state.seed(
                req_pool_indices[seed_rows].to(torch.int64), replay_token_ids
            )
        self._slots = rows.take(req_pool_indices).to(torch.int64)
        temperatures = rows.take(sampling_info.temperatures)
        # The stock sampler sanitizes too, but after the candidate top-k here.
        sanitize_nan_logits(logits, "watermark: next_token_logits")

        if self._can_fuse(logits, keep_logits):
            return SampleStep(
                sampling_info=sampling_info,
                logprob_logits=None,
                token_ids=self.sampler.sample(logits, temperatures, self._slots),
            )
        self.sampler.apply(
            logits,
            temperatures,
            self._slots,
            rows=rows.partial_indices,
            filter_probs=_request_filter(sampling_info, rows),
        )
        return SampleStep(
            sampling_info=_without_request_filters(sampling_info, rows),
            logprob_logits=logprob_logits,
        )

    def commit(self, token_ids: torch.Tensor) -> None:
        self.sampler.state.commit(self._slots, self.rows.take(token_ids))

    def _can_fuse(self, logits: torch.Tensor, keep_logits: bool) -> bool:
        """Whether the stock sampler has nothing to do: every row watermarked,
        no request filter, seed, logprob, or sampling-mask output."""
        sampling_info = self.forward_batch.sampling_info
        return (
            self.allow_fused_sample
            and self.rows.all_on
            and not keep_logits
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_min_p_sampling
            and sampling_info.sampling_seed is None
            and sampling_info.sampling_mask_batch_indices is None
            and self.sampler.can_fuse_sample(logits)
        )


class SynthIDVerify:
    """One EAGLE verify's transform. ``check_server_args`` requires rejection
    sampling, which allows only topk=1 drafts, so each row's draft is a chain
    whatever ``custom_mask`` holds."""

    def __init__(self, sampler: SynthIDSampler, rows: SynthIDRows) -> None:
        self.sampler = sampler
        self.rows = rows

    def apply(
        self,
        logits: torch.Tensor,
        *,
        sampling_info: SamplingBatchInfo,
        req_pool_indices: torch.Tensor,
        draft_tokens: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        custom_mask: Optional[torch.Tensor],
        keep_logits: bool,
    ) -> tuple[torch.Tensor, SamplingBatchInfo]:
        if keep_logits:
            logits = logits.clone()
        rows = self.rows
        self.sampler.apply_verify(
            logits,
            rows.take(sampling_info.temperatures),
            rows.take(req_pool_indices).to(torch.int64),
            rows.take(draft_tokens.reshape(-1, draft_token_num)),
            requests=rows.partial_indices,
            filter_probs=_request_filter(sampling_info, rows, width=draft_token_num),
        )
        return logits, _without_request_filters(sampling_info, rows)

    def record(
        self, accept_index: Optional[torch.Tensor], accept_lens: torch.Tensor
    ) -> None:
        # The state advances only by append_accept_tokens.
        pass


def watermark_view_fields(rows: SynthIDRows) -> dict[str, Any]:
    # check_server_args refuses the NV mega runtime, the fields' only reader.
    return NO_WATERMARK_VIEW_FIELDS


def check_server_args(cfg: ResolvingConfig) -> None:
    if cfg.watermark_keys is None:
        raise ValueError("--watermark-algorithm synthid requires watermark config keys")
    # SynthIDConfig validates the keys and the other fields.
    _synthid_config(cfg)
    if not is_cuda():
        raise ValueError("--watermark-algorithm synthid requires NVIDIA CUDA")
    if cfg.rl_on_policy_target is not None:
        raise ValueError(
            "--watermark-algorithm synthid is incompatible with "
            "--rl-on-policy-target: it changes the sampling distribution"
        )
    if envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get():
        raise ValueError(
            "--watermark-algorithm synthid is not supported with "
            "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
        )
    if cfg.speculative_algorithm is None:
        return
    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        raise ValueError(
            "--watermark-algorithm synthid supports speculative decoding only "
            f"with EAGLE or EAGLE3, got {cfg.speculative_algorithm!r}"
        )
    # Rejection sampling emits from the watermarked target distribution
    # whatever the draft proposes, from whichever vocabulary.
    if not cfg.speculative_use_rejection_sampling:
        raise ValueError(
            "--watermark-algorithm synthid with speculative decoding requires "
            "--speculative-use-rejection-sampling"
        )
