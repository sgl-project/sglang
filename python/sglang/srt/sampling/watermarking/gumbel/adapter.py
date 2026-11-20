"""The Aaronson-Gumbel watermark behind ``interface``'s contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from ..batch_layout import num_prefill_rows
from ..config import (
    MAX_WATERMARK_CONTEXT_WINDOW,
    WatermarkRequestConfig,
    parse_watermark_key,
)
from ..interface import SampleStep
from .detector import hash_context
from .state import WatermarkState

if TYPE_CHECKING:
    from sglang.srt.arg_groups.fields.exec_ import ExecFeatures
    from sglang.srt.arg_groups.model_override_base import ResolvingConfig
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams


def _as_signed_int32(value: int) -> int:
    return value if value < (1 << 31) else value - (1 << 32)


class GumbelRows(msgspec.Struct, frozen=True):
    keys: torch.Tensor
    context_windows: torch.Tensor
    enabled: torch.Tensor
    candidates_host: list[bool]

    def __repr__(self) -> str:
        return f"GumbelRows(rows={len(self.candidates_host)!r})"

    def filter(
        self, keep_indices: list[int], keep_indices_device: torch.Tensor
    ) -> Optional[GumbelRows]:
        candidates_host = [self.candidates_host[index] for index in keep_indices]
        if not any(candidates_host):
            return None
        return GumbelRows(
            keys=self.keys[keep_indices_device],
            context_windows=self.context_windows[keep_indices_device],
            enabled=self.enabled[keep_indices_device],
            candidates_host=candidates_host,
        )

    def concat(
        self, other: Optional[GumbelRows], self_len: int, other_len: int
    ) -> GumbelRows:
        if other is None:
            other = _off_rows(other_len, self.keys.device)
        return GumbelRows(
            keys=torch.cat([self.keys, other.keys]),
            context_windows=torch.cat([self.context_windows, other.context_windows]),
            enabled=torch.cat([self.enabled, other.enabled]),
            candidates_host=self.candidates_host + other.candidates_host,
        )

    def prepend_off(self, num_rows: int) -> GumbelRows:
        return _off_rows(num_rows, self.keys.device).concat(
            self, num_rows, len(self.candidates_host)
        )


def _off_rows(num_rows: int, device: torch.device) -> GumbelRows:
    # Disabled rows are never forced, so their key and window are never used.
    return GumbelRows(
        keys=torch.zeros(num_rows, dtype=torch.int64, device=device),
        context_windows=torch.ones(num_rows, dtype=torch.int32, device=device),
        enabled=torch.zeros(num_rows, dtype=torch.bool, device=device),
        candidates_host=[False] * num_rows,
    )


def resolve_watermark_request(
    config: Optional[WatermarkRequestConfig],
    *,
    server_enabled: bool,
    default_key: Optional[str],
    default_context_window: int,
    default_enabled: bool,
    enforce_all: bool,
) -> tuple[Optional[str], int, bool]:
    context_window = (
        config.context_window
        if config is not None and config.context_window is not None
        else default_context_window
    )
    if config is not None and config.context_window is not None:
        if config.context_window > default_context_window:
            raise ValueError(
                "request watermark context_window cannot exceed the server's "
                "watermark context_window"
            )

    if config is None:
        enabled = default_enabled or enforce_all
        request_key = None
    else:
        if config.enabled is False:
            if config.key is not None:
                raise ValueError(
                    "request watermark key cannot be combined with enabled=false"
                )
            if enforce_all:
                raise ValueError(
                    "request watermark enabled=false is not allowed when the "
                    "server sets watermark enforce_all"
                )
            return None, context_window, False
        enabled = config.enabled is True or config.key is not None
        request_key = config.key
        if not enabled:
            raise ValueError("request watermark must set enabled or key")

    if not enabled:
        return None, context_window, False
    if not server_enabled:
        raise ValueError(
            "request watermarking requires the server to enable --enable-watermark"
        )
    resolved_key = request_key if request_key is not None else default_key
    if resolved_key is None:
        raise ValueError(
            "request watermark enabled=true requires a server default key or "
            "request key"
        )
    return resolved_key, context_window, True


def build_rows(
    requests: Sequence[Req],
    *,
    features: ExecFeatures,
    device: torch.device | str,
) -> Optional[GumbelRows]:
    keys = []
    context_windows = []
    enabled = []
    candidates = []
    for request in requests:
        config = request.sampling_params.watermark
        key, context_window, request_enabled = resolve_watermark_request(
            config,
            server_enabled=True,
            default_key=features.watermark_key,
            default_context_window=features.watermark_context_window,
            default_enabled=features.watermark_default_enabled,
            enforce_all=features.watermark_enforce_all,
        )
        keys.append(parse_watermark_key(key) if key is not None else 0)
        context_windows.append(context_window)
        enabled.append(request_enabled)
        candidates.append(request_enabled and request.sampling_params.top_k > 1)
    if not any(candidates):
        return None
    return GumbelRows(
        keys=torch.tensor(keys, dtype=torch.int64, device=device),
        context_windows=torch.tensor(context_windows, dtype=torch.int32, device=device),
        enabled=torch.tensor(enabled, dtype=torch.bool, device=device),
        candidates_host=candidates,
    )


def request_enabled(sampling_params: SamplingParams, *, features: ExecFeatures) -> bool:
    _, _, enabled = resolve_watermark_request(
        sampling_params.watermark,
        server_enabled=features.enable_watermark,
        default_key=features.watermark_key,
        default_context_window=features.watermark_context_window,
        default_enabled=features.watermark_default_enabled,
        enforce_all=features.watermark_enforce_all,
    )
    return enabled


def make_engine(
    features: ExecFeatures,
    *,
    max_num_reqs: int,
    max_contexts_per_req: int,
    vocab_size: int,
    device: str,
) -> GumbelEngine:
    state = WatermarkState(
        max_num_reqs=max_num_reqs,
        context_window=features.watermark_context_window,
        max_contexts_per_req=max_contexts_per_req,
        vocab_size=vocab_size,
        key_b=features.watermark_key_b,
        mixing_probability=features.watermark_mixing_probability,
        max_probability=features.watermark_max_probability,
        device=device,
    )
    return GumbelEngine(state, features)


class GumbelEngine:
    def __init__(self, state: WatermarkState, features: ExecFeatures) -> None:
        self.state = state
        self._features = features

    def seed_payload(
        self, batch: ScheduleBatch
    ) -> Optional[
        tuple[list[Optional[list[int]]], Optional[list[Optional[list[int]]]]]
    ]:
        if batch.sampling_info.watermark is None:
            return None
        prompt_tails = self._prompt_tails(batch)
        if prompt_tails is None:
            return None
        return prompt_tails, self._retracted_context_hashes(batch)

    def bind_sample(self, forward_batch: ForwardBatch) -> Optional[GumbelSample]:
        rows = forward_batch.sampling_info.watermark
        if rows is None:
            return None
        return GumbelSample(self.state, rows, forward_batch)

    def bind_verify(
        self, rows: Optional[GumbelRows], *, full_mask: bool
    ) -> Optional[GumbelVerify]:
        if rows is None:
            return None
        return GumbelVerify(self.state, rows, full_mask)

    def append_accept_tokens(
        self,
        rows: GumbelRows,
        req_pool_indices: torch.Tensor,
        accept_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        self.state.append_speculative(req_pool_indices, accept_tokens, accept_lens)

    def _prompt_tails(
        self, batch: ScheduleBatch
    ) -> Optional[list[Optional[list[int]]]]:
        if not batch.forward_mode.is_extend_without_speculative():
            return None

        num_prefill_reqs = num_prefill_rows(batch)
        if num_prefill_reqs == 0:
            return None

        tails: list[Optional[list[int]]] = []
        for index, req in enumerate(batch.reqs):
            if index >= num_prefill_reqs:
                tails.append(None)
                continue
            tails.append(list(req.get_fill_ids()[-self.state.context_window :]))
        return tails

    def _retracted_context_hashes(
        self, batch: ScheduleBatch
    ) -> Optional[list[Optional[list[int]]]]:
        if not batch.forward_mode.is_extend_without_speculative():
            return None

        histories: list[Optional[list[int]]] = []
        has_retracted_request = False
        for req in batch.reqs:
            if not req.retracted_stain:
                histories.append(None)
                continue

            has_retracted_request = True
            request_config = req.sampling_params.watermark
            _, context_window, request_enabled = resolve_watermark_request(
                request_config,
                server_enabled=True,
                default_key=self._features.watermark_key,
                default_context_window=self.state.context_window,
                default_enabled=self._features.watermark_default_enabled,
                enforce_all=self._features.watermark_enforce_all,
            )
            if not request_enabled or req.sampling_params.top_k <= 1:
                histories.append([])
                continue

            token_ids = list(req.origin_input_ids) + list(req.output_ids)
            seen = set()
            history = []
            for position in range(len(req.origin_input_ids), len(token_ids)):
                context = token_ids[max(0, position - context_window) : position]
                if not context:
                    continue
                context_hash = hash_context(context)
                if context_hash in seen:
                    continue
                seen.add(context_hash)
                history.append(_as_signed_int32(context_hash))
                if len(history) == self.state.watermarked_context_hashes.shape[1]:
                    break
            histories.append(history)

        return histories if has_retracted_request else None


class GumbelSample:
    def __init__(
        self, state: WatermarkState, rows: GumbelRows, forward_batch: ForwardBatch
    ) -> None:
        self.state = state
        self.rows = rows
        self.forward_batch = forward_batch

    def apply(self, logits: torch.Tensor, *, keep_logits: bool) -> SampleStep:
        forward_batch = self.forward_batch
        logprob_logits = logits.clone() if keep_logits else None
        req_pool_indices = forward_batch.req_pool_indices[: logits.shape[0]]
        if forward_batch.watermark_seed is not None:
            self.state.init_from_prompt(req_pool_indices, *forward_batch.watermark_seed)
        self.state.force(
            logits,
            req_pool_indices,
            forward_batch.sampling_info,
            self.rows.keys,
            self.rows.context_windows,
            self.rows.enabled,
        )
        return SampleStep(
            sampling_info=forward_batch.sampling_info, logprob_logits=logprob_logits
        )

    def commit(self, token_ids: torch.Tensor) -> None:
        req_pool_indices = self.forward_batch.req_pool_indices[: token_ids.shape[0]]
        self.state.append(req_pool_indices, token_ids)


class GumbelVerify:
    """One verify's forcing; also the provider ABI v2 ``WatermarkStateView``."""

    def __init__(
        self, state: WatermarkState, rows: GumbelRows, full_mask: bool
    ) -> None:
        self.state = state
        self.rows = rows
        self.full_mask = full_mask
        # (req_pool_indices, context_hashes, selected) of the applied verify.
        self._forced: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._chain_width: Optional[int] = None

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
        if custom_mask is None:
            self._chain_width = draft_token_num
            batch_size = draft_tokens.shape[0]
            custom_mask = self.state.chain_mask(draft_token_num)[:batch_size].reshape(
                -1
            )
            draft_tokens = draft_tokens.reshape(-1)
            positions = positions.reshape(-1)
        if logits.is_cuda:
            context_hashes, selected = self.force_speculative_from_tree(
                logits=logits,
                req_pool_indices=req_pool_indices,
                draft_tokens=draft_tokens,
                custom_mask=custom_mask,
                positions=positions,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
                full_mask=self.full_mask,
            )
        else:
            contexts, context_lengths = self.speculative_contexts(
                req_pool_indices=req_pool_indices,
                draft_tokens=draft_tokens,
                custom_mask=custom_mask,
                positions=positions,
                draft_token_num=draft_token_num,
                full_mask=self.full_mask,
                context_windows=self.context_windows(sampling_info),
            )
            context_hashes, selected = self.force_speculative(
                logits=logits,
                req_pool_indices=req_pool_indices,
                contexts=contexts,
                context_lengths=context_lengths,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
            )
        self._forced = (req_pool_indices, context_hashes, selected)
        return logits, sampling_info

    def record(
        self, accept_index: Optional[torch.Tensor], accept_lens: torch.Tensor
    ) -> None:
        req_pool_indices, context_hashes, selected = self._forced
        if accept_index is None:
            assert self._chain_width is not None
            accept_index = torch.arange(
                selected.numel(), dtype=torch.int32, device=selected.device
            ).view(-1, self._chain_width)
        self.record_speculative(
            req_pool_indices, context_hashes, selected, accept_index, accept_lens
        )

    # Provider ABI v2 WatermarkStateView: the state's operations on these rows.
    def context_windows(self, sampling: Any) -> torch.Tensor:
        return self.rows.context_windows

    def speculative_contexts(
        self,
        req_pool_indices: torch.Tensor,
        draft_tokens: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        full_mask: bool,
        context_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state.speculative_contexts(
            req_pool_indices=req_pool_indices,
            draft_tokens=draft_tokens,
            custom_mask=custom_mask,
            positions=positions,
            draft_token_num=draft_token_num,
            full_mask=full_mask,
            context_windows=context_windows,
        )

    def force_speculative(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        contexts: torch.Tensor,
        context_lengths: torch.Tensor,
        sampling_info: Any,
        draft_token_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state.force_speculative(
            logits=logits,
            req_pool_indices=req_pool_indices,
            contexts=contexts,
            context_lengths=context_lengths,
            sampling_info=sampling_info,
            draft_token_num=draft_token_num,
            keys=self.rows.keys,
            watermark_enabled=self.rows.enabled,
        )

    def force_speculative_from_tree(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        draft_tokens: torch.Tensor,
        custom_mask: torch.Tensor,
        positions: torch.Tensor,
        sampling_info: Any,
        draft_token_num: int,
        full_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state.force_speculative_from_tree(
            logits=logits,
            req_pool_indices=req_pool_indices,
            draft_tokens=draft_tokens,
            custom_mask=custom_mask,
            positions=positions,
            sampling_info=sampling_info,
            draft_token_num=draft_token_num,
            full_mask=full_mask,
            keys=self.rows.keys,
            context_windows=self.rows.context_windows,
            watermark_enabled=self.rows.enabled,
        )

    def record_speculative(
        self,
        req_pool_indices: torch.Tensor,
        context_hashes: torch.Tensor,
        selected: torch.Tensor,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        self.state.record_speculative(
            req_pool_indices, context_hashes, selected, accept_index, accept_lens
        )


def watermark_view_fields(rows: GumbelRows) -> dict[str, Any]:
    return {
        "watermark_keys": rows.keys,
        "watermark_context_windows": rows.context_windows,
        "watermark_enabled": rows.enabled,
        "has_watermark_candidates": True,
    }


def check_server_args(cfg: ResolvingConfig) -> None:
    if cfg.watermark_key is not None:
        parse_watermark_key(cfg.watermark_key)
    if cfg.watermark_key_b is not None:
        parse_watermark_key(cfg.watermark_key_b)
    if (
        isinstance(cfg.watermark_context_window, bool)
        or not isinstance(cfg.watermark_context_window, int)
        or not 1 <= cfg.watermark_context_window <= MAX_WATERMARK_CONTEXT_WINDOW
    ):
        raise ValueError("watermark context_window must be an integer from 1 to 64")
    if cfg.watermark_key_b is not None and cfg.watermark_key is None:
        raise ValueError("watermark key_b requires a server key")
    if not 0 < cfg.watermark_mixing_probability < 1:
        raise ValueError(
            "watermark mixing_probability must be strictly between 0 and 1"
        )
    if cfg.watermark_key_b is None and cfg.watermark_mixing_probability != 0.5:
        raise ValueError("watermark mixing_probability requires key_b")
    if not 0 < cfg.watermark_max_probability <= 1:
        raise ValueError(
            "watermark max_probability must be greater than 0 and at most 1"
        )
    if (
        cfg.watermark_default_enabled or cfg.watermark_enforce_all
    ) and cfg.watermark_key is None:
        raise ValueError(
            "watermark default_enabled and enforce_all require a server key"
        )
    if cfg.speculative_use_rejection_sampling:
        raise ValueError(
            "--enable-watermark is incompatible with "
            "--speculative-use-rejection-sampling"
        )
