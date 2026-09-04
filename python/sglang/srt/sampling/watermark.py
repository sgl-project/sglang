from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.sampling.watermark_config import parse_watermark_key

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = float(1 << 32)


def redact_watermark_secrets(value: Any, *, in_watermark_config: bool = False) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if field.name in {"watermark_key", "watermark_config"}:
                object.__setattr__(
                    result,
                    field.name,
                    "<redacted>" if item is not None else None,
                )
            elif field.name in {
                "watermark",
                "sampling_params",
                "preferred_sampling_params",
            }:
                object.__setattr__(
                    result,
                    field.name,
                    redact_watermark_secrets(
                        item, in_watermark_config=field.name == "watermark"
                    ),
                )
        return result
    if isinstance(value, WatermarkRequestConfig):
        return WatermarkRequestConfig(
            key="<redacted>" if value.key is not None else None,
            context_window=value.context_window,
        )
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key in {"watermark_key", "watermark_config"}
                or (in_watermark_config and key == "key")
                else redact_watermark_secrets(
                    item,
                    in_watermark_config=in_watermark_config or key == "watermark",
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            redact_watermark_secrets(item, in_watermark_config=in_watermark_config)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            redact_watermark_secrets(item, in_watermark_config=in_watermark_config)
            for item in value
        )
    return value


def redact_watermark_command_line(argv: Sequence[str]) -> str:
    result = []
    redact_next = False
    for argument in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif argument in {"--watermark-key", "--watermark-config"}:
            result.append(argument)
            redact_next = True
        elif argument.startswith("--watermark-key="):
            result.append("--watermark-key=<redacted>")
        elif argument.startswith("--watermark-config="):
            result.append("--watermark-config=<redacted>")
        else:
            result.append(argument)
    return " ".join(result)


class WatermarkRequestConfig(msgspec.Struct, frozen=True, kw_only=True):
    key: Optional[str] = None
    context_window: Optional[int] = None

    def __repr__(self) -> str:
        return (
            "WatermarkRequestConfig(key=<redacted>, "
            f"context_window={self.context_window!r})"
        )


def normalize_watermark_request(value: Any) -> Optional[WatermarkRequestConfig]:
    if value is None:
        return None
    if isinstance(value, WatermarkRequestConfig):
        key = value.key
        context_window = value.context_window
    else:
        if not isinstance(value, dict):
            raise ValueError("watermark must be an object")
        unknown = set(value) - {"key", "context_window"}
        if unknown:
            raise ValueError("watermark contains unknown fields")
        key = value.get("key")
        context_window = value.get("context_window")
    if key is not None:
        parse_watermark_key(key)
    if context_window is not None and (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or context_window < 1
    ):
        raise ValueError("watermark context_window must be a positive integer")
    return WatermarkRequestConfig(key=key, context_window=context_window)


def build_watermark_batch_config(
    requests: Sequence[Any],
    *,
    default_key: Optional[str],
    default_context_window: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = []
    context_windows = []
    enabled = []
    for request in requests:
        config = request.sampling_params.watermark
        key = (
            config.key if config is not None and config.key is not None else default_key
        )
        context_window = (
            config.context_window
            if config is not None and config.context_window is not None
            else default_context_window
        )
        keys.append(parse_watermark_key(key) if key is not None else 0)
        context_windows.append(context_window)
        enabled.append(key is not None)
    return (
        torch.tensor(keys, dtype=torch.int64, device=device),
        torch.tensor(context_windows, dtype=torch.int32, device=device),
        torch.tensor(enabled, dtype=torch.bool, device=device),
    )


def _rotl32(value: torch.Tensor, shift: int) -> torch.Tensor:
    return ((value << shift) | (value >> (32 - shift))) & _MASK32


def _murmur3_mix(state: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    value = (value * 0xCC9E2D51) & _MASK32
    value = _rotl32(value, 15)
    value = (value * 0x1B873593) & _MASK32
    state = state ^ value
    state = _rotl32(state, 13)
    return (state * 5 + 0xE6546B64) & _MASK32


def _fmix32(value: torch.Tensor) -> torch.Tensor:
    value = value ^ (value >> 16)
    value = (value * 0x85EBCA6B) & _MASK32
    value = value ^ (value >> 13)
    value = (value * 0xC2B2AE35) & _MASK32
    return value ^ (value >> 16)


def _hash_context_token_ids(token_ids: Sequence[int]) -> int:
    state = 0
    for token_id in token_ids:
        value = (int(token_id) * 0xCC9E2D51) & _MASK32
        value = ((value << 15) | (value >> 17)) & _MASK32
        value = (value * 0x1B873593) & _MASK32
        state ^= value
        state = ((state << 13) | (state >> 19)) & _MASK32
        state = (state * 5 + 0xE6546B64) & _MASK32
    state ^= len(token_ids) * 4
    state ^= state >> 16
    state = (state * 0x85EBCA6B) & _MASK32
    state ^= state >> 13
    state = (state * 0xC2B2AE35) & _MASK32
    state ^= state >> 16
    return state


def _as_signed_int32(value: int) -> int:
    return value if value < (1 << 31) else value - (1 << 32)


def _hash_contexts(
    context_token_ids: torch.Tensor, context_lengths: torch.Tensor
) -> torch.Tensor:
    state = torch.zeros(
        context_token_ids.shape[0], dtype=torch.int64, device=context_token_ids.device
    )
    lengths = context_lengths.to(torch.int64)
    for index in range(context_token_ids.shape[1]):
        mixed = _murmur3_mix(state, context_token_ids[:, index].to(torch.int64))
        state = torch.where(index < lengths, mixed, state)
    return _fmix32(state ^ (lengths * 4))


def _watermark_hash32_torch(
    keys: torch.Tensor, context_hashes: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    state = torch.zeros(
        (keys.shape[0], token_ids.shape[0]),
        dtype=torch.int64,
        device=keys.device,
    )
    keys = keys.view(-1, 1)
    state = _murmur3_mix(state, keys & _MASK32)
    state = _murmur3_mix(state, (keys >> 32) & _MASK32)
    state = _murmur3_mix(state, context_hashes.view(-1, 1) & _MASK32)
    state = _murmur3_mix(state, token_ids.view(1, -1) & _MASK32)
    return _fmix32(state ^ 16)


def select_watermark_tokens_torch(
    probabilities: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.arange(probabilities.shape[-1], device=probabilities.device)
    hashed = _watermark_hash32_torch(keys, context_hashes, token_ids)
    uniform = (hashed.to(torch.float32) + 0.5) / _UINT32_SCALE
    scores = torch.where(
        probabilities > 0,
        uniform.log() / probabilities.to(torch.float32),
        -torch.inf,
    )
    return scores.argmax(dim=-1)


def _truncate_probabilities(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits / temperatures, dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    ranks = torch.arange(logits.shape[-1], device=logits.device).view(1, -1)
    keep = ranks < top_ks.view(-1, 1)
    keep &= (cumulative_probabilities - sorted_probabilities) <= top_ps.view(-1, 1)
    keep &= sorted_probabilities >= (sorted_probabilities[:, :1] * min_ps.view(-1, 1))
    sorted_probabilities = torch.where(keep, sorted_probabilities, 0.0)
    sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter_(
        dim=-1, index=sorted_indices, src=sorted_probabilities
    )


def force_watermark_tokens(
    logits: torch.Tensor,
    context_hashes: torch.Tensor,
    eligible: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    keys: torch.Tensor,
) -> None:
    if logits.is_cuda:
        try:
            from sglang.kernels.ops.sampling.textseal_selector import (
                force_watermark_tokens_triton,
            )
        except ImportError:
            pass
        else:
            force_watermark_tokens_triton(
                logits,
                context_hashes,
                eligible,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                keys,
            )
            return

    probabilities = _truncate_probabilities(
        logits, temperatures, top_ks, top_ps, min_ps
    )
    rows = eligible.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return
    candidate_probabilities = probabilities[rows].to(torch.float32).contiguous()
    candidate_context_hashes = context_hashes[rows].contiguous()
    candidate_keys = keys[rows].contiguous()
    if candidate_probabilities.is_cuda:
        try:
            from sglang.kernels.ops.sampling.textseal_selector import (
                select_watermark_tokens_triton,
            )
        except ImportError:
            selected = select_watermark_tokens_torch(
                candidate_probabilities,
                candidate_context_hashes,
                candidate_keys,
            )
        else:
            selected = select_watermark_tokens_triton(
                candidate_probabilities,
                candidate_context_hashes,
                candidate_keys,
            )
    else:
        selected = select_watermark_tokens_torch(
            candidate_probabilities,
            candidate_context_hashes,
            candidate_keys,
        )
    logits[rows] = -torch.inf
    logits[rows, selected] = 0.0


class WatermarkState:
    def __init__(
        self,
        *,
        max_num_reqs: int,
        context_window: int,
        max_contexts_per_req: int,
        key: Optional[str],
        device: str,
    ) -> None:
        self.default_key = parse_watermark_key(key) if key is not None else None
        self.context_window = context_window
        self.token_ids = torch.zeros(
            (max_num_reqs, context_window), dtype=torch.int32, device=device
        )
        self.lengths = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.write_positions = torch.zeros(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.watermarked_context_hashes = torch.empty(
            (max_num_reqs, max_contexts_per_req), dtype=torch.int32, device=device
        )
        self.num_watermarked_contexts = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.context_history_positions = torch.arange(
            max_contexts_per_req, dtype=torch.int32, device=device
        )
        self.context_hash_buffer = torch.empty(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.eligible_buffer = torch.empty(
            max_num_reqs, dtype=torch.bool, device=device
        )

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        max_num_reqs: int,
        context_window: int,
        max_contexts_per_req: int,
        key: Optional[str],
        device: str,
    ) -> Optional[WatermarkState]:
        if not enabled:
            return None
        return cls(
            max_num_reqs=max_num_reqs,
            context_window=context_window,
            max_contexts_per_req=max_contexts_per_req,
            key=key,
            device=device,
        )

    def prompt_tails(self, batch: ScheduleBatch) -> Optional[list[Optional[list[int]]]]:
        if not batch.forward_mode.is_extend_without_speculative():
            return None

        num_prefill_reqs = len(batch.reqs)
        if batch.forward_mode.is_mixed():
            num_prefill_reqs -= len(batch.mix_running_indices_cpu)
        elif batch.decoding_reqs is not None:
            num_prefill_reqs = 0

        if num_prefill_reqs == 0:
            return None

        tails: list[Optional[list[int]]] = []
        for index, req in enumerate(batch.reqs):
            if index >= num_prefill_reqs:
                tails.append(None)
                continue
            tails.append(list(req.get_fill_ids()[-self.context_window :]))
        return tails

    def retracted_context_hashes(
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
            request_key = (
                request_config.key
                if request_config is not None and request_config.key is not None
                else self.default_key
            )
            if request_key is None or req.sampling_params.top_k <= 1:
                histories.append([])
                continue

            context_window = (
                request_config.context_window
                if request_config is not None
                and request_config.context_window is not None
                else self.context_window
            )
            token_ids = list(req.origin_input_ids) + list(req.output_ids)
            seen = set()
            history = []
            for position in range(len(req.origin_input_ids), len(token_ids)):
                context = token_ids[max(0, position - context_window) : position]
                if not context:
                    continue
                context_hash = _hash_context_token_ids(context)
                if context_hash in seen:
                    continue
                seen.add(context_hash)
                history.append(_as_signed_int32(context_hash))
            histories.append(history)

        return histories if has_retracted_request else None

    def init_from_prompt(
        self,
        req_pool_indices: torch.Tensor,
        prompt_tail_ids: Optional[Sequence[Optional[Sequence[int]]]],
        context_hash_history: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> None:
        if prompt_tail_ids is None:
            return
        assert len(prompt_tail_ids) == req_pool_indices.shape[0]

        valid_positions = [
            index
            for index, token_ids in enumerate(prompt_tail_ids)
            if token_ids is not None
        ]
        if not valid_positions:
            return

        device = self.token_ids.device
        batch_positions = torch.tensor(
            valid_positions, dtype=torch.int64, device=device
        )
        pool_indices = req_pool_indices[batch_positions].to(torch.int64)
        tails = [list(prompt_tail_ids[index]) for index in valid_positions]
        lengths = torch.tensor(
            [len(token_ids) for token_ids in tails], dtype=torch.int32, device=device
        )
        padded = torch.tensor(
            [
                token_ids + [0] * (self.context_window - len(token_ids))
                for token_ids in tails
            ],
            dtype=torch.int32,
            device=device,
        )
        self.token_ids[pool_indices] = padded
        self.lengths[pool_indices] = lengths
        self.write_positions[pool_indices] = (
            lengths.to(torch.int64) % self.context_window
        )
        self.num_watermarked_contexts[pool_indices] = 0
        if context_hash_history is None:
            return
        assert len(context_hash_history) == req_pool_indices.shape[0]
        for batch_position, pool_index in zip(
            valid_positions, pool_indices.tolist(), strict=True
        ):
            history = context_hash_history[batch_position]
            if history is None:
                continue
            history = list(history)[: self.watermarked_context_hashes.shape[1]]
            if history:
                self.watermarked_context_hashes[pool_index, : len(history)] = (
                    torch.tensor(history, dtype=torch.int32, device=device)
                )
            self.num_watermarked_contexts[pool_index] = len(history)

    def contexts_tail(
        self,
        req_pool_indices: torch.Tensor,
        context_windows: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool_indices = req_pool_indices.to(torch.int64)
        lengths = self.lengths[pool_indices]
        write_positions = self.write_positions[pool_indices]
        starts = torch.where(lengths == self.context_window, write_positions, 0)
        offsets = torch.arange(self.context_window, device=self.token_ids.device)
        gather_indices = (
            starts.view(-1, 1) + offsets.view(1, -1)
        ) % self.context_window
        contexts = self.token_ids[pool_indices].gather(1, gather_indices)
        if context_windows is None:
            return contexts, lengths

        context_lengths = torch.minimum(lengths, context_windows.to(torch.int32))
        source_starts = lengths - context_lengths
        output_positions = torch.arange(
            self.context_window, device=self.token_ids.device
        ).view(1, -1)
        suffix_indices = (source_starts.view(-1, 1) + output_positions).clamp(
            max=self.context_window - 1
        )
        contexts = contexts.gather(1, suffix_indices.to(torch.int64))
        contexts = torch.where(
            output_positions < context_lengths.view(-1, 1), contexts, 0
        )
        return contexts, context_lengths

    def _watermark_batch_config(
        self, sampling_info: SamplingBatchInfo
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = sampling_info.top_ks.shape[0]
        keys = getattr(sampling_info, "watermark_keys", None)
        context_windows = getattr(sampling_info, "watermark_context_windows", None)
        enabled = getattr(sampling_info, "watermark_enabled", None)
        if keys is not None and context_windows is not None and enabled is not None:
            return keys, context_windows, enabled

        key = self.default_key if self.default_key is not None else 0
        device = sampling_info.top_ks.device
        return (
            torch.full((batch_size,), key, dtype=torch.int64, device=device),
            torch.full(
                (batch_size,),
                self.context_window,
                dtype=torch.int32,
                device=device,
            ),
            torch.full(
                (batch_size,),
                self.default_key is not None,
                dtype=torch.bool,
                device=device,
            ),
        )

    def context_windows(self, sampling_info: SamplingBatchInfo) -> torch.Tensor:
        return self._watermark_batch_config(sampling_info)[1]

    def _new_context_mask(
        self,
        req_pool_indices: torch.Tensor,
        context_hashes: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        pool_indices = req_pool_indices.to(torch.int64)
        counts = self.num_watermarked_contexts[pool_indices]
        repeated = (
            (
                self.watermarked_context_hashes[pool_indices]
                == context_hashes.to(torch.int32).view(-1, 1)
            )
            & (self.context_history_positions.view(1, -1) < counts.view(-1, 1))
        ).any(dim=1)
        return (
            eligible
            & repeated.logical_not()
            & (counts < self.watermarked_context_hashes.shape[1])
        )

    def _record_contexts(
        self,
        req_pool_indices: torch.Tensor,
        context_hashes: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        rows = selected.nonzero(as_tuple=True)[0]
        pool_indices = req_pool_indices[rows].to(torch.int64)
        counts = self.num_watermarked_contexts[pool_indices]
        self.watermarked_context_hashes[pool_indices, counts.to(torch.int64)] = (
            context_hashes[rows].to(torch.int32)
        )
        self.num_watermarked_contexts[pool_indices] = counts + 1

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
        batch_size = req_pool_indices.shape[0]
        draft_tokens = (
            draft_tokens[: batch_size * draft_token_num]
            .view(batch_size, draft_token_num)
            .to(torch.int32)
        )
        positions = positions[: batch_size * draft_token_num].view(
            batch_size, draft_token_num
        )

        if full_mask:
            prefix_lengths = positions[:, 0].to(torch.int64)
            request_spans = draft_token_num * (prefix_lengths + draft_token_num)
            request_offsets = torch.cumsum(request_spans, dim=0) - request_spans
            rows = torch.arange(
                draft_token_num, dtype=torch.int64, device=positions.device
            ).view(1, -1, 1)
            columns = torch.arange(
                draft_token_num, dtype=torch.int64, device=positions.device
            ).view(1, 1, -1)
            tree_indices = (
                request_offsets.view(-1, 1, 1)
                + rows * (prefix_lengths.view(-1, 1, 1) + draft_token_num)
                + prefix_lengths.view(-1, 1, 1)
                + columns
            )
            tree_mask = custom_mask[tree_indices]
        else:
            tree_mask = custom_mask[: batch_size * draft_token_num**2].view(
                batch_size, draft_token_num, draft_token_num
            )

        base_contexts, base_lengths = self.contexts_tail(
            req_pool_indices, context_windows
        )
        base_contexts = base_contexts.unsqueeze(1).expand(-1, draft_token_num, -1)
        base_valid = (
            torch.arange(self.context_window, device=positions.device).view(1, 1, -1)
            < base_lengths.view(-1, 1, 1)
        ).expand(-1, draft_token_num, -1)
        ancestor_tokens = (
            draft_tokens[:, 1:].unsqueeze(1).expand(-1, draft_token_num, -1)
        )
        candidate_tokens = torch.cat((base_contexts, ancestor_tokens), dim=-1)
        candidate_valid = torch.cat((base_valid, tree_mask[:, :, 1:].bool()), dim=-1)
        ranks = candidate_valid.cumsum(dim=-1)
        total_lengths = candidate_valid.sum(dim=-1)
        context_lengths = torch.minimum(
            total_lengths,
            context_windows.view(-1, 1).expand(-1, draft_token_num),
        ).to(torch.int32)
        output_positions = torch.arange(
            self.context_window, device=positions.device
        ).view(1, 1, -1)
        desired_ranks = (
            (total_lengths - context_lengths.to(total_lengths.dtype)).unsqueeze(-1)
            + output_positions
            + 1
        )
        matches = candidate_valid.unsqueeze(-2) & (
            ranks.unsqueeze(-2) == desired_ranks.unsqueeze(-1)
        )
        gather_indices = matches.to(torch.int32).argmax(dim=-1)
        contexts = candidate_tokens.gather(dim=-1, index=gather_indices)
        contexts = torch.where(
            output_positions < context_lengths.unsqueeze(-1), contexts, 0
        )
        return contexts.flatten(0, 1), context_lengths.flatten()

    def force_speculative(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        contexts: torch.Tensor,
        context_lengths: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        draft_token_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_hashes = _hash_contexts(contexts, context_lengths)
        expanded_req_pool_indices = req_pool_indices.repeat_interleave(draft_token_num)
        keys, _, watermark_enabled = self._watermark_batch_config(sampling_info)
        keys = keys.repeat_interleave(draft_token_num)
        watermark_enabled = watermark_enabled.repeat_interleave(draft_token_num)
        top_ks = sampling_info.top_ks.repeat_interleave(draft_token_num, dim=0)
        eligible = (
            watermark_enabled & (top_ks <= 1).logical_not() & (context_lengths > 0)
        )
        selected = self._new_context_mask(
            expanded_req_pool_indices, context_hashes, eligible
        )
        context_hash_matrix = context_hashes.view(-1, draft_token_num)
        prior_rows = torch.tril(
            torch.ones(
                (draft_token_num, draft_token_num),
                dtype=torch.bool,
                device=logits.device,
            ),
            diagonal=-1,
        )
        repeated_in_tree = (
            (context_hash_matrix.unsqueeze(2) == context_hash_matrix.unsqueeze(1))
            & prior_rows.view(1, draft_token_num, draft_token_num)
        ).any(dim=2)
        selected &= repeated_in_tree.flatten().logical_not()
        force_watermark_tokens(
            logits=logits,
            context_hashes=context_hashes,
            eligible=selected,
            temperatures=sampling_info.temperatures.repeat_interleave(
                draft_token_num, dim=0
            ),
            top_ks=top_ks,
            top_ps=sampling_info.top_ps.repeat_interleave(draft_token_num, dim=0),
            min_ps=sampling_info.min_ps.repeat_interleave(draft_token_num, dim=0),
            keys=keys,
        )
        return context_hashes, selected

    def record_speculative(
        self,
        req_pool_indices: torch.Tensor,
        context_hashes: torch.Tensor,
        selected: torch.Tensor,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        for position in range(accept_index.shape[1]):
            valid = position < accept_lens
            rows = accept_index[:, position].clamp(min=0).to(torch.int64)
            self._record_contexts(
                req_pool_indices,
                context_hashes[rows],
                valid & selected[rows],
            )

    def append_speculative(
        self,
        req_pool_indices: torch.Tensor,
        accept_tokens: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        for position in range(accept_tokens.shape[1]):
            valid = position < accept_lens
            self.append(req_pool_indices[valid], accept_tokens[valid, position])

    def force(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        sampling_info: SamplingBatchInfo,
    ) -> None:
        keys, context_windows, watermark_enabled = self._watermark_batch_config(
            sampling_info
        )
        if logits.is_cuda:
            try:
                from sglang.kernels.ops.sampling.textseal_selector import (
                    prepare_watermark_contexts_triton,
                )
            except ImportError:
                pass
            else:
                batch_size = req_pool_indices.shape[0]
                context_hashes = self.context_hash_buffer[:batch_size]
                eligible = self.eligible_buffer[:batch_size]
                prepare_watermark_contexts_triton(
                    self.token_ids,
                    self.lengths,
                    self.write_positions,
                    self.watermarked_context_hashes,
                    self.num_watermarked_contexts,
                    req_pool_indices,
                    context_windows,
                    watermark_enabled,
                    sampling_info.top_ks,
                    context_hashes,
                    eligible,
                )
                force_watermark_tokens(
                    logits=logits,
                    context_hashes=context_hashes,
                    eligible=eligible,
                    temperatures=sampling_info.temperatures,
                    top_ks=sampling_info.top_ks,
                    top_ps=sampling_info.top_ps,
                    min_ps=sampling_info.min_ps,
                    keys=keys,
                )
                return

        contexts, context_lengths = self.contexts_tail(
            req_pool_indices, context_windows
        )
        context_hashes = _hash_contexts(contexts, context_lengths)
        eligible = (
            watermark_enabled
            & (sampling_info.top_ks <= 1).logical_not()
            & (context_lengths > 0)
        )
        selected = self._new_context_mask(req_pool_indices, context_hashes, eligible)
        force_watermark_tokens(
            logits=logits,
            context_hashes=context_hashes,
            eligible=selected,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
            min_ps=sampling_info.min_ps,
            keys=keys,
        )
        self._record_contexts(req_pool_indices, context_hashes, selected)

    def append(self, req_pool_indices: torch.Tensor, token_ids: torch.Tensor) -> None:
        pool_indices = req_pool_indices.to(torch.int64)
        write_positions = self.write_positions[pool_indices]
        self.token_ids[pool_indices, write_positions] = token_ids.to(torch.int32)
        self.write_positions[pool_indices] = (write_positions + 1) % self.context_window
        self.lengths[pool_indices] = torch.clamp(
            self.lengths[pool_indices] + 1, max=self.context_window
        )
