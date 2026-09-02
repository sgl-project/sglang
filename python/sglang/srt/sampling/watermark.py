from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = float(1 << 32)


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
    key: int,
) -> None:
    probabilities = _truncate_probabilities(
        logits, temperatures, top_ks, top_ps, min_ps
    )
    seeds = torch.full((logits.shape[0],), key, dtype=torch.int64, device=logits.device)
    token_ids = torch.arange(logits.shape[-1], device=logits.device)
    hashed = murmur_hash32(seeds, context_hashes, token_ids)
    uniform = (hashed.to(torch.float64) + 0.5) / _UINT32_SCALE
    scores = torch.where(
        probabilities > 0,
        uniform.log() / probabilities.to(torch.float64),
        -torch.inf,
    )
    selected = scores.argmax(dim=-1)
    rows = eligible.nonzero(as_tuple=True)[0]
    logits[rows] = -torch.inf
    logits[rows, selected[rows]] = 0.0


class WatermarkState:
    def __init__(
        self,
        *,
        max_num_reqs: int,
        context_window: int,
        max_contexts_per_req: int,
        key: str,
        device: str,
    ) -> None:
        key_value = int(key, 16)
        self.key = key_value if key_value < (1 << 63) else key_value - (1 << 64)
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

    def init_from_prompt(
        self,
        req_pool_indices: torch.Tensor,
        prompt_tail_ids: Optional[Sequence[Optional[Sequence[int]]]],
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

    def contexts_tail(
        self, req_pool_indices: torch.Tensor
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
        return contexts, lengths

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

        base_contexts, base_lengths = self.contexts_tail(req_pool_indices)
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
        context_lengths = total_lengths.clamp(max=self.context_window).to(torch.int32)
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
        top_ks = sampling_info.top_ks.repeat_interleave(draft_token_num, dim=0)
        eligible = (top_ks <= 1).logical_not() & (context_lengths > 0)
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
            key=self.key,
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
        contexts, context_lengths = self.contexts_tail(req_pool_indices)
        context_hashes = _hash_contexts(contexts, context_lengths)
        eligible = (sampling_info.top_ks <= 1).logical_not() & (context_lengths > 0)
        selected = self._new_context_mask(req_pool_indices, context_hashes, eligible)
        force_watermark_tokens(
            logits=logits,
            context_hashes=context_hashes,
            eligible=selected,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
            min_ps=sampling_info.min_ps,
            key=self.key,
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
