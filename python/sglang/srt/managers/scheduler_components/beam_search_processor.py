# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam search post-processor for the scheduler.

Handles beam expansion, pruning, KV-cache management, and completion detection
after each inference round.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.managers.beam_search_type import BeamSearchSequence
from sglang.srt.managers.io_struct import BeamSearchOutput
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SchedulerBeamSearchProcessor:
    """Beam-search result processor (composed scheduler component)."""

    scheduler: "Scheduler"

    def process_beam_search_prefill_result(
        self, batch: ScheduleBatch, logits_output
    ) -> None:
        """Initialize beam candidates from prefill logprobs for every request."""
        assert all(req.is_beam_search for req in batch.reqs)

        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            self._process_beam_search_prefill_result_single_req(
                req=req,
                batch=batch,
                logprobs=logits_output.logprobs[i],
                device=logits_output.logprobs.device,
            )

            if req.finished():
                release_kv_cache(req, self.scheduler.tree_cache)
                req.time_stats.completion_time = time.perf_counter()
            elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                self.scheduler.tree_cache.cache_unfinished_req(req)

        self.scheduler.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, None
        )

    def process_beam_search_decode_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ):
        """Process decode results: expand/prune beams, manage KV cache, stream output.

        Batch layout per request i is [beam_1, ..., beam_width]. Grammar is not
        supported for beam search.
        """
        self.scheduler.metrics_reporter.num_generated_tokens += len(
            batch.req_pool_indices
        )

        beam_output_top_tokens, beam_output_top_logprobs = self._extract_beam_topk_data(
            batch, result
        )

        reqs_for_kv_copy = []
        last_batch_slot_indices_list = []

        offset = 0
        for req in batch.reqs:
            if req.is_retracted:
                continue

            start_idx = offset
            end_idx = offset + req.beam_width
            offset = end_idx
            topk = req.beam_candidates
            top_tokens = beam_output_top_tokens[start_idx:end_idx, :topk]
            top_logprobs = beam_output_top_logprobs[start_idx:end_idx, :topk]
            last_batch_slot_indices = self._process_beam_search_expansion(
                req,
                batch,
                req.beam_width,
                topk,
                top_tokens,
                top_logprobs,
            )

            if last_batch_slot_indices is not None:
                reqs_for_kv_copy.append(req)
                last_batch_slot_indices_list.append(last_batch_slot_indices)

            if req.finished():
                for beam in req.beam_list.incomplete:
                    beam.beam_score = self._calculate_beam_score(
                        beam.cum_logprob, len(beam.tokens)
                    )

                completed = req.beam_list.completed + req.beam_list.incomplete
                completed = sorted(completed, key=lambda x: x.beam_score, reverse=True)
                req.beam_list.completed = completed[: req.beam_width]
                req.beam_list.incomplete = []

        self.scheduler.output_streamer.stream_output(batch.reqs, batch.return_logprob)

        self.scheduler.token_to_kv_pool_allocator.free_group_begin()
        if any([req.finished() for req in batch.reqs]):
            self._cache_finished_beam_search(batch)
        if reqs_for_kv_copy:
            self._handle_beam_kv_cache(
                batch, reqs_for_kv_copy, last_batch_slot_indices_list
            )
        self.scheduler.token_to_kv_pool_allocator.free_group_end()

        # report_decode_stats self-gates on metrics-enabled and decode_log_interval.
        self.scheduler.metrics_reporter.forward_ct_decode = (
            self.scheduler.metrics_reporter.forward_ct_decode + 1
        ) % (1 << 30)
        self.scheduler.metrics_reporter.report_decode_stats(
            result.can_run_cuda_graph,
            running_batch=batch,
        )

    @staticmethod
    def sum_beam_completion_tokens(req: Req) -> int:
        """Calculate total completion tokens for beam search request."""
        return sum(len(beam_seq.tokens) for beam_seq in req.beam_list.completed)

    @staticmethod
    def convert_beam_sequences_to_output(req: Req):
        """Convert beam search completed sequences to output format."""
        sequences_with_json_finish_reason = []
        for beam in req.beam_list.completed:
            seq_with_json = replace(
                beam,
                finish_reason=(
                    beam.finish_reason.to_json() if beam.finish_reason else None
                ),
            )
            sequences_with_json_finish_reason.append(seq_with_json)

        return BeamSearchOutput(sequences=sequences_with_json_finish_reason)

    def _process_beam_search_prefill_result_single_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        logprobs: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Create initial beam candidates for one request from prefill top-k tokens."""
        logprobs = self._apply_custom_logit_processor(req, logprobs)
        topk_result = logprobs.topk(req.beam_candidates, dim=0, sorted=True)
        top_logprobs_val = topk_result.values.tolist()
        top_logprobs_idx = topk_result.indices.tolist()

        finish_by_len = False
        all_new_tokens = topk_result.indices
        if req.sampling_params.max_new_tokens <= 1:
            finish_mask = torch.ones(
                all_new_tokens.size(0), dtype=torch.bool, device=device
            )
            finish_by_len = True
        else:
            finish_mask = self._batch_check_prefill_generated_tokens_stop_conditions(
                req, all_new_tokens, device
            )
        finish_mask_cpu = finish_mask.cpu().tolist()

        # Request end: insufficient candidates, take top beam_width tokens to respond
        if (~finish_mask).sum().item() < req.beam_width:
            self._create_completed_beams_for_insufficient_candidates(
                req,
                top_logprobs_val,
                top_logprobs_idx,
                finish_mask_cpu,
                finish_by_len,
            )
            return

        # Normal case: sufficient unfinished candidates
        self._create_initial_beam_sequences(
            req,
            top_logprobs_val,
            top_logprobs_idx,
            finish_mask_cpu,
            device,
        )

    def _batch_check_prefill_generated_tokens_stop_conditions(
        self,
        req: Req,
        generated_token_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a [num_beams] bool mask of prefill tokens that hit a stop condition."""
        if not req.sampling_params.ignore_eos and req.stop_token_ids:
            stop_token_ids_tensor = torch.tensor(
                list(req.stop_token_ids),
                dtype=generated_token_ids.dtype,
                device=device,
            )
            finish_mask = (
                generated_token_ids.unsqueeze(1) == stop_token_ids_tensor.unsqueeze(0)
            ).any(dim=1)
        else:
            finish_mask = torch.zeros(
                generated_token_ids.size(0), dtype=torch.bool, device=device
            )

        if (
            req.sampling_params.stop_strs
            and len(req.sampling_params.stop_strs) > 0
            and req.tokenizer is not None
        ):
            generated_token_ids_cpu = generated_token_ids.cpu().tolist()
            decoded_tokens = []
            for token_id in generated_token_ids_cpu:
                decoded = req.tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_tokens.append(decoded)

            for i, decoded_token in enumerate(decoded_tokens):
                if decoded_token and decoded_token in req.sampling_params.stop_strs:
                    finish_mask[i] = True

        return finish_mask

    def _create_completed_beams_for_insufficient_candidates(
        self,
        req: Req,
        top_logprobs_val: List[float],
        top_logprobs_idx: List[int],
        finish_mask_cpu: List[bool],
        finish_by_len: bool,
    ) -> None:
        """End beam search early by marking the top beam_width prefill candidates completed."""
        completed = [
            BeamSearchSequence(
                tokens=[top_token],
                cum_logprob=top_logprob,
                beam_score=self._calculate_beam_score(top_logprob, 1),
                finish_reason=(
                    (
                        FINISH_LENGTH(length=req.sampling_params.max_new_tokens)
                        if finish_by_len
                        else FINISH_MATCHED_TOKEN(matched=top_logprobs_idx[i])
                    )
                    if finish_mask_cpu[i]
                    else None
                ),
            )
            for i, (top_logprob, top_token) in enumerate(
                zip(
                    top_logprobs_val[: req.beam_width],
                    top_logprobs_idx[: req.beam_width],
                )
            )
        ]
        req.beam_list.completed = completed
        req.beam_list.incomplete = []
        if req.finished_reason is None:
            # completed[0] may itself be unfinished, which would leave req with no beam
            # state and crash the next decode tick; adopt the first real finish reason
            # instead (guaranteed to exist since unfinished < beam_width; length is a
            # defensive fallback).
            req.finished_reason = next(
                (b.finish_reason for b in completed if b.finish_reason is not None),
                FINISH_LENGTH(length=req.sampling_params.max_new_tokens),
            )

    def _create_initial_beam_sequences(
        self,
        req: Req,
        top_logprobs_val: List[float],
        top_logprobs_idx: List[int],
        finish_mask_cpu: List[bool],
        device: torch.device,
    ) -> None:
        """Build initial beam_list state from prefill when >= beam_width candidates remain."""
        completed_beams = []
        incomplete_beams = []
        last_token_ids = []
        cum_logprobs = []

        for i, (top_logprob, top_token) in enumerate(
            zip(top_logprobs_val, top_logprobs_idx)
        ):
            is_finished = finish_mask_cpu[i]
            beam_sequence = BeamSearchSequence(
                tokens=[top_token],
                cum_logprob=top_logprob,
                finish_reason=(
                    FINISH_MATCHED_TOKEN(matched=top_logprobs_idx[i])
                    if is_finished
                    else None
                ),
            )
            if is_finished:
                beam_sequence.beam_score = self._calculate_beam_score(top_logprob, 1)
                completed_beams.append(beam_sequence)
            else:
                incomplete_beams.append(beam_sequence)
                cum_logprobs.append(top_logprob)
                last_token_ids.append(top_token)
                if len(incomplete_beams) == req.beam_width:
                    break

        req.beam_list.completed = completed_beams
        req.beam_list.incomplete = incomplete_beams
        req.beam_list.prompt_lens = torch.tensor(
            [len(req.origin_input_ids)] * req.beam_width,
            dtype=torch.long,
            device=device,
        )
        req.beam_list.cum_logprobs = torch.tensor(
            cum_logprobs,
            device=device,
        )
        req.beam_list.last_tokens = torch.tensor(
            last_token_ids,
            device=device,
        )

    def _check_beam_finished(self, req: Req, beam: BeamSearchSequence) -> bool:
        """Check beam against stop tokens/strings/regex, set finish_reason, return done?."""
        if not req.sampling_params.ignore_eos:
            last_token_id = beam.tokens[-1]
            if req.stop_token_ids and last_token_id in req.stop_token_ids:
                beam.finish_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return True

        if (
            len(req.sampling_params.stop_strs) == 0
            and len(req.sampling_params.stop_regex_strs) == 0
        ) or req.tokenizer is None:
            return False

        tail_str = self._tail_str(req, beam.tokens)
        if not tail_str:
            return False

        if len(req.sampling_params.stop_strs) > 0:
            for stop_str in req.sampling_params.stop_strs:
                if stop_str in tail_str:
                    beam.finish_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return True

        if len(req.sampling_params.stop_regex_strs) > 0:
            for stop_regex_str in req.sampling_params.stop_regex_strs:
                if re.search(stop_regex_str, tail_str):
                    beam.finish_reason = FINISHED_MATCHED_REGEX(matched=stop_regex_str)
                    return True

    def _tail_str(self, req, tokens: List[int]) -> str:
        """Decode the trailing tokens needed to test stop strings/regex.

        Caller must ensure stop_strs or stop_regex_strs is non-empty and tokenizer
        is not None.
        """
        max_len_tail_str = max(
            req.sampling_params.stop_str_max_len + 1,
            req.sampling_params.stop_regex_max_len + 1,
        )
        tail_len = min((max_len_tail_str + 1), len(tokens))
        return req.tokenizer.decode(tokens[-tail_len:])

    @staticmethod
    def _apply_custom_logit_processor(
        req: Req, logprobs: torch.Tensor
    ) -> torch.Tensor:
        """Apply one request's processor to its beam-search score rows."""
        if not req.custom_logit_processor:
            return logprobs

        processor = CustomLogitProcessor.from_str(req.custom_logit_processor)
        custom_params = dict(req.sampling_params.custom_params or {})
        custom_params["__req__"] = req

        processor_input = logprobs if logprobs.ndim == 2 else logprobs.unsqueeze(0)
        processed = processor(processor_input, [custom_params])
        return processed if logprobs.ndim == 2 else processed.squeeze(0)

    def _apply_decode_custom_logit_processors(
        self, batch: ScheduleBatch, logprobs: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-request processors to contiguous per-beam score slices."""
        offset = 0
        for req in batch.reqs:
            if req.is_retracted:
                continue

            num_beams = len(req.beam_list.incomplete)
            if num_beams == 0:
                continue

            beam_slice = logprobs[offset : offset + num_beams]
            processed = self._apply_custom_logit_processor(req, beam_slice)
            if processed is not beam_slice:
                beam_slice.copy_(processed)
            offset += num_beams

        return logprobs

    def _extract_beam_topk_data(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (top_tokens, top_logprobs), each [total_beams, max_k], from decode logprobs.

        Unlike transformers (which scores full vocab), we take top-k first; this can
        shrink the candidate pool and slightly affect quality.
        """
        max_k = max([req.beam_candidates for req in batch.reqs])
        logprobs = result.logits_output.logprobs
        if any(req.custom_logit_processor for req in batch.reqs):
            logprobs = self._apply_decode_custom_logit_processors(batch, logprobs)

        # sorted=True is required so mixed beam_width requests select the right top-k.
        beam_top_token_logprobs = logprobs.topk(max_k, dim=1, sorted=True)
        return beam_top_token_logprobs.indices, beam_top_token_logprobs.values

    def _process_beam_search_expansion(
        self,
        req: Req,
        batch: ScheduleBatch,
        beam_width: int,
        topk: int,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Expand/prune one request's beams and prepare KV-cache copy info.

        From the beam_width * topk candidate paths, keeps the top-k by cumulative
        logprob, then up to beam_width incomplete paths as new beams. Returns the
        surviving beams' batch slot indices for KV copy, or None if no copy is needed.
        """
        all_cum_logprobs = req.beam_list.cum_logprobs.unsqueeze(1) + top_logprobs

        all_cum_logprobs_flat = all_cum_logprobs.flatten()
        all_tokens_flat = top_tokens.flatten()
        topk_values, topk_indices = torch.topk(
            all_cum_logprobs_flat,
            k=topk,
            largest=True,
        )

        # All incomplete beams share the same length; check the first.
        current_generated = len(req.beam_list.incomplete[0].tokens)
        will_finish_reason = (
            req.to_finish
            if req.to_finish
            else (
                FINISH_LENGTH(length=req.sampling_params.max_new_tokens)
                if current_generated + 1 >= req.sampling_params.max_new_tokens
                else None
            )
        )
        if will_finish_reason:
            self._create_completed_beams_for_finished_request(
                req,
                beam_width,
                topk,
                topk_indices,
                topk_values,
                all_tokens_flat,
                will_finish_reason,
            )
            req.finished_reason = will_finish_reason
            return None

        keep_last_beam_indices = self._expand_and_prune_beams(
            req, beam_width, topk, topk_indices, topk_values, all_tokens_flat
        )
        if keep_last_beam_indices is None:
            return None

        last_batch_slot_indices = req.beam_list.batch_slot_start_idx + torch.tensor(
            keep_last_beam_indices,
            dtype=torch.int32,
            device=batch.device,
        )

        return last_batch_slot_indices

    def _expand_and_prune_beams(
        self,
        req: Req,
        beam_width: int,
        topk: int,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        all_tokens_flat: torch.Tensor,
    ) -> Optional[List[int]]:
        """Split top-k candidates into new incomplete/completed beams and update state.

        Picks a fast path by stop condition: none (no checks), stop tokens only
        (vectorized EOS), or stop strings/regex (sequential). Returns surviving
        prev-round beam indices, or None if the request finished.
        """

        last_beam_indices = topk_indices // topk
        top_tokens = all_tokens_flat[topk_indices]

        has_stop_strs = len(req.sampling_params.stop_strs) > 0
        ignore_eos = req.sampling_params.ignore_eos

        last_beam_indices_cpu = last_beam_indices.cpu().tolist()
        top_tokens_cpu = top_tokens.cpu().tolist()
        topk_values_cpu = topk_values.cpu().tolist()

        incomplete = []
        completed = []
        keep_last_beam_indices = []
        last_token_ids = []
        incomplete_cum_logprobs = []

        if not has_stop_strs and (ignore_eos or not req.stop_token_ids):
            for last_beam_idx, top_token, cum_logprob in zip(
                last_beam_indices_cpu, top_tokens_cpu, topk_values_cpu
            ):
                beam_seq = req.beam_list.incomplete[last_beam_idx]
                new_beam = BeamSearchSequence(
                    tokens=beam_seq.tokens + [top_token],
                    cum_logprob=cum_logprob,
                )
                incomplete.append(new_beam)
                keep_last_beam_indices.append(last_beam_idx)
                last_token_ids.append(top_token)
                incomplete_cum_logprobs.append(cum_logprob)
                if len(incomplete) >= beam_width:
                    break
        elif not has_stop_strs and not ignore_eos and req.stop_token_ids:
            stop_token_ids_tensor = torch.tensor(
                list(req.stop_token_ids),
                dtype=top_tokens.dtype,
                device=top_tokens.device,
            )
            eos_mask_cpu = torch.isin(top_tokens, stop_token_ids_tensor).cpu().tolist()
            for last_beam_idx, top_token, cum_logprob, is_eos in zip(
                last_beam_indices_cpu,
                top_tokens_cpu,
                topk_values_cpu,
                eos_mask_cpu,
            ):
                beam_seq = req.beam_list.incomplete[last_beam_idx]
                new_beam = BeamSearchSequence(
                    tokens=beam_seq.tokens + [top_token],
                    cum_logprob=cum_logprob,
                )
                if is_eos:
                    new_beam.finish_reason = FINISH_MATCHED_TOKEN(matched=top_token)
                    new_beam.beam_score = self._calculate_beam_score(
                        new_beam.cum_logprob, len(new_beam.tokens)
                    )
                    completed.append(new_beam)
                else:
                    incomplete.append(new_beam)
                    keep_last_beam_indices.append(last_beam_idx)
                    last_token_ids.append(top_token)
                    incomplete_cum_logprobs.append(cum_logprob)
                    if len(incomplete) >= beam_width:
                        break
        else:
            for last_beam_idx, top_token, cum_logprob in zip(
                last_beam_indices_cpu, top_tokens_cpu, topk_values_cpu
            ):
                beam_seq = req.beam_list.incomplete[last_beam_idx]
                new_beam = BeamSearchSequence(
                    tokens=beam_seq.tokens + [top_token],
                    cum_logprob=cum_logprob,
                )
                if self._check_beam_finished(req, new_beam):
                    new_beam.beam_score = self._calculate_beam_score(
                        new_beam.cum_logprob, len(new_beam.tokens)
                    )
                    completed.append(new_beam)
                else:
                    incomplete.append(new_beam)
                    keep_last_beam_indices.append(last_beam_idx)
                    last_token_ids.append(top_token)
                    incomplete_cum_logprobs.append(cum_logprob)
                    if len(incomplete) >= beam_width:
                        break

        req.beam_list.incomplete = incomplete
        req.beam_list.completed += completed

        # Too few incomplete beams left to continue: finish the request.
        if len(req.beam_list.incomplete) < beam_width:
            req.finished_reason = req.beam_list.completed[0].finish_reason
            return None

        req.beam_list.last_tokens = torch.tensor(
            last_token_ids,
            device=req.beam_list.last_tokens.device,
        )
        if req.beam_list.incomplete:
            req.beam_list.cum_logprobs = torch.tensor(
                incomplete_cum_logprobs,
                dtype=torch.float32,
                device=req.beam_list.cum_logprobs.device,
            )

        return keep_last_beam_indices

    def _create_completed_beams_for_finished_request(
        self,
        req: Req,
        beam_width: int,
        topk: int,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        all_tokens_flat: torch.Tensor,
        will_finish_reason: BaseFinishReason,
    ) -> None:
        """Finalize a finishing request: mark its top beam_width candidates completed."""
        selected_indices = topk_indices[:beam_width]
        last_beam_indices = selected_indices // topk
        top_tokens = all_tokens_flat[selected_indices]

        last_beam_indices_cpu = last_beam_indices.cpu().tolist()
        top_tokens_cpu = top_tokens.cpu().tolist()
        topk_values_cpu = topk_values[:beam_width].cpu().tolist()

        # Finishing by length, so no stop-token check is needed during scoring.
        completed = [
            BeamSearchSequence(
                tokens=req.beam_list.incomplete[last_beam_idx].tokens + [top_token],
                beam_score=self._calculate_beam_score(
                    cum_logprob,
                    len(req.beam_list.incomplete[last_beam_idx].tokens) + 1,
                ),
                cum_logprob=cum_logprob,
                finish_reason=will_finish_reason,
            )
            for last_beam_idx, top_token, cum_logprob in zip(
                last_beam_indices_cpu, top_tokens_cpu, topk_values_cpu
            )
        ]

        req.beam_list.completed += completed
        req.beam_list.incomplete = []
        req.beam_list.cum_logprobs = torch.empty(
            0, dtype=torch.float32, device=req.beam_list.cum_logprobs.device
        )

    def _handle_beam_kv_cache(
        self,
        batch: ScheduleBatch,
        reqs: List[Req],
        last_batch_slot_indices_list: List[torch.Tensor],
    ):
        """Batch-copy KV cache to surviving beams and free pruned indices.

        Copies the newly generated KV (prefix stays shared) from surviving beams to
        their new positions, then frees indices present in the old beams but not kept
        (identified via unique counts == 1).
        """
        prompt_lens = torch.cat([req.beam_list.prompt_lens for req in reqs])

        beam_indices_list = []
        for req in reqs:
            base_idx = req.beam_list.batch_slot_start_idx
            beam_indices = (
                torch.arange(req.beam_width, dtype=torch.int64, device=batch.device)
                + base_idx
            )
            beam_indices_list.append(beam_indices)
        beam_indices_flatten = torch.cat(beam_indices_list)

        beam_req_pool_indices = batch.req_pool_indices[beam_indices_flatten]
        beam_req_seq_lens = batch.seq_lens[beam_indices_flatten]
        last_beam_kv_indices = self._batch_collect_range_kv_indices(
            beam_req_pool_indices,
            beam_req_seq_lens,
            batch.device,
            prompt_lens,
        )

        last_beam_indices = torch.cat(last_batch_slot_indices_list)

        src_pool_indices = batch.req_pool_indices[last_beam_indices]
        seq_lens_batch = batch.seq_lens[last_beam_indices]

        # Copy all beams including src == dst: redundant reads are cheaper than branching.
        keep_kv_indices = self._copy_kvcache_for_beams(
            src_pool_indices,
            beam_req_pool_indices,
            prompt_lens,
            seq_lens_batch,
            batch.device,
        )

        uniques, counts = torch.unique(
            torch.cat([last_beam_kv_indices, keep_kv_indices]), return_counts=True
        )
        free_kv_indices = uniques[counts == 1]
        self.scheduler.token_to_kv_pool_allocator.free(free_kv_indices)

    def _batch_collect_range_kv_indices(
        self, pool_indices, seq_lens, device, prefix_lens=None
    ):
        """Vectorized: return unique KV indices over each req's [prefix_len, seq_len) range.

        prefix_lens defaults to 0 for all requests when None.
        """
        num_reqs = len(pool_indices)
        if prefix_lens is None:
            prefix_lens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        max_range_len = (seq_lens - prefix_lens).max().item()
        # Position matrix [num_reqs, max_range_len], each row offset by its prefix_len.
        position_indices = torch.arange(
            max_range_len, dtype=torch.int64, device=device
        ).unsqueeze(0) + prefix_lens.unsqueeze(1)

        mask = position_indices < seq_lens.unsqueeze(1)
        batch_kv_indices = self.scheduler.req_to_token_pool.req_to_token[
            pool_indices.unsqueeze(1), position_indices
        ]
        return batch_kv_indices[mask].unique()

    def _cache_finished_beam_search(self, batch: ScheduleBatch):
        """Free decode-portion KV cache, pool slots, and radix entries for finished reqs.

        Only [prefix_len:seq_len] is freed since the prefix may be shared.
        """
        finished_reqs = [req for req in batch.reqs if req.finished()]
        beam_decode_kv_indices, beam_pool_indices = (
            self._collect_beam_req_decode_kv_indices(batch, finished_reqs)
        )

        self.scheduler.token_to_kv_pool_allocator.free(beam_decode_kv_indices)
        self.scheduler.req_to_token_pool.free_by_indices(beam_pool_indices.tolist())

        for req in finished_reqs:
            release_kv_cache(req, self.scheduler.tree_cache)

    def _copy_kvcache_for_beams(
        self,
        src_pool_indices,
        dst_pool_indices,
        prompt_lens,
        seq_lens_batch,
        device,
    ):
        """Copy KV cache from src to dst pool indices for all beams; return read indices.

        Fast path when all beams share dimensions; otherwise groups by
        (prompt_len, seq_len). Handles src == dst (redundant reads are intentional).
        Groups run serially, so src/dst overlap *across* groups could read stale data;
        overlap within a group is safe since copies are parallel.
        """
        if len(prompt_lens.unique()) == 1 and len(seq_lens_batch.unique()) == 1:
            prompt_len = prompt_lens[0].item()
            seq_len = seq_lens_batch[0].item()
            kvcache_batch_unique = self._copy_kvcache_group(
                src_pool_indices,
                dst_pool_indices,
                prompt_len,
                seq_len,
            )
            return kvcache_batch_unique
        else:
            seq_lens_cpu = seq_lens_batch.cpu()
            prompt_lens_cpu = prompt_lens.cpu()

            copy_groups = {}
            for beam_idx, (prompt_len, seq_len) in enumerate(
                zip(prompt_lens_cpu.tolist(), seq_lens_cpu.tolist())
            ):
                key = (prompt_len, seq_len)
                if key not in copy_groups:
                    copy_groups[key] = []
                copy_groups[key].append(beam_idx)

            kvcache_indices_list = []
            for (prompt_len, seq_len), beam_indices in copy_groups.items():
                if len(beam_indices) == 0:
                    continue

                beam_indices_tensor = torch.tensor(
                    beam_indices, dtype=torch.int64, device=device
                )
                src_indices = src_pool_indices[beam_indices_tensor]
                dst_indices = dst_pool_indices[beam_indices_tensor]
                kvcache_indices = self._copy_kvcache_group(
                    src_indices, dst_indices, prompt_len, seq_len
                )
                kvcache_indices_list.append(kvcache_indices)

            return torch.cat(kvcache_indices_list).unique()

    def _copy_kvcache_group(
        self, src_indices, dst_indices, prefix_len: int, seq_len: int
    ):
        """Copy [prefix_len:seq_len] KV for one group; dedup src reads (beams share parents)."""
        unique_src_indices, inverse_indices = torch.unique(
            src_indices, return_inverse=True
        )
        kvcache_batch_unique = self.scheduler.req_to_token_pool.req_to_token[
            unique_src_indices, prefix_len:seq_len
        ].clone()
        kvcache_batch = kvcache_batch_unique[inverse_indices]
        self.scheduler.req_to_token_pool.req_to_token[
            dst_indices, prefix_len:seq_len
        ] = kvcache_batch
        return kvcache_batch_unique.flatten().unique()

    def _collect_beam_req_decode_kv_indices(
        self, batch: ScheduleBatch, finished_reqs: List[Req]
    ):
        """Return (unique decode-portion KV indices, beam pool indices) for finished reqs.

        Collects only [prefix_len:seq_len] across every beam_width candidate; the prefix
        (possibly shared) is skipped.
        """

        beam_indices_list = []
        for req in finished_reqs:
            beam_indices = (
                torch.arange(req.beam_width, dtype=torch.int64, device=batch.device)
                + req.beam_list.batch_slot_start_idx
            )
            beam_indices_list.append(beam_indices)
        beam_indices_flatten = torch.cat(beam_indices_list)
        beam_pool_indices = batch.req_pool_indices[beam_indices_flatten]
        beam_pool_seq_len = batch.seq_lens[beam_indices_flatten]
        beam_prompt_lens = torch.cat(
            [req.beam_list.prompt_lens for req in finished_reqs], dim=0
        )

        beam_decode_kv_indices = self._batch_collect_range_kv_indices(
            beam_pool_indices,
            beam_pool_seq_len,
            batch.device,
            beam_prompt_lens,
        )

        return beam_decode_kv_indices, beam_pool_indices

    @staticmethod
    def _calculate_beam_score(
        cum_logprob: float,
        seq_len: int,
        length_penalty: float = 1.0,
    ):
        """Length-normalized score: cum_logprob / seq_len**length_penalty.

        seq_len is the actual length (EOS included if present), matching transformers.
        length_penalty >1 favors longer sequences, <1 shorter, 1.0 is no bias.
        """
        return cum_logprob / (seq_len**length_penalty)
