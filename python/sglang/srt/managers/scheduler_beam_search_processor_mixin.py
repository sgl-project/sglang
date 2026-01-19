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
"""Beam search post-processor mixin for managing beam search logic during inference.

This module handles beam search expansion, pruning, and state updates after each
inference round. It manages KV cache operations for beam candidates and determines
when beam search requests are complete.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import replace
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

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


class SchedulerBeamSearchProcessorMixin:
    """Mixin class for beam search post-processing logic in Scheduler.

    Handles beam search expansion, pruning, and state updates after each inference round.
    Manages KV cache operations and completion detection for beam search requests.
    """

    def process_beam_search_prefill_result(
        self: Scheduler, batch: ScheduleBatch, logits_output
    ) -> None:
        """Process beam search for all requests in prefill batch.

        Handles beam search initialization for all beam search requests in the batch.
        For each request, initializes beam candidates from prefill logprobs and checks
        for early termination conditions.

        Args:
            batch: Schedule batch containing beam search requests
            logits_output: Logits processor output containing logprobs for all requests
        """
        assert all(req.is_beam_search for req in batch.reqs)
        beam_width_list = [req.beam_width for req in batch.reqs]
        logger.debug(f"[prefill process]beam_width_list: {beam_width_list}")

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
                release_kv_cache(req, self.tree_cache)
                req.time_stats.completion_time = time.perf_counter()
            elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob, None)

    def process_beam_search_decode_result(
        self: Scheduler, batch: ScheduleBatch, result: GenerationBatchResult
    ):
        """Process batch results from beam search decode phase.

        During beam search decode, batch data is organized as follows:
        For each request i: [beam_1, beam_2, ..., beam_width]

        Main responsibilities:
        1. Extract top-k logprobs for all beam branches
        2. Update beam search state for each request (expand candidates, prune, check completion)
        3. Manage KV cache operations (copy for surviving beams, free for pruned beams)
        4. Stream output to users and log decode statistics

        Args:
            batch: Schedule batch containing beam search requests
            result: Generation batch result from model inference

        Note:
            beam search does not support grammar
        """
        self.num_generated_tokens += len(batch.req_pool_indices)

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
                    beam.beam_score = self._calculate_beam_score_with_eos_check(
                        beam, req.stop_token_ids
                    )

                completed = req.beam_list.completed + req.beam_list.incomplete
                completed = sorted(completed, key=lambda x: x.beam_score, reverse=True)
                req.beam_list.completed = completed[: req.beam_width]
                req.beam_list.incomplete = []

        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_begin()
        if any([req.finished() for req in batch.reqs]):
            self._cache_finished_beam_search(batch)
        if reqs_for_kv_copy:
            self._handle_beam_kv_cache(
                batch, reqs_for_kv_copy, last_batch_slot_indices_list
            )
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.current_scheduler_metrics_enabled
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats(result.can_run_cuda_graph, running_batch=batch)

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
        self: Scheduler,
        req: Req,
        batch: ScheduleBatch,
        logprobs: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Initialize beam search from prefill results.

        Creates initial beam candidates from the top-k tokens generated during prefill.
        Checks for early termination conditions and sets up beam state.

        Args:
            req: Request to initialize beam search for
            batch: Schedule batch containing the request
            logprobs: Log probabilities tensor for all tokens [vocab_size]
            device: Device where tensors are located
        """
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
        self: Scheduler,
        req: Req,
        generated_token_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Batch check if tokens generated during prefill should finish.

        Checks multiple tokens in parallel against stop conditions (stop tokens and stop strings).
        Creates and returns a boolean mask indicating which sequences should finish.

        Args:
            req: Request object containing stop condition configuration
            generated_token_ids: Token IDs generated during prefill phase, Shape: [num_beams]
            device: Device where tensors are located

        Returns:
            torch.Tensor: Boolean mask, Shape: [num_beams], True indicates sequence should finish
        """
        if not req.sampling_params.ignore_eos and req.stop_token_ids:
            stop_token_ids_tensor = torch.tensor(
                list(req.stop_token_ids),
                dtype=generated_token_ids.dtype,
                device=device,
            )
            # [num_beams, 1] == [1, num_stop_tokens] -> [num_beams, num_stop_tokens]
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
        self: Scheduler,
        req: Req,
        top_logprobs_val: List[float],
        top_logprobs_idx: List[int],
        finish_mask_cpu: List[bool],
        finish_by_len: bool,
    ) -> None:
        """Create completed beam sequences when insufficient candidates are available.

        When the number of unfinished candidates is less than beam_width during prefill,
        this function selects the top beam_width candidates and marks them as completed,
        effectively ending the beam search early.

        Args:
            req: Request object
            top_logprobs_val: List of top-k log probability values
            top_logprobs_idx: List of top-k token indices
            finish_mask_cpu: Boolean mask indicating which candidates are finished
            finish_by_len: Whether the finish is due to length constraint
        """
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
            req.finished_reason = completed[0].finish_reason

    def _create_initial_beam_sequences(
        self: Scheduler,
        req: Req,
        top_logprobs_val: List[float],
        top_logprobs_idx: List[int],
        finish_mask_cpu: List[bool],
        device: torch.device,
    ) -> None:
        """Create initial beam sequences from prefill results with sufficient candidates.

        When there are enough unfinished candidates (>= beam_width) during prefill,
        this function creates initial beam sequences, separating them into completed
        and incomplete lists, and initializes the beam_list state.

        Args:
            req: Request object
            top_logprobs_val: List of top-k log probability values
            top_logprobs_idx: List of top-k token indices
            finish_mask_cpu: Boolean mask indicating which candidates are finished
            device: Device where tensors are located
        """
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
                logger.debug(f"completed beam: {beam_sequence}")
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

    def _check_beam_finished(
        self: Scheduler, req: Req, beam: BeamSearchSequence
    ) -> bool:
        """Check if a beam search sequence should be marked as finished.

        Checks against stop tokens, stop strings, and stop regex patterns.
        Updates beam.finish_reason if a stop condition is met.

        Args:
            req: Request object containing stop conditions
            beam: Beam search sequence to check

        Returns:
            bool: True if sequence is finished, False otherwise
        """
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

    def _tail_str(self: Scheduler, req, tokens: List[int]) -> str:
        """Get tail string from token sequence for stop condition checking.

        Decodes the last N tokens where N is determined by the maximum length
        of stop strings and stop regex patterns.

        Args:
            req: Request object containing stop condition configuration
            tokens: List of token IDs

        Returns:
            str: Decoded tail string

        Note:
            Caller must ensure that either req.sampling_params.stop_strs or
            req.sampling_params.stop_regex_strs is non-empty, and req.tokenizer
            is not None.
        """
        max_len_tail_str = max(
            req.sampling_params.stop_str_max_len + 1,
            req.sampling_params.stop_regex_max_len + 1,
        )
        tail_len = min((max_len_tail_str + 1), len(tokens))
        return req.tokenizer.decode(tokens[-tail_len:])

    def _extract_beam_topk_data(
        self: Scheduler, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract top-k candidate data for all beam branches.

        Args:
            batch: Schedule batch
            result: Generation batch result containing logits_output.logprobs with shape
                [total_beams, vocab_size] where total_beams = sum(req.beam_width for req in batch)

        Returns:
            tuple: (beam_output_top_tokens, beam_output_top_logprobs)
                - beam_output_top_tokens: Top-k token IDs, Shape: [total_beams, max_k]
                - beam_output_top_logprobs: Top-k logprobs, Shape: [total_beams, max_k]

        Note:
            Transformers implementation doesn't take topk here, but uses full vocab_size
            logprobs and adds them to historical cum_logprob before selecting top candidates.
            Our implementation takes top-k first, which may result in a smaller candidate pool
            for the first path selection and potentially affect beam search quality.

        # beam search does not support custom logit processor
        """
        max_k = max([req.beam_candidates for req in batch.reqs])
        # Use sorted=True: when requests with different beam_width are batched together,
        # sorted results are needed to correctly select top beam_width candidates.
        beam_top_token_logprobs = result.logits_output.logprobs.topk(
            max_k, dim=1, sorted=True
        )
        return beam_top_token_logprobs.indices, beam_top_token_logprobs.values

    def _process_beam_search_expansion(
        self: Scheduler,
        req: Req,
        batch: ScheduleBatch,
        beam_width: int,
        topk: int,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Process beam search expansion and KV cache preparation for a single request.

        Selects optimal candidate paths from current beam branches, updates beam state,
        and prepares KV cache copy information.

        Expansion logic:
        1. Currently have beam_width beam branches, each with topk candidate tokens
        2. Total of beam_width * topk candidate paths
        3. Select topk candidates with highest cumulative logprob
        4. From these topk candidates, select up to beam_width incomplete paths as new beam branches
        5. Prepare KV cache copy information (if needed)

        Args:
            req: Request object
            batch: Schedule batch
            beam_width: Width of beam search
            topk: Top-k value for this request
            top_tokens: Top-k token IDs for current beam branches, Shape: [beam_width, topk]
            top_logprobs: Top-k logprobs for current beam branches, Shape: [beam_width, topk]

        Returns:
            Optional[torch.Tensor]: If KV cache processing needed, returns last_batch_slot_indices
                                   (positions of surviving beams in batch), otherwise None
        """
        # [beam_width, 1] + [beam_width, topK] -> [beam_width, topK]
        all_cum_logprobs = req.beam_list.cum_logprobs.unsqueeze(1) + top_logprobs

        all_cum_logprobs_flat = all_cum_logprobs.flatten()
        all_tokens_flat = top_tokens.flatten()
        topk_values, topk_indices = torch.topk(
            all_cum_logprobs_flat,
            k=topk,
            largest=True,
        )

        # Determine if length is sufficient by checking the first beam request
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
        self: Scheduler,
        req: Req,
        beam_width: int,
        topk: int,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        all_tokens_flat: torch.Tensor,
    ) -> Optional[List[int]]:
        """Expand and prune beam candidate sequences, and update beam state.

        Selects incomplete beams from top-k candidates as new candidate sequences,
        moves completed beams to the completed list, and updates req.beam_list state.

        Optimization: Uses different strategies based on stop conditions:
        - No stop conditions: Fast path without checks
        - Only stop tokens: Vectorized EOS checking
        - Stop strings/regex: Sequential checking with full condition evaluation

        Args:
            req: Request object
            beam_width: Width of beam search
            topk: Top-k value
            topk_indices: Indices of top-k candidates in flattened array
            topk_values: Cumulative logprob values of top-k candidates
            all_tokens_flat: Flattened array of all candidate tokens

        Returns:
            Optional[List[int]]: List of surviving beam indices from previous round,
                                None if request is finished
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
                    new_beam.beam_score = self._calculate_beam_score_with_eos_check(
                        new_beam, req.stop_token_ids
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
                    new_beam.beam_score = self._calculate_beam_score_with_eos_check(
                        new_beam, req.stop_token_ids
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

        # Early termination condition: incomplete beams less than beam_width (many paths ended)
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
        self: Scheduler,
        req: Req,
        beam_width: int,
        topk: int,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        all_tokens_flat: torch.Tensor,
        will_finish_reason: BaseFinishReason,
    ) -> None:
        """Create completed beam sequences for a request about to finish.

        When a request is about to finish (reached max_new_tokens or other completion condition),
        selects the top beam_width candidates from top-k and creates completed beam sequences.

        Args:
            req: Request object
            beam_width: Width of beam search
            topk: Top-k value
            topk_indices: Indices of top-k candidates in flattened array
            topk_values: Cumulative logprob values of top-k candidates
            all_tokens_flat: Flattened array of all candidate tokens
            will_finish_reason: Reason for completion
        """

        # Select top beam_width candidates
        selected_indices = topk_indices[:beam_width]
        last_beam_indices = selected_indices // topk
        top_tokens = all_tokens_flat[selected_indices]

        last_beam_indices_cpu = last_beam_indices.cpu().tolist()
        top_tokens_cpu = top_tokens.cpu().tolist()
        topk_values_cpu = topk_values[:beam_width].cpu().tolist()

        # Terminated due to length, no need to check stop tokens during scoring
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
        self: Scheduler,
        batch: ScheduleBatch,
        reqs: List[Req],
        last_batch_slot_indices_list: List[torch.Tensor],
    ):
        """Batch process KV cache copying and releasing for beam requests after inference.

        Called after all requests' beam search expansion is complete. Performs KV cache
        operations in batch for better performance.

        Algorithm:
        1. Collect all beam branches' KV cache indices from last round (from prefix_len to seq_len)
        2. Copy KV cache from surviving beams to new beam positions (keep prefix unchanged, copy newly generated part)
        3. Free KV cache indices that are no longer needed (using unique counts, indices appearing only once are pruned)

        Memory optimization:
        - Use torch.unique and counts to identify indices to be freed
        - Only free indices that appear in last_beam_kv_indices but not in keep_kv_indices
        - Batch operations reduce overhead from multiple calls

        Args:
            batch: Schedule batch containing all requests' state information
            reqs: List of requests to process (only includes requests needing KV cache operations)
            last_batch_slot_indices_list: List of surviving beam positions in batch for each request
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

        # Copy KV cache for all beams (including src == dst cases)
        # Although this may read some indices redundantly, it simplifies the code
        # and performs better in practice by avoiding branching overhead
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
        self.token_to_kv_pool_allocator.free(free_kv_indices)

    def _batch_collect_range_kv_indices(
        self: Scheduler, pool_indices, seq_lens, device, prefix_lens=None
    ):
        """Batch collect KV cache indices within specified ranges (vectorized, high-performance).

        Efficiently reads KV cache indices from req_to_token_pool for multiple requests,
        each with its own [prefix_len, seq_len) range. Uses vectorized operations to
        avoid loops and maximize GPU utilization.

        Args:
            pool_indices: Request pool indices, Shape: [num_reqs]
            seq_lens: Sequence length for each request (range end), Shape: [num_reqs]
            device: torch device for tensor operations
            prefix_lens: Prefix length for each request (range start), Shape: [num_reqs]
                        If None, defaults to 0 for all requests

        Returns:
            torch.Tensor: Unique KV cache indices across all requests in their ranges,
                         Shape: [num_unique_tokens], dtype: int64

        Example:
            >>> pool_indices = torch.tensor([0, 1, 2])
            >>> seq_lens = torch.tensor([10, 15, 12])
            >>> prefix_lens = torch.tensor([5, 8, 6])
            >>> indices = _batch_collect_range_kv_indices(
            ...     pool_indices, seq_lens, device, prefix_lens
            ... )
            # Returns unique KV indices from ranges [5:10], [8:15], [6:12]
        """
        num_reqs = len(pool_indices)
        if prefix_lens is None:
            prefix_lens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        max_range_len = (seq_lens - prefix_lens).max().item()
        # Create position index matrix [num_reqs, max_range_len]
        # Add corresponding prefix_len offset to each position
        position_indices = torch.arange(
            max_range_len, dtype=torch.int64, device=device
        ).unsqueeze(0) + prefix_lens.unsqueeze(1)

        mask = position_indices < seq_lens.unsqueeze(1)
        batch_kv_indices = self.req_to_token_pool.req_to_token[
            pool_indices.unsqueeze(1), position_indices
        ]
        return batch_kv_indices[mask].unique()

    def _cache_finished_beam_search(self: Scheduler, batch: ScheduleBatch):
        """Release KV cache for decode portion of finished beam search requests.

        Collects and frees KV cache indices for the decode portion (excluding prefix)
        of all beam candidates, releases their request pool slots, and cleans up
        radix cache entries.

        Args:
            batch: Batch object containing reqs, seq_lens, device, etc.

        Note:
            Only the decode portion [prefix_len:seq_len] is freed, as the prefix
            may be shared across multiple requests. After releasing KV cache,
            radix cache entries are also cleaned up for each finished request.
        """
        finished_reqs = [req for req in batch.reqs if req.finished()]
        beam_decode_kv_indices, beam_pool_indices = (
            self._collect_beam_req_decode_kv_indices(batch, finished_reqs)
        )

        self.token_to_kv_pool_allocator.free(beam_decode_kv_indices)
        self.req_to_token_pool.free(beam_pool_indices.tolist())

        for req in finished_reqs:
            release_kv_cache(req, self.tree_cache)

    def _copy_kvcache_for_beams(
        self: Scheduler,
        src_pool_indices,
        dst_pool_indices,
        prompt_lens,
        seq_lens_batch,
        device,
    ):
        """Copy KV cache for all beam candidates in parallel.

        Efficiently copies KV cache from source to destination pool indices,
        with automatic optimization based on beam dimensions.

        Args:
            src_pool_indices: Source request pool indices, Shape: [num_beams]
            dst_pool_indices: Destination request pool indices, Shape: [num_beams]
            prompt_lens: Prompt length for each beam, Shape: [num_beams]
            seq_lens_batch: Sequence length for each beam, Shape: [num_beams]
            device: torch device for tensor operations

        Returns:
            torch.Tensor: Unique KV cache indices that were read during copy,
                         Shape: [num_unique_tokens], dtype: int64

        Implementation:
            - Fast path: When all beams have identical dimensions, processes in one batch
            - Slow path: When beams have different dimensions, groups by (prefix_len, seq_len)
                        and processes each group separately for efficiency

        Note:
            This function handles all cases including src_pool_indices == dst_pool_indices
            (where data is already in the correct position). Although this may result in
            redundant reads, it simplifies the code and performs better in practice by
            avoiding branching overhead.

            Copy operations are grouped by (prompt_len, seq_len) and executed serially between
            different groups. If src and dst indices overlap across different groups, it may
            cause incorrect copy behavior (e.g., group 1 writes to index X, then group 2 reads
            from index X expecting old data). Within the same group, overlapping is safe as
            copies are parallel.
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
            # Group by (prompt_len, seq_len) for processing
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
        self: Scheduler, src_indices, dst_indices, prefix_len: int, seq_len: int
    ):
        """Copy KV cache for a single group with deduplication optimization.

        Optimizes memory access by deduplicating source indices before reading,
        which is crucial when multiple beams share the same parent.

        Args:
            src_indices: Source request pool indices, Shape: [group_size]
            dst_indices: Destination request pool indices, Shape: [group_size]
            prefix_len: Prefix length (copy start position)
            seq_len: Sequence length (copy end position)

        Returns:
            torch.Tensor: Unique KV cache indices that were copied,
                         Shape: [num_unique_tokens], dtype: int64

        Algorithm:
            1. Deduplicate source indices to avoid redundant reads
            2. Batch read unique source KV cache entries
            3. Use inverse_indices to map back to original indices
            4. Batch write to destination positions

        Example:
            If src_indices = [0, 0, 1], only reads from indices 0 and 1 once,
            then replicates the result for the duplicate source.
        """
        unique_src_indices, inverse_indices = torch.unique(
            src_indices, return_inverse=True
        )
        kvcache_batch_unique = self.req_to_token_pool.req_to_token[
            unique_src_indices, prefix_len:seq_len
        ].clone()
        kvcache_batch = kvcache_batch_unique[inverse_indices]
        self.req_to_token_pool.req_to_token[dst_indices, prefix_len:seq_len] = (
            kvcache_batch
        )
        return kvcache_batch_unique.flatten().unique()

    def _collect_beam_req_decode_kv_indices(
        self: Scheduler, batch: ScheduleBatch, finished_reqs: List[Req]
    ):
        """Collect decode portion KV cache indices for beam requests (deduplicated).

        Collects KV cache indices only from the decode portion [prefix_len:seq_len],
        skipping the prefix which may be shared. Uses vectorized operations for efficiency.

        Args:
            batch: Batch object containing reqs, seq_lens, device, etc.
            finished_reqs: List of finished beam search requests to process

        Returns:
            tuple: A tuple containing:
                - beam_decode_kv_indices (torch.Tensor): Unique KV cache indices from
                  decode portions of all beam requests, Shape: [num_unique_decode_tokens]
                - beam_pool_indices (torch.Tensor): Pool indices of all beam requests,
                  Shape: [sum(req.beam_width for req in finished_reqs)], used for subsequent cleanup

        Note:
            Only processes finished requests. For each finished request, collects
            indices from all req.beam_width candidates. Different requests may have
            different beam_width values.
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

        # decode part's KV cache indices (prefix_len ~ seq_len)
        beam_decode_kv_indices = self._batch_collect_range_kv_indices(
            beam_pool_indices,
            beam_pool_seq_len,
            batch.device,
            beam_prompt_lens,
        )

        return beam_decode_kv_indices, beam_pool_indices

    @staticmethod
    def _calculate_beam_score_with_eos_check(
        beam,
        stop_token_ids: set[int],
        length_penalty: float = 1.0,
    ):
        """Calculate normalized beam search score with automatic stop token handling.

        Computes the beam search score while automatically excluding stop tokens
        (EOS, etc.) from the sequence length calculation. This ensures that stop
        tokens don't unfairly penalize finished sequences during scoring.

        Args:
            beam: BeamSearchSequence object containing tokens and cum_logprob
            stop_token_ids: Set of token IDs that indicate sequence termination
                (e.g., EOS token, custom stop tokens)
            length_penalty: Exponent for length normalization (default 1.0)
                - 1.0: No length bias (standard normalization)
                - >1.0: Favor longer sequences
                - <1.0: Favor shorter sequences

        Returns:
            float: Normalized score (cum_logprob / adjusted_seq_len^length_penalty)
                where adjusted_seq_len excludes the final stop token if present

        Example:
            >>> beam = BeamSearchSequence(tokens=[1, 2, 3, 50256], cum_logprob=-5.2)
            >>> stop_token_ids = {50256}  # EOS token
            >>> score = _calculate_beam_score_with_eos_check(beam, stop_token_ids)
            # Uses seq_len=3 instead of 4, excluding the EOS token

        Note:
            This is the recommended function for scoring completed beam sequences,
            as it handles the common case where sequences end with stop tokens.
        """
        seq_len = len(beam.tokens)
        if beam.tokens[-1] in stop_token_ids:
            seq_len = len(beam.tokens) - 1

        return SchedulerBeamSearchProcessorMixin._calculate_beam_score(
            beam.cum_logprob, seq_len, length_penalty
        )

    @staticmethod
    def _calculate_beam_score(
        cum_logprob: float,
        seq_len: int,
        length_penalty: float = 1.0,
    ):
        """Calculate normalized beam search score from raw parameters.

        Applies length penalty to cumulative log probability to avoid bias toward
        shorter sequences. This is a low-level utility function that performs the
        core scoring calculation.

        Args:
            cum_logprob: Cumulative log probability of the sequence
            seq_len: Sequence length (number of tokens) for normalization
            length_penalty: Exponent for length normalization (default 1.0)
                - 1.0: No length bias (standard normalization)
                - >1.0: Favor longer sequences
                - <1.0: Favor shorter sequences

        Returns:
            float: Normalized score (cum_logprob / seq_len^length_penalty)

        Note:
            This function does not handle stop token adjustments. Use
            _calculate_beam_score_with_eos_check for automatic handling of
            stop tokens in finished sequences.
        """
        return cum_logprob / (seq_len**length_penalty)
