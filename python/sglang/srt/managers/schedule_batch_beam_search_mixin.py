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
"""Mixin classes for beam search operations in ScheduleBatch and Req."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.managers.beam_search_type import BeamSearchList
from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class BeamSearchAdmissionError(RuntimeError):
    """Req-to-token pool can't fit the beam branches a decode tick is allocating.

    Fail-fast assertion: unreachable when ``Scheduler.get_num_allocatable_reqs``
    admits correctly (and oversized beam_width is rejected up front), so if it
    fires it signals an admission-accounting bug and crashes with structured
    context rather than being silently caught.
    """

    def __init__(self, *, failing_reqs, total_slots, available):
        self.failing_reqs = failing_reqs
        self.total_slots = total_slots
        self.available = available
        super().__init__(
            f"Beam search decode needs {total_slots} req-to-token slots but only "
            f"{available} are free (#reqs={len(failing_reqs)})."
        )


class ScheduleBatchBeamSearchMixin:
    """Mixin class for beam search related operations in ScheduleBatch."""

    def prepare_for_beam_search_decode(self: ScheduleBatch):
        """Set up the batch for a beam-search decode step: collect last tokens,
        init new beam reqs (alloc KV slots), alloc out_cache_loc, bump seq_lens."""
        # Pre-flight pool check before any mutation: prefill-stage reqs
        # (batch_slot_start_idx == -1) each claim `beam_width` branch slots now;
        # if they don't fit, raise first so the scheduler can skip the tick with
        # the batch in a clean state.
        pending_new_beam_reqs = [
            req for req in self.reqs if req.beam_list.batch_slot_start_idx == -1
        ]
        if pending_new_beam_reqs:
            total_slots = sum(req.beam_width for req in pending_new_beam_reqs)
            available = self.req_to_token_pool.available_size()
            if total_slots > available:
                raise BeamSearchAdmissionError(
                    failing_reqs=self.reqs,
                    total_slots=total_slots,
                    available=available,
                )

        self.forward_mode = ForwardMode.DECODE

        beam_ids = torch.cat(
            [req.beam_list.last_tokens.to(torch.int32) for req in self.reqs]
        )

        # beam search not need self.output_ids
        self.input_ids = beam_ids
        self.output_ids = None

        self._prepare_for_new_beam_search()

        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)
        # alloc_for_decode charges 1 token per tick to each Req's linear KV
        # accounting, but beam decode tokens land in beam-branch rows whose KV
        # is tracked and freed by the beam machinery, not on the leader row.
        # Undo the phantom charge so release_kv_cache sees
        # kv_committed_len == kv_allocated_len (prompt only) on finish.
        for req in self.reqs:
            req.kv.kv_allocated_len -= 1
        self.seq_lens.add_(1)
        self.seq_lens_cpu.add_(1)
        self.orig_seq_lens.add_(1)
        self.seq_lens_sum = self.seq_lens.sum().item()

    def filter_beam_search_batch(
        self: ScheduleBatch,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        """Filter beam search batch to keep only specified requests.

        This method handles the special filtering logic for beam search batches,
        where each request occupies beam_width slots in the batch tensors.

        Args:
            chunked_req_to_exclude: Requests to exclude from the batch
            keep_indices: Indices of requests to keep (if None, computed from chunked_req_to_exclude)
        """
        if keep_indices is None:
            if chunked_req_to_exclude is not None and not isinstance(
                chunked_req_to_exclude, list
            ):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            return

        old_pool_indices = []
        extend_idx = 0
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx != -1:
                old_pool_indices.append(
                    torch.tensor(
                        [
                            req.beam_list.batch_slot_start_idx + i
                            for i in range(req.beam_width)
                        ],
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
                extend_idx += req.beam_width
            else:
                old_pool_indices.append(
                    torch.tensor([extend_idx], dtype=torch.int64, device=self.device)
                )
                extend_idx += 1
        keep_pool_indices = torch.concat([old_pool_indices[i] for i in keep_indices])
        self.reqs = [self.reqs[i] for i in keep_indices]

        old_pool_indices_for_debug = torch.concat(old_pool_indices)
        assert len(self.req_pool_indices) == len(old_pool_indices_for_debug)
        old_to_new_pool_indices = torch.arange(
            len(self.req_pool_indices), dtype=torch.int64, device=self.device
        )
        new_pool_indices = torch.arange(
            len(keep_pool_indices), dtype=torch.int64, device=self.device
        )
        old_to_new_pool_indices[keep_pool_indices] = new_pool_indices
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx != -1:
                req.beam_list.batch_slot_start_idx = old_to_new_pool_indices[
                    req.beam_list.batch_slot_start_idx
                ].item()

        self.req_pool_indices = self.req_pool_indices[keep_pool_indices]
        self.seq_lens = self.seq_lens[keep_pool_indices]
        self.seq_lens_cpu = self.seq_lens.cpu()
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.orig_seq_lens = self.orig_seq_lens[keep_pool_indices]

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        # beam search does not support parameters in sampling_info
        # beam search does not support spec info

    def _prepare_for_new_beam_search(self: ScheduleBatch):
        """Allocate `beam_width` req_to_token slots per new request, copy the
        prefill KV cache into each branch, and replace the request's slot in
        req_pool_indices with the beam-branch slots (extending seq_lens to match).

        The original prefill slot stays on req.req_pool_idx so release_kv_cache()
        can free it on finish, letting beam and non-beam reqs share cache_finished_req.
        Supports a different beam_width per request.
        """
        new_reqs = []
        old_reqs = []
        for req in self.reqs:
            if req.beam_list.batch_slot_start_idx == -1:
                new_reqs.append(req)
            else:
                old_reqs.append(req)

        if not new_reqs:
            return

        new_pool_slot_list = [req.beam_width for req in new_reqs]
        total_slots = sum(new_pool_slot_list)

        beam_req_pool_indices = self.req_to_token_pool.alloc_by_count(total_slots)
        if beam_req_pool_indices is None:
            # Should be unreachable when the admission gate is sized correctly
            # (see Scheduler.get_num_allocatable_reqs); the typed exception gives
            # the scheduler structured context to skip the tick instead of a
            # generic OOM that tears down the subprocess.
            raise BeamSearchAdmissionError(
                failing_reqs=new_reqs,
                total_slots=total_slots,
                available=self.req_to_token_pool.available_size(),
            )
        beam_req_pool_indices = torch.tensor(
            beam_req_pool_indices, dtype=torch.int64, device=self.device
        )

        skip_idx = sum(req.beam_width for req in old_reqs)

        new_req_pool_indices_list = []
        new_seq_lens_list = []
        new_orig_seq_lens_list = []

        beam_offset = 0
        req_pool = self.req_to_token_pool.req_to_token
        for i, req in enumerate(new_reqs):
            normal_idx = self.req_pool_indices[skip_idx + i : skip_idx + i + 1]
            seq_len_tensor = self.seq_lens[skip_idx + i : skip_idx + i + 1]
            seq_len = seq_len_tensor.squeeze()

            beam_start = beam_offset
            beam_end = beam_offset + req.beam_width

            normal_kvcache = req_pool[normal_idx, :seq_len].squeeze(0)
            beam_indices = beam_req_pool_indices[beam_start:beam_end]
            req_pool[beam_indices, :seq_len] = normal_kvcache

            beam_offset = beam_end

            # Use all beam indices (normal_idx is preserved in req.req_pool_idx for release_kv_cache)
            new_req_pool_indices_list.append(beam_indices)

            expanded_seq_lens = seq_len_tensor.repeat(req.beam_width)
            new_seq_lens_list.append(expanded_seq_lens)

            orig_seq_len = self.orig_seq_lens[skip_idx + i : skip_idx + i + 1]
            expanded_orig_seq_lens = orig_seq_len.repeat(req.beam_width)
            new_orig_seq_lens_list.append(expanded_orig_seq_lens)

        new_req_pool_indices = torch.cat(new_req_pool_indices_list)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices[:skip_idx], new_req_pool_indices]
        )

        new_seq_lens = torch.cat(new_seq_lens_list)
        self.seq_lens = torch.cat([self.seq_lens[:skip_idx], new_seq_lens])
        self.seq_lens_cpu = self.seq_lens.cpu()

        new_orig_seq_lens = torch.cat(new_orig_seq_lens_list)
        self.orig_seq_lens = torch.cat(
            [self.orig_seq_lens[:skip_idx], new_orig_seq_lens]
        )

        current_idx = skip_idx
        for req in new_reqs:
            req.beam_list.batch_slot_start_idx = current_idx
            current_idx += req.beam_width


class ReqBeamSearchMixin:
    """Mixin class for beam search related operations in Req.

    This mixin provides beam search specific attributes and initialization logic
    that can be mixed into the Req class.
    """

    def _init_beam_search_attributes(self, is_beam_search, sampling_params):
        """Initialize beam search related attributes.

        This method should be called from Req.__init__() to set up beam search state.
        """
        self.is_beam_search = is_beam_search
        if self.is_beam_search:
            # sampling_params.n has already been validated in tokenizermanager
            self.beam_width = sampling_params.n
            self.beam_list = BeamSearchList()
            # Path expansion candidate count
            self.beam_candidates = self.beam_width * 2

        self._stop_token_ids_cache: Optional[set] = None

    @property
    def stop_token_ids(self):
        """Get the stop token ids (cached).

        This property is only used in beam search scenarios.
        """
        if self._stop_token_ids_cache is None:
            stop_token_ids = set()
            if self.sampling_params.stop_token_ids:
                stop_token_ids.update(self.sampling_params.stop_token_ids)
            if self.eos_token_ids:
                stop_token_ids.update(self.eos_token_ids)
            if self.tokenizer is not None:
                if self.tokenizer.eos_token_id is not None:
                    stop_token_ids.add(self.tokenizer.eos_token_id)
                if self.tokenizer.additional_stop_token_ids:
                    stop_token_ids.update(self.tokenizer.additional_stop_token_ids)
            self._stop_token_ids_cache = stop_token_ids
        return self._stop_token_ids_cache
