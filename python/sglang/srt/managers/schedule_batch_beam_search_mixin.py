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
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class ScheduleBatchBeamSearchMixin:
    """Mixin class for beam search related operations in ScheduleBatch."""

    def prepare_for_beam_search_decode(self: ScheduleBatch):
        """Prepare batch for beam search decode phase.

        This method sets up the batch for beam search decoding by:
        1. Collecting last tokens from all beam branches
        2. Initializing new beam search requests (allocating KV cache slots)
        3. Allocating output cache locations
        4. Updating sequence lengths
        """
        self.forward_mode = ForwardMode.DECODE

        beam_ids = torch.cat(
            [req.beam_list.last_tokens.to(torch.int32) for req in self.reqs]
        )

        # beam search not need self.output_ids
        self.input_ids = beam_ids
        self.output_ids = None

        self._prepare_for_new_beam_search()

        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)
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
        """
        Initialize beam search inference for new requests.

        Features:
            1. Allocate req_to_token slots for all beam branches (beam_width slots per request)
            2. Copy KV cache from normal slot to all beam branches in parallel
            3. Record the beam request slots in req_to_token to the local req_pool_indices
            4. Extend seq_lens and orig_seq_lens, replicating the same sequence length for each beam branch
            5. Set batch_slot_start_idx for each request (pointing to the request's starting position in the batch)

        Batch req_pool_indices layout for a single beam search request:
            [beam0_pool_idx | beam1_pool_idx | beam2_pool_idx | ... | beam(K-1)_pool_idx]
             └─────────────────────── beam_width slots ───────────────────────┘

            Each pool_idx points to a row in req_to_token_pool.req_to_token that stores
            the KV cache token indices for that beam branch.

        Memory management details:
            - Before extend: batch.req_pool_indices[i] points to the prefill slot (normal_idx)
            - After extend: batch.req_pool_indices[i:i+beam_width] are replaced with new beam slots
            - Important: The original prefill slot in batch.req_pool_indices is OVERWRITTEN and no
              longer accessible through batch.req_pool_indices
            - However: req.req_pool_idx still preserves the original prefill slot index, which is
              used by release_kv_cache() to free the prefill KV cache when the request finishes
            - This design allows both beam search and non-beam search requests to use the same
              cache_finished_req mechanism

        Notes:
            - Supports different beam_width for each request
            - All beam branches share the same prefill KV cache (implemented through parallel copying)
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
        beam_req_pool_indices = self.req_to_token_pool.alloc(sum(new_pool_slot_list))
        if not beam_req_pool_indices:
            raise RuntimeError(
                "Out of memory. Please set a smaller number for `--max-running-requests` or `--beam-width`."
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

    def _init_beam_search_attributes(self, sampling_params):
        """Initialize beam search related attributes.

        This method should be called from Req.__init__() to set up beam search state.

        Args:
            sampling_params: The sampling parameters that may contain beam search settings
        """
        self.is_beam_search = False
        self.beam_width = 0
        self.beam_candidates = 0

        if get_global_server_args().enable_beam_search:
            # sampling_params.n has already been validated in tokenizermanager
            self.is_beam_search = True
            self.beam_width = sampling_params.n
            self.beam_list = BeamSearchList()
            # Path expansion candidate count
            self.beam_candidates = self.beam_width * 2

        # stop_token_ids cache (only used in beam search)
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
