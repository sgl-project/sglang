from __future__ import annotations

import logging
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


class ScheduleBatchDisaggregationDecodeMixin:

    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """

        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_range.length for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)
            pre_len = len(req.prefix_indices)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                pre_len : pre_len + req.extend_range.length
            ]
            assert (
                offset + req.extend_range.length <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_range.length={req.extend_range.length}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_range.length] = chunk
            offset += req.extend_range.length

            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_range.length
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_range.length={req.extend_range.length}"

            if not req.retracted_stain:
                # Clamp to avoid double-counting: already_computed is seeded from
                # the prefill-reported cached_tokens in _commit_transfer_to_req, so
                # a decode-side prefix shorter than the prefill report must not
                # subtract from cached_tokens.
                delta = max(0, pre_len - req.already_computed)
                req.cached_tokens += delta
                req.cached_tokens_device += delta
                req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)

        # Set fields
        self.input_ids = torch.tensor(
            sum(input_ids, array("q")), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_range.length for r in reqs]
        self.extend_logprob_start_lens = None
        self.extend_input_logprob_token_ids = None
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _maybe_setup_mamba_prebuilt_checkpoint(self: ScheduleBatch, req):
        """Decode PD + mamba extra_buffer: seed the ping-pong track slot at PREBUILT
        admission so the normal cache_unfinished_req insert path runs.

        The prefill-transferred end-of-prompt SSM state lands in the request's main mamba
        slot (get_mamba_indices(req_pool_idx)). Under extra_buffer, cache_unfinished_req
        reads the ping-pong track slot get_mamba_ping_pong_other_idx(mamba_next_track_idx)
        and forks it; at admission that slot is still unwritten and mamba_last_track_seqlen
        is None, so the insert is skipped (prefix_len stays 0). We mirror the main slot into
        exactly that ping-pong slot and set the tracked seqlen to the page-aligned prompt
        length so cache_unfinished_req inserts a matchable node.

        The planted mamba state is at prompt_len (end-of-prompt), while the node depth is
        aligned_len (≤ prompt_len). In PD decode this is safe: every request's working mamba
        slot is overwritten by its own prefill transfer, so the cached mamba_value is only
        used as a "node has valid checkpoint" flag for prefix matching, never for computation.
        """
        tree_cache = self.tree_cache
        if tree_cache is None or not hasattr(tree_cache, "supports_mamba"):
            return
        if not tree_cache.supports_mamba():
            return
        if not getattr(tree_cache, "enable_mamba_extra_buffer", False):
            return
        pool = self.req_to_token_pool
        if req.mamba_ping_pong_track_buffer is None or req.mamba_next_track_idx is None:
            return
        page_size = tree_cache.page_size
        prompt_len = len(req.origin_input_ids)
        aligned_len = (
            (prompt_len // page_size * page_size) if page_size > 1 else prompt_len
        )
        if aligned_len == 0:
            return
        keep_idx = pool.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)
        dst_slot = req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1)
        main_slot = pool.get_mamba_indices(req.req_pool_idx).unsqueeze(-1)
        pool.mamba_pool.copy_from(main_slot, dst_slot)
        req.mamba_last_track_seqlen = aligned_len

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        last_tokens: List[int] = []
        for req in self.reqs:
            last_tokens.append(req.output_ids[-1])
            self._maybe_setup_mamba_prebuilt_checkpoint(req)
            maybe_cache_unfinished_req(req, self.tree_cache)
            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    # if it is not None, then the grammar is from a retracted request, and we should not
                    # accept the token as it's already accepted
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    from sglang.srt.managers.schedule_batch import FINISH_ABORT

                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    # Use to_finish (not finished_reason) so that process_batch_result_prebuilt
                    # handles the release via update_finish_state -> release_kv_cache in one place.
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    req.to_finish = FINISH_ABORT(
                        error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        last_tokens_tensor = torch.tensor(
            last_tokens, dtype=torch.int64, device=self.device
        )

        spec_info = self.spec_algorithm.build_disagg_draft_input(
            self,
            server_args,
            last_tokens_tensor,
            future_map,
        )
        if spec_info is not None:
            self.spec_info = spec_info
        else:
            # Non-spec: stash last token into the relay so the first DECODE's
            # resolve_forward_inputs gathers it like any other decode iter.
            future_map.stash(
                self.req_pool_indices, RelayPayload(bonus_tokens=last_tokens_tensor)
            )
            self.input_ids = None
