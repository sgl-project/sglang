from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
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
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_input_len for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : req.extend_input_len
            ]
            assert (
                offset + req.extend_input_len <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_input_len={req.extend_input_len}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_input_len] = chunk
            offset += req.extend_input_len

            pre_len = len(req.prefix_indices)
            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_input_len
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_input_len={req.extend_input_len}"

            if not req.retracted_stain:
                req.cached_tokens += pre_len - req.already_computed
                req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            req.extend_logprob_start_len = 0

        extend_input_logprob_token_ids = None

        # Set fields
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        self.output_ids = []
        for req in self.reqs:
            self.output_ids.append(req.output_ids[-1])
            self.tree_cache.cache_unfinished_req(req)
            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    # if it is not None, then the grammar is from a retracted request, and we should not
                    # accept the token as it's already accepted
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    release_kv_cache(req, self.tree_cache)
                    prepare_abort(
                        req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        self.output_ids = torch.tensor(self.output_ids, device=self.device)

        # Simulate the eagle run.
        if self.spec_algorithm.is_eagle():

            b = len(self.reqs)
            topk = server_args.speculative_eagle_topk
            topk_p = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_p[:topk],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for req in self.reqs
                ],
                dim=0,
            )
            topk_index = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_index[:topk],
                        device=self.device,
                        dtype=torch.int64,
                    )
                    for req in self.reqs
                ],
                dim=0,
            )

            hidden_states_list = [req.hidden_states_tensor for req in self.reqs]
            hidden_states = torch.stack(hidden_states_list, dim=0).to(self.device)

            # local import to avoid circular import
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            spec_info = EagleDraftInput(
                topk_p=topk_p,
                topk_index=topk_index,
                hidden_states=hidden_states,
                verified_id=self.output_ids,
                new_seq_lens=self.seq_lens,
                allocate_lens=self.seq_lens,
            )
            spec_info.prepare_for_extend(self)
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
            if self.enable_overlap:
                spec_info.future_indices = future_map.alloc_future_indices(
                    len(self.seq_lens)
                )
                future_map.store_to_map_for_new_batch(
                    spec_info.future_indices, spec_info
                )
            self.spec_info = spec_info
