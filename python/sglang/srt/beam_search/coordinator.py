"""Scheduler wiring for one-logical-request beam search.

Each user request remains one Req. BeamGroup owns the vectorized beam state,
while ScheduleBatch forwards beam_width rows through the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.beam_search.beam_group import BeamGroup, BeamGroupState
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def prepare_vectorized_beam_decode(batch: ScheduleBatch) -> None:
    """Prepare one decode step for logical requests backed by beam rows."""
    batch.forward_mode = ForwardMode.DECODE
    batch.input_embeds = None
    batch.input_ids = torch.cat(
        [req.beam_group.next_tokens for req in batch.reqs]
    ).to(dtype=torch.int64)
    leader_allocated_lens = [req.kv.kv_allocated_len for req in batch.reqs]
    batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
    for req, allocated_len in zip(batch.reqs, leader_allocated_lens):
        req.kv.kv_allocated_len = allocated_len

    for req in batch.reqs:
        req.decode_batch_idx += 1
        req.beam_group.beam_seq_len += 1

    batch.seq_lens = batch.seq_lens + 1
    batch.seq_lens_cpu = batch.seq_lens_cpu + 1
    batch.orig_seq_lens = batch.orig_seq_lens + 1
    batch.seq_lens_sum = None


def release_vectorized_beam_resources(
    group: BeamGroup,
    req_to_token_pool,
    token_to_kv_pool_allocator,
) -> None:
    """Release vectorized decode rows while leaving the leader's prompt row."""
    rows = group.beam_req_pool_indices
    if rows is None or group.resources_released:
        return

    if group.beam_seq_len > group.prompt_len:
        slots = req_to_token_pool.req_to_token[
            rows, group.prompt_len : group.beam_seq_len
        ].flatten()
        token_to_kv_pool_allocator.free(slots.unique())

    req_to_token_pool.free_by_indices(rows.tolist())
    group.beam_req_pool_indices = None
    group.resources_released = True


@dataclass(kw_only=True)
class BeamCoordinator:
    server_args: ServerArgs
    model_config: ModelConfig
    spec_algorithm: SpeculativeAlgorithm
    enable_overlap: bool
    dllm_enabled: bool
    max_req_len: int
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache

    _pending_expand_groups: List[BeamGroup] = field(default_factory=list)

    @staticmethod
    def request_beam_width(recv_req) -> int:
        return getattr(recv_req.sampling_params, "beam_width", None) or 1

    def validate_and_init(self, req: Req, recv_req) -> Optional[str]:
        user_params = req.sampling_params
        beam_width = user_params.beam_width

        if self.enable_overlap:
            return (
                "Beam search currently requires the server to run with "
                "--disable-overlap-schedule."
            )
        if not self.spec_algorithm.is_none():
            return "Beam search is not supported with speculative decoding."
        if self.server_args.disaggregation_mode != "null":
            return "Beam search is not supported with PD disaggregation."
        if self.server_args.page_size > 1:
            return "Beam search currently requires --page-size 1."
        if self.dllm_enabled:
            return "Beam search is not supported with diffusion LLM."
        if getattr(self.server_args, "enable_hisparse", False):
            return "Beam search is not supported with hisparse."
        if self.server_args.pp_size > 1:
            return "Beam search is not supported with pipeline parallelism."
        if self.server_args.enable_hierarchical_cache:
            return "Beam search is not supported with hierarchical cache."

        if req.session_id is not None or recv_req.session_params is not None:
            return "Beam search is not supported for session requests."
        if recv_req.return_logprob:
            return "Beam search does not support return_logprob."
        if recv_req.return_hidden_states:
            return "Beam search does not support return_hidden_states."
        if recv_req.return_sampling_mask or recv_req.return_routed_experts:
            return "Beam search does not support sampling-mask/routed-experts returns."
        if any(
            value is not None
            for value in (
                user_params.json_schema,
                user_params.regex,
                user_params.ebnf,
                user_params.structural_tag,
            )
        ):
            return "Beam search is not supported with constrained decoding."
        if user_params.stop_strs or user_params.stop_regex_strs:
            return "Beam search does not support stop strings/regex yet."
        if user_params.min_new_tokens > 0:
            return "Beam search does not support min_new_tokens yet."
        if user_params.n > beam_width:
            return f"n ({user_params.n}) cannot exceed beam_width ({beam_width})."
        if 2 * beam_width > self.model_config.vocab_size:
            return f"beam_width ({beam_width}) is too large for the vocabulary."
        if beam_width + 1 > self.req_to_token_pool.size:
            return (
                f"beam_width ({beam_width}) needs {beam_width + 1} "
                f"req-to-token slots but the pool holds only "
                f"{self.req_to_token_pool.size}."
            )

        prompt_len = len(req.origin_input_ids)
        max_new_tokens = min(
            (
                user_params.max_new_tokens
                if user_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - prompt_len - 1,
        )
        if max_new_tokens < 1:
            return (
                "Beam search needs at least one generated token within the "
                f"context budget (prompt_len={prompt_len})."
            )

        group = BeamGroup(
            beam_width=beam_width,
            stop_token_ids=self._collect_stop_token_ids(req, user_params),
            max_new_tokens=max_new_tokens,
            num_return=user_params.n if user_params.n > 1 else beam_width,
            device=self.req_to_token_pool.device,
        )
        group.leader = req
        group.prompt_len = prompt_len
        group.beam_seq_len = prompt_len
        req.beam_group = group
        req.skip_radix_cache_insert = True
        return None

    @staticmethod
    def _collect_stop_token_ids(req: Req, user_params) -> List[int]:
        if user_params.ignore_eos:
            return []
        stop_ids = set(user_params.stop_token_ids or ())
        stop_ids |= set(req.eos_token_ids or ())
        tokenizer = req.tokenizer
        if tokenizer is not None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                stop_ids.add(tokenizer.eos_token_id)
            stop_ids |= set(
                getattr(tokenizer, "additional_stop_token_ids", None) or ()
            )
        return sorted(stop_ids)

    def on_leader_prefill(
        self, req: Req, i: int, logits_output: LogitsProcessorOutput
    ) -> None:
        group = req.beam_group
        if req.to_finish is not None:
            self._abort_group(group)
            return

        row_logprobs = logits_output.logprobs[i]
        top = row_logprobs.topk(group.num_candidates, sorted=True)
        selected = self._advance(group, top.values.unsqueeze(0), top.indices.unsqueeze(0))
        if selected is None:
            self._finish_group(group)
            return

        req.output_ids.append(selected[0])
        group.next_tokens = torch.tensor(
            selected,
            dtype=torch.int64,
            device=self.req_to_token_pool.device,
        )
        self._pending_expand_groups.append(group)

    def expand_pending_beams(self, batch: ScheduleBatch) -> bool:
        """Replace logical prefill rows with vectorized beam decode rows."""
        if not self._pending_expand_groups:
            return False
        if any(req.beam_group is None for req in batch.reqs):
            raise RuntimeError("Beam and normal requests must use separate batches.")

        pending_ids = {id(group) for group in self._pending_expand_groups}
        self._pending_expand_groups = []
        row_tensors = []
        seq_lens = []
        input_ids = []

        for req in batch.reqs:
            group = req.beam_group
            if id(group) not in pending_ids:
                if group.beam_req_pool_indices is None:
                    raise RuntimeError("Beam batch contains an unexpanded request.")
                rows = group.beam_req_pool_indices
            else:
                raw_rows = self.req_to_token_pool.alloc_by_count(group.beam_width)
                if raw_rows is None:
                    raise RuntimeError(
                        f"Beam expansion needs {group.beam_width} request rows."
                    )
                rows = torch.tensor(
                    raw_rows,
                    dtype=torch.int64,
                    device=self.req_to_token_pool.device,
                )
                for row in raw_rows:
                    self.req_to_token_pool.req_generation[row] += 1
                self.req_to_token_pool.req_to_token[
                    rows, : group.prompt_len
                ] = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : group.prompt_len
                ]
                group.beam_req_pool_indices = rows

            row_tensors.append(rows)
            seq_lens.extend([group.beam_seq_len] * group.beam_width)
            input_ids.append(group.next_tokens)

        batch.req_pool_indices = torch.cat(row_tensors)
        batch.req_pool_indices_cpu = batch.req_pool_indices.cpu()
        batch.seq_lens = torch.tensor(
            seq_lens, dtype=torch.int64, device=batch.device
        )
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=batch.device
        )
        batch.input_ids = torch.cat(input_ids)
        batch.out_cache_loc = None
        batch.seq_lens_sum = sum(seq_lens)
        batch.return_logprob = False
        batch.top_logprobs_nums = [0] * len(batch.reqs)
        batch.token_ids_logprobs = [None] * len(batch.reqs)
        return True

    def process_decode(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput
    ) -> None:
        offset = 0
        for req in batch.reqs:
            group = req.beam_group
            if group is None or group.state != BeamGroupState.DECODING:
                continue
            width = group.beam_width
            row_logprobs = logits_output.logprobs[offset : offset + width]
            offset += width

            top = row_logprobs.topk(group.num_candidates, dim=1, sorted=True)
            selected = self._advance(
                group,
                top.values,
                top.indices,
                return_parents=True,
            )
            if selected is None:
                self._finish_group(group)
                self._release_group_decode_resources(group)
                continue

            tokens, parents = selected
            self._reparent_group(group, parents)
            group.next_tokens = torch.tensor(
                tokens,
                dtype=torch.int64,
                device=self.req_to_token_pool.device,
            )

    def _advance(
        self,
        group: BeamGroup,
        top_logprobs: torch.Tensor,
        top_tokens: torch.Tensor,
        return_parents: bool = False,
    ):
        width = group.beam_width
        if group.next_step_is_final():
            group.advance_final(
                select_final_topk(
                    group.frontier_cum_logprobs,
                    top_logprobs,
                    top_tokens,
                    width,
                )
            )
            return None

        selected = joint_select(
            group.frontier_cum_logprobs,
            top_logprobs,
            top_tokens,
            group.stop_token_ids,
            width,
        )
        if group.advance(selected):
            return None
        tokens = selected.next_tokens[:width].tolist()
        if return_parents:
            return tokens, selected.parent_idx[:width].tolist()
        return tokens

    def _reparent_group(self, group: BeamGroup, parents: List[int]) -> None:
        rows = group.beam_req_pool_indices
        parent_idx = torch.tensor(
            parents, dtype=torch.int64, device=rows.device
        )
        old_mapping = self.req_to_token_pool.req_to_token[
            rows, group.prompt_len : group.beam_seq_len
        ].clone()
        new_mapping = old_mapping[parent_idx]
        self.req_to_token_pool.req_to_token[
            rows, group.prompt_len : group.beam_seq_len
        ] = new_mapping

        old_slots = old_mapping.flatten().unique()
        new_slots = new_mapping.flatten().unique()
        self.token_to_kv_pool_allocator.free(
            old_slots[~torch.isin(old_slots, new_slots)]
        )

    def _finish_group(self, group: BeamGroup) -> None:
        group.final_results = group.finalize()
        top = group.final_results[0]
        leader = group.leader
        if top.matched_token is not None:
            leader.finished_reason = FINISH_MATCHED_TOKEN(
                matched=top.matched_token
            )
        else:
            leader.finished_reason = FINISH_LENGTH(length=group.num_generated)

    def _abort_group(self, group: BeamGroup) -> None:
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        leader = group.leader
        leader.finished_reason = leader.to_finish or FINISH_ABORT(
            "Beam group aborted."
        )
        leader.to_finish = None
        self._release_group_decode_resources(group)

    def _release_group_decode_resources(self, group: BeamGroup) -> None:
        release_vectorized_beam_resources(
            group,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
        )
