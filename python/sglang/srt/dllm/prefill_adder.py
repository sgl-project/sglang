"""The dLLM block budget: admission of one block per request in a dLLM step."""

from __future__ import annotations

from typing import Optional, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    AddReqResult,
    PrefillAdder,
    _PrefillAdmission,
)


class DllmPrefillAdder(PrefillAdder):
    """A PrefillAdder for dLLM steps: a request extends one block, never a
    chunk, and reserves no decode tokens; the dLLM step budget
    `rem_dllm_tokens` takes the chunk budget's place."""

    def __init__(self, *args, dllm_config: DllmConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.dllm_config = dllm_config
        self.exact_chunk_fill = False

        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = (
            self.rem_input_tokens
            if dllm_config.requires_separate_context_encoding
            else max_running_reqs * self.dllm_block_size
        )

    def budget_state(self):
        # The step budget closes admission as the chunk budget does.
        state = super().budget_state()
        if state == AddReqResult.CONTINUE and self.rem_dllm_tokens <= 0:
            return AddReqResult.OTHER
        return state

    def _charge_chunk_budget(self, compute_charge: int) -> None:
        self.rem_dllm_tokens -= compute_charge

    def _check_prefill_budget(
        self,
        req: Req,
        *,
        extend_input_len: int,
        total_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: Optional[int],
    ) -> tuple[bool, Optional[int]]:
        # No chunk cap: the block is the extend.
        can_admit, _ = super()._check_prefill_budget(
            req,
            extend_input_len=extend_input_len,
            total_tokens=total_tokens,
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=chunk_limit,
        )
        return can_admit, None

    def _select_prefill_admission(
        self,
        req: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: Optional[int],
        has_chunked_req: bool = False,
    ) -> Union[_PrefillAdmission, AddReqResult]:
        assert truncation_align_size is None, (
            "truncation_align_size is not supported for dllm prefill"
        )
        admission = super()._select_prefill_admission(
            req,
            total_tokens=total_tokens,
            host_hit_length=host_hit_length,
            swa_host_hit_length=swa_host_hit_length,
            truncation_align_size=truncation_align_size,
            has_chunked_req=has_chunked_req,
        )
        if isinstance(admission, AddReqResult):
            return admission
        extend_len = self._get_dllm_extend_len(req, admission.prefix_len)
        if extend_len <= 0:
            return AddReqResult.OTHER
        return _PrefillAdmission(
            admission.prefix_len, extend_len, 0, False, admission.replay_len
        )

    def _admit_ignore_eos(
        self, req: Req, cand_extend_input_len: int
    ) -> Optional[AddReqResult]:
        if self.rem_dllm_tokens <= 0:
            return AddReqResult.OTHER

        if (
            tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
        ) is not None:
            return tile_stop

        self._add_dllm_req(req)
        return None

    def _get_dllm_remain_tokens(self, req: Optional[Req] = None) -> int:
        _rem_tokens = min(self.rem_dllm_tokens, int(self.rem_total_tokens))
        if not (
            self.dllm_config.requires_separate_context_encoding
            and req is not None
            and req.dllm.is_prefill()
        ):
            _rem_tokens = min(_rem_tokens, self.dllm_block_size)
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        if self.dllm_config.requires_separate_context_encoding:
            _rem_tokens = min(_rem_tokens, int(self.cur_rem_tokens) - self.page_size)
            if self.is_hybrid_swa:
                _rem_tokens = min(
                    _rem_tokens, int(self.memory_budget.remaining_swa) - self.page_size
                )
        return _rem_tokens

    def _get_dllm_extend_len(self, req: Req, prefix_len: int) -> int:
        if self.dllm_config.requires_separate_context_encoding:
            remaining = (
                req.dllm.block_offset - prefix_len
                if req.dllm.is_prefill()
                else len(req.full_untruncated_fill_ids) - prefix_len
            )
            limit = self.rem_dllm_tokens
            if not req.dllm.is_prefill():
                limit = min(limit, self.dllm_block_size)
            trunc_len = min(limit, remaining)
        else:
            trunc_len = (
                min(self.rem_dllm_tokens, self.dllm_block_size)
                // self.page_size
                * self.page_size
            )

        return trunc_len

    def _add_dllm_req(self, req: Req):
        prefix_len = req.prefix_len
        trunc_len = self._get_dllm_extend_len(req, prefix_len)
        req.extend_end = prefix_len + trunc_len

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            replay_len=self._replay_len(req, prefix_len),
        )
        self._account_prefill_cache_admission(req, prefix_len)

    def add_dllm_staging_req(self, req: Req):
        _rem_tokens = self._get_dllm_remain_tokens(req)

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - req.prefix_len
        if (
            self.dllm_config.requires_separate_context_encoding
            and req.dllm.is_prefill()
        ):
            cand_extend_input_len = min(
                cand_extend_input_len,
                req.dllm.block_offset - req.prefix_len,
            )
        if (
            req.dllm.incomplete_ids
            or (
                self.dllm_config.requires_separate_context_encoding
                and not req.dllm.is_prefill()
            )
        ) and cand_extend_input_len > _rem_tokens:
            return AddReqResult.NO_TOKEN
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.extend_end = req.prefix_len + new_len
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0,
            req.extend_len,
            max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens(req) <= 0
            else AddReqResult.CONTINUE
        )
