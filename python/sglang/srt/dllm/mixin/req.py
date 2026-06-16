from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            if self.dllm_config.causal_context:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.full_untruncated_fill_ids) < min_required_length:
            # still incoming stage
            return

        if self.dllm_config.causal_context:
            latest_block_start = (
                len(self.full_untruncated_fill_ids) - self.dllm_config.block_size
            )
            input_block = self.full_untruncated_fill_ids[latest_block_start:]
        else:
            input_block = self.full_untruncated_fill_ids[
                prefix_length:min_required_length
            ]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_config.causal_context:
            self.dllm_block_offset = len(self.origin_input_ids) + len(self.output_ids)
        else:
            self.dllm_block_offset = (
                0
                if not self.dllm_initialized
                else self.dllm_block_offset + self.dllm_config.block_size
            )
        self.full_untruncated_fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
        )
        self.dllm_initialized = True

    def _set_dllm_extend_range_to_fill_len(self: Req):
        prefix_len = len(self.prefix_indices)
        self.set_extend_range(prefix_len, len(self.full_untruncated_fill_ids))

    def init_prompt_cache_input(self: Req):
        """Prepare the causal prompt-cache pass."""
        # flatten_arrays_to_int64_tensor expects a buffer-backed array.
        self.full_untruncated_fill_ids = array("q", self.origin_input_ids)
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self._set_dllm_extend_range_to_fill_len()

    def check_finished_stop_before_length(self: Req, new_accepted_len: int = 1):
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if new_accepted_len <= 0:
            return

        from sglang.srt.managers.schedule_batch import (
            FINISH_LENGTH,
            FINISH_MATCHED_TOKEN,
        )

        if self.grammar is not None and self.grammar.is_terminated():
            self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
            return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish(new_accepted_len):
            return

        if self._check_token_based_finish(new_accepted_tokens):
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        # LinearSpec can later accept partial blocks, so only the prompt is aligned.
        if not self.output_ids and prefix_len % self.dllm_config.block_size != 0:
            raise ValueError(
                f"DLLM prefix len {prefix_len} is not aligned to "
                f"block_size {self.dllm_config.block_size}"
            )
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
