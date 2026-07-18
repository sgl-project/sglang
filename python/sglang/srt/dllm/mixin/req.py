from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Optional

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
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_step_map_state = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
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
        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.full_untruncated_fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.full_untruncated_fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_incomplete_ids:
            prefix_len = len(self.prefix_indices)
            assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
            self.full_untruncated_fill_ids = (
                self.full_untruncated_fill_ids[:prefix_len] + self.dllm_incomplete_ids
            )
            # extend_range is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        self.dllm_block_offset = (
            0
            if not self.dllm_initialized
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.dllm_step_map_state = None
        self.full_untruncated_fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
        )
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
