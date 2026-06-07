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
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is None:
            return
        if self.dllm_config.is_uniform:
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

        if self.dllm_config.is_uniform:
            # Uniform: the canvas begins once the cached prefix covers the real context.
            in_prefill = prefix_length < self.seqlen
        else:
            # Masked: the staging block still contains mask tokens to fill.
            input_block = self.full_untruncated_fill_ids[
                prefix_length:min_required_length
            ]
            in_prefill = self.dllm_config.mask_id not in input_block

        self.dllm_phase = (
            DllmReqPhase.STAGING_PREFILL if in_prefill else DllmReqPhase.STAGING_DECODE
        )

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_config.is_uniform:
            # Uniform (DiffusionGemma renoise): append a placeholder canvas (zeros)
            # only once the real context is fully cached, else encode context with no
            # canvas. Re-expressed in main's dllm_initialized idiom (#28054 used fill_len).
            context_len = self.seqlen
            self.dllm_block_offset = context_len
            if self.dllm_initialized and len(self.prefix_indices) >= context_len:
                self.full_untruncated_fill_ids = (
                    self.origin_input_ids
                    + self.output_ids
                    + array("q", [0] * self.dllm_config.block_size)
                )
            else:
                self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids
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

    def _update_block_offset_for_dllm(self: Req):
        prefix_len = len(self.prefix_indices)
        if self.dllm_config.is_uniform:
            context_len = self.seqlen
            assert (
                prefix_len <= context_len
            ), f"Unexpected uniform prefix len {prefix_len} > context {context_len}"
        else:
            assert (
                prefix_len % self.dllm_config.block_size == 0
            ), f"Unexpected prefix len: {prefix_len}"
            if prefix_len > self.dllm_block_offset:
                self.dllm_block_offset = prefix_len
