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
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.dllm_has_literal_mask = False

        if self.dllm_config is not None:
            if self.dllm_config.requires_separate_context_encoding:
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

    def get_cacheable_fill_ids(self: Req) -> array:
        """The fill ids a prefix cache may take ownership of.

        A dLLM request's in-flight denoise block is rewritten in full on every
        step (``dllm_incomplete_ids`` is reassigned in the scheduler mixin), so
        its token ids are not final. Because the dLLM page size is pinned to the
        block size, that block is a whole trailing page of the radix key: caching
        it makes the next step's key diverge there, which drives the match below
        ``cache_protected_len`` and leaves the tree owning two nodes for the same
        pages. Only the settled prefix is cacheable; the block is inserted on a
        later step, once it resolves and moves into the committed fill ids.
        """
        fill_ids = self.get_fill_ids()
        if self.dllm_incomplete_ids:
            return fill_ids[: len(fill_ids) - len(self.dllm_incomplete_ids)]
        return fill_ids

    def cacheable_checkpoint_len(self: Req, up_to: int) -> int:
        """Clamp a checkpoint length so the in-flight denoise block stays out.

        The block is the tail of ``[:extend_end]``, so the cacheable part ends
        where the block starts (see ``get_cacheable_fill_ids``).
        """
        if self.dllm_incomplete_ids:
            return min(up_to, self.extend_end - len(self.dllm_incomplete_ids))
        return up_to

    def determine_dllm_phase(self: Req):
        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = self.prefix_len
        if self.dllm_config.requires_separate_context_encoding:
            self.dllm_phase = (
                DllmReqPhase.STAGING_PREFILL
                if prefix_length < self.dllm_block_offset
                else DllmReqPhase.STAGING_DECODE
            )
            return

        min_required_length = prefix_length + self.dllm_config.block_size
        if len(self.full_untruncated_fill_ids) < min_required_length:
            return

        # Retraction preserves output_ids while releasing KV. Re-encode all
        # committed tokens, including literal mask IDs, before generating more.
        committed_len = len(self.origin_input_ids) + len(self.output_ids)
        is_prefill_phase = min_required_length <= committed_len

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_incomplete_ids:
            prefix_len = self.prefix_len
            assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
            self.full_untruncated_fill_ids = (
                self.full_untruncated_fill_ids[:prefix_len] + self.dllm_incomplete_ids
            )
            # extend_end is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        if not self.dllm_initialized:
            # Committed tokens are immutable, so a literal mask ID among them
            # needs the explicit prompt mask. Recomputed after retraction.
            mask_id = self.dllm_config.mask_id
            self.dllm_has_literal_mask = (
                mask_id in self.origin_input_ids or mask_id in self.output_ids
            )

        if self.dllm_config.requires_separate_context_encoding:
            self.refresh_fill_ids()
            self.dllm_block_offset = self.seqlen
            if self.dllm_initialized:
                self.full_untruncated_fill_ids.extend([0] * self.dllm_config.block_size)
        else:
            self.dllm_block_offset = (
                0
                if not self.dllm_initialized
                else self.dllm_block_offset + self.dllm_config.block_size
            )
            self.full_untruncated_fill_ids = (
                self.origin_input_ids
                + self.output_ids
                + array(
                    "q",
                    [self.dllm_config.mask_id] * self.dllm_config.block_size,
                )
            )
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = self.prefix_len
        if self.dllm_config.requires_separate_context_encoding:
            assert prefix_len <= self.dllm_block_offset
        else:
            assert prefix_len % self.dllm_config.block_size == 0, (
                f"Unexpected prefix len: {prefix_len}"
            )
            if prefix_len > self.dllm_block_offset:
                self.dllm_block_offset = prefix_len
