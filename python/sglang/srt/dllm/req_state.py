from __future__ import annotations

import dataclasses
import enum
from array import array
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


@dataclasses.dataclass(slots=True, kw_only=True)
class DllmReqState:
    """A dLLM request's block state, ``req.dllm``; the dLLM owner attaches it
    when it takes the request and writes it at host commit."""

    config: DllmConfig
    phase: Optional[DllmReqPhase] = None
    # FDFO: the unresolved block and its algorithm state, kept for the next round.
    incomplete_ids: array = dataclasses.field(default_factory=lambda: array("q"))
    algo_state: Any = None
    block_offset: int = 0
    initialized: bool = False
    # A literal mask id among the committed tokens: committed tokens are
    # immutable, so the block then needs an explicit prompt mask.
    has_literal_mask: bool = False
    # The owner's waiting list keeps the order it took requests in.
    intake_seq: int = 0

    @classmethod
    def for_req(cls, req: Req, config: DllmConfig, intake_seq: int) -> DllmReqState:
        if config.requires_separate_context_encoding:
            phase = DllmReqPhase.INCOMING_PREFILL
        elif len(req.origin_input_ids) < config.block_size:
            phase = DllmReqPhase.INCOMING_DECODE
        else:
            phase = DllmReqPhase.INCOMING_PREFILL
        return cls(config=config, phase=phase, intake_seq=intake_seq)

    def init_next_round_input(
        self, req: Req, tree_cache: Optional[BasePrefixCache] = None
    ) -> None:
        """The dLLM form of ``Req.init_next_round_input``: the fill ids end with
        the next block, and a separate-context model matches no prefix past
        its context."""
        self.init_fill_ids(req)
        req.match_prefix(tree_cache, max_prefix_len=self.max_prefix_len())
        if tree_cache is not None:
            self.update_block_offset(req)
        self.determine_phase(req)

    def max_prefix_len(self) -> Optional[int]:
        if self.config.requires_separate_context_encoding:
            return self.block_offset
        return None

    def bound_prefix_len(self, req: Req, max_prefix_len: int) -> int:
        """Appended masks and a partial prompt/output block are not reusable
        KV: bidirectional attention depends on the rest of that block. Bound
        the lookup by real tokens, even if another cached prompt happens to
        contain the same token IDs as the synthetic masks."""
        block_size = self.config.block_size
        committed_len = len(req.origin_input_ids) + len(req.output_ids)
        max_prefix_len = min(max_prefix_len, committed_len)
        return max_prefix_len // block_size * block_size

    def is_prefill(self) -> bool:
        return self.phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_phase(self, req: Req):
        if self.incomplete_ids:
            self.phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = req.prefix_len
        if self.config.requires_separate_context_encoding:
            self.phase = (
                DllmReqPhase.STAGING_PREFILL
                if prefix_length < self.block_offset
                else DllmReqPhase.STAGING_DECODE
            )
            return

        min_required_length = prefix_length + self.config.block_size
        if len(req.full_untruncated_fill_ids) < min_required_length:
            return

        # Retraction preserves output_ids while releasing KV. Re-encode all
        # committed tokens, including literal mask IDs, before generating more.
        committed_len = len(req.origin_input_ids) + len(req.output_ids)
        if min_required_length <= committed_len:
            self.phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.phase = DllmReqPhase.STAGING_DECODE

    def init_fill_ids(self, req: Req):
        if self.incomplete_ids:
            prefix_len = req.prefix_len
            assert len(self.incomplete_ids) == self.config.block_size
            req.full_untruncated_fill_ids = (
                req.full_untruncated_fill_ids[:prefix_len] + self.incomplete_ids
            )
            # extend_end is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        if not self.initialized:
            # Recomputed after retraction, which resets the state.
            mask_id = self.config.mask_id
            self.has_literal_mask = (
                mask_id in req.origin_input_ids or mask_id in req.output_ids
            )

        if self.config.requires_separate_context_encoding:
            req.refresh_fill_ids()
            self.block_offset = req.seqlen
            if self.initialized:
                req.full_untruncated_fill_ids.extend([0] * self.config.block_size)
        else:
            self.block_offset = (
                0
                if not self.initialized
                else self.block_offset + self.config.block_size
            )
            req.full_untruncated_fill_ids = (
                req.origin_input_ids
                + req.output_ids
                + array(
                    "q",
                    [self.config.mask_id] * self.config.block_size,
                )
            )
        self.initialized = True

    def update_block_offset(self, req: Req):
        prefix_len = req.prefix_len
        if self.config.requires_separate_context_encoding:
            assert prefix_len <= self.block_offset
        else:
            assert prefix_len % self.config.block_size == 0, (
                f"Unexpected prefix len: {prefix_len}"
            )
            if prefix_len > self.block_offset:
                self.block_offset = prefix_len
