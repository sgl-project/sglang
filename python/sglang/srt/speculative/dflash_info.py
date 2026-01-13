# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DFlash speculative decoding input/output - minimal version."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
from sglang.srt.utils import next_power_of_2


@dataclass
class DFlashDraftInput(SpecInput):
    """Input for DFlash decode phase."""

    hidden_states: Optional[torch.Tensor]
    verified_id: torch.Tensor
    block_size: int = 16
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    def __post_init__(self):
        super().__init__(SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.verified_id is not None and self.verified_id.numel() > 0:
            if new_indices.numel() == 0:
                self.verified_id = self.verified_id[:0]
            elif has_been_filtered:
                self.verified_id = self.verified_id[:len(new_indices)]
            else:
                self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, other: "DFlashDraftInput"):
        if other is None:
            return
        if self.verified_id is None or self.verified_id.numel() == 0:
            self.verified_id = other.verified_id
        elif other.verified_id is not None and other.verified_id.numel() > 0:
            self.verified_id = torch.cat([self.verified_id, other.verified_id], dim=0)


@dataclass
class DFlashVerifyInput(SpecInput):
    """Input for DFlash verification."""

    def __init__(
        self,
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL,
    ):
        super().__init__(SpecInputType.DFLASH_VERIFY)
        self.draft_token = draft_token
        self.positions = positions
        self.draft_token_num = block_size
        self.capture_hidden_mode = capture_hidden_mode
        self.device = draft_token.device if draft_token is not None else "cuda"
        self.accept_length: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        """Allocate KV cache slots."""
        if batch.forward_mode.is_idle():
            return
        batch.input_ids = self.draft_token
        batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
        end_offset = batch.seq_lens + self.draft_token_num
        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int]:
        """Verify draft tokens using cumprod comparison."""
        bs = batch.batch_size()
        block = self.draft_token_num

        # Target predictions
        target = torch.argmax(logits_output.next_token_logits, dim=-1).reshape(bs, block)
        draft = self.draft_token.reshape(bs, block)

        # DFlash verification: draft[1:] should match target[:-1]
        matches = draft[:, 1:] == target[:, :-1]
        cumprod = matches.cumprod(dim=1)
        self.accept_length = cumprod.sum(dim=1).to(torch.int32)

        # Get verified tokens
        verified_id = target[torch.arange(bs, device=self.device), self.accept_length]

        # Update requests
        draft_flat = draft.flatten().tolist()
        accept_cpu = self.accept_length.cpu().tolist()
        accepted_indices = []

        for i, req in enumerate(batch.reqs):
            base = i * block
            acc = accept_cpu[i]
            for j in range(acc + 1):
                accepted_indices.append(base + j)
                req.output_ids.append(draft_flat[base + j])
                req.check_finished()
                if req.finished():
                    self.accept_length[i] = j
                    break
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += acc

        accepted = torch.tensor(accepted_indices, device=self.device, dtype=torch.long)

        # Free rejected cache
        evict = torch.ones(bs * block, dtype=torch.bool, device=self.device)
        evict[accepted] = False
        batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict])
        batch.out_cache_loc = batch.out_cache_loc[accepted]

        # Update lengths
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_cpu[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(self.accept_length.cpu() + 1)

        # Filter outputs
        logits_output.next_token_logits = logits_output.next_token_logits[accepted]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[accepted]

        return logits_output, verified_id, self.accept_length.sum().item()

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, other: "DFlashVerifyInput"):
        pass
