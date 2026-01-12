# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DFlash speculative decoding dataclasses with cumprod verification."""

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

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

    hidden_states: torch.Tensor  # [batch, seq_len, hidden*num_layers]
    verified_id: torch.Tensor  # [batch]
    block_size: int = 16
    ctx_lens: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    ALLOC_LEN_PER_DECODE: ClassVar[int] = 16

    def __post_init__(self):
        super().__init__(SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        for attr in ("hidden_states", "verified_id"):
            tensor = getattr(self, attr)
            if tensor is not None and tensor.numel() > 0:
                if new_indices.numel() == 0:
                    setattr(self, attr, tensor[:0])
                elif has_been_filtered:
                    setattr(self, attr, tensor[: len(new_indices)])
                else:
                    setattr(self, attr, tensor[new_indices])

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        if spec_info is None:
            return
        if self.verified_id is None or self.verified_id.numel() == 0:
            self.verified_id = spec_info.verified_id
        elif spec_info.verified_id is not None and spec_info.verified_id.numel() > 0:
            self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
        elif spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states[:, -1:, :], spec_info.hidden_states[:, -1:, :]], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Input for DFlash target model verification with cumprod algorithm."""

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
        self.accept_index: Optional[torch.Tensor] = None
        self.predict: Optional[torch.Tensor] = None
        self.accepted_indices: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return
        batch.input_ids = self.draft_token
        batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices, batch.req_to_token_pool.req_to_token,
            batch.seq_lens, batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc, batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

    def verify(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput, page_size: int
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int]:
        """Verify draft tokens: compare draft[:,1:] == target[:,:-1], accept until mismatch."""
        bs, block_size = batch.batch_size(), self.draft_token_num

        # Compute matches
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).reshape(bs, block_size)
        candidates = self.draft_token.reshape(bs, block_size).to(torch.int64)
        matches = (candidates[:, 1:] == target_predict[:, :-1]).cumprod(dim=1)
        self.accept_length = matches.sum(dim=1).to(torch.int32)
        self.predict = candidates.to(torch.int32).flatten()

        # Build accept_index
        self.accept_index = torch.full((bs, block_size), -1, dtype=torch.int32, device=self.device)
        for i in range(bs):
            for j in range(self.accept_length[i].item() + 1):
                self.accept_index[i, j] = i * block_size + j
        self.accepted_indices = self.accept_index[self.accept_index != -1]

        # Fill requests
        predict_cpu = self.predict.tolist()
        for i, req in enumerate(batch.reqs):
            for j in range(self.accept_length[i].item() + 1):
                req.output_ids.append(predict_cpu[self.accept_index[i, j].item()])
                req.check_finished()
                if req.finished():
                    self.accept_index[i, j + 1:] = -1
                    break
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += (self.accept_index[i] != -1).sum().item() - 1

        # Recompute after potential early finish
        self.accept_length = (self.accept_index != -1).sum(dim=1).to(torch.int32) - 1
        self.accepted_indices = self.accept_index[self.accept_index != -1]

        # Free rejected cache
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[self.accepted_indices] = False
        batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        batch.out_cache_loc = batch.out_cache_loc[self.accepted_indices]

        # Update request state
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += self.accept_length[i].item() + 1
            req.kv_allocated_len = req.kv_committed_len

        # Get verified_id (last accepted token per request)
        last_indices = [(self.accept_index[i][self.accept_index[i] != -1][-1] if (self.accept_index[i] != -1).any() else i * block_size) for i in range(bs)]
        self.verified_id = self.predict[torch.tensor(last_indices, device=self.device, dtype=torch.long)]

        # Filter logits
        logits_output.next_token_logits = logits_output.next_token_logits[self.accepted_indices]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[self.accepted_indices]

        # Update batch seq_lens
        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(self.accept_length.cpu() + 1)

        return logits_output, self.verified_id, self.accept_length.cpu().sum().item()

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: "DFlashVerifyInput"):
        pass
