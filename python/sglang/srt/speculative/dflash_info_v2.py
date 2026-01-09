# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash V2-specific extensions for spec info classes.

Adds V2-specific methods for overlapped verification preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@triton.jit
def assign_extend_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    req_to_token_stride,
    draft_token_num: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """Triton kernel to assign cache locations for draft tokens."""
    pid = tl.program_id(0)
    if pid >= bs_upper:
        return
    
    req_pool_idx = tl.load(req_pool_indices + pid)
    start = tl.load(start_offset + pid)
    end = tl.load(end_offset + pid)
    
    for i in range(draft_token_num):
        if start + i < end:
            loc = tl.load(req_to_token + req_pool_idx * req_to_token_stride + start + i)
            tl.store(out_cache_loc + pid * draft_token_num + i, loc)


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign cache locations for draft token verification."""
    from sglang.srt.utils import next_power_of_2
    
    out_cache_loc = torch.empty(
        batch_size * draft_token_num,
        dtype=torch.int64,
        device=device,
    )
    
    assign_extend_cache_locs_kernel[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        draft_token_num,
        next_power_of_2(batch_size),
    )
    
    return out_cache_loc


@triton.jit
def fill_new_verified_id_kernel(
    all_verified_id,
    accept_length,
    verified_id_out,
    bs_upper: tl.constexpr,
):
    """Fill verified_id from all_verified_id based on accept_length."""
    pid = tl.program_id(0)
    if pid >= bs_upper:
        return
    
    acc_len = tl.load(accept_length + pid)
    # Get the token at position acc_len in all_verified_id
    # all_verified_id is flattened [total_verified]
    # Need cumsum to find position
    tl.store(verified_id_out + pid, tl.load(all_verified_id + pid + acc_len))


@dataclass
class DFlashVerifyInputV2Mixin:
    """
    V2-specific methods for DFlashVerifyInput.
    
    Adds support for overlapped verification preparation.
    """
    
    def prepare_for_v2_verify(
        self: DFlashVerifyInput,
        req_to_token_pool: "ReqToTokenPool",
        batch: "ModelWorkerBatch",
        target_worker: "TpModelWorker",
    ) -> Tuple[ForwardBatch, bool]:
        """
        Prepare for overlapped verification.
        
        This can run on a separate stream while draft is executing.
        
        Args:
            req_to_token_pool: Token pool mapping
            batch: Model worker batch
            target_worker: Target model worker
            
        Returns:
            Tuple of (forward_batch, can_run_cuda_graph)
        """
        if not batch.forward_mode.is_idle():
            # Assign cache locations for draft tokens
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )
        
        # Create forward batch for verification
        batch.forward_mode = (
            ForwardMode.IDLE if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(
            batch, target_worker.model_runner
        )
        
        # Check CUDA graph compatibility
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )
        
        return verify_forward_batch, can_run_cuda_graph
    
    def sample_v2(
        self: DFlashVerifyInput,
        batch: "ModelWorkerBatch",
        logits_output,
        vocab_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        V2 sampling with separated predict, accept_length, and accept_index.
        
        Returns:
            predict: Predicted tokens [bs]
            accept_length: Number of accepted tokens per request [bs]
            accept_index: Indices of accepted tokens in flattened draft
        """
        bs = len(batch.req_pool_indices)
        device = self.device
        
        # Get target predictions
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)
        
        # Get draft tokens
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        
        # Sequential verification:
        # candidates[:, 1:] should match target_predict[:, :-1]
        matches = (candidates[:, 1:] == target_predict[:, :-1])
        accept_mask = matches.cumprod(dim=1)
        accept_length = accept_mask.sum(dim=1).to(torch.int32)
        
        # Get bonus token (target's prediction at position accept_length)
        batch_indices = torch.arange(bs, device=device)
        predict = target_predict[batch_indices, accept_length]
        
        # Build accept_index (indices of accepted tokens in flattened draft)
        # For DFlash, this is straightforward since we accept sequentially
        accept_index = torch.zeros(
            bs * (self.draft_token_num + 1),
            dtype=torch.int32,
            device=device,
        )
        
        # Fill accept_index: for each request, indices 0..accept_length are accepted
        for i in range(bs):
            acc_len = accept_length[i].item()
            for j in range(acc_len + 1):  # +1 for bonus token
                accept_index[i * (self.draft_token_num + 1) + j] = (
                    i * self.draft_token_num + j
                )
        
        return predict, accept_length, accept_index


# Monkey-patch the V2 mixin methods onto DFlashVerifyInput
DFlashVerifyInput.prepare_for_v2_verify = DFlashVerifyInputV2Mixin.prepare_for_v2_verify
DFlashVerifyInput.sample_v2 = DFlashVerifyInputV2Mixin.sample_v2







