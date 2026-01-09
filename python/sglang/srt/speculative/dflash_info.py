# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DFlash speculative decoding input/output dataclasses.

Following the NGRAM pattern for fixed-length draft token verification.
Uses fused verify_tree_greedy kernel for efficient verification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool
from sglang.srt.utils import next_power_of_2

# Import fused verification kernel
try:
    from sgl_kernel import verify_tree_greedy
    FUSED_KERNEL_AVAILABLE = True
except ImportError:
    FUSED_KERNEL_AVAILABLE = False

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureIndices


@dataclass
class DFlashDraftInput(SpecInput):
    """
    Input for DFlash decode phase - stores hidden states between iterations.
    Similar to EagleDraftInput.
    """
    
    # Multi-layer hidden states from target model [batch, seq_len, hidden*num_layers]
    hidden_states: torch.Tensor
    
    # Verified token from previous iteration [batch]
    verified_id: torch.Tensor
    
    # Block size
    block_size: int = 16
    
    # Capture mode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    
    # Future indices for overlap scheduling (V2)
    future_indices: Optional["FutureIndices"] = None
    
    # New sequence lengths after this batch (for V2 overlap)
    new_seq_lens: Optional[torch.Tensor] = None
    
    # CUDA event to synchronize verification completion (for V2 overlap)
    verify_done: Optional[torch.cuda.Event] = None
    
    def __post_init__(self):
        super().__init__(SpecInputType.DFLASH_DRAFT)
    
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1
    
    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """Filter batch by new indices."""
        if self.hidden_states is not None and self.hidden_states.numel() > 0:
            if new_indices.numel() == 0:
                self.hidden_states = self.hidden_states[:0]
            elif has_been_filtered:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            else:
                self.hidden_states = self.hidden_states[new_indices]
        
        if self.verified_id is not None and self.verified_id.numel() > 0:
            if new_indices.numel() == 0:
                self.verified_id = self.verified_id[:0]
            elif has_been_filtered:
                self.verified_id = self.verified_id[: len(new_indices)]
            else:
                self.verified_id = self.verified_id[new_indices]
    
    def merge_batch(self, spec_info: "DFlashDraftInput"):
        """Merge another batch into this one.
        
        CRITICAL FIX: Always concatenate verified_id, regardless of hidden_states.
        DFlash sets hidden_states=None, so the old logic would REPLACE verified_id
        instead of concatenating, causing decode requests to lose their verified_ids.
        """
        if spec_info is None:
            return
        
        # ALWAYS concatenate verified_id - this is critical for batching correctness
        if self.verified_id is None or self.verified_id.numel() == 0:
            self.verified_id = spec_info.verified_id
        elif spec_info.verified_id is not None and spec_info.verified_id.numel() > 0:
            self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        
        # Handle hidden_states separately (DFlash manages these per-request, not in spec_info)
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
        elif spec_info.hidden_states is not None:
            # For hidden_states, we need to handle varying sequence lengths
            # Take the last token's hidden states from each
            self.hidden_states = torch.cat(
                [self.hidden_states[:, -1:, :], spec_info.hidden_states[:, -1:, :]], dim=0
            )
    
    def prepare_for_decode(self, batch: "ScheduleBatch"):
        """
        Prepare for decode phase (V2 overlap).
        
        Allocates token slots and updates batch state for the decode phase.
        DFlash uses simpler allocation than EAGLE since it doesn't use top-k tree.
        """
        from sglang.srt.mem_cache.common import alloc_token_slots
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
        
        bs = batch.batch_size()
        
        # Wait for verification to complete
        batch.maybe_wait_verify_done()
        
        # Allocate token slots for next decode iteration
        # DFlash uses block_size tokens per request per iteration
        num_needed_tokens = bs * self.block_size
        
        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        
        for r in batch.reqs:
            # Allocate block_size tokens for draft speculation
            alloc_len = self.block_size * 2  # Extra headroom
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + alloc_len)
            r.kv_allocated_len += alloc_len
        
        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")
        
        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, sum(nxt_kv_lens_cpu - cur_kv_lens_cpu).item())
        else:
            from sglang.srt.mem_cache.common import alloc_paged_token_slots_extend, get_last_loc
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                (nxt_kv_lens_cpu - cur_kv_lens_cpu).sum().item(),
            )
        
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )
        
        # Sync seq_lens to CPU
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()


@dataclass
class DFlashVerifyInput(SpecInput):
    """
    Input for DFlash target model verification.
    
    Uses fused verify_tree_greedy kernel for efficient linear chain verification.
    The linear chain is represented as a degenerate tree with no siblings.
    """

    def __init__(
        self,
        draft_token: torch.Tensor,  # [bs * block_size]
        positions: torch.Tensor,     # [bs * block_size]
        block_size: int,
    ):
        super().__init__(SpecInputType.DFLASH_VERIFY)
        self.draft_token = draft_token
        self.positions = positions
        self.draft_token_num = block_size
        
        # Set device from draft_token or default to cuda
        if draft_token is not None:
            self.device = draft_token.device
        else:
            self.device = "cuda"
        
        # Pre-allocated tensors for fused kernel (built lazily)
        self.retrive_index: Optional[torch.Tensor] = None
        self.retrive_next_token: Optional[torch.Tensor] = None
        self.retrive_next_sibling: Optional[torch.Tensor] = None
        
        # Output tensors from verification
        self.accepted_indices: Optional[torch.Tensor] = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        """Allocate KV cache slots for draft tokens."""
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        # Allocate KV cache slots
        batch.out_cache_loc = alloc_token_slots(
            batch.tree_cache,
            len(batch.input_ids),
        )
        end_offset = batch.seq_lens + self.draft_token_num

        # Assign cache locations to request token pool
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

    def prepare_for_verify_v2(
        self,
        batch: ModelWorkerBatch,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        req_to_token_pool,
    ):
        """
        Allocate KV cache slots for draft tokens (V2 version).
        
        Works with ModelWorkerBatch instead of ScheduleBatch.
        """
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        # Allocate KV cache slots directly using the allocator
        num_tokens = len(batch.input_ids)
        out_cache_loc = token_to_kv_pool_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            raise RuntimeError(
                f"Out of memory in V2 verify. Failed to allocate {num_tokens} tokens."
            )
        batch.out_cache_loc = out_cache_loc
        
        end_offset = batch.seq_lens + self.draft_token_num

        # Assign cache locations to request token pool
        bs = len(batch.seq_lens)
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int]:
        """
        Verify draft tokens using the verify_tree_greedy CUDA kernel.
        
        Uses fused CUDA kernel for efficient tree/linear verification,
        matching EAGLE/NGRAM performance patterns.
        
        Returns:
            logits_output: Modified logits output
            verified_id: Next token IDs [batch]
            num_accepted_tokens: Total accepted tokens
        """
        bs = batch.batch_size()
        
        # 1. Run kernel verification
        self._kernel_verify(batch, logits_output)
        
        # 2. Fill requests with accepted tokens
        has_finished = self._fill_requests(batch)
        
        # 3. Free rejected KV cache (vectorized)
        self._free_cache(batch, page_size)
        
        # 4. Filter logits to accepted positions (vectorized)
        self._filter_logits(logits_output)
        
        # 5. Update sequence lengths
        accept_length_cpu = self.accept_length.cpu()
        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_length_cpu + 1)
        
        num_accepted_tokens = accept_length_cpu.sum().item()
        
        return logits_output, self.verified_id, num_accepted_tokens

    def _build_linear_chain_indices(self, bs: int):
        """Build tree traversal indices for linear chain verification.
        
        For linear chains, the structure is simple:
        - retrive_index: sequential indices [0, 1, 2, ...] per request
        - retrive_next_token: each token points to next [0→1, 1→2, ..., n-1→-1]
        - retrive_next_sibling: all -1 (no siblings in linear chain)
        
        These indices allow the verify_tree_greedy kernel to work with linear chains.
        """
        block_size = self.draft_token_num
        device = self.device
        
        # retrive_index: maps logical position to flat index
        # For request i, position j: index = i * block_size + j
        self.retrive_index = torch.arange(
            bs * block_size, device=device, dtype=torch.int64
        ).reshape(bs, block_size)
        
        # retrive_next_token: linear chain [0→1, 1→2, ..., n-2→n-1, n-1→-1]
        next_token = torch.arange(1, block_size + 1, device=device, dtype=torch.int64)
        next_token[-1] = -1  # Last token has no successor
        self.retrive_next_token = next_token.unsqueeze(0).expand(bs, -1).contiguous()
        
        # retrive_next_sibling: no siblings for linear chain (all -1)
        self.retrive_next_sibling = torch.full(
            (bs, block_size), -1, dtype=torch.int64, device=device
        )

    def _kernel_verify(self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput):
        """
        Verify draft tokens using fused verify_tree_greedy CUDA kernel.
        
        This is much faster than the Python-based verification because:
        1. Single fused kernel vs 10+ separate kernels
        2. No CPU-GPU sync (.item() calls) during verification
        3. All tensor operations done on GPU
        
        The linear chain is represented as a degenerate tree with no siblings.
        """
        bs = batch.batch_size()
        block_size = self.draft_token_num
        
        # Get target predictions
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, block_size).to(torch.int64)
        
        # Get draft tokens [bs, block_size]
        candidates = self.draft_token.reshape(bs, block_size).to(torch.int64)
        
        # Build linear chain indices if not already built or batch size changed
        if self.retrive_index is None or self.retrive_index.shape[0] != bs:
            self._build_linear_chain_indices(bs)
        
        # Output tensors - predict is flat [bs * block_size]
        self.predict = torch.empty(bs * block_size, dtype=torch.int32, device=self.device)
        self.accept_index = torch.full(
            (bs, block_size), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.zeros(bs, dtype=torch.int32, device=self.device)
        
        if FUSED_KERNEL_AVAILABLE:
            # Single fused kernel call - no CPU-GPU sync!
            verify_tree_greedy(
                predicts=self.predict,
                accept_index=self.accept_index,
                accept_token_num=self.accept_length,
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # Fallback to Python implementation if kernel not available
            self._kernel_verify_fallback(batch, logits_output, candidates, target_predict)
        
        # Flatten accepted_indices for downstream compatibility (like NGram)
        self.accepted_indices = self.accept_index[self.accept_index != -1]

    def _kernel_verify_fallback(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
    ):
        """
        Fallback Python implementation when fused kernel is not available.
        
        Uses the original DFlash verification logic with cumprod.
        """
        bs = batch.batch_size()
        block_size = self.draft_token_num
        
        # EXACT DFlash comparison: draft[:, 1:] == target_predict[:, :-1]
        draft_predictions = candidates[:, 1:]  # [bs, block_size-1]
        target_for_comparison = target_predict[:, :-1]  # [bs, block_size-1]
        
        # Compute matches and cumulative product (stops at first mismatch)
        matches = (draft_predictions == target_for_comparison)
        cumprod_matches = matches.cumprod(dim=1)
        
        # acceptance_length[i] = number of consecutive matches for request i
        self.accept_length = cumprod_matches.sum(dim=1).to(torch.int32)
        
        # Build predict tensor (flat format to match fused kernel output)
        self.predict = candidates.to(torch.int32).flatten()
        
        # Build accept_index to match fused kernel output format
        # accept_index[i, j] = global flat index of j-th accepted token for request i
        for i in range(bs):
            acc_len = self.accept_length[i].item()
            base_idx = i * block_size
            for j in range(acc_len + 1):  # +1 for the accepted tokens (not bonus)
                self.accept_index[i, j] = base_idx + j

    def _fill_requests(self, batch: ScheduleBatch) -> bool:
        """Fill accepted tokens into requests - following NGram pattern.
        
        accept_index[i] contains GLOBAL flat indices into predict[:].
        predict is flat [bs * block_size] matching the fused kernel output.
        """
        # Convert to CPU ONCE before loop
        accept_index_cpu = self.accept_index.tolist()
        predict_cpu = self.predict.tolist()  # Flat: [bs * block_size]
        has_finished = False

        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                # idx is a GLOBAL flat index into predict
                token_id = predict_cpu[idx]
                req.output_ids.append(token_id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # Mark remaining as rejected
                    self.accept_index[i, j + 1:] = -1
                    break
            
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += sum(1 for x in accept_index_row if x != -1) - 1

        if has_finished:
            self.accept_length = (self.accept_index != -1).sum(dim=1).to(torch.int32) - 1
        
        # Rebuild accepted_indices after potential modifications from finished requests
        self.accepted_indices = self.accept_index[self.accept_index != -1]
        
        return has_finished

    def _free_cache(self, batch: ScheduleBatch, page_size: int):
        """Free KV cache for rejected tokens - following NGram pattern.
        
        Uses accepted_indices (global flat indices) to determine which cache slots to keep.
        """
        bs = batch.batch_size()
        
        # Free the KV cache for unaccepted tokens using accepted_indices
        # accepted_indices contains global flat indices of accepted tokens
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[self.accepted_indices] = False
        batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        batch.out_cache_loc = batch.out_cache_loc[self.accepted_indices]

        # Update kv_committed_len
        accept_length_cpu = self.accept_length.cpu().tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_cpu[i] + 1
            req.kv_allocated_len = req.kv_committed_len
        
        # FIX: Get LAST accepted token per request (the bonus token)
        # accept_index has shape [bs, block_size], find last valid index per row
        # verified_id must have shape [bs] - one token per request
        last_indices = []
        for i in range(bs):
            row = self.accept_index[i]
            valid_mask = row != -1
            if valid_mask.any():
                # Get the last valid global index in this row
                last_pos = valid_mask.nonzero()[-1].item()
                last_idx = row[last_pos].item()
                last_indices.append(last_idx)
            else:
                # Fallback (shouldn't happen)
                last_indices.append(i * self.draft_token_num)
        
        self.verified_id = self.predict[torch.tensor(last_indices, device=self.device, dtype=torch.long)]

    def _filter_logits(self, logits_output: LogitsProcessorOutput):
        """Filter logits to accepted positions only - following NGram pattern.
        
        Uses accepted_indices (global flat indices) directly.
        Hidden states are also filtered if present.
        """
        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accepted_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                self.accepted_indices
            ]

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: "DFlashVerifyInput"):
        pass

