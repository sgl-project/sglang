"""
SDAR Interleaved Speculative Decoding Worker
=============================================

Implements interleaved large+small model drafting for SDAR dLLM (diffusion LM) models.
Based on eval_interleaved_draft.py algorithm.

Algorithm per decode step (one SDAR block of `block_size` positions):
  Round loop until block is fully filled:
    1. Large model drafts `draft_length` high-confidence mask positions
    2. Small model creates `num_branches` branch states
       (k=0 assumes 0 large accepted, k=1 assumes 1, ...)
    3. Build verification batch: (draft_length+1) large nodes + all small branch nodes
    4. Large model ONE batch verification forward
    5. Verify interleaved: accept large drafts by confidence order, then best small branch
    6. Apply accepted + bonus tokens to block_tokens

  After block filled:
    Commit final block to KV caches of both models (one EXTEND forward each).

Key differences from AR speculative decoding (EAGLE/STANDALONE):
  - Drafts positions by confidence (not sequential left-to-right)
  - Verification compares (position, token) pairs, not token chains
  - Uses SDAR's block-diffusion bidirectional attention (ENCODER_ONLY)
  - Both models are full SDAR variants (no special draft head needed)

Usage (via ServerArgs):
  speculative_algorithm = "SDAR_INTERLEAVED"
  speculative_draft_model_path = "JetLM/SDAR-1.7B-Chat"
  speculative_num_draft_tokens = 16    # block_size
  speculative_num_steps = 4            # draft_length per round
  speculative_eagle_topk = 3           # num_branches
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType, SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _SmallModelBranch:
    """One speculative branch: assumes `assumed_large_accepted` large candidates accepted."""
    assumed_large_accepted: int
    # Tokens assumed accepted: [(abs_position, token_id), ...]
    assumed_tokens: List[Tuple[int, int]]
    # Small model's draft candidates sorted by confidence desc: [(abs_pos, tok, conf), ...]
    draft_candidates: List[Tuple[int, int, float]]


@dataclass
class SDARInterleavedDraftState(SpecInput):
    """
    Per-batch state kept in batch.spec_info between decode steps.

    pending_large_drafts[i] is either:
      - None  → fresh large-model draft needed at round start
      - List  → draft candidates reused from last round's verification logits
    """
    pending_large_drafts: List[Optional[List[Tuple[int, int, float]]]]

    # Config (same for all requests in batch)
    draft_length: int = 4
    num_branches: int = 3
    block_size: int = 16
    mask_token_id: int = 0

    def __post_init__(self):
        # Use DFLASH_DRAFT as a placeholder type; no actual attention path is
        # reached through spec_info for this worker.
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        idx = new_indices.tolist()
        self.pending_large_drafts = [self.pending_large_drafts[i] for i in idx]

    def merge_batch(self, other: "SDARInterleavedDraftState"):
        if self.pending_large_drafts and other.pending_large_drafts:
            self.pending_large_drafts = self.pending_large_drafts + other.pending_large_drafts
        else:
            self.pending_large_drafts = [None] * (
                len(self.pending_large_drafts or []) + len(other.pending_large_drafts or [])
            )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class SDARInterleavedWorker:
    """
    Speculative decoding for SDAR dLLM using interleaved large+small model drafting.

    target_worker  : large SDAR model  (e.g. SDAR-8B-Chat)
    draft_worker   : small SDAR model  (e.g. SDAR-1.7B-Chat)

    KV cache architecture (mirrors StandaloneWorker):
      - Shared req_to_token_pool (same logical pos → slot mapping for both models)
      - Shared token_to_kv_pool_allocator (slot allocation counter)
      - Separate physical KV storage in each model's token_to_kv_pool
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.target_worker = target_worker
        self.device = target_worker.device
        self.page_size = server_args.page_size

        # Algorithm hyper-parameters
        # speculative_num_draft_tokens → block_size (positions per SDAR block)
        # speculative_num_steps        → draft_length (candidates per round)
        # speculative_eagle_topk       → num_branches
        self.block_size = int(server_args.speculative_num_draft_tokens or 16)
        self.draft_length = int(server_args.speculative_num_steps or 4)
        self.num_branches = int(server_args.speculative_eagle_topk or 3)

        # Shared memory pools (same as StandaloneWorker approach)
        target_req_to_token_pool, target_kv_allocator = target_worker.get_memory_pool()

        # Build draft server_args: same model path as speculative_draft_model_path,
        # context length aligned with target.
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )

        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=target_req_to_token_pool,
            token_to_kv_pool_allocator=target_kv_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )
        set_global_server_args_for_scheduler(saved_server_args)

        self.draft_model_runner = self.draft_worker.model_runner

        # Resolve mask token id
        self._mask_token_id = self._resolve_mask_token_id()

        # Pre-allocated position offset buffer [block_size]
        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )

        if tp_rank == 0:
            logger.info(
                "SDARInterleavedWorker ready. "
                "block_size=%d, draft_length=%d, num_branches=%d, mask_token_id=%d",
                self.block_size,
                self.draft_length,
                self.num_branches,
                self._mask_token_id,
            )

    # ------------------------------------------------------------------
    # Public interface (same as DFlashWorker / StandaloneWorker)
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        """Delegate attributes not found here to the target worker."""
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        pass

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_prefill(batch, **kwargs)
        else:
            return self._forward_decode(batch, **kwargs)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _forward_prefill(self, batch: ScheduleBatch, **kwargs) -> GenerationBatchResult:
        """
        Prefill both models.

        Target model is run first (standard path) to populate its KV cache.
        Draft model is then run in EXTEND mode to populate its own KV cache
        from the same input so that its prefix context is ready for decode.
        """
        bs = batch.batch_size()

        # --- Target prefill (standard) ---
        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, **kwargs
        )

        # --- Draft prefill ---
        # Re-prepare a fresh ModelWorkerBatch from the ScheduleBatch because
        # target forward may have modified model_worker_batch in place.
        draft_model_worker_batch = batch.get_model_worker_batch()
        with torch.inference_mode():
            self.draft_worker.forward_batch_generation(draft_model_worker_batch)

        # Initialise spec state for decode phase
        batch.spec_info = SDARInterleavedDraftState(
            pending_large_drafts=[None] * bs,
            draft_length=self.draft_length,
            num_branches=self.num_branches,
            block_size=self.block_size,
            mask_token_id=self._mask_token_id,
        )

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    # ------------------------------------------------------------------
    # Decode  (interleaved speculative block)
    # ------------------------------------------------------------------

    def _forward_decode(self, batch: ScheduleBatch, **kwargs) -> GenerationBatchResult:
        """Fill one SDAR block using the interleaved algorithm."""
        spec_state = batch.spec_info
        if not isinstance(spec_state, SDARInterleavedDraftState):
            # Fallback: plain target forward
            mwb = batch.get_model_worker_batch()
            return self.target_worker.forward_batch_generation(mwb, **kwargs)

        bs = batch.batch_size()
        block_size = self.block_size
        mask_id = self._mask_token_id

        # Initialize current block: all MASK tokens
        block_tokens = torch.full(
            (bs, block_size), mask_id, dtype=torch.long, device=self.device
        )
        # Absolute positions for each slot in the new block [bs, block_size]
        block_positions = (
            batch.seq_lens.unsqueeze(1).long() + self._block_pos_offsets
        )  # [bs, block_size]

        pending_large_drafts: List[Optional[List]] = list(spec_state.pending_large_drafts)
        total_accepted = 0

        # ---------------------------------------------------------------
        # Interleaved rounds until block is fully filled
        # ---------------------------------------------------------------
        max_rounds = block_size  # worst case: 1 token per round
        for _round in range(max_rounds):
            mask_counts = (block_tokens == mask_id).sum(dim=1)
            if int(mask_counts.max().item()) == 0:
                break  # block complete

            # --- Phase 1: Large-model draft candidates (per request) ---
            large_candidates_list = self._get_large_candidates(
                batch, block_tokens, block_positions, pending_large_drafts
            )

            # --- Phases 2-5: Per-request interleaved verify ---
            new_pending: List[Optional[List]] = [None] * bs
            for b in range(bs):
                if int(mask_counts[b].item()) == 0:
                    continue
                large_cands = large_candidates_list[b]
                if not large_cands:
                    continue

                # Phase 2: small-model branches
                small_branches = self._build_small_branches(
                    batch, b, block_tokens[b], block_positions[b], large_cands
                )

                # Phase 3: verification states
                all_states, large_node_range, small_branch_ranges = (
                    self._build_verification_states(
                        block_tokens[b], block_positions[b], large_cands, small_branches
                    )
                )

                # Phase 4: large-model batch verification
                verif_logits = self._run_verification_states(
                    self.target_worker.model_runner,
                    batch, b,
                    all_states, block_positions[b],
                )
                # verif_logits: [num_states, block_size, vocab_size]

                verif_probs = F.softmax(verif_logits.float(), dim=-1)
                verif_conf, verif_tokens = verif_probs.max(dim=-1)

                # Phase 5: verify interleaved
                result = _verify_interleaved(
                    all_states=all_states,
                    large_candidates=large_cands,
                    small_branches=small_branches,
                    large_node_range=large_node_range,
                    small_branch_ranges=small_branch_ranges,
                    verif_conf=verif_conf,
                    verif_tokens=verif_tokens,
                    block_positions=block_positions[b],
                    mask_id=mask_id,
                )

                # Apply accepted tokens to block_tokens[b]
                for abs_pos, tok in result["accepted_tokens"]:
                    rel = int(abs_pos) - int(block_positions[b, 0].item())
                    if 0 <= rel < block_size:
                        block_tokens[b, rel] = tok

                if result["bonus"] is not None:
                    abs_pos, tok = result["bonus"]
                    rel = int(abs_pos) - int(block_positions[b, 0].item())
                    if 0 <= rel < block_size:
                        block_tokens[b, rel] = tok

                n = len(result["accepted_tokens"]) + (1 if result["bonus"] else 0)
                total_accepted += n

                # Reuse verification logits as next round's large draft (if there are
                # remaining masks and a valid last_state_idx from the algorithm).
                last_idx = result["last_state_idx"]
                if last_idx is not None:
                    remaining = (block_tokens[b] == mask_id)
                    if remaining.any():
                        rem_rel = remaining.nonzero(as_tuple=True)[0]
                        new_cands = []
                        for rp in rem_rel.tolist():
                            ap = int(block_positions[b, rp].item())
                            t = int(verif_tokens[last_idx, rp].item())
                            c = float(verif_conf[last_idx, rp].item())
                            new_cands.append((ap, t, c))
                        new_cands.sort(key=lambda x: x[2], reverse=True)
                        new_pending[b] = new_cands[: self.draft_length]

            pending_large_drafts = new_pending

            # Safety: if no tokens were accepted in this round, avoid infinite loop
            # (fall through; remaining masks will stay as-is and block_tokens may
            # still contain mask_id — we'll overwrite them with a final greedy pass)
            all_accepted_now = int((block_tokens == mask_id).sum().item())
            if all_accepted_now == 0:
                break

        # If any masks remain (should not happen under normal operation, but guard
        # against pathological cases), fill them with a fresh large-model greedy pass.
        remaining_mask = (block_tokens == mask_id).any(dim=1)
        if remaining_mask.any():
            self._fill_remaining_masks(batch, block_tokens, block_positions, remaining_mask)

        # ---------------------------------------------------------------
        # Commit the accepted block to both KV caches
        # ---------------------------------------------------------------
        self._commit_block(batch, block_tokens, block_positions)

        # Update batch seq_lens (SGLang expects this to be updated after commit)
        batch.seq_lens = batch.seq_lens + block_size

        # Reset pending drafts
        spec_state.pending_large_drafts = [None] * bs
        batch.spec_info = spec_state
        # Reset forward_mode to DECODE for next step
        batch.forward_mode = ForwardMode.DECODE

        # ---------------------------------------------------------------
        # Update req.output_ids and check finish conditions (Spec V1 requires
        # the worker to do this; the scheduler's process_batch_result_decode
        # skips these steps for Spec V1, assuming the worker already handled them).
        # ---------------------------------------------------------------
        for i, req in enumerate(batch.reqs):
            # kv_allocated: we committed all block_size slots to the KV pool in _commit_block
            # kv_committed: only the tokens actually added in this block (kv_before + num_added).
            #
            # IMPORTANT: do NOT use (len(origin_ids) + len(output_ids)) for kv_committed because
            # output_ids[0] was produced by prefill and its KV is already counted in the prefill
            # allocation; using raw output_ids length would be off by 1 and cause cache_finished_req
            # to read req_to_token_pool one slot past the end (garbage → double-count in evictable).
            #
            # release_kv_cache frees slots in [kv_committed_len, kv_allocated_len) automatically,
            # which covers the trailing slots from blocks where EOS / max_new_tokens fired early.
            kv_before = req.kv_committed_len  # KV length before this block
            output_before = len(req.output_ids)
            block_ids = block_tokens[i].tolist()
            for tok in block_ids:
                req.output_ids.append(tok)
                req.check_finished()
                if req.finished():
                    break
            num_added = len(req.output_ids) - output_before
            # All block_size slots were permanently written to both model KV caches
            req.kv_allocated_len = kv_before + block_size
            # Only count the tokens we actually consumed (avoids off-by-one vs output_ids length)
            req.kv_committed_len = kv_before + num_added

        next_token_ids = block_tokens[:, -1]

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            num_accepted_tokens=bs * block_size,
            can_run_cuda_graph=False,
        )

    # ------------------------------------------------------------------
    # Phase 1 helper: large-model draft (batched)
    # ------------------------------------------------------------------

    def _get_large_candidates(
        self,
        batch: ScheduleBatch,
        block_tokens: torch.Tensor,
        block_positions: torch.Tensor,
        pending: List[Optional[List]],
    ) -> List[Optional[List[Tuple[int, int, float]]]]:
        """
        Return per-request large-model draft candidates.
        Requests with pending drafts from a previous round skip the model forward.
        The others are batched into a single TARGET_VERIFY forward.
        """
        bs = batch.batch_size()
        result: List[Optional[List]] = [None] * bs

        needs_fresh = [b for b in range(bs) if pending[b] is None]

        if needs_fresh:
            # Run target model on current block for these requests (temporary)
            fresh_logits = self._run_model_on_block_subset(
                self.target_worker.model_runner,
                batch,
                block_tokens[needs_fresh],
                block_positions[needs_fresh],
                [batch.req_pool_indices[b] for b in needs_fresh],
                batch.seq_lens[needs_fresh],
            )
            # fresh_logits: [len(needs_fresh), block_size, vocab_size]
            probs = F.softmax(fresh_logits.float(), dim=-1)
            conf, tokens = probs.max(dim=-1)  # each [len(needs_fresh), block_size]

            for i, b in enumerate(needs_fresh):
                mask_rel = (block_tokens[b] == self._mask_token_id).nonzero(as_tuple=True)[0]
                cands = []
                for rp in mask_rel.tolist():
                    ap = int(block_positions[b, rp].item())
                    t = int(tokens[i, rp].item())
                    c = float(conf[i, rp].item())
                    cands.append((ap, t, c))
                cands.sort(key=lambda x: x[2], reverse=True)
                result[b] = cands[: self.draft_length]

        for b in range(bs):
            if pending[b] is not None:
                result[b] = pending[b]

        return result

    # ------------------------------------------------------------------
    # Phase 2 helper: small-model branches
    # ------------------------------------------------------------------

    def _build_small_branches(
        self,
        batch: ScheduleBatch,
        req_idx: int,
        block_tokens: torch.Tensor,   # [block_size]
        block_positions: torch.Tensor, # [block_size]
        large_cands: List[Tuple[int, int, float]],
    ) -> List[_SmallModelBranch]:
        """
        Run small model on `min(num_branches, len(large_cands)+1)` branch states.
        Branch k assumes the first k large candidates were accepted.
        """
        num_large = len(large_cands)
        actual_branches = min(self.num_branches, num_large + 1)
        block_size = int(block_tokens.shape[0])
        mask_id = self._mask_token_id

        # Build hypothesis inputs: [actual_branches, block_size]
        hyp_states = []
        assumed_lists = []
        for k in range(actual_branches):
            state = block_tokens.clone()
            assumed = []
            for i in range(k):
                ap, tok, _ = large_cands[i]
                rel = ap - int(block_positions[0].item())
                if 0 <= rel < block_size:
                    state[rel] = tok
                    assumed.append((ap, tok))
            hyp_states.append(state)
            assumed_lists.append(assumed)

        hyp_batch = torch.stack(hyp_states, dim=0)  # [actual_branches, block_size]
        # Expand positions and req_pool_indices
        exp_positions = block_positions.unsqueeze(0).expand(actual_branches, -1)  # [k, bs]
        exp_req_pool = batch.req_pool_indices[req_idx].unsqueeze(0).expand(actual_branches)
        exp_seq_lens = batch.seq_lens[req_idx].unsqueeze(0).expand(actual_branches)

        # Run small model on all branches (temporary)
        small_logits = self._run_model_on_block_subset(
            self.draft_model_runner,
            batch,
            hyp_batch,
            exp_positions,
            [batch.req_pool_indices[req_idx]] * actual_branches,
            exp_seq_lens,
        )
        # small_logits: [actual_branches, block_size, vocab_size]
        s_probs = F.softmax(small_logits.float(), dim=-1)
        s_conf, s_tokens = s_probs.max(dim=-1)  # [actual_branches, block_size]

        branches = []
        for k in range(actual_branches):
            cur_state = hyp_states[k]
            mask_rel = (cur_state == mask_id).nonzero(as_tuple=True)[0]
            cands = []
            for rp in mask_rel.tolist():
                ap = int(block_positions[rp].item())
                t = int(s_tokens[k, rp].item())
                c = float(s_conf[k, rp].item())
                cands.append((ap, t, c))
            cands.sort(key=lambda x: x[2], reverse=True)
            branches.append(
                _SmallModelBranch(
                    assumed_large_accepted=k,
                    assumed_tokens=assumed_lists[k],
                    draft_candidates=cands[: self.draft_length],
                )
            )
        return branches

    # ------------------------------------------------------------------
    # Phase 3 helper: build verification states
    # ------------------------------------------------------------------

    def _build_verification_states(
        self,
        block_tokens: torch.Tensor,    # [block_size]
        block_positions: torch.Tensor, # [block_size]
        large_cands: List[Tuple[int, int, float]],
        small_branches: List[_SmallModelBranch],
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], Dict[int, Tuple[int, int]]]:
        """
        Returns:
          all_states        : list of [block_size] tensors (one per verification node)
          large_node_range  : (start, end) index into all_states
          small_branch_ranges: {k: (start, end)} index into all_states
        """
        block_size = int(block_tokens.shape[0])
        base_pos = int(block_positions[0].item())
        all_states: List[torch.Tensor] = []

        # Large verification nodes: states with i=0..len(large_cands) of the large draft applied
        large_start = 0
        for i in range(len(large_cands) + 1):
            state = block_tokens.clone()
            for j in range(i):
                ap, tok, _ = large_cands[j]
                rel = ap - base_pos
                if 0 <= rel < block_size:
                    state[rel] = tok
            all_states.append(state)
        large_end = len(all_states)

        # Small branch verification nodes
        small_branch_ranges: Dict[int, Tuple[int, int]] = {}
        for branch in small_branches:
            k = branch.assumed_large_accepted
            branch_start = len(all_states)

            # Base state for this branch (large_cands[:k] accepted)
            base_state = block_tokens.clone()
            for ap, tok in branch.assumed_tokens:
                rel = ap - base_pos
                if 0 <= rel < block_size:
                    base_state[rel] = tok

            for i in range(len(branch.draft_candidates) + 1):
                state = base_state.clone()
                for j in range(i):
                    ap, tok, _ = branch.draft_candidates[j]
                    rel = ap - base_pos
                    if 0 <= rel < block_size:
                        state[rel] = tok
                all_states.append(state)

            small_branch_ranges[k] = (branch_start, len(all_states))

        return all_states, (large_start, large_end), small_branch_ranges

    # ------------------------------------------------------------------
    # Phase 4 helper: run model on multiple verification states
    # ------------------------------------------------------------------

    def _run_verification_states(
        self,
        model_runner,
        batch: ScheduleBatch,
        req_idx: int,
        all_states: List[torch.Tensor],   # list of [block_size]
        block_positions: torch.Tensor,    # [block_size]
    ) -> torch.Tensor:
        """
        Run model on all states in one batched forward for request req_idx.
        Returns tensor [num_states, block_size, vocab_size].
        """
        n = len(all_states)
        state_batch = torch.stack(all_states, dim=0)         # [n, block_size]
        exp_positions = block_positions.unsqueeze(0).expand(n, -1)  # [n, block_size]
        exp_req_pool = [batch.req_pool_indices[req_idx]] * n
        exp_seq_lens = batch.seq_lens[req_idx : req_idx + 1].expand(n)

        return self._run_model_on_block_subset(
            model_runner,
            batch,
            state_batch,
            exp_positions,
            exp_req_pool,
            exp_seq_lens,
        )  # [n, block_size, vocab_size]

    # ------------------------------------------------------------------
    # Low-level forward helpers
    # ------------------------------------------------------------------

    def _run_model_on_block_subset(
        self,
        model_runner,
        batch: ScheduleBatch,
        block_tokens: torch.Tensor,    # [n, block_size]
        block_positions: torch.Tensor, # [n, block_size]
        req_pool_indices_list,         # list/tensor of length n
        seq_lens,                      # tensor [n]
    ) -> torch.Tensor:
        """
        Run model on `n` requests each with `block_size` block tokens.
        Uses temporary KV allocation (backup/restore).
        Returns [n, block_size, vocab_size].
        """
        n = block_tokens.shape[0]
        block_size = block_tokens.shape[1]
        device = self.device

        if isinstance(req_pool_indices_list, list):
            req_pool_indices = torch.stack([
                r if isinstance(r, torch.Tensor) else torch.tensor(r, device=device)
                for r in req_pool_indices_list
            ])
        else:
            req_pool_indices = req_pool_indices_list.to(device)

        if not isinstance(seq_lens, torch.Tensor):
            seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        seq_lens = seq_lens.to(device, dtype=torch.int32)

        allocator = model_runner.token_to_kv_pool_allocator
        backup = allocator.backup_state()
        try:
            cache_loc = self._alloc_block_cache(
                model_runner, batch, req_pool_indices, seq_lens, n, block_size
            )
            if cache_loc is None:
                raise RuntimeError(
                    "SDARInterleavedWorker: OOM during temporary block KV allocation"
                )

            # Register temporary positions in req_to_token_pool so attention
            # backend can look them up during the forward.
            end_lens = seq_lens.to(torch.int32) + block_size
            assign_req_to_token_pool_func(
                req_pool_indices,
                model_runner.req_to_token_pool.req_to_token,
                seq_lens,
                end_lens,
                cache_loc,
                n,
            )

            spec_info = DFlashVerifyInput(
                draft_token=block_tokens.reshape(-1),
                positions=block_positions.reshape(-1),
                draft_token_num=block_size,
                custom_mask=None,  # ENCODER_ONLY gives full attention
            )

            seq_lens_sum = int(seq_lens.sum().item())
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=n,
                input_ids=block_tokens.reshape(-1),
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=block_positions.reshape(-1),
                req_to_token_pool=model_runner.req_to_token_pool,
                token_to_kv_pool=model_runner.token_to_kv_pool,
                attn_backend=model_runner.attn_backend,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                logits_output = model_runner.forward(forward_batch)

            # next_token_logits: [n * block_size, vocab_size]
            logits = logits_output.logits_output.next_token_logits
            return logits.view(n, block_size, -1)

        finally:
            allocator.restore_state(backup)

    def _run_single_state(
        self,
        model_runner,
        batch: ScheduleBatch,
        req_idx: int,
        state: torch.Tensor,           # [block_size]
        block_positions: torch.Tensor, # [block_size]
    ) -> torch.Tensor:
        """
        Run model on a single block state for one request.
        Returns [block_size, vocab_size].
        """
        block_size = int(state.shape[0])
        logits_2d = self._run_model_on_block_subset(
            model_runner,
            batch,
            state.unsqueeze(0),
            block_positions.unsqueeze(0),
            [batch.req_pool_indices[req_idx]],
            batch.seq_lens[req_idx : req_idx + 1],
        )
        return logits_2d[0]  # [block_size, vocab_size]

    def _alloc_block_cache(
        self,
        model_runner,
        batch: ScheduleBatch,
        req_pool_indices: torch.Tensor,  # [n]
        seq_lens: torch.Tensor,          # [n] int32
        n: int,
        block_size: int,
    ) -> Optional[torch.Tensor]:
        """Allocate KV slots for n * block_size positions."""
        allocator = model_runner.token_to_kv_pool_allocator
        if self.page_size == 1:
            return allocator.alloc(n * block_size)
        else:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)
            end_lens = seq_lens + block_size
            end_lens_cpu = seq_lens_cpu + block_size
            last_loc = get_last_loc(
                model_runner.req_to_token_pool.req_to_token,
                req_pool_indices,
                seq_lens,
            )
            return alloc_paged_token_slots_extend(
                batch.tree_cache,
                seq_lens,
                seq_lens_cpu,
                end_lens,
                end_lens_cpu,
                last_loc,
                n * block_size,
            )

    # ------------------------------------------------------------------
    # Fill remaining masks (safety fallback)
    # ------------------------------------------------------------------

    def _fill_remaining_masks(
        self,
        batch: ScheduleBatch,
        block_tokens: torch.Tensor,    # [bs, block_size]
        block_positions: torch.Tensor, # [bs, block_size]
        remaining_mask: torch.Tensor,  # [bs] bool
    ):
        """Greedily fill any positions still containing mask_token_id."""
        bs = block_tokens.shape[0]
        needs = remaining_mask.nonzero(as_tuple=True)[0].tolist()
        if not needs:
            return

        logits = self._run_model_on_block_subset(
            self.target_worker.model_runner,
            batch,
            block_tokens[needs],
            block_positions[needs],
            [batch.req_pool_indices[b] for b in needs],
            batch.seq_lens[needs],
        )
        # logits: [len(needs), block_size, vocab]
        greedy = logits.argmax(dim=-1)  # [len(needs), block_size]
        mask_id = self._mask_token_id
        for i, b in enumerate(needs):
            still_mask = (block_tokens[b] == mask_id)
            block_tokens[b][still_mask] = greedy[i][still_mask]

    # ------------------------------------------------------------------
    # Commit block to KV caches (both models, permanent allocation)
    # ------------------------------------------------------------------

    def _commit_block(
        self,
        batch: ScheduleBatch,
        block_tokens: torch.Tensor,    # [bs, block_size]  (all masks filled)
        block_positions: torch.Tensor, # [bs, block_size]
    ):
        """
        Run both models with the accepted block tokens to permanently store their KV.
        Uses TARGET_VERIFY mode WITHOUT backup/restore so the KV cache persists.
        Also updates req_to_token_pool for the committed block.

        NOTE: target and draft share the SAME token_to_kv_pool_allocator and
        req_to_token_pool (allocated once in StandaloneWorker.__init__).  We must
        allocate and register the KV slots exactly ONCE; both models then write
        their own KV tensors into their respective token_to_kv_pool buffers at
        the same slot indices.
        """
        bs = batch.batch_size()
        block_size = self.block_size

        # Use the shared allocator / req_to_token_pool (both models point to the same object)
        shared_runner = self.target_worker.model_runner
        allocator = shared_runner.token_to_kv_pool_allocator
        seq_lens = batch.seq_lens.to(torch.int32)
        end_lens = seq_lens + block_size

        if self.page_size == 1:
            cache_loc = allocator.alloc(bs * block_size)
        else:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)
            end_lens_cpu = seq_lens_cpu + block_size
            last_loc = get_last_loc(
                shared_runner.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                seq_lens,
            )
            cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                seq_lens,
                seq_lens_cpu,
                end_lens,
                end_lens_cpu,
                last_loc,
                bs * block_size,
            )

        if cache_loc is None:
            raise RuntimeError(
                "SDARInterleavedWorker: OOM during block commit KV allocation"
            )

        # Register slots in req_to_token_pool once (shared by both models)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            shared_runner.req_to_token_pool.req_to_token,
            seq_lens,
            end_lens,
            cache_loc,
            bs,
        )

        seq_lens_sum = int(seq_lens.sum().item())
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)
        spec_info = DFlashVerifyInput(
            draft_token=block_tokens.reshape(-1),
            positions=block_positions.reshape(-1),
            draft_token_num=block_size,
            custom_mask=None,
        )

        # Run both models with the same cache_loc slots; each model writes its own
        # KV tensors into its own token_to_kv_pool buffer at those indices.
        for model_runner in (self.target_worker.model_runner, self.draft_model_runner):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_tokens.reshape(-1),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=block_positions.reshape(-1),
                req_to_token_pool=model_runner.req_to_token_pool,
                token_to_kv_pool=model_runner.token_to_kv_pool,
                attn_backend=model_runner.attn_backend,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                model_runner.forward(forward_batch)

    # ------------------------------------------------------------------
    # Mask token resolution
    # ------------------------------------------------------------------

    def _resolve_mask_token_id(self) -> int:
        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is not None:
            mid = getattr(tokenizer, "mask_token_id", None)
            if mid is not None:
                return int(mid)
            vocab = tokenizer.get_vocab()
            for s in ("[MASK]", "<mask>", "<MASK>"):
                if s in vocab:
                    return int(vocab[s])

        hf_cfg = self.target_worker.model_runner.model_config.hf_config
        mid = getattr(hf_cfg, "mask_token_id", None)
        if mid is not None:
            return int(mid)

        raise RuntimeError(
            "SDARInterleavedWorker: cannot resolve mask_token_id. "
            "Ensure the model config or tokenizer defines a mask token."
        )


# ---------------------------------------------------------------------------
# Standalone verification logic (ported from eval_interleaved_draft.py)
# ---------------------------------------------------------------------------

def _verify_interleaved(
    all_states: List[torch.Tensor],
    large_candidates: List[Tuple[int, int, float]],
    small_branches: List[_SmallModelBranch],
    large_node_range: Tuple[int, int],
    small_branch_ranges: Dict[int, Tuple[int, int]],
    verif_conf: torch.Tensor,   # [num_states, block_size]
    verif_tokens: torch.Tensor, # [num_states, block_size]
    block_positions: torch.Tensor, # [block_size]
    mask_id: int,
) -> Dict:
    """
    Two-stage verification (port of eval_interleaved_draft.verify_interleaved).

    Returns dict with keys:
      accepted_tokens : List[(abs_pos, token_id)]
      bonus           : Optional[(abs_pos, token_id)]
      large_accepted  : int
      small_accepted  : int
      last_state_idx  : Optional[int]
    """
    large_start, large_end = large_node_range
    num_large = len(large_candidates)
    base_pos = int(block_positions[0].item())
    block_size = int(block_positions.shape[0])

    def _rel(abs_pos):
        return int(abs_pos) - base_pos

    # ---- Stage 1: verify large draft ----
    large_accepted = 0
    large_failed_at = -1

    for i in range(num_large):
        state_idx = large_start + i
        node_state = all_states[state_idx]
        mask_rel = (node_state == mask_id).nonzero(as_tuple=True)[0]
        if not mask_rel.numel():
            break

        node_conf = verif_conf[state_idx, mask_rel]
        node_toks = verif_tokens[state_idx, mask_rel]
        best_i = int(node_conf.argmax().item())
        pred_pos = int(block_positions[mask_rel[best_i]].item())
        pred_tok = int(node_toks[best_i].item())

        exp_pos, exp_tok, _ = large_candidates[i]
        if pred_pos == exp_pos and pred_tok == exp_tok:
            large_accepted += 1
        else:
            large_failed_at = state_idx
            break

    # ---- Stage 2: select matching small branch and verify ----
    accepted_tokens = [(p, t) for p, t, _ in large_candidates[:large_accepted]]
    small_accepted = 0
    bonus: Optional[Tuple[int, int]] = None
    last_state_idx: Optional[int] = None

    def _extract_bonus(state_idx: int) -> Optional[Tuple[int, int]]:
        """Pick the highest-confidence prediction from verification node state_idx."""
        node_state = all_states[state_idx]
        mask_rel = (node_state == mask_id).nonzero(as_tuple=True)[0]
        if not mask_rel.numel():
            return None
        nc = verif_conf[state_idx, mask_rel]
        nt = verif_tokens[state_idx, mask_rel]
        bi = int(nc.argmax().item())
        return (int(block_positions[mask_rel[bi]].item()), int(nt[bi].item()))

    if large_accepted in small_branch_ranges:
        branch_start, branch_end = small_branch_ranges[large_accepted]
        selected_branch = next(
            (br for br in small_branches if br.assumed_large_accepted == large_accepted),
            None,
        )

        if selected_branch is not None and selected_branch.draft_candidates:
            num_small = len(selected_branch.draft_candidates)
            all_passed = True
            for i in range(num_small):
                state_idx = branch_start + i
                node_state = all_states[state_idx]
                mask_rel = (node_state == mask_id).nonzero(as_tuple=True)[0]
                if not mask_rel.numel():
                    all_passed = False
                    break

                nc = verif_conf[state_idx, mask_rel]
                nt = verif_tokens[state_idx, mask_rel]
                bi = int(nc.argmax().item())
                pred_pos = int(block_positions[mask_rel[bi]].item())
                pred_tok = int(nt[bi].item())

                exp_pos, exp_tok, _ = selected_branch.draft_candidates[i]
                if pred_pos == exp_pos and pred_tok == exp_tok:
                    small_accepted += 1
                    accepted_tokens.append((exp_pos, exp_tok))
                    last_state_idx = state_idx
                else:
                    bonus = (pred_pos, pred_tok)
                    last_state_idx = state_idx
                    all_passed = False
                    break

            if all_passed:
                # All small candidates accepted → take bonus from leaf node
                leaf_idx = branch_start + num_small
                last_state_idx = leaf_idx
                bonus = _extract_bonus(leaf_idx)
        else:
            # No small candidates: fall back to bonus from large failure/leaf
            if large_failed_at >= 0:
                last_state_idx = large_failed_at
                bonus = _extract_bonus(large_failed_at)
            elif large_accepted == num_large:
                leaf_idx = large_start + num_large
                last_state_idx = leaf_idx
                bonus = _extract_bonus(leaf_idx)
    else:
        # No matching small branch
        if large_failed_at >= 0:
            last_state_idx = large_failed_at
            bonus = _extract_bonus(large_failed_at)
        elif large_accepted == num_large:
            leaf_idx = large_start + num_large
            last_state_idx = leaf_idx
            bonus = _extract_bonus(leaf_idx)

    return {
        "accepted_tokens": accepted_tokens,
        "bonus": bonus,
        "large_accepted": large_accepted,
        "small_accepted": small_accepted,
        "last_state_idx": last_state_idx,
    }
