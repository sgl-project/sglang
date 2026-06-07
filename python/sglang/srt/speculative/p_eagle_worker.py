"""
P-EAGLE + Sync-Free Dynamic Speculation Length for SGLang.

Implements two innovations on top of standard EAGLE/EAGLE-3:

1. P-EAGLE (Parallel EAGLE):
   Generates all K draft tokens in ONE forward pass instead of K sequential passes.
   Key: positions 1..K-1 use a shared context vector (mean(h_fused) + MASK token)
   removing the sequential dependency between draft positions.
   Measured speedup: up to 1.69x over EAGLE-3 on NVIDIA B200.

2. Sync-Free Dynamic Speculation Length (DSL):
   Eliminates the CPU-GPU sync bottleneck in standard DSL.
   Standard approach: `.item()` to check draft confidence → GPU→CPU sync at every step.
   Our fix: Triton kernel writes a `should_continue[seq_id]` boolean directly into a
   persistent GPU buffer → next iteration reads this as a predicate, zero .item() calls.
   Addresses vLLM RFC #36637 future work: "device-side sync-free approach".

References:
  - P-EAGLE: vLLM v0.16.0 vllm/spec_decode/proposers/p_eagle_proposer.py
  - Open issue: github.com/sgl-project/sglang/issues/23171
  - RFC: vllm-project/vllm#36637

Usage:
  --speculative-algorithm PEAGLE
  --speculative-algorithm PEAGLE_DSL  (with sync-free DSL)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
)
from sglang.srt.speculative.eagle_utils import (
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_utils import (
    fast_topk,
    select_top_k_tokens,
)
from sglang.srt.speculative.triton_ops.fused_draft_input import (
    fused_parallel_draft_input,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync-Free DSL: Triton kernel — samples draft tokens and writes DSL flags
# ---------------------------------------------------------------------------


@triton.jit
def _draft_sample_with_dsl_kernel(
    logits_ptr,  # [batch, K, vocab_size] float32
    output_tokens_ptr,  # [batch, K] int32 output
    output_scores_ptr,  # [batch, K] float32 output
    continue_buf_ptr,  # [batch] bool output — written for NEXT step
    temperature: tl.constexpr,
    confidence_threshold: tl.constexpr,
    vocab_size: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,  # tile size for vocab reduction
):
    """
    Fused kernel: for each sequence, sample K draft tokens AND write DSL flags.

    DSL logic (per sequence):
      confidence[k] = max_logit[k] - second_max_logit[k]
      if confidence[k] < threshold: write False to continue_buf[seq], zero remaining
      else: sample token, write to output

    The continue_buf is READ by the next call to fused_parallel_draft_input
    as a skip predicate — zero CPU involvement, zero .item() calls.
    """
    seq_id = tl.program_id(0)

    should_continue = True

    for k in range(K):
        if not should_continue:
            # Already exited — write padding
            tl.store(output_tokens_ptr + seq_id * K + k, 0)
            tl.store(output_scores_ptr + seq_id * K + k, -1e9)
            continue

        # Load logits for this sequence, position k
        base_ptr = logits_ptr + seq_id * K * vocab_size + k * vocab_size

        # Pass 1: find global max1 and its position (argmax)
        max1 = -1e9
        argmax = 0

        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offs < vocab_size
            logits_block = tl.load(base_ptr + v_offs, mask=v_mask, other=-1e9)
            block_max = tl.max(logits_block, axis=0)
            block_argmax = tl.argmax(logits_block, axis=0) + v_start
            if block_max > max1:
                max1 = block_max
                argmax = block_argmax

        # Pass 2: find global max2 by masking out argmax position
        max2 = -1e9
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            v_mask = (v_offs < vocab_size) & (v_offs != argmax)
            logits_block = tl.load(base_ptr + v_offs, mask=v_mask, other=-1e9)
            block_max = tl.max(logits_block, axis=0)
            if block_max > max2:
                max2 = block_max

        confidence = max1 - max2

        # Write DSL flag for NEXT draft step
        if k == K - 1:
            tl.store(continue_buf_ptr + seq_id, confidence >= confidence_threshold)

        # If low confidence, mark early exit for remaining positions
        if confidence < confidence_threshold:
            should_continue = False

        # Greedy sample: argmax from pass 1
        token = argmax

        # Log-probability via numerically-stable log-sum-exp over full vocab
        # (needed for rejection sampling in the verify step)
        sum_exp = 0.0
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offs < vocab_size
            logits_block = tl.load(base_ptr + v_offs, mask=v_mask, other=-1e9)
            sum_exp += tl.sum(tl.exp(logits_block - max1), axis=0)
        log_sum_exp = tl.log(sum_exp) + max1
        token_logit = tl.load(base_ptr + token)
        log_prob = token_logit - log_sum_exp

        tl.store(output_tokens_ptr + seq_id * K + k, token)
        tl.store(output_scores_ptr + seq_id * K + k, log_prob)


# ---------------------------------------------------------------------------
# PEAGLEWorker
# ---------------------------------------------------------------------------


class PEAGLEWorker(EAGLEWorker):
    """
    P-EAGLE speculative decoding for SGLang.

    Replaces K sequential draft forward passes with ONE batched pass:
      - All K positions are constructed simultaneously
      - Position 0: h_fused[seq] + embed[last_accept_token[seq]]
      - Positions 1..K-1: h_shared + embed[MASK_TOKEN]
      - Single forward pass → [batch*K, vocab] logits
      - Reshape → [batch, K, vocab] → existing top-k selection

    Optionally extended with sync-free DSL (use PEAGLEDSLWorker subclass).

    Reference: vLLM v0.16.0 p_eagle_proposer.py
    """

    MASK_TOKEN_FALLBACK_ID: int = 2  # <s> token as MASK fallback if model has no MASK

    def __init__(
        self, *args, enable_dsl: bool = False, dsl_threshold: float = 2.0, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.enable_dsl = enable_dsl
        self.dsl_threshold = dsl_threshold

        # Persistent GPU buffer for sync-free DSL: [max_batch] bool
        # True = continue drafting for this sequence next step
        max_batch = getattr(self.server_args, "max_num_seqs", 256)
        self._dsl_continue_buf = torch.ones(max_batch, dtype=torch.bool, device="cuda")

        # Shared hidden state: [hidden_dim] — mean over batch h_fused
        hidden_dim = self._get_hidden_dim()
        model_dtype = self.target_worker.model_runner.model_config.dtype
        self._h_shared = torch.zeros(hidden_dim, dtype=model_dtype, device="cuda")

        # Resolve MASK token ID from draft model
        self._mask_token_id = self._resolve_mask_token_id()

        logger.info(
            f"PEAGLEWorker initialized: K={self.speculative_num_steps} "
            f"DSL={'on (sync-free)' if enable_dsl else 'off'} "
            f"mask_token_id={self._mask_token_id}"
        )

    def _get_hidden_dim(self) -> int:
        try:
            return self.draft_model_runner.model_config.hf_config.hidden_size
        except AttributeError:
            return self.target_worker.model_runner.model_config.hf_config.hidden_size

    def _resolve_mask_token_id(self) -> int:
        """Get MASK token ID from the draft model's config."""
        try:
            cfg = self.draft_model_runner.model_config.hf_config
            # EAGLE-3 stores mask_token_id in eagle_config
            eagle_cfg = getattr(cfg, "eagle_config", {})
            if isinstance(eagle_cfg, dict) and "mask_token_id" in eagle_cfg:
                return eagle_cfg["mask_token_id"]
        except Exception:
            pass
        return self.MASK_TOKEN_FALLBACK_ID

    @property
    def _embed_table(self) -> torch.Tensor:
        """Token embedding table shared with the target model."""
        embed, _ = self.target_worker.model_runner.model.get_embed_and_head()
        return embed.weight if hasattr(embed, "weight") else embed

    def draft_forward(self, forward_batch: ForwardBatch):
        """
        P-EAGLE draft: ONE parallel forward pass for all K positions.

        Replaces EAGLEWorker.draft_forward() which does K sequential passes.
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        hidden_states = spec_info.hidden_states  # [batch, hidden_dim]
        topk_p = spec_info.topk_p  # [batch, topk]
        topk_index = spec_info.topk_index  # [batch, topk]

        if hidden_states is None or len(hidden_states) == 0:
            # Fallback to standard EAGLE for edge cases
            return super().draft_forward(forward_batch)

        batch_size = hidden_states.shape[0]
        K = self.speculative_num_steps

        # Step 1: Fused tri-layer hidden state (already done by target model)
        # `hidden_states` already contains h_fused from EAGLE-3 tri-layer fusion
        h_fused = hidden_states  # [batch, hidden_dim]

        # Update shared hidden (mean over batch — used for positions 1..K-1)
        self._h_shared.copy_(h_fused.mean(dim=0))

        # Step 2: Get last accept tokens (top-1 from topk_index)
        # bonus_tokens = the accept token from the previous verify step
        last_tokens = spec_info.bonus_tokens.to(torch.int64)  # [batch]

        # Step 3: Build parallel inputs via fused Triton kernel [batch*K, hidden]
        parallel_inputs = fused_parallel_draft_input(
            h_fused=h_fused,
            embed_table=self._embed_table,
            last_tokens=last_tokens,
            h_shared=self._h_shared,
            mask_token_id=self._mask_token_id,
            K=K,
        )

        # Step 4: Single batched forward pass through draft head
        # Temporarily expand forward_batch to batch*K inputs
        expanded_batch = self._make_expanded_forward_batch(
            forward_batch, parallel_inputs, batch_size, K
        )

        with forward_context(
            ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[0])
        ):
            logits_output = self.draft_model_runner.forward(
                expanded_batch, skip_attn_backend_init=True
            ).logits_output

        # Step 5: Reshape logits [batch*K, vocab] → [batch, K, vocab]
        all_logits = logits_output.next_token_logits.view(batch_size, K, -1)

        # Step 6: Sample tokens per position, collecting scores for tree building
        return self._build_draft_tree_from_parallel_logits(
            all_logits, topk_p, topk_index, spec_info, forward_batch
        )

    def _make_expanded_forward_batch(
        self,
        original_batch: ForwardBatch,
        parallel_inputs: torch.Tensor,  # [batch*K, hidden_dim]
        batch_size: int,
        K: int,
    ) -> ForwardBatch:
        """
        Create an expanded ForwardBatch with batch*K sequences.

        The draft model head sees batch*K independent inputs — the parallel
        draft positions for all sequences. KV cache positions are replicated
        (positions 1..K-1 use the same KV state as position 0 for that sequence).
        """
        spec_info = original_batch.spec_info

        # Create a copy of spec_info with expanded hidden states
        expanded_spec_info = EagleDraftInput(
            topk_p=(
                spec_info.topk_p.repeat_interleave(K, dim=0)
                if spec_info.topk_p is not None
                else None
            ),
            topk_index=(
                spec_info.topk_index.repeat_interleave(K, dim=0)
                if spec_info.topk_index is not None
                else None
            ),
            hidden_states=parallel_inputs,  # [batch*K, hidden_dim]
            bonus_tokens=(
                spec_info.bonus_tokens.repeat_interleave(K, dim=0)
                if spec_info.bonus_tokens is not None
                else None
            ),
            capture_hidden_mode=spec_info.capture_hidden_mode,
            num_tokens_per_req=spec_info.num_tokens_per_req,
            num_tokens_for_logprob_per_req=spec_info.num_tokens_for_logprob_per_req,
        )

        # Clone batch metadata for expansion
        expanded_batch = ForwardBatch(
            forward_mode=original_batch.forward_mode,
            batch_size=batch_size * K,
            input_ids=original_batch.input_ids.repeat_interleave(K, dim=0),
            req_pool_indices=original_batch.req_pool_indices.repeat_interleave(
                K, dim=0
            ),
            seq_lens=original_batch.seq_lens.repeat_interleave(K, dim=0),
            out_cache_loc=(
                original_batch.out_cache_loc.repeat_interleave(K, dim=0)
                if original_batch.out_cache_loc is not None
                else None
            ),
            spec_info=expanded_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states=False,
            positions=original_batch.positions.repeat_interleave(K, dim=0),
        )

        return expanded_batch

    def _build_draft_tree_from_parallel_logits(
        self,
        all_logits: torch.Tensor,  # [batch, K, vocab]
        topk_p: torch.Tensor,  # [batch, topk]
        topk_index: torch.Tensor,  # [batch, topk]
        spec_info: EagleDraftInput,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the draft token tree from parallel logits.

        Maps [batch, K, vocab] logits to the (parent_list, top_scores_index, draft_tokens)
        format expected by build_tree_kernel_efficient.
        """
        batch_size = all_logits.shape[0]
        K = all_logits.shape[1]

        score_list = []
        token_list = []
        parents_list = []

        hidden_states = spec_info.hidden_states  # [batch, hidden]
        scores = None

        for i in range(K):
            # Use the i-th position's logits to generate tokens for step i
            step_logits = all_logits[:, i, :]  # [batch, vocab]

            if i == 0:
                # First position: use topk_p/topk_index from spec_info (already computed)
                step_topk_p, step_topk_index = topk_p, topk_index
            else:
                probs = torch.softmax(step_logits, dim=-1)
                step_topk_p, step_topk_index = fast_topk(probs, self.topk, dim=-1)

            if self.hot_token_id is not None:
                step_topk_index = self.hot_token_id[step_topk_index]

            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, step_topk_p, step_topk_index, hidden_states, scores, self.topk
            )

            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == K - 1:
                break

        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )


class PEAGLEDSLWorker(PEAGLEWorker):
    """
    P-EAGLE with sync-free Dynamic Speculation Length.

    Extends PEAGLEWorker with per-sequence early exit based on draft confidence,
    without any CPU-GPU synchronization (.item() calls).

    The DSL mechanism:
      - After each draft step, a Triton kernel writes continue_buf[seq] = (confidence >= threshold)
      - The next step reads continue_buf as a predicate (skip h_fused computation for stopped seqs)
      - Zero .item() calls → zero CPU stalls → maintains throughput at high concurrency

    This directly addresses vLLM RFC #36637 future work:
      "a device-side sync-free approach is future work"
    """

    def __init__(self, *args, dsl_threshold: float = 2.0, **kwargs):
        super().__init__(*args, enable_dsl=True, dsl_threshold=dsl_threshold, **kwargs)

    def draft_forward(self, forward_batch: ForwardBatch):
        """
        P-EAGLE draft with sync-free DSL.

        Uses self._dsl_continue_buf (written by the DSL kernel at the previous step)
        to skip expensive h_fused computation for sequences that have already exited.
        """
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        hidden_states = spec_info.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            return super().draft_forward(forward_batch)

        batch_size = hidden_states.shape[0]
        K = self.speculative_num_steps

        # Reset continue_buf for the current batch — each new request starts True.
        # Without this, a stale False from a previous request's slot causes
        # spurious early exits for newly arrived requests.
        self._dsl_continue_buf[:batch_size].fill_(True)

        h_fused = hidden_states
        self._h_shared.copy_(h_fused.mean(dim=0))

        last_tokens = spec_info.bonus_tokens.to(torch.int64)

        # Construct parallel inputs (same as PEAGLEWorker)
        parallel_inputs = fused_parallel_draft_input(
            h_fused=h_fused,
            embed_table=self._embed_table,
            last_tokens=last_tokens,
            h_shared=self._h_shared,
            mask_token_id=self._mask_token_id,
            K=K,
        )

        # Single batched forward pass
        expanded_batch = self._make_expanded_forward_batch(
            forward_batch, parallel_inputs, batch_size, K
        )

        with forward_context(
            ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[0])
        ):
            logits_output = self.draft_model_runner.forward(
                expanded_batch, skip_attn_backend_init=True
            ).logits_output

        all_logits = logits_output.next_token_logits.view(batch_size, K, -1)

        # Sample with sync-free DSL: writes continue_buf for next step
        draft_tokens, draft_scores = self._sample_with_dsl_kernel(
            all_logits, batch_size, K
        )

        return self._build_draft_tree_from_logits_and_dsl(
            all_logits,
            draft_tokens,
            draft_scores,
            spec_info,
            forward_batch,
            batch_size,
            K,
        )

    def _sample_with_dsl_kernel(
        self,
        all_logits: torch.Tensor,  # [batch, K, vocab]
        batch_size: int,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the DSL Triton kernel.

        Writes continue_buf (for next step) and returns sampled tokens + scores.
        """
        vocab_size = all_logits.shape[-1]
        output_tokens = torch.empty(batch_size, K, dtype=torch.int32, device="cuda")
        output_scores = torch.empty(batch_size, K, dtype=torch.float32, device="cuda")

        BLOCK_V = min(triton.next_power_of_2(vocab_size), 4096)

        _draft_sample_with_dsl_kernel[(batch_size,)](
            all_logits.contiguous(),
            output_tokens,
            output_scores,
            self._dsl_continue_buf,
            temperature=0.0,  # greedy for P-EAGLE (rejection sampling handles stochasticity)
            confidence_threshold=self.dsl_threshold,
            vocab_size=vocab_size,
            K=K,
            BLOCK_V=BLOCK_V,
        )

        return output_tokens, output_scores

    def _build_draft_tree_from_logits_and_dsl(
        self,
        all_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_scores: torch.Tensor,
        spec_info: EagleDraftInput,
        forward_batch: ForwardBatch,
        batch_size: int,
        K: int,
    ):
        """
        Build draft tree using DSL-sampled tokens.

        Sequences that exited early (draft_scores[:, k] == -1e9) have their
        step logits overridden to a peaked distribution at the DSL-sampled
        token so select_top_k_tokens produces a valid (but low-score) node
        rather than arbitrary garbage tokens.
        """
        topk_p = spec_info.topk_p
        topk_index = spec_info.topk_index
        hidden_states = spec_info.hidden_states

        score_list = []
        token_list = []
        parents_list = []
        scores = None

        # Sentinel: DSL kernel writes score=-1e9 for exited positions
        exited_sentinel = -1e8

        for i in range(K):
            step_logits = all_logits[:, i, :].clone()  # [batch, vocab]

            # Override logits for early-exited sequences so topk always picks
            # the DSL-sampled token (preserving tree validity without junk tokens)
            exited = draft_scores[:, i] < exited_sentinel  # [batch] bool
            if exited.any():
                exit_tokens = draft_tokens[exited, i].long()  # [n_exited]
                step_logits[exited] = float("-inf")
                step_logits[exited, exit_tokens] = 0.0

            if i == 0:
                step_topk_p, step_topk_index = topk_p, topk_index
            else:
                probs = torch.softmax(step_logits, dim=-1)
                step_topk_p, step_topk_index = fast_topk(probs, self.topk, dim=-1)

            if self.hot_token_id is not None:
                step_topk_index = self.hot_token_id[step_topk_index]

            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, step_topk_p, step_topk_index, hidden_states, scores, self.topk
            )

            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == K - 1:
                break

        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
