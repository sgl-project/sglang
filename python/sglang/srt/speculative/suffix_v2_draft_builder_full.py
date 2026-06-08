"""SUFFIX V2 standalone draft builder — M3b.

Wraps `arctic_inference.SuffixDecodingCache` with a V2-friendly tensor
interface. Handles per-request state lifecycle (start/update/stop) and
returns padded GPU tensors ready to feed into the M2 EagleVerifyInput
builder.

Key design decisions:
- Uses arctic's default `use_tree_spec=False` → linear chain (matches our
  V2 design's "topk=1 linear" assumption).
- Keys state by sglang `rid` (Hashable string). Never by req_pool_index
  (per design doc §3.4: pool slots get reused, would corrupt history).
- Padding: short matches (< K-1) are padded with token_id=0 + valid_mask
  False. Downstream verify will reject the padded slots — accept length
  naturally cuts at the real match length.
- Pattern construction: last `max_tree_depth` tokens (matching V1's
  policy for parity).

This builder is stateful (suffix tree per rid). Caller must invoke
`start_request` for each new rid, `stop_request` on finish, and
`update_with_accepted` after each verify step to push accepted tokens
into the suffix tree (so future speculations include them).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SuffixV2DraftBuilder:
    """Standalone V2 draft builder over `arctic_inference.SuffixDecodingCache`.

    Returns padded GPU tensors:
      draft_tokens: (bs, K_minus_1) int32 — speculated tokens, padded with 0
      draft_lens:   (bs,) int32           — actual match length per req
      valid_mask:   (bs, K_minus_1) bool  — True for [:draft_lens[i]]
    """

    def __init__(
        self,
        max_cached_requests: int = 10000,
        max_tree_depth: int = 64,
        max_spec_factor: float = 2.0,
        min_token_prob: float = 0.1,
    ):
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        warmup_path = os.environ.get("ARCTIC_SUFFIX_GLOBAL_WARMUP_PATH")
        warmup_count = getattr(self.cache, "_warmup_count", 0)
        if warmup_path:
            logger.info(
                "SuffixV2DraftBuilder: arctic global tree warmup loaded "
                "%d records from %s (max_tree_depth=%d max_cached=%d)",
                warmup_count,
                warmup_path,
                max_tree_depth,
                max_cached_requests,
            )
        else:
            logger.info(
                "SuffixV2DraftBuilder: no warmup "
                "(ARCTIC_SUFFIX_GLOBAL_WARMUP_PATH unset); "
                "max_tree_depth=%d max_cached=%d",
                max_tree_depth,
                max_cached_requests,
            )
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_tree_depth = max_tree_depth

        # rid → length of tokens already pushed into the suffix cache.
        # Lets `update_with_accepted` push only the delta since last call.
        self._pushed_len: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Per-request lifecycle
    # ------------------------------------------------------------------
    def start_request(self, rid: str, prompt_tokens: Sequence[int]) -> None:
        """Register a new request with its prompt. Must be called before any
        query/update for this rid."""
        self.cache.start_request(rid, list(prompt_tokens))
        self._pushed_len[rid] = len(prompt_tokens)

    def update_with_accepted(self, rid: str, all_tokens: Sequence[int]) -> None:
        """Push the delta of accepted tokens since last call into the suffix
        tree. Idempotent if all_tokens hasn't grown.

        Caller passes the FULL accumulated tokens (prompt + all output so
        far); we slice out the new tail using our tracked `_pushed_len`.
        """
        pushed = self._pushed_len.get(rid, 0)
        cur = len(all_tokens)
        if cur > pushed:
            new = list(all_tokens[pushed:cur])
            if rid in self.cache.active_requests:
                self.cache.add_active_response(rid, new)
            self._pushed_len[rid] = cur

    def stop_request(self, rid: str) -> None:
        """Finish a request — release local tree state. Global tree retains
        the response (with FIFO eviction governed by max_cached_requests)."""
        if rid in self.cache.active_requests:
            self.cache.stop_request(rid)
        self._pushed_len.pop(rid, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_batch(
        self,
        rids: List[str],
        current_tokens: List[Sequence[int]],
        K_minus_1: int,
        device: torch.device,
        return_scores: bool = False,
        max_remaining_per_req: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query the suffix cache for each req's next draft.

        Args:
            rids: per-req identifiers (must have been registered via
                  `start_request`).
            current_tokens: full prompt+output for each req (used to form
                  the suffix pattern for matching).
            K_minus_1: number of draft tokens to request per req (chain
                  width K minus the bonus slot).
            device: target device for returned tensors.
            return_scores: when True, also returns per-req scores (Python
                  list, for the HYBRID selector).
            max_remaining_per_req: optional per-req ceiling on draft length
                  (e.g. ctx_len - cur_seq_len - 1). Prevents the verify
                  step from overshooting context_len and producing an
                  overlap-schedule "zombie" verify on the next iteration
                  (see patch_nsa_zombie_overflow.py for the symptom).
                  When None, no per-req cap; each req gets up to K_minus_1.

        Returns:
            draft_tokens: (bs, K_minus_1) int32, padded with 0
            draft_lens:   (bs,) int32, actual length per req
            valid_mask:   (bs, K_minus_1) bool
            scores:       List[float] (bs,) — only if return_scores=True
        """
        bs = len(rids)
        assert len(current_tokens) == bs
        if max_remaining_per_req is not None:
            assert len(max_remaining_per_req) == bs, (
                f"max_remaining_per_req has {len(max_remaining_per_req)} entries, "
                f"expected bs={bs}"
            )

        draft_tokens_np = np.zeros((bs, K_minus_1), dtype=np.int32)
        draft_lens_np = np.zeros(bs, dtype=np.int32)
        valid_mask_np = np.zeros((bs, K_minus_1), dtype=bool)
        scores: List[float] = [0.0] * bs if return_scores else []

        for i, (rid, tokens) in enumerate(zip(rids, current_tokens)):
            # Per-req draft ceiling: K_minus_1 floor by remaining ctx room
            # (if provided). 0 means this req is at the boundary — skip
            # drafting entirely; verify still runs with bonus-only.
            req_K = K_minus_1
            if max_remaining_per_req is not None:
                req_K = min(req_K, max(0, int(max_remaining_per_req[i])))
            if req_K == 0:
                if return_scores:
                    scores[i] = 0.0
                continue

            # Take last max_tree_depth tokens as pattern. Arctic also clips
            # internally but doing it here keeps pattern shorter for arctic.
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = list(tokens[pattern_start:])

            draft = self.cache.speculate(
                rid,
                pattern,
                max_spec_tokens=req_K,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
                # use_tree_spec=False (default) → linear chain
            )

            actual_len = min(len(draft.token_ids), req_K)
            if actual_len > 0:
                draft_tokens_np[i, :actual_len] = list(draft.token_ids[:actual_len])
                draft_lens_np[i] = actual_len
                valid_mask_np[i, :actual_len] = True
            if return_scores:
                scores[i] = float(getattr(draft, "score", 0.0))

        result = (
            torch.from_numpy(draft_tokens_np).to(device),
            torch.from_numpy(draft_lens_np).to(device),
            torch.from_numpy(valid_mask_np).to(device),
        )
        if return_scores:
            return result + (scores,)
        return result
