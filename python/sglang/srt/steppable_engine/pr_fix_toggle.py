"""Reverse-apply historical PR fixes for scripted regression tests.

For each PR fix listed in _PR_FIX_TOGGLES, provides a revert(scheduler) /
reapply(scheduler) pair via class-level monkey-patches. SteppableEngine
calls revert(...) when SteppableEngineConfig.apply_pr_NNNNN_fix is False
(i.e., the test wants to reproduce the original bug), and reapply(...) when
True (restore the patched method, idempotent on a fresh launch).

Currently supports PR #25015 only.

Why monkey-patch vs source diff: tests need apply/revert to be reversible
within a single launch call; source diff would require recompile or full
restart per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


@dataclass(frozen=True, slots=True, kw_only=True)
class _PrFixToggle:
    revert: Callable[["Scheduler"], None]
    reapply: Callable[["Scheduler"], None]


# ---- PR #25015: EAGLE draft decode positions ----
#
# Upstream fix moved `forward_batch.positions.add_(1)` from BEFORE each draft
# decode forward to AFTER, so the current forward uses the intended RoPE
# position. Reverting puts the increment back BEFORE the forward call, so
# each step sees positions one step ahead of what it should be. This is the
# behavior that surfaces as a position canary mismatch.


def _revert_pr_25015(scheduler: "Scheduler") -> None:
    """Re-introduce the off-by-one in EAGLE draft decode positions."""
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    if getattr(scheduler, "_pr_25015_reverted", False):
        return

    scheduler._pr_25015_orig_eagle_worker_draft_forward = EAGLEWorker.draft_forward
    scheduler._pr_25015_orig_eagle_draft_worker_v2_draft_forward = (
        EagleDraftWorker.draft_forward
    )

    EAGLEWorker.draft_forward = _broken_eagle_worker_draft_forward
    EagleDraftWorker.draft_forward = _broken_eagle_draft_worker_v2_draft_forward

    scheduler._pr_25015_reverted = True


def _reapply_pr_25015(scheduler: "Scheduler") -> None:
    """Restore the fixed implementations."""
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    if not getattr(scheduler, "_pr_25015_reverted", False):
        return

    EAGLEWorker.draft_forward = scheduler._pr_25015_orig_eagle_worker_draft_forward
    EagleDraftWorker.draft_forward = (
        scheduler._pr_25015_orig_eagle_draft_worker_v2_draft_forward
    )

    del scheduler._pr_25015_orig_eagle_worker_draft_forward
    del scheduler._pr_25015_orig_eagle_draft_worker_v2_draft_forward
    scheduler._pr_25015_reverted = False


def _broken_eagle_worker_draft_forward(self, forward_batch):
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.eagle_utils import organize_draft_results
    from sglang.srt.speculative.spec_utils import (
        fast_topk,
        maybe_detect_nan,
        maybe_detect_oob,
        select_top_k_tokens,
    )

    spec_info = forward_batch.spec_info
    assert isinstance(spec_info, EagleDraftInput)
    out_cache_loc = forward_batch.out_cache_loc
    topk_p, topk_index, hidden_states = (
        spec_info.topk_p,
        spec_info.topk_index,
        spec_info.hidden_states,
    )

    maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

    if self.hot_token_id is not None:
        topk_index = self.hot_token_id[topk_index]
    out_cache_loc = out_cache_loc.reshape(
        forward_batch.batch_size, self.topk, self.speculative_num_steps
    )
    out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
        self.speculative_num_steps, -1
    )

    score_list: List[torch.Tensor] = []
    token_list: List[torch.Tensor] = []
    parents_list: List[torch.Tensor] = []

    scores = None
    for i in range(self.speculative_num_steps):
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden_states, scores, self.topk
        )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

        if i == self.speculative_num_steps - 1:
            break

        forward_batch.input_ids = input_ids
        if (
            self.server_args.speculative_algorithm == "STANDALONE"
            and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
        ):
            out_cache_loc = out_cache_loc.contiguous()
        forward_batch.out_cache_loc = out_cache_loc[i]
        forward_batch.positions.add_(1)
        forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
        spec_info.hidden_states = hidden_states

        logits_output = self.draft_model_runner.forward(
            forward_batch, skip_attn_backend_init=True
        ).logits_output
        maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        maybe_detect_oob(
            topk_index,
            0,
            logits_output.next_token_logits.shape[-1],
            f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        hidden_states = logits_output.hidden_states

    parent_list, top_scores_index, draft_tokens = organize_draft_results(
        score_list, token_list, parents_list, self.speculative_num_draft_tokens
    )

    return parent_list, top_scores_index, draft_tokens


def _broken_eagle_draft_worker_v2_draft_forward(self, forward_batch):
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_utils import (
        maybe_detect_nan,
        maybe_detect_oob,
        select_top_k_tokens,
    )
    from sglang.srt.utils.common import fast_topk

    spec_info: EagleDraftInput = forward_batch.spec_info
    out_cache_loc = forward_batch.out_cache_loc
    topk_p, topk_index, hidden_states = (
        spec_info.topk_p,
        spec_info.topk_index,
        spec_info.hidden_states,
    )

    maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

    if self.hot_token_id is not None:
        topk_index = self.hot_token_id[topk_index]

    out_cache_loc = out_cache_loc.reshape(
        forward_batch.batch_size, self.topk, self.speculative_num_steps
    )
    out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
        self.speculative_num_steps, -1
    )

    score_list: List[torch.Tensor] = []
    token_list: List[torch.Tensor] = []
    parents_list: List[torch.Tensor] = []

    scores = None
    for i in range(self.speculative_num_steps):
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
            i, topk_p, topk_index, hidden_states, scores, self.topk
        )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

        if i == self.speculative_num_steps - 1:
            break

        forward_batch.input_ids = input_ids
        forward_batch.out_cache_loc = out_cache_loc[i]
        forward_batch.positions.add_(1)
        forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
        spec_info.hidden_states = hidden_states

        logits_output = self.draft_runner.forward(
            forward_batch, skip_attn_backend_init=True
        ).logits_output
        maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        maybe_detect_oob(
            topk_index,
            0,
            logits_output.next_token_logits.shape[-1],
            f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        hidden_states = logits_output.hidden_states

    score_list = torch.cat(score_list, dim=1).flatten(
        1
    )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
    ss_token_list = torch.cat(
        token_list, dim=1
    )  # b, (self.topk + (num_steps-1) * self.topk)
    top_scores = torch.topk(score_list, self.speculative_num_draft_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "draft_forward: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


# ---- registry ----

_PR_FIX_TOGGLES: Dict[int, _PrFixToggle] = {
    25015: _PrFixToggle(revert=_revert_pr_25015, reapply=_reapply_pr_25015),
}


def apply_pr_fix_toggles(
    scheduler: "Scheduler", *, choices: Dict[int, Optional[bool]]
) -> None:
    """Apply requested PR fix toggles. None means 'don't touch'."""
    for pr_num, choice in choices.items():
        if choice is None:
            continue
        toggle = _PR_FIX_TOGGLES.get(pr_num)
        if toggle is None:
            raise ValueError(
                f"PR #{pr_num} fix toggle not registered; available: "
                f"{sorted(_PR_FIX_TOGGLES.keys())}"
            )
        if choice is False:
            toggle.revert(scheduler)
        else:
            toggle.reapply(scheduler)
