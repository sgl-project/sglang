"""Reverse-apply historical PR fixes for regression-style tests.

Each registered PR is a YAML config consumed by
sglang.srt.debug_utils.source_patcher.apply_patches_from_config which does
text-level edits on the function source, compiles, and swaps __code__.
PatchState.restore() puts back the original __code__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.debug_utils.source_patcher import (
    PatchState,
    apply_patches_from_config,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


# Reverts PR #25015 ("Fix Eagle draft decode positions"). Re-introduces the
# pre-fix positions.add_(1) location (start of inner loop body) and removes
# the post-fix positions.add_(1) at the loop bottom and the cuda-graph
# capture-side positions.sub_ tail compensation.
_PR_25015_REVERT_YAML = """
patches:
  - target: sglang.srt.speculative.eagle_worker.EAGLEWorker.draft_forward
    edits:
      - match: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
        replacement: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          forward_batch.positions.add_(1)
          forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
      - match: |
          hidden_states = logits_output.hidden_states
          forward_batch.positions.add_(1)
        replacement: |
          hidden_states = logits_output.hidden_states

  - target: sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward
    edits:
      - match: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
        replacement: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          forward_batch.positions.add_(1)
          forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
      - match: |
          hidden_states = logits_output.hidden_states
          forward_batch.positions.add_(1)
        replacement: |
          hidden_states = logits_output.hidden_states

  - target: sglang.srt.speculative.eagle_draft_cuda_graph_runner.EAGLEDraftCudaGraphRunner.capture_one_batch_size
    edits:
      - match: |
          forward_batch.spec_info.hidden_states = hidden_states_backup
          forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)
          return ret
        replacement: |
          forward_batch.spec_info.hidden_states = hidden_states_backup
          return ret
"""


_PR_FIX_REVERT_YAML: Dict[int, str] = {
    25015: _PR_25015_REVERT_YAML,
}


def apply_pr_fix_toggles(
    scheduler: "Scheduler", *, choices: Dict[int, Optional[bool]]
) -> None:
    for pr_num, choice in choices.items():
        if choice is None:
            continue

        attr = f"_pr_{pr_num}_patch_states"

        if choice is True:
            states: Optional[List[PatchState]] = getattr(scheduler, attr, None)
            if states is not None:
                for state in reversed(states):
                    state.restore()
                delattr(scheduler, attr)
            continue

        if pr_num not in _PR_FIX_REVERT_YAML:
            raise NotImplementedError(
                f"PR #{pr_num} revert is not registered; "
                f"available: {sorted(_PR_FIX_REVERT_YAML.keys())}"
            )
        if hasattr(scheduler, attr):
            continue
        states = apply_patches_from_config(_PR_FIX_REVERT_YAML[pr_num])
        setattr(scheduler, attr, states)
