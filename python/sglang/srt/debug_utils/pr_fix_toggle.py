"""Reverse-apply historical PR fixes for regression-style tests."""

from __future__ import annotations

from typing import Dict

from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

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


def revert_pr_fix(pr_num: int) -> None:
    if pr_num not in _PR_FIX_REVERT_YAML:
        raise NotImplementedError(
            f"PR #{pr_num} revert is not registered; "
            f"available: {sorted(_PR_FIX_REVERT_YAML.keys())}"
        )
    apply_patches_from_config(_PR_FIX_REVERT_YAML[pr_num])
