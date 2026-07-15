"""Reverse-apply historical PR fixes for regression-style tests."""

from __future__ import annotations

from typing import Dict

from sglang.srt.debug_utils.source_patcher import apply_patches_from_config
from sglang.srt.environ import envs

_PR_REVERT_YAML_25015 = """
patches:
  - target: sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward
    edits:
      - match: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          spec_info.hidden_states = hidden_states
        replacement: |
          forward_batch.out_cache_loc = out_cache_loc[i]
          forward_batch.positions.add_(1)
          spec_info.hidden_states = hidden_states
      - match: |
          hidden_states = logits_output.hidden_states
          forward_batch.positions.add_(1)
        replacement: |
          hidden_states = logits_output.hidden_states

  - target: sglang.srt.speculative.eagle_draft_cuda_graph_runner.EAGLEDraftCudaGraphRunner.capture_one_shape
    edits:
      - match: |
          forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)
        replacement: ""
"""


_PR_REVERT_YAML_26329 = """
patches:
  - target: sglang.srt.speculative.eagle_utils._eagle_prefill_tail_tokens
    edits:
      - match: |
          tail_tokens = next_token_ids.to(batch.input_ids.dtype)
        prepend: |
          return next_token_ids.to(batch.input_ids.dtype)
"""


_PR_REVERT_YAML_27338 = """
patches:
  - target: sglang.srt.layers.attention.flashinfer_backend.FlashInferMultiStepDraftBackend.init_cuda_graph_state
    edits:
      - match: |
          (self.speculative_num_steps, max_bs * self.topk * self.max_context_len),
        replacement: |
          (self.speculative_num_steps, max_bs * self.max_context_len),
"""


_PR_REVERT_YAML_27360 = """
patches:
  - target: sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend._apply_cuda_graph_metadata
    edits:
      - match: |
          cache_loc = cache_loc[:, :decode_length]
        replacement: ""
"""


_PR_REVERT_YAML_26972 = """
patches:
  - target: sglang.srt.mem_cache.allocation_sizing.get_req_to_token_extra_context_len
    edits:
      - match: |
          if server_args.speculative_algorithm is not None:
              extra = max(extra, get_alloc_reserve_per_decode(server_args))
        replacement: ""
"""


_PR_REVERT_YAML_27460 = """
patches:
  - target: sglang.srt.layers.attention.flashinfer_mla_backend.FlashInferMLAMultiStepDraftBackend.init_cuda_graph_state
    edits:
      - match: |
          (self.speculative_num_steps, max_bs * self.topk * self.max_context_len),
        replacement: |
          (self.speculative_num_steps, max_bs * self.max_context_len),
"""


_PR_FIX_REVERT_YAML: Dict[int, str] = {
    25015: _PR_REVERT_YAML_25015,
    26329: _PR_REVERT_YAML_26329,
    27338: _PR_REVERT_YAML_27338,
    27360: _PR_REVERT_YAML_27360,
    26972: _PR_REVERT_YAML_26972,
    27460: _PR_REVERT_YAML_27460,
}


def maybe_revert_pr_fix() -> None:
    if pr_num := envs.SGLANG_DEBUG_REVERT_PR.get():
        _revert_pr_fix(pr_num)


def _revert_pr_fix(pr_num: int) -> None:
    if pr_num not in _PR_FIX_REVERT_YAML:
        raise NotImplementedError(
            f"PR #{pr_num} revert is not registered; "
            f"available: {sorted(_PR_FIX_REVERT_YAML.keys())}"
        )
    apply_patches_from_config(_PR_FIX_REVERT_YAML[pr_num])
