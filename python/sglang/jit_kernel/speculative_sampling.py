from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import flashinfer
import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_speculative_sampling_module() -> Module:
    flashinfer_dir = pathlib.Path(flashinfer.__file__).parent.resolve()
    assert (
        flashinfer_dir / "data" / "include"
    ).exists(), (
        f"flashinfer headers are missing: {str(flashinfer_dir / 'data' / 'include')}"
    )
    flashinfer_include_path = (flashinfer_dir / "data" / "include").resolve()
    return load_jit(
        "speculative_sampling",
        cuda_files=["speculative/speculative_sampling.cuh"],
        cuda_wrappers=[
            (
                "tree_speculative_sampling_target_only",
                "tree_speculative_sampling_target_only",
            )
        ],
        extra_include_paths=[str(flashinfer_include_path)],
    )


@register_custom_op(
    op_name="tree_speculative_sampling_target_only_out",
    mutates_args=["predicts", "accept_index", "accept_token_num", "draft_probs"],
)
def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool = True,
) -> None:
    """
    Tree-based speculative sampling (target-only variant).

    Performs tree speculative decoding acceptance/rejection in-place.

    Args:
        predicts:              [tot_num_draft_tokens] int32 — filled with accepted/sampled tokens
        accept_index:          [bs, num_spec_step] int32 — filled with accepted token indices
        accept_token_num:      [bs] int32 — filled with number of accepted tokens per sequence
        candidates:            [bs, num_draft_tokens] int64 — draft token IDs
        retrive_index:         [bs, num_draft_tokens] int64 — tree node indices
        retrive_next_token:    [bs, num_draft_tokens] int64 — next token in tree path
        retrive_next_sibling:  [bs, num_draft_tokens] int64 — next sibling in tree
        uniform_samples:       [bs, num_draft_tokens] float32 — uniform random samples
        uniform_samples_for_final_sampling: [bs] float32 — uniform sample for bonus token
        target_probs:          [bs, num_draft_tokens, vocab_size] float32
        draft_probs:           [bs, num_draft_tokens, vocab_size] float32 — modified in-place
        threshold_single:      float in [0, 1] — per-token acceptance threshold
        threshold_acc:         float in [0, 1] — cumulative acceptance threshold
        deterministic:         bool — deterministic sampling (default True)
    """
    module = _jit_speculative_sampling_module()
    module.tree_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        1 if deterministic else 0,
    )
