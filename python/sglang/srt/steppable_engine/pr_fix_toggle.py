"""Reverse-apply historical PR fixes for regression-style tests.

Uses sglang.srt.debug_utils.source_patcher to do precise text-level edits on
the affected function source, compile, and swap __code__. Reverts on
scheduler shutdown / on explicit reapply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from sglang.srt.debug_utils.source_patcher import CodePatcher, EditSpec, PatchSpec

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def _pr_25015_revert_patches() -> List[PatchSpec]:
    eager_edit_move_add_to_loop_top = EditSpec(
        match=(
            "forward_batch.out_cache_loc = out_cache_loc[i]\n"
            "            forward_batch.attn_backend"
        ),
        replacement=(
            "forward_batch.out_cache_loc = out_cache_loc[i]\n"
            "            forward_batch.positions.add_(1)\n"
            "            forward_batch.attn_backend"
        ),
    )
    eager_edit_drop_post_loop_add = EditSpec(
        match=(
            "hidden_states = logits_output.hidden_states\n"
            "            forward_batch.positions.add_(1)\n"
        ),
        replacement="hidden_states = logits_output.hidden_states\n",
    )

    return [
        PatchSpec(
            target="sglang.srt.speculative.eagle_worker.EAGLEWorker.draft_forward",
            edits=[eager_edit_move_add_to_loop_top, eager_edit_drop_post_loop_add],
        ),
        PatchSpec(
            target=(
                "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward"
            ),
            edits=[eager_edit_move_add_to_loop_top, eager_edit_drop_post_loop_add],
        ),
        PatchSpec(
            target=(
                "sglang.srt.speculative.eagle_draft_cuda_graph_runner."
                "EAGLEDraftCudaGraphRunner.capture_one_batch_size"
            ),
            edits=[
                EditSpec(
                    match=(
                        "forward_batch.spec_info.hidden_states = hidden_states_backup\n"
                        "            forward_batch.positions.sub_(self.eagle_worker.speculative_num_steps - 1)\n"
                        "            return ret"
                    ),
                    replacement=(
                        "forward_batch.spec_info.hidden_states = hidden_states_backup\n"
                        "            return ret"
                    ),
                ),
            ],
        ),
    ]


_PR_FIX_REVERT_PATCHES: Dict[int, Callable[[], List[PatchSpec]]] = {
    25015: _pr_25015_revert_patches,
}


def apply_pr_fix_toggles(
    scheduler: "Scheduler", *, choices: Dict[int, Optional[bool]]
) -> None:
    for pr_num, choice in choices.items():
        if choice is None:
            continue

        attr = f"_pr_{pr_num}_patcher"

        if choice is True:
            patcher: Optional[CodePatcher] = getattr(scheduler, attr, None)
            if patcher is not None:
                patcher.__exit__(None, None, None)
                delattr(scheduler, attr)
            continue

        if pr_num not in _PR_FIX_REVERT_PATCHES:
            raise NotImplementedError(
                f"PR #{pr_num} revert is not registered; "
                f"available: {sorted(_PR_FIX_REVERT_PATCHES.keys())}"
            )
        if hasattr(scheduler, attr):
            continue
        patcher = CodePatcher(patches=_PR_FIX_REVERT_PATCHES[pr_num]())
        patcher.__enter__()
        setattr(scheduler, attr, patcher)
