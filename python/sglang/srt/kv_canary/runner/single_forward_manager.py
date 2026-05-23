"""SingleForwardManager (SFM) and its per-step snapshot dataclass.

One SFM owns the per-step state of one inner ``model.forward`` invocation
inside an outer canary cycle. The outer ``CanaryManager`` holds a static
list of SFMs (length ``max(1, speculative_num_steps - 1)``) and dispatches
the monkey-patched ``model.forward`` wrap to the active SFM through a
context manager.

Lifecycle (per SFM, enforced by ``SimplePhaseChecker``):

    IDLE
      ── pre_ops_outside_graph(maybe_non_mature_forward_batch)
      → AFTER_PRE_OUT
      ── pre_ops_maybe_inside_graph(forward_batch)
      → AFTER_PRE_MAYBE_IN
      ── (original model.forward runs)
      ── post_ops_maybe_inside_graph(forward_batch)
      → AFTER_POST_MAYBE_IN
      ── post_ops_outside_graph(snapshot=self.snapshot)
      → IDLE

Phase 1 and 4 are host-side outside any cuda graph; phase 2 and 3 are
"maybe inside graph" — captured on the DECODE path, eager on EXTEND
fallback. The source code in phase 2/3 is the same regardless: every op
must be capture-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class PostOpsInsideGraphOutputSnapshot:
    """Per-SFM cloned view of the tensors produced by phases 2-3.

    The snapshot is written by phase 3 (``post_ops_maybe_inside_graph``)
    and read by phase 4 (``post_ops_outside_graph``). Phase 4 must NOT
    read ``ForwardBatch`` directly — by then the outer cycle may have
    advanced the batch to the next inner step, mutating shared fields
    (seq_lens, out_cache_loc, positions).

    All fields are device tensors holding immutable cloned snapshots of
    the per-step output. We prefer over-cloning to guarantee phase 4 sees
    a dead snapshot of "what this SFM produced", not a live view that
    later steps might overwrite.
    """

    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor
    swa_verify_total_count: torch.Tensor | None
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
