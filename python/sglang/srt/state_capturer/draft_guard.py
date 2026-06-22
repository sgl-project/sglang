"""Draft-worker opt-out + fail-closed guard for R3 routed-experts capture.

Role invariant: R3 capture is target-only, so a draft model's MoE `TopK`
must never write the target's routed-experts capture buffer. The only
thing the runtime chokepoint (`capture_routed_experts_if_allowed`)
consults is `topk_config.allow_routed_experts_capture`, so enforcement is
purely a matter of that flag on each draft-side `TopK`.

`ModelRunner.initialize()` runs both passes here on every draft worker
(unconditionally, before backend/graph init):

  - `disable_routed_experts_capture_for_draft(model)` walks the freshly
    loaded draft model and sets the flag False on every `TopK`. This is
    where the opt-out happens -- not at block construction.
  - `check_draft_capture_optout(model, ...)` re-walks and asserts the
    invariant, turning any un-opted-out draft `TopK` (flag left at the
    default `True`) into a loud startup failure instead of silent buffer
    pollution. It runs only when capture is actually enabled
    (`--enable-return-routed-experts`); otherwise it is a no-op.

Dense drafts have zero `TopK` and both passes are no-ops. `HashTopK` is
intentionally NOT covered: it has no `topk_config` and never calls the R3
capturer (it only writes the EPLB expert-distribution recorder).
"""

from __future__ import annotations

from typing import Any, List


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft-side MoE `TopK` out of R3 routed-experts capture.

    Role invariant: a draft model's MoE `TopK` must never write the
    target's R3 capture buffer. Walks `model` and sets
    `topk_config.allow_routed_experts_capture = False` on every `TopK`.
    Runs before the guard and before any graph recording. Dense-safe
    (zero `TopK` -> no-op) and idempotent. `HashTopK` is intentionally
    NOT covered: no `topk_config`, never calls the R3 capturer.
    """
    # Imported lazily to avoid an import-time cycle between
    # `state_capturer.*` and `layers.moe.topk` during module initialization.
    from sglang.srt.layers.moe.topk import TopK

    for m in model.modules():
        if isinstance(m, TopK):
            m.topk_config.allow_routed_experts_capture = False


def check_draft_capture_optout(
    model: Any,
    *,
    routed_experts_capture_enabled: bool,
) -> None:
    """Fail-closed guard. See module docstring for the contract.

    `model` is the freshly constructed draft `nn.Module`. The
    `routed_experts_capture_enabled` argument is True iff the process
    intends to capture routed experts (e.g. the global capturer has been
    installed); when False, the guard is a no-op.
    """
    if not routed_experts_capture_enabled:
        return

    # Imported lazily to avoid an import-time cycle between
    # `state_capturer.*` and `layers.moe.topk` during module initialization.
    from sglang.srt.layers.moe.topk import TopK

    offenders: List[Any] = [
        m
        for m in model.modules()
        if isinstance(m, TopK)
        and getattr(m.topk_config, "allow_routed_experts_capture", True)
    ]
    if offenders:
        raise RuntimeError(
            f"draft worker model {type(model).__name__!r} has {len(offenders)} "
            "MoE TopK module(s) with allow_routed_experts_capture=True; every "
            "draft-side MoE block must set allow_routed_experts_capture=False to "
            "avoid polluting the target's routed-experts (R3) capture buffer."
        )
