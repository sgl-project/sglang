"""Fail-closed runtime guard for draft-worker R3 routed-experts capture.

Called once per draft `ModelRunner` construction (when the draft worker
loads its model and before any forward runs). The guard walks the freshly
constructed draft model and asserts the single safety invariant that the
R3 routed-experts capture relies on:

    every `TopK` on a draft model must carry
    `topk_config.allow_routed_experts_capture == False`.

That flag is the only thing the runtime capture chokepoint
(`capture_routed_experts_if_allowed`) consults, so checking it directly on
the real model instance is both necessary and sufficient: an un-opted-out
draft MoE layer (flag left at the default `True`) would silently pollute
the target's R3 capture buffer, and the walk turns that into a loud
startup failure instead. Dense drafts simply have zero `TopK` and pass.

The guard runs only when routed-experts capture is actually enabled (i.e.
`--enable-return-routed-experts`); otherwise no capture can happen and the
guard is a no-op.
"""

from __future__ import annotations

from typing import Any, List


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
