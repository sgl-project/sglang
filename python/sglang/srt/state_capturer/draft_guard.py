"""Fail-closed runtime guard for draft-worker R3 routed-experts capture.

Called once per draft `ModelRunner` construction (when the draft worker
loads its model and before any forward runs). The guard:

  1. Resolves the actual draft architecture from the loaded model class
     name; cross-checks against `hf_config.architectures[0]` if available.
  2. Consults `draft_inventory.lookup_draft_arch(arch_name)`:
       - Unknown architecture                   -> RuntimeError (refuse start).
       - MoE-bearing but `opted_out=False`      -> RuntimeError (pending plumbing).
       - MoE-bearing and `opted_out=True`       -> walk modules; every
         `TopK.topk_config.allow_routed_experts_capture` must be False.
       - Dense allowlist                        -> walk modules; assert 0 TopK.
  3. Failure modes raise `RuntimeError` with a message that names the
     architecture and the specific contract violated, so the operator
     can identify the missing opt-out at a glance.

The guard is invoked only when routed-experts capture is actually enabled
(i.e. when the global capturer is present). Without `-­-enable-return-
routed-experts`, the guard is a no-op so users running normal speculative
decoding aren't blocked by pending opt-out work for families they don't
use under R3.
"""

from __future__ import annotations

from typing import Any, List, Optional

from sglang.srt.state_capturer.draft_inventory import (
    DraftInventoryEntry,
    lookup_draft_arch,
)


def _collect_topk_modules(model: Any) -> List[Any]:
    """Return every `TopK` instance reachable via `model.modules()`.

    Imported lazily to avoid an import-time cycle between
    `state_capturer.*` and `layers.moe.topk` during module initialization.
    """
    from sglang.srt.layers.moe.topk import TopK

    return [m for m in model.modules() if isinstance(m, TopK)]


def _resolved_architecture(model: Any, hf_config: Any) -> str:
    """Prefer the actually loaded class name; fall back to the HF config
    when the model object cannot be introspected. `type(self.model).__name__`
    is stronger evidence than `hf_config.architectures[0]` because
    `_config_draft_model()` may rewrite the config but the model is what
    actually got constructed."""
    if model is not None:
        cls_name = type(model).__name__
        if cls_name:
            return cls_name
    if hf_config is not None:
        archs = getattr(hf_config, "architectures", None) or []
        if archs:
            return archs[0]
    return "<unknown>"


def check_draft_capture_optout(
    model: Any,
    hf_config: Any,
    *,
    routed_experts_capture_enabled: bool,
) -> None:
    """Fail-closed guard. See module docstring for the contract.

    `model` is the freshly constructed draft `nn.Module`. `hf_config` is
    the HuggingFace-style config consulted for the cross-check. The
    `routed_experts_capture_enabled` argument is True iff the process
    intends to capture routed experts (e.g. the global capturer has been
    installed); when False, the guard is a no-op.
    """
    if not routed_experts_capture_enabled:
        return

    arch = _resolved_architecture(model, hf_config)
    entry: Optional[DraftInventoryEntry] = lookup_draft_arch(arch)

    if entry is None:
        raise RuntimeError(
            f"draft architecture {arch!r} not registered in draft_inventory; "
            "refusing to start to prevent silent R3 pollution. Add an entry "
            "to python/sglang/srt/state_capturer/draft_inventory.py before "
            "running with --enable-return-routed-experts."
        )

    if entry.moe_bearing and not entry.opted_out:
        raise RuntimeError(
            f"draft architecture {arch!r} is registered as MoE-bearing but "
            "its per-TopKConfig allow_routed_experts_capture opt-out is not yet "
            "plumbed (opted_out=False). Refusing to start to prevent silent "
            "R3 pollution. See "
            f"{entry.opt_out_injection_point!r} for the planned injection site."
        )

    topks = _collect_topk_modules(model)

    if entry.moe_bearing:
        # Every TopK on this draft model must carry allow_routed_experts_capture=False.
        offenders = [
            t for t in topks if getattr(t.topk_config, "allow_routed_experts_capture", True)
        ]
        if offenders:
            raise RuntimeError(
                f"draft architecture {arch!r} has {len(offenders)} TopK "
                "module(s) with allow_routed_experts_capture=True; expected all "
                "False on a draft worker. The per-model opt-out at "
                f"{entry.opt_out_injection_point!r} is missing or incomplete."
            )
        return

    # Dense allowlist: zero TopK modules are expected.
    if topks:
        raise RuntimeError(
            f"draft architecture {arch!r} is allowlisted as dense "
            f"(dense_no_topk) but the constructed model contains "
            f"{len(topks)} TopK module(s). The dense classification in "
            "draft_inventory.py is wrong, or the model wrapper started "
            "constructing MoE blocks that need an opt-out plumbed."
        )
