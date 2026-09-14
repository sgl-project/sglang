"""Static capability pre-flight — the gate that is BOTH guarantee and generator.

This is the mechanical core of "refine, not replace": Attune reuses SGLang's existing
eligibility logic (the ``_get_default_attn_backend`` / ``overrides.py`` predicates) to
produce the candidate list, then empirically ranks only among those survivors. It never
expands the legal set and never bypasses a predicate.

Static checks are necessary but NOT sufficient — they never launch a kernel, so they miss
"no kernel image available" wheel-coverage failures (vLLM's static-only selector shipped
exactly this class of sm_120 regression). That is why the survivors still go through the
empirical subprocess probe (see isolation.py). Static prune first (cheap, avoid spawning
doomed candidates), empirical probe second (catches what static checks structurally can't).

Returns, per backend, either eligible or a human-readable reason it was pruned — the
vLLM ``validate_configuration`` "reasons list" pattern.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

from .device import DeviceInfo
from .shapes import AttnProfile


@dataclasses.dataclass
class Candidate:
    name: str
    eligible: bool
    reason: str = ""  # why pruned, when not eligible


# Minimal, CUDA-only v1 backend set (ROCm/AITER is v2 — its integration is gated by an
# AITER_MOE env + _is_hip check, NOT the registry, so registry-style enumeration wouldn't
# even see it; that's an explicit v1 scope boundary).
_MHA_POOL = ["fa3", "fa4", "flashinfer", "trtllm_mha", "triton", "torch_native"]
_MLA_POOL = ["fa3", "flashmla", "cutlass_mla", "trtllm_mla", "flashinfer", "triton"]


def _supported(
    backend: str, dev: DeviceInfo, profile: AttnProfile, phase: str
) -> Tuple[bool, str]:
    """Declarative per-backend capability predicate (mirror of SGLang's gate).

    NOTE: these predicates are illustrative and must be reconciled with the current
    ``server_args.py`` / ``overrides.py`` before landing — they encode the *shape* of the
    gate, not a frozen copy of it.
    """
    sm = dev.sm
    if backend in ("triton", "torch_native"):
        return True, ""
    if backend == "flashinfer":
        # Broad coverage sm75+; can't do attention sinks (out of scope here).
        return (sm >= 75, "" if sm >= 75 else "flashinfer requires sm>=75")
    if backend == "fa3":
        # Hopper-only (sm90a). Does NOT run on Ampere or Blackwell.
        return (sm == 90, "" if sm == 90 else "fa3 is Hopper-only (sm90)")
    if backend == "fa4":
        # Hopper + Blackwell.
        return (
            sm in (90, 100, 103),
            "" if sm in (90, 100, 103) else "fa4 requires Hopper/Blackwell",
        )
    if backend == "trtllm_mha":
        ok = (phase == "prefill" and sm == 100) or (
            phase == "decode" and sm in (90, 100, 120)
        )
        return ok, "" if ok else f"trtllm_mha {phase} unsupported on sm{sm}"
    if backend in ("flashmla", "cutlass_mla", "trtllm_mla"):
        if not profile.is_mla:
            return False, f"{backend} is MLA-only"
        if backend == "cutlass_mla":
            return (sm >= 100, "" if sm >= 100 else "cutlass_mla requires Blackwell")
        return (sm >= 90, "" if sm >= 90 else f"{backend} requires sm>=90")
    return False, f"unknown backend {backend}"


def candidate_backends(
    dev: DeviceInfo, profile: AttnProfile, phase: str
) -> Tuple[List[str], Dict[str, str]]:
    """Return (eligible_backends, {pruned_backend: reason}) for a (device, profile, phase)."""
    pool = _MLA_POOL if profile.is_mla else _MHA_POOL
    eligible: List[str] = []
    pruned: Dict[str, str] = {}
    for b in pool:
        ok, why = _supported(b, dev, profile, phase)
        if ok:
            eligible.append(b)
        else:
            pruned[b] = why
    # There is ALWAYS a runnable fallback (triton) so an empty survivor set never happens
    # for CUDA — the fail-open-to-triton guarantee the v0.14.0 vLLM "no valid backend"
    # regression teaches us to keep.
    if not eligible:
        eligible = ["triton"]
    return eligible, pruned
