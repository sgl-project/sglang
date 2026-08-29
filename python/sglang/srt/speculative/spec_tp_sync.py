from __future__ import annotations

import logging
from enum import Flag, auto

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SpecTpSyncSite(Flag):
    """Kinds of cross-rank divergence a speculative step can suffer.

    TP ranks run in lockstep -- one broadcast seed, all-reduced logits, the same
    code -- so each site is a *hypothesis* about how that lockstep breaks, not a
    known break. Grouping by kind lets a deployment drop one hypothesis at a
    time via ``SGLANG_SPEC_TP_SYNC`` and observe whether the hang returns.
    """

    NONE = 0
    # Startup probe of a per-rank quantity (free GPU memory). Measured to differ
    # by ~10% between ranks; crossing a capture threshold then makes ranks
    # capture structurally different graphs. The one confirmed divergence.
    INIT = auto()
    # Draft tokens drawn with RNG (exponential noise, multinomial).
    DRAFT_SAMPLE = auto()
    # Draft tokens from argmax: deterministic given identical logits.
    DRAFT_GREEDY = auto()
    # Target-model sampled tokens (the spec analogue of SYNC_TOKEN_IDS_ACROSS_TP).
    TARGET = auto()
    # Accept length and bonus from rejection sampling: consumes RNG.
    VERIFY_SAMPLE = auto()
    # Accept length and bonus from argmax or a deterministic Triton kernel.
    VERIFY_GREEDY = auto()
    # Per-step verify-length schedule: a pure function of the sites above.
    PLAN = auto()


_ALL = (
    SpecTpSyncSite.INIT
    | SpecTpSyncSite.DRAFT_SAMPLE
    | SpecTpSyncSite.DRAFT_GREEDY
    | SpecTpSyncSite.TARGET
    | SpecTpSyncSite.VERIFY_SAMPLE
    | SpecTpSyncSite.VERIFY_GREEDY
    | SpecTpSyncSite.PLAN
)

_SITES = {site.name.lower(): site for site in SpecTpSyncSite if site.name != "NONE"}

_PRESETS = {
    "all": _ALL,
    # Only the sites whose value depends on RNG, plus the confirmed init probe.
    # Holds if logits and kernels are bit-identical across ranks.
    "rng": (
        SpecTpSyncSite.INIT
        | SpecTpSyncSite.DRAFT_SAMPLE
        | SpecTpSyncSite.TARGET
        | SpecTpSyncSite.VERIFY_SAMPLE
    ),
    "off": SpecTpSyncSite.NONE,
    "none": SpecTpSyncSite.NONE,
}


def parse_spec_tp_sync(spec: str) -> SpecTpSyncSite:
    """Parse ``SGLANG_SPEC_TP_SYNC``: comma-separated preset or site names, each
    optionally negated with a leading ``-`` (e.g. ``all,-plan,-verify_greedy``)."""
    sites = SpecTpSyncSite.NONE
    for token in spec.replace(" ", "").lower().split(","):
        if not token:
            continue
        negate = token.startswith("-")
        name = token[1:] if negate else token
        value = _PRESETS.get(name, _SITES.get(name))
        if value is None:
            raise ValueError(
                f"SGLANG_SPEC_TP_SYNC: unknown token {token!r}. "
                f"Presets: {sorted(_PRESETS)}. Sites: {sorted(_SITES)}."
            )
        sites = sites & ~value if negate else sites | value
    return sites


class SpecTpSync:
    """Broadcasts a speculative decision from rank 0 to its TP group.

    Which sites broadcast is set by ``SGLANG_SPEC_TP_SYNC`` so the coverage can
    be narrowed under live traffic to find where ranks actually diverge.
    """

    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        # Parsed even on a single rank so a typo fails on every deployment.
        sites = parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.get())
        self._sites = sites if tp_group.world_size > 1 else SpecTpSyncSite.NONE
        if (
            self._sites != _ALL
            and tp_group.world_size > 1
            and tp_group.rank_in_group == 0
        ):
            logger.warning("Speculative TP sync reduced to %s.", self._sites)

    def enabled(self, site: SpecTpSyncSite) -> bool:
        return bool(self._sites & site)

    def sync(self, site: SpecTpSyncSite, values: torch.Tensor) -> torch.Tensor:
        if self._sites & site:
            self._tp_group.broadcast(values, src=0)
        return values

    def available_memory_gb(self, device, gpu_id, *, group) -> float:
        """Free GPU memory for a capture decision, reduced to the group minimum
        under the INIT site so that every rank decides identically."""
        distributed = self.enabled(SpecTpSyncSite.INIT) and group.world_size > 1
        return get_available_gpu_memory(
            device,
            gpu_id,
            distributed=distributed,
            cpu_group=group.cpu_group if distributed else None,
        )
