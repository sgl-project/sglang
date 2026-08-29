from __future__ import annotations

import logging
from enum import IntEnum

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SpecTpSyncSite(IntEnum):
    """Every place a speculative step broadcasts a decision from rank 0.

    Enumerated rather than grouped: any grouping is itself a claim about which
    sites share a cause, and that claim is what we are trying to establish. Each
    site has a stable number and slug so coverage can be bisected under live
    traffic through ``SGLANG_SPEC_TP_SYNC``.
    """

    # -- DSpark --
    DSPARK_MEM = 1  # free-memory probe gating capture and folded sampling
    DSPARK_DRAFT_GREEDY = 2  # eager argmax over draft step logits
    DSPARK_DRAFT_SAMPLE = 3  # eager SampleStepTokens, freshly drawn noise
    DSPARK_DRAFT_MULTINOMIAL = 4  # eager multinomial mixed with argmax
    DSPARK_GRAPH_SAMPLE = 5  # SampleStepTokens on in-graph philox noise
    DSPARK_GRAPH_GREEDY = 6  # argmax inside the captured draft graph
    DSPARK_PLAN = 7  # per-step verify-length schedule
    DSPARK_ACCEPT_GREEDY = 8  # AcceptGreedy correct_len/bonus/cap_trim
    DSPARK_ACCEPT_SAMPLE = 9  # AcceptSampling correct_len/bonus/cap_trim
    DSPARK_ACCEPT_GRAPH = 10  # accept_greedy_triton in the verify epilogue
    DSPARK_TARGET = 11  # target-model sampled tokens

    # -- DFlash --
    DFLASH_MEM = 12  # free-memory probe gating capture
    DFLASH_SELECTOR = 13  # selector accept_len/bonus
    DFLASH_ACCEPT_SAMPLE = 14  # sampling-verify accept_len/bonus
    DFLASH_ACCEPT_GREEDY = 15  # argmax over target logits
    DFLASH_TARGET = 16  # target-model sampled tokens

    @property
    def slug(self) -> str:
        return self.name.lower().replace("_", "-")


_ALL = frozenset(SpecTpSyncSite)
_INIT = frozenset({SpecTpSyncSite.DSPARK_MEM, SpecTpSyncSite.DFLASH_MEM})
# Sites whose value is drawn from the RNG, so they can differ across ranks even
# with bit-identical logits and deterministic kernels.
_RNG = frozenset(
    {
        SpecTpSyncSite.DSPARK_DRAFT_SAMPLE,
        SpecTpSyncSite.DSPARK_DRAFT_MULTINOMIAL,
        SpecTpSyncSite.DSPARK_GRAPH_SAMPLE,
        SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE,
        SpecTpSyncSite.DSPARK_TARGET,
        SpecTpSyncSite.DFLASH_SELECTOR,
        SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE,
        SpecTpSyncSite.DFLASH_TARGET,
    }
)

_PRESETS = {
    "all": _ALL,
    "off": frozenset(),
    "none": frozenset(),
    # The one measured per-rank input. Never drop it except to test it.
    "init": _INIT,
    "rng": _INIT | _RNG,
}

_BY_SLUG = {site.slug: site for site in SpecTpSyncSite}
_BY_NUMBER = {str(int(site)): site for site in SpecTpSyncSite}


def _resolve(name: str) -> frozenset[SpecTpSyncSite]:
    if name in _PRESETS:
        return _PRESETS[name]
    site = _BY_SLUG.get(name) or _BY_NUMBER.get(name)
    if site is None:
        raise ValueError(
            f"SGLANG_SPEC_TP_SYNC: unknown token {name!r}. "
            f"Presets: {sorted(_PRESETS)}. "
            f"Sites: {[(int(s), s.slug) for s in SpecTpSyncSite]}."
        )
    return frozenset({site})


def parse_spec_tp_sync(spec: str) -> frozenset[SpecTpSyncSite]:
    """Parse ``SGLANG_SPEC_TP_SYNC``: comma-separated preset names, site slugs or
    site numbers, each negatable with a leading ``-`` -- ``all,-dspark-plan,-6``.
    Underscores in a slug are accepted for its enum spelling."""
    sites: frozenset[SpecTpSyncSite] = frozenset()
    for token in spec.replace(" ", "").replace("_", "-").lower().split(","):
        if not token:
            continue
        negate = token.startswith("-")
        value = _resolve(token[1:] if negate else token)
        sites = sites - value if negate else sites | value
    return sites


class SpecTpSync:
    """Broadcasts a speculative decision from rank 0 to its TP group.

    Which sites broadcast is set by ``SGLANG_SPEC_TP_SYNC`` so coverage can be
    narrowed under live traffic to find where ranks actually diverge.
    """

    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        # Parsed even on a single rank so a typo fails on every deployment.
        sites = parse_spec_tp_sync(envs.SGLANG_SPEC_TP_SYNC.get())
        self._sites = sites if tp_group.world_size > 1 else frozenset()
        if sites != _ALL and tp_group.world_size > 1 and tp_group.rank_in_group == 0:
            logger.warning(
                "Speculative TP sync limited to %s.",
                [f"{int(s)}:{s.slug}" for s in sorted(sites)] or "no site",
            )

    def enabled(self, site: SpecTpSyncSite) -> bool:
        return site in self._sites

    def sync(self, site: SpecTpSyncSite, values: torch.Tensor) -> torch.Tensor:
        if site in self._sites:
            self._tp_group.broadcast(values, src=0)
        return values

    def available_memory_gb(self, site: SpecTpSyncSite, device, gpu_id, *, group):
        """Free GPU memory for a capture decision, reduced to the group minimum
        when ``site`` is enabled so that every rank decides identically."""
        distributed = self.enabled(site) and group.world_size > 1
        return get_available_gpu_memory(
            device,
            gpu_id,
            distributed=distributed,
            cpu_group=group.cpu_group if distributed else None,
        )
