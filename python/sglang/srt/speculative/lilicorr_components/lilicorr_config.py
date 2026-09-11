"""Geometry of the LiLiCorr candidate-lattice reranker head.

Parsed here rather than as a field on ``DFlashDraftConfig``, following
``dspark_components/dspark_config.py``, so the shared DFLASH config carries no
knowledge of this head. The cost is one extra read of ``dflash_config``, at
model build.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import msgspec

from sglang.kernels.ops.speculative.lilicorr import MAX_FUSED_CANDIDATE_TOPK
from sglang.srt.speculative.dflash_utils import _get_dflash_config

logger = logging.getLogger(__name__)

# ── PROBABILISTIC DRAFT, RESOLVED ONCE AT IMPORT ───────────────────────────────
# OFF by default, and off is byte-identical to the greedy commit that ships: the same
# kernel, the same argmax, no proposal emitted. ON samples each slot from
# ``softmax(psi_s / T)`` over that slot's candidates and publishes the row it drew
# from, so verify pays ``sum_c min(p, q)`` rather than ``p(argmax)`` -- the contract
# DSpark and the DFlash2 selector already serve under.
#
# Read at import, not per call, and that is load-bearing. ``LiLiCorrHead.select`` is
# ``torch.compile``d with ``dynamic=False`` and replayed from a captured draft CUDA
# graph, so a per-call ``os.environ`` read is a graph break in the compiled body and a
# host-side branch inside a replay. One module constant means one compiled body per
# process either way, and never a mixture.
# It lives here, not on the head, because the head and the graph-folded sampler both
# need it and this is the module they can both import without a cycle.
# It changes nothing about the checkpoint: the same trained factors are read either
# way, so a pair of runs differing only in this variable is a clean A/B on one set of
# weights. It is also orthogonal to the DFlash backbone, convolutions included -- the
# conv feeds ``pass_hidden``, which ``score`` consumes identically under both rules.
_SAMPLING_RAW = os.environ.get("LILICORR_SAMPLING", "0").strip().lower()
if _SAMPLING_RAW not in ("0", "1", "off", "on", "false", "true"):
    raise ValueError(
        f"LILICORR_SAMPLING={_SAMPLING_RAW!r} is not a boolean. Refusing rather than "
        "defaulting: a typo here would report the greedy number under the sampled "
        "arm's name."
    )
SAMPLING_ENABLED = _SAMPLING_RAW in ("1", "on", "true")
logger.info(
    "LiLiCorr draft commit: %s",
    "sampled (T>0 aware)" if SAMPLING_ENABLED else "greedy",
)

# The require gate catches the operator error the raise above cannot. A typo in
# ``LILICORR_SAMPLING`` already raises; FORGETTING it entirely is silent and serves the
# greedy commit under a sampled arm's name -- believable numbers, wrong arm. Set
# ``SGLANG_LILICORR_REQUIRE_SAMPLING=1`` on any cell whose name claims to be sampled and
# that mismatch becomes a startup failure instead of a plausible row.
# It cannot catch an absent overlay, and nothing here can: this file is part of the
# overlay, so with the overlay absent none of this runs and ``LILICORR_SAMPLING`` is read
# by nobody. The only proof of PRESENCE is the log line above -- assert it per cell
# (``harvest_sampling_pair.py --log`` does exactly that).
_REQUIRE_RAW = os.environ.get("SGLANG_LILICORR_REQUIRE_SAMPLING", "0").strip().lower()
if _REQUIRE_RAW not in ("0", "1", "off", "on", "false", "true"):
    raise ValueError(
        f"SGLANG_LILICORR_REQUIRE_SAMPLING={_REQUIRE_RAW!r} is not a boolean. Refusing "
        "rather than defaulting: a gate that silently reads as off is not a gate."
    )
if _REQUIRE_RAW in ("1", "on", "true") and not SAMPLING_ENABLED:
    raise RuntimeError(
        "SGLANG_LILICORR_REQUIRE_SAMPLING is set but LILICORR_SAMPLING is not enabled. "
        "This process would serve the GREEDY commit under a sampled arm's name. Refusing."
    )


class LiLiCorrConfig(msgspec.Struct, frozen=True):
    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, *, model_hidden_size: int) -> int:
        # hidden_size 0 means "as wide as the draft"; the exporter records it for
        # a head that carries no token_proj.
        return int(self.hidden_size) if self.hidden_size else int(model_hidden_size)


def _parse_lilicorr_config(dflash_cfg: dict) -> Optional[LiLiCorrConfig]:
    # Absence and an explicit lilicorr_enabled: false both mean "no head", so a
    # DFLASH checkpoint that merely mentions the flag still parses.
    if not any(key.startswith("lilicorr_") for key in dflash_cfg):
        return None
    enabled = dflash_cfg.get("lilicorr_enabled")
    if enabled is not None and not bool(enabled):
        return None

    def required(key: str, cast, *, positive: bool = True):
        # No field may be defaulted. Most change a tensor shape and would be
        # caught at weight load, but logit_scale and vector_eps would not: a
        # guessed value builds a head that loads cleanly and scores a different
        # function of the same weights.
        full_key = f"lilicorr_{key}"
        if full_key not in dflash_cfg:
            raise ValueError(
                f"DFLASH dflash_config.{full_key} is required to rebuild the "
                "LiLiCorr head. The checkpoint does not carry it, so the head "
                "this would construct is not the head that was trained."
            )
        try:
            value = cast(dflash_cfg[full_key])
        except Exception as e:
            raise ValueError(
                f"Invalid dflash_config.{full_key}={dflash_cfg[full_key]!r}."
            ) from e
        if positive and value <= 0:
            raise ValueError(f"dflash_config.{full_key} must be positive, got {value}.")
        return value

    candidate_topk = required("candidate_topk", int)
    if candidate_topk & (candidate_topk - 1):
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk must be a power of two, got "
            f"{candidate_topk}. The tiled candidate top-k selects its tiles inside a "
            "single Triton lane group, and tl.arange requires a power-of-two extent, "
            "so a head trained at another width could only be served on the slow "
            "reference path."
        )
    if candidate_topk > MAX_FUSED_CANDIDATE_TOPK:
        # A wider pool loads and serves correctly but falls off the fused greedy
        # commit onto the torch path, which is roughly three kernel launches per
        # slot. Refuse it here for the same reason the power-of-two case is
        # refused: silently deoptimized is worse than not served.
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk={candidate_topk} exceeds the "
            f"fused greedy commit's width of {MAX_FUSED_CANDIDATE_TOPK}, which is "
            "one Triton lane group. A wider head would serve on the reference path "
            "at a large throughput cost."
        )

    return LiLiCorrConfig(
        candidate_topk=candidate_topk,
        hidden_size=required("hidden_size", int, positive=False),
        num_layers=required("num_layers", int),
        num_heads=required("num_heads", int),
        mlp_ratio=required("mlp_ratio", float),
        factor_dim=required("factor_dim", int),
        vector_eps=required("vector_eps", float),
        logit_scale=required("logit_scale", float),
    )


def parse_lilicorr_draft_config(*, draft_hf_config: Any) -> LiLiCorrConfig:
    config = _parse_lilicorr_config(_get_dflash_config(draft_hf_config))
    if config is None:
        raise ValueError(
            "LiLiCorr requires the lilicorr_* geometry fields in dflash_config. "
            'A checkpoint declaring architectures=["LiLiCorrDraftModel"] without '
            "them cannot be rebuilt into the head that was trained."
        )
    return config
