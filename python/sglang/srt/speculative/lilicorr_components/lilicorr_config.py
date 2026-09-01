"""Geometry of the LiLiCorr candidate-lattice reranker head.

Parsed here rather than as a field on ``DFlashDraftConfig`` so that the shared
DFLASH config carries no knowledge of this head. This follows
``dspark_components/dspark_config.py``, which extends the base draft config from
its own package instead of adding fields to it. The cost is one extra read of
``dflash_config``, which happens once, at model build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.speculative.dflash_utils import _get_dflash_config


@dataclass(frozen=True)
class LiLiCorrConfig:
    """Geometry of the LiLiCorr candidate-lattice reranker head.

    Every field is read from the checkpoint with no default. Most of them change a
    tensor shape and would be caught at weight load, but ``logit_scale`` and
    ``vector_eps`` do not: a guessed value there builds a head that loads cleanly
    and scores a different function of the same weights, which surfaces as a small
    believable acceptance delta rather than as an error.
    """

    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, *, model_hidden_size: int) -> int:
        """Head width, where 0 means "as wide as the draft"."""
        return int(self.hidden_size) if self.hidden_size else int(model_hidden_size)


def _parse_lilicorr_config(dflash_cfg: dict) -> Optional[LiLiCorrConfig]:
    """Parse the LiLiCorr head geometry, or None when this is a plain DFLASH config.

    Absence and an explicit ``lilicorr_enabled: false`` both mean "no head", so a
    DFLASH checkpoint that merely mentions the flag still parses.
    """
    if not any(key.startswith("lilicorr_") for key in dflash_cfg):
        return None
    enabled = dflash_cfg.get("lilicorr_enabled")
    if enabled is not None and not bool(enabled):
        return None

    def required(key: str, cast, *, positive: bool = True):
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
            "reference path. Refused here, at load, rather than either failing inside "
            "a kernel on the first decode or silently serving deoptimized."
        )

    return LiLiCorrConfig(
        candidate_topk=candidate_topk,
        # The one field allowed to be zero: it means "as wide as the draft", which
        # is what the exporter records for a head that carries no token_proj.
        hidden_size=required("hidden_size", int, positive=False),
        num_layers=required("num_layers", int),
        num_heads=required("num_heads", int),
        mlp_ratio=required("mlp_ratio", float),
        factor_dim=required("factor_dim", int),
        vector_eps=required("vector_eps", float),
        logit_scale=required("logit_scale", float),
    )


def parse_lilicorr_draft_config(*, draft_hf_config: Any) -> LiLiCorrConfig:
    """The head geometry a LiLiCorr checkpoint must carry.

    Called from ``LiLiCorrDraftModel``, so the architecture string is what a
    missing geometry contradicts: the checkpoint asked for this head and did not
    say which one.
    """
    config = _parse_lilicorr_config(_get_dflash_config(draft_hf_config))
    if config is None:
        raise ValueError(
            "LiLiCorr requires the lilicorr_* geometry fields in dflash_config. "
            'A checkpoint declaring architectures=["LiLiCorrDraftModel"] without '
            "them cannot be rebuilt into the head that was trained."
        )
    return config
