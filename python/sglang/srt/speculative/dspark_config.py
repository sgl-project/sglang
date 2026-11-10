from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


@dataclass(frozen=True)
class DSparkDraftConfig:
    gamma: int
    markov_rank: int
    markov_head_type: str
    mask_token_id: int


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _nested_dflash_config(config: Any) -> dict:
    value = _cfg_get(config, "dflash_config", None)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def parse_dspark_draft_config(config: Any) -> DSparkDraftConfig:
    base = parse_dflash_draft_config(draft_hf_config=config)
    nested = _nested_dflash_config(config)

    gamma = int(base.resolve_block_size(default=7))
    markov_rank = int(nested.get("markov_rank", _cfg_get(config, "markov_rank", 0)))
    markov_head_type = str(
        nested.get("markov_head_type", _cfg_get(config, "markov_head_type", "vanilla"))
    ).lower()
    mask_token_id: Optional[int] = base.mask_token_id

    if gamma <= 0:
        raise ValueError(f"DSPARK block_size (gamma) must be positive, got {gamma}.")
    if markov_rank <= 0:
        raise ValueError(f"DSPARK markov_rank must be positive, got {markov_rank}.")
    if markov_head_type != "vanilla":
        raise ValueError(
            "The V100 DSPARK worker currently supports the vanilla Markov head; "
            f"got markov_head_type={markov_head_type!r}."
        )
    if mask_token_id is None:
        raise ValueError(
            "DSPARK requires dflash_config.mask_token_id in the draft config."
        )

    return DSparkDraftConfig(
        gamma=gamma,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        mask_token_id=int(mask_token_id),
    )
