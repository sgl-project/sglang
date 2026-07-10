from __future__ import annotations

import logging
from typing import Any, List, Optional

import msgspec

from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_GAMMA = 7
SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")


class DSparkLengthContract(msgspec.Struct, frozen=True):

    gamma: int

    @property
    def num_draft_positions(self) -> int:
        return int(self.gamma)

    @property
    def verify_num_draft_tokens(self) -> int:
        return int(self.gamma) + 1

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self.verify_num_draft_tokens

    def validate(self) -> None:
        if int(self.gamma) < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")


def make_dspark_length_contract(*, gamma: int) -> DSparkLengthContract:
    contract = DSparkLengthContract(gamma=int(gamma))
    contract.validate()
    return contract


def dspark_gamma_from_num_draft_tokens(num_draft_tokens: int) -> int:
    gamma = int(num_draft_tokens) - 1
    if gamma < 1:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {num_draft_tokens}."
        )
    return gamma


class DSparkDraftConfig(msgspec.Struct, frozen=True):
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    gamma: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]
    markov_rank: int
    markov_head_type: Optional[str]

    def resolve_gamma(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.gamma if self.gamma is not None else default

    def require_markov(self) -> bool:
        return int(self.markov_rank) > 0


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    return config


def _get_dspark_config(config: Any) -> dict:
    cfg = _cfg_get(config, "dspark_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_layer_ids(raw_layer_ids: Any, *, field_name: str) -> Optional[List[int]]:
    if raw_layer_ids is None:
        return None
    if not isinstance(raw_layer_ids, (list, tuple)) or not len(raw_layer_ids):
        raise ValueError(
            f"DSpark {field_name} must be a non-empty list of ints, "
            f"got {raw_layer_ids!r}."
        )
    return [int(x) for x in raw_layer_ids]


def get_dspark_target_layer_ids(config: Any) -> Optional[List[int]]:
    """Read DSpark target aux-hidden layers from any supported config layout."""
    dspark_cfg = _get_dspark_config(config)
    text_config = _get_text_config(config)

    candidates = (
        ("dspark_target_layer_ids", _cfg_get(config, "dspark_target_layer_ids", None)),
        (
            "text_config.dspark_target_layer_ids",
            _cfg_get(text_config, "dspark_target_layer_ids", None),
        ),
        ("dspark_config.target_layer_ids", dspark_cfg.get("target_layer_ids", None)),
        (
            "dspark_config.dspark_target_layer_ids",
            dspark_cfg.get("dspark_target_layer_ids", None),
        ),
        (
            "dspark_config.aux_hidden_state_layer_ids",
            dspark_cfg.get("aux_hidden_state_layer_ids", None),
        ),
        (
            "aux_hidden_state_layer_ids",
            _cfg_get(config, "aux_hidden_state_layer_ids", None),
        ),
        (
            "text_config.aux_hidden_state_layer_ids",
            _cfg_get(text_config, "aux_hidden_state_layer_ids", None),
        ),
    )
    for field_name, layer_ids in candidates:
        parsed = _parse_layer_ids(layer_ids, field_name=field_name)
        if parsed is not None:
            return parsed
    return None


def parse_dspark_draft_config(*, draft_hf_config: Any) -> DSparkDraftConfig:
    base = parse_dflash_draft_config(draft_hf_config=draft_hf_config)

    dspark_cfg = _get_dspark_config(draft_hf_config)
    text_config = _get_text_config(draft_hf_config)

    prefixed_block_size = _cfg_get(draft_hf_config, "dspark_block_size", None)
    prefixed_markov_rank = _cfg_get(draft_hf_config, "dspark_markov_rank", None)
    prefixed_markov_head_type = _cfg_get(
        draft_hf_config, "dspark_markov_head_type", None
    )
    prefixed_noise_token_id = _cfg_get(draft_hf_config, "dspark_noise_token_id", None)
    prefixed_target_layer_ids = _cfg_get(
        draft_hf_config, "dspark_target_layer_ids", None
    )
    if prefixed_target_layer_ids is None:
        prefixed_target_layer_ids = _cfg_get(
            text_config, "dspark_target_layer_ids", None
        )
    uses_prefixed = any(
        value is not None
        for value in (
            prefixed_block_size,
            prefixed_markov_rank,
            prefixed_noise_token_id,
            prefixed_target_layer_ids,
        )
    )

    raw_markov_rank = (
        prefixed_markov_rank
        if prefixed_markov_rank is not None
        else dspark_cfg.get(
            "markov_rank",
            _cfg_get(
                text_config, "markov_rank", _cfg_get(draft_hf_config, "markov_rank", 0)
            ),
        )
    )
    markov_rank = int(raw_markov_rank) if raw_markov_rank is not None else 0
    if markov_rank < 0:
        raise ValueError(f"DSpark markov_rank must be >= 0, got {markov_rank}.")

    markov_head_type = (
        prefixed_markov_head_type
        if prefixed_markov_head_type is not None
        else dspark_cfg.get(
            "markov_head_type",
            _cfg_get(
                text_config,
                "markov_head_type",
                _cfg_get(draft_hf_config, "markov_head_type", None),
            ),
        )
    )
    if markov_rank > 0 and markov_head_type is None and not uses_prefixed:
        raise ValueError(
            "DSpark requires markov_head_type when markov_rank > 0, got None."
        )
    if markov_head_type is not None:
        markov_head_type = str(markov_head_type).lower()
        if markov_head_type not in SUPPORTED_DSPARK_MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unsupported DSpark markov_head_type={markov_head_type!r}. "
                f"Supported: {SUPPORTED_DSPARK_MARKOV_HEAD_TYPES}."
            )

    raw_mask_token_id = (
        prefixed_noise_token_id
        if prefixed_noise_token_id is not None
        else dspark_cfg.get(
            "mask_token_id",
            _cfg_get(
                text_config,
                "mask_token_id",
                _cfg_get(draft_hf_config, "mask_token_id", base.mask_token_id),
            ),
        )
    )
    mask_token_id = int(raw_mask_token_id) if raw_mask_token_id is not None else None
    if mask_token_id is not None and mask_token_id < 0:
        raise ValueError(
            f"DSpark mask_token_id must be non-negative, got {mask_token_id}."
        )

    gamma = (
        int(prefixed_block_size) if prefixed_block_size is not None else base.block_size
    )

    target_layer_ids = get_dspark_target_layer_ids(draft_hf_config)
    if target_layer_ids is None:
        target_layer_ids = base.target_layer_ids

    return DSparkDraftConfig(
        num_hidden_layers=base.num_hidden_layers,
        num_target_layers=base.num_target_layers,
        gamma=gamma,
        target_layer_ids=target_layer_ids,
        mask_token=base.mask_token,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
    )
