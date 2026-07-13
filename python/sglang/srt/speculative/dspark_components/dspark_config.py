from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec

from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_DSPARK_GAMMA = 7
SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")

# The dsv4 self-drafting checkpoint runs its draft attention on the dedicated
# DeepSeek-V4 backend instead of the generic draft-backend fallback.
DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"


def draft_is_deepseek_v4(*, server_args: ServerArgs) -> bool:
    from sglang.srt.configs.model_config import is_deepseek_v4
    from sglang.srt.utils.hf_transformers_utils import get_config

    draft_model_path = server_args.speculative_draft_model_path
    if not draft_model_path:
        return False
    draft_hf_config = get_config(
        draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.speculative_draft_model_revision,
        model_override_args=json.loads(server_args.json_model_override_args),
        model_config_parser=server_args.model_config_parser,
    )
    return draft_hf_config is not None and is_deepseek_v4(draft_hf_config)


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
    speculators_convention: bool

    def resolve_gamma(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.gamma if self.gamma is not None else default

    def require_markov(self) -> bool:
        return int(self.markov_rank) > 0


class DSparkRuntimeConfig(msgspec.Struct, frozen=True):
    gamma: int
    verify_num_draft_tokens: int
    mask_token_id: int


def resolve_runtime_config(
    *,
    draft_hf_config: Any,
    speculative_num_draft_tokens: Optional[int],
    target_vocab_size: int,
) -> DSparkRuntimeConfig:
    """Resolve and validate the worker-facing DSpark runtime knobs (gamma,
    verify window, mask token) from the draft checkpoint config, with
    server_args.speculative_num_draft_tokens taking precedence for gamma."""
    draft_config = parse_dspark_draft_config(draft_hf_config=draft_hf_config)
    if not draft_config.require_markov():
        raise ValueError(
            "DSpark draft requires markov_rank > 0; got "
            f"markov_rank={draft_config.markov_rank}."
        )

    if speculative_num_draft_tokens is None:
        gamma = int(draft_config.resolve_gamma(default=None) or 0)
        if gamma < 1:
            raise ValueError(
                "DSpark could not resolve gamma from the draft config and "
                "speculative_num_draft_tokens is unset."
            )
    else:
        gamma = dspark_gamma_from_num_draft_tokens(int(speculative_num_draft_tokens))
        config_gamma = draft_config.resolve_gamma(default=None)
        if config_gamma is not None and int(config_gamma) != gamma:
            logger.warning(
                "DSpark gamma mismatch: using gamma=%s (from "
                "speculative_num_draft_tokens=%s) but draft config block_size=%s.",
                gamma,
                speculative_num_draft_tokens,
                config_gamma,
            )

    if draft_config.mask_token_id is None:
        raise ValueError(
            "DSpark requires mask_token_id to be set in the draft model config."
        )
    mask_token_id = int(draft_config.mask_token_id)
    if mask_token_id >= target_vocab_size:
        raise ValueError(
            f"DSpark mask_token_id={mask_token_id} is outside the target "
            f"vocab size {target_vocab_size}."
        )

    return DSparkRuntimeConfig(
        gamma=gamma,
        verify_num_draft_tokens=gamma + 1,
        mask_token_id=mask_token_id,
    )


def read_draft_checkpoint_gamma(*, server_args: ServerArgs) -> Optional[int]:
    """Load the draft checkpoint's hf config and read its DSpark gamma
    (block_size). Raises on config-load failure; callers pick the fallback."""
    from sglang.srt.utils.hf_transformers_utils import get_config

    draft_hf_config = get_config(
        server_args.speculative_draft_model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.speculative_draft_model_revision,
        model_override_args=json.loads(server_args.json_model_override_args),
    )
    return parse_dspark_draft_config(draft_hf_config=draft_hf_config).resolve_gamma(
        default=None
    )


def checkpoint_bundles_dspark_draft(hf_config: Any) -> bool:
    """The checkpoint carries a bundled DSpark draft head, marked by the
    prefixed dspark_* keys on the target hf config. Single source of truth
    for the bundling convention (draft-path defaulting, draft-arch remap)."""
    return any(
        _cfg_get(hf_config, key, None) is not None
        for key in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        )
    )


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

    if prefixed_target_layer_ids is not None:
        if not isinstance(prefixed_target_layer_ids, (list, tuple)) or not len(
            prefixed_target_layer_ids
        ):
            raise ValueError(
                "DSpark dspark_target_layer_ids must be a non-empty list of ints, "
                f"got {prefixed_target_layer_ids!r}."
            )
        target_layer_ids: Optional[List[int]] = [
            int(x) for x in prefixed_target_layer_ids
        ]
    else:
        target_layer_ids = base.target_layer_ids

    # speculators (github.com/vllm-project/speculators)-trained DSpark
    # checkpoints use a different block-slot convention than the
    # DeepSpec-trained ones this file was written against: DeepSpec trains
    # block slot k to predict anchor+k+1 (every slot trained), while
    # speculators trains slot j to predict anchor+j with slot 0 loss-masked
    # (its first slot is never trained to predict anything useful). Every
    # `run_markov_block` slot is therefore read one position early for a
    # speculators checkpoint, degrading accept length to ~1 regardless of
    # the underlying model's real speculative quality. Detected here, not
    # silently corrected, until the shift is implemented and validated
    # end-to-end (it isn't yet) -- see sgl-project/sglang#30261 (comment by
    # jessiewei7, 2026-07-09) for the confirmed diagnosis and reproduction.
    speculators_model_type = _cfg_get(draft_hf_config, "speculators_model_type", None)
    speculators_convention = speculators_model_type == "dspark"

    return DSparkDraftConfig(
        num_hidden_layers=base.num_hidden_layers,
        num_target_layers=base.num_target_layers,
        gamma=gamma,
        target_layer_ids=target_layer_ids,
        mask_token=base.mask_token,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        speculators_convention=speculators_convention,
    )
