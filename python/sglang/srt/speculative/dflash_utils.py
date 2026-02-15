from __future__ import annotations

import logging
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
logger = logging.getLogger(__name__)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Mirrors the upstream DFlash helper in `docs/dflash/model/utils.py`, but keeps the
    logic local to SGLang.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def resolve_dflash_block_size(
    *,
    draft_hf_config: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve DFLASH block size from draft config.

    Precedence:
      1) `dflash_config.block_size`
      2) top-level `block_size`
      3) `default`
    """
    dflash_cfg = get_dflash_config(draft_hf_config)
    dflash_block_size = dflash_cfg.get("block_size", None)
    if isinstance(draft_hf_config, dict):
        top_level_block_size = draft_hf_config.get("block_size", None)
    else:
        top_level_block_size = getattr(draft_hf_config, "block_size", None)

    parsed_dflash_block_size = None
    if dflash_block_size is not None:
        try:
            parsed_dflash_block_size = int(dflash_block_size)
        except Exception as e:
            raise ValueError(
                f"Invalid DFLASH dflash_config.block_size={dflash_block_size!r}."
            ) from e

    parsed_top_level_block_size = None
    if top_level_block_size is not None:
        try:
            parsed_top_level_block_size = int(top_level_block_size)
        except Exception as e:
            raise ValueError(
                f"Invalid DFLASH block_size={top_level_block_size!r}."
            ) from e

    if (
        parsed_dflash_block_size is not None
        and parsed_top_level_block_size is not None
        and parsed_dflash_block_size != parsed_top_level_block_size
    ):
        logger.warning(
            "DFLASH draft config has both block_size=%s and dflash_config.block_size=%s; using dflash_config.block_size.",
            top_level_block_size,
            dflash_block_size,
        )

    block_size = (
        parsed_dflash_block_size
        if parsed_dflash_block_size is not None
        else parsed_top_level_block_size
    )
    if block_size is None:
        return default

    if block_size <= 0:
        raise ValueError(f"DFLASH block_size must be positive, got {block_size}.")
    return block_size


def resolve_dflash_target_layer_ids(
    *,
    draft_hf_config: Any,
    target_num_layers: int,
    draft_num_layers: int,
) -> List[int]:
    """Resolve target layer ids used to build DFlash context features.

    Precedence:
      1) `draft_hf_config.dflash_config.target_layer_ids`
      2) `draft_hf_config.target_layer_ids` (fallback to base config)
      3) default `build_target_layer_ids(target_num_layers, draft_num_layers)`

    Notes:
        The number of draft transformer layers is *not* fundamentally tied to the number
        of target-layer features (K) used as DFlash context. We treat
        `len(target_layer_ids)` as K when explicitly provided. For backward compatibility
        (and for current released checkpoints), the default still uses K == draft_num_layers.
    """
    cfg = get_dflash_config(draft_hf_config)
    layer_ids = cfg.get("target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(draft_hf_config, "target_layer_ids", None)
    if layer_ids is None:
        return build_target_layer_ids(target_num_layers, draft_num_layers)

    if not isinstance(layer_ids, (list, tuple)):
        raise ValueError(
            "DFLASH dflash_config.target_layer_ids must be a list of ints, "
            f"got type={type(layer_ids).__name__}."
        )

    resolved: List[int] = [int(x) for x in layer_ids]
    if len(resolved) <= 0:
        raise ValueError(
            "DFLASH dflash_config.target_layer_ids must be non-empty. "
            f"Got len(target_layer_ids)={len(resolved)}."
        )

    for idx, val in enumerate(resolved):
        if val < 0 or val >= int(target_num_layers):
            raise ValueError(
                "DFLASH target_layer_ids contains an out-of-range layer id. "
                f"target_layer_ids[{idx}]={val}, target_num_layers={int(target_num_layers)}."
            )
    return resolved


def resolve_dflash_mask_token(*, draft_hf_config: Any) -> str:
    cfg = get_dflash_config(draft_hf_config)
    mask_token = cfg.get("mask_token", None)
    if mask_token is None:
        return DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )
    return mask_token


def resolve_dflash_mask_token_id(*, draft_hf_config: Any) -> Optional[int]:
    cfg = get_dflash_config(draft_hf_config)
    mask_token_id = cfg.get("mask_token_id", None)
    if mask_token_id is None:
        return None
    if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
        raise ValueError(
            "DFLASH dflash_config.mask_token_id must be an integer, "
            f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
        )
    mask_token_id = int(mask_token_id)
    if mask_token_id < 0:
        raise ValueError(
            "DFLASH dflash_config.mask_token_id must be non-negative, "
            f"got {mask_token_id}."
        )
    return mask_token_id


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index accept_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)
