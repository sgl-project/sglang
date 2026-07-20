from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.utils import is_cuda, is_musa
from sglang.srt.utils.common import fast_topk

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

logger = logging.getLogger(__name__)

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TritonAttnBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda() or is_musa():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def apply_dflash_verify_logits_adjustments(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
) -> None:
    """Apply sampling-time logit adjustments for DFlash verify in place.

    This keeps v1 and v2 verify semantics aligned while letting overlap scheduling
    use the cheaper precomputed `acc_linear_penalties` path instead of allocating a
    repeated `[bs * draft_token_num, vocab]` penalty tensor every step.
    """
    if sampling_info is None:
        return
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")

    bs = len(sampling_info)
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch for DFlash verify adjustments. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )

    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            next_token_logits,
            sampling_info,
            num_tokens_in_batch=draft_token_num,
        )

    acc_linear_penalties = getattr(sampling_info, "acc_linear_penalties", None)
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    vocab_mask = getattr(sampling_info, "vocab_mask", None)
    logit_bias = getattr(sampling_info, "logit_bias", None)

    logits_3d: Optional[torch.Tensor] = None

    def get_logits_3d() -> torch.Tensor:
        nonlocal logits_3d
        if logits_3d is None:
            logits_3d = next_token_logits.reshape(bs, draft_token_num, -1)
        return logits_3d

    # Dense fallback only when we need live penalizer application or a vocab mask.
    # In overlap scheduling the common path is `acc_linear_penalties`, which can be
    # broadcast over the verify block without materializing a repeated buffer.
    if (
        penalizer is not None and penalizer.is_required and acc_linear_penalties is None
    ) or vocab_mask is not None:
        linear_penalty = torch.zeros(
            (bs, next_token_logits.shape[1]),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        sampling_info.apply_logits_bias(linear_penalty)
        get_logits_3d().add_(
            linear_penalty[:, None, :].to(dtype=next_token_logits.dtype)
        )
        return

    if acc_linear_penalties is not None:
        if (
            acc_linear_penalties.device != next_token_logits.device
            or acc_linear_penalties.dtype != next_token_logits.dtype
        ):
            acc_linear_penalties = acc_linear_penalties.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(acc_linear_penalties[:, None, :])

    if logit_bias is not None:
        if (
            logit_bias.device != next_token_logits.device
            or logit_bias.dtype != next_token_logits.dtype
        ):
            logit_bias = logit_bias.to(
                device=next_token_logits.device,
                dtype=next_token_logits.dtype,
            )
        get_logits_3d().add_(logit_bias[:, None, :])


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

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


def get_dflash_layer_types(config: Any) -> Optional[Sequence[str]]:
    text_config = _get_text_config(config)
    layer_types = _cfg_get(text_config, "layer_types", _cfg_get(config, "layer_types"))
    if layer_types is None:
        return None
    if isinstance(layer_types, str) or not isinstance(layer_types, Sequence):
        raise ValueError(
            "DFLASH config.layer_types must be a sequence of attention type strings."
        )
    return layer_types


def get_dflash_attention_sliding_window_size(config: Any) -> Optional[int]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None

    text_config = _get_text_config(config)
    sliding_window = _cfg_get(
        text_config, "sliding_window", _cfg_get(config, "sliding_window")
    )
    if sliding_window is None:
        raise ValueError(
            "DFLASH sliding_attention layers require config.sliding_window."
        )

    # HF sliding windows include the current token; SGLang stores window_left.
    return int(sliding_window) - 1


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
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


def _get_dflash_config(config: Any) -> dict:
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


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
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

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


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


def compute_dflash_correct_drafts_and_bonus(
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
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

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
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


def compute_dflash_sampling_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_single = get_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_acc = get_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    target_probs = build_dflash_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
        bs=bs,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
        use_sparse_topk=use_sparse_topk,
    )
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    correct_len = accept_token_num
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


def compute_dflash_tree_sampling_accept(
    *,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-greedy accept over the DFlash verify TREE via rejection sampling.

    Wraps ``tree_speculative_sampling_target_only`` (already tree-native) with the
    pruned tree's retrieve structures. Returns ``(predicts, accept_index,
    accept_token_num)`` in the SAME layout as ``verify_tree_greedy_func``, so the
    downstream ``dflash_tree_accept_compact`` path is unchanged. `candidates` is
    [bs, draft_token_num] in the kernel layout (prepare_for_verify order).
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    bs, draft_token_num = candidates.shape
    device = next_token_logits.device
    if threshold_single is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_single = get_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.runtime_context import get_server_args

        threshold_acc = get_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    target_probs = build_dflash_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
        bs=bs,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
        use_sparse_topk=use_sparse_topk,
    )
    draft_probs = torch.zeros_like(target_probs)
    uniform_samples = torch.rand(
        (bs, draft_token_num), dtype=torch.float32, device=device
    )
    uniform_samples_for_final_sampling = torch.rand(
        (bs,), dtype=torch.float32, device=device
    )
    predicts = torch.empty((bs * draft_token_num,), device=device, dtype=torch.int32)
    accept_index = torch.full(
        (bs, draft_token_num), -1, device=device, dtype=torch.int32
    )
    accept_token_num = torch.zeros((bs,), device=device, dtype=torch.int32)
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
    return predicts, accept_index, accept_token_num


def build_dflash_verify_target_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
    bs: int,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    use_sparse_topk: bool = True,
) -> torch.Tensor:
    device = next_token_logits.device
    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        if max_top_k is None:
            max_top_k = int(repeated_top_ks.max().item())
        else:
            max_top_k = int(max_top_k)
        if max_top_k < 1:
            max_top_k = 1
        elif max_top_k > vocab_size:
            max_top_k = vocab_size

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
    return target_probs.view(bs, draft_token_num, -1).contiguous()


def validate_dflash_request(req: Req, enable_overlap: bool) -> Optional[str]:
    if req.return_logprob:
        return "DFLASH speculative decoding does not support return_logprob yet."

    if enable_overlap and req.return_hidden_states:
        return "DFLASH speculative decoding does not support return_hidden_states yet."

    if (
        req.sampling_params.json_schema is not None
        or req.sampling_params.regex is not None
        or req.sampling_params.ebnf is not None
        or req.sampling_params.structural_tag is not None
    ):
        return (
            "DFLASH speculative decoding does not support "
            "grammar-constrained decoding yet."
        )

    return None


def build_tree_verify_tokens(
    *,
    verified_id: torch.Tensor,
    topk_probs: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_draft_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the pruned draft tree for DFlash EAGLE-style tree verify.

    Runs the fused single-pass GPU kernel to expand + top-``topk`` the draft block,
    then selects the top ``num_draft_tokens - 1`` nodes by score. Requires the fused
    path: CUDA + power-of-2 ``topk`` in {4, 8, 16} (the kernel indexes ``topk**2``).

    Args:
        verified_id: Current token per request, [bs].
        topk_probs / topk_ids: Draft top-k probs / ids, [bs, num_steps, topk].
        num_draft_tokens: Tree node budget (including the root).

    Returns:
        draft_tokens [bs * num_draft_tokens], parent_list [bs, topk*(depth-1)+1],
        selected_index [bs, num_draft_tokens - 1] (all EAGLE-kernel layout).
    """
    if not (is_cuda() and topk_probs.is_cuda and topk in (4, 8, 16)):
        raise ValueError(
            "DFLASH tree verify requires CUDA and --speculative-eagle-topk in "
            f"{{4, 8, 16}} (fused tree kernel is power-of-2 only); got topk={topk}."
        )
    from sglang.kernels.ops.speculative.dflash_tree_expand_topk import (
        dflash_tree_verify_select_topk4_fused,
    )

    score_list_cat, ss_token_list, parent_list = dflash_tree_verify_select_topk4_fused(
        topk_probs, topk_ids
    )
    max_candidates = score_list_cat.shape[1]
    if num_draft_tokens - 1 > max_candidates:
        raise ValueError(
            "num_draft_tokens exceeds available candidates: "
            f"requested={num_draft_tokens - 1}, available={max_candidates}."
        )
    top_scores_index = torch.sort(
        fast_topk(score_list_cat, num_draft_tokens - 1, dim=-1).indices
    ).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    draft_tokens = torch.cat([verified_id[:, None], draft_tokens], dim=1).flatten()
    return draft_tokens, parent_list, top_scores_index
