from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.distributed.utils import get_pp_indices

logger = logging.getLogger(__name__)


class DSparkDisaggMetadataConfig(msgspec.Struct, frozen=True):
    hidden_pool_size: int = 0
    hidden_size: int = 0
    hidden_device: str = "cpu"
    hidden_states_dtype: Optional[torch.dtype] = None
    target_layer_ids: Tuple[int, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.hidden_size > 0 and bool(self.target_layer_ids)

    def metadata_buffer_kwargs(self) -> dict:
        return {
            "pd_hidden_pool_size": self.hidden_pool_size,
            "pd_hidden_size": self.hidden_size,
            "pd_hidden_device": self.hidden_device,
        }


class PDHiddenBootstrapPlan(msgspec.Struct):
    hidden_start: int
    hidden_len: int
    streaming_hidden: bool
    dst_indices: List[int]
    local_layer_ids: List[int]
    local_slice_len: int
    source_window_rows: int
    pool: object


def _infer_target_layer_ids(model_runner: Any, hf_config: Any) -> List[int]:
    spec_aux_config = getattr(model_runner, "spec_aux_config", None)
    target_layer_ids = getattr(spec_aux_config, "dflash_target_layer_ids", None)
    if target_layer_ids:
        return [int(x) for x in target_layer_ids]

    env_layer_ids = os.getenv("SGLANG_DSPARK_PD_TARGET_LAYER_IDS")
    if env_layer_ids:
        return [int(x.strip()) for x in env_layer_ids.split(",") if x.strip()]

    try:
        from sglang.srt.speculative.dspark_components.dspark_config import (
            parse_dspark_draft_config,
        )

        cfg = parse_dspark_draft_config(draft_hf_config=hf_config)
        if cfg.target_layer_ids:
            return [int(x) for x in cfg.target_layer_ids]
    except Exception:
        logger.debug("Could not infer DSpark target layer ids.", exc_info=True)
    return []


def resolve_disagg_metadata_config(
    *,
    disaggregation_mode: Any,
    transfer_backend: Any,
    spec_algorithm: Any,
    model_config: Any,
    server_args: Any,
    model_runner: Any,
    pp_rank: int,
    pp_size: int,
    gpu_id: int,
    max_prefill_tokens: int,
) -> DSparkDisaggMetadataConfig:
    mode_value = getattr(disaggregation_mode, "value", disaggregation_mode)
    if mode_value not in ("decode", "prefill"):
        return DSparkDisaggMetadataConfig()

    should_probe = (
        spec_algorithm.is_dspark()
        or mode_value == "prefill"
        or os.getenv("SGLANG_DSPARK_PD_TARGET_LAYER_IDS")
    )
    if not should_probe:
        return DSparkDisaggMetadataConfig()

    target_layer_ids = _infer_target_layer_ids(model_runner, model_config.hf_config)
    if not target_layer_ids:
        return DSparkDisaggMetadataConfig()

    backend_value = getattr(transfer_backend, "value", transfer_backend)
    if backend_value not in ("mooncake", "fake"):
        raise NotImplementedError(
            "DSpark PD hidden transfer is implemented only for Mooncake/Fake "
            f"backends, got {backend_value}."
        )

    hidden_size = len(target_layer_ids) * int(model_config.hidden_size)
    default_pool_rows = max(
        1,
        int(server_args.max_prefill_buffer_tokens() or 0),
        int(max_prefill_tokens or 0),
    )
    pool_env_value = os.getenv("SGLANG_PD_HIDDEN_POOL_TOKENS")
    if mode_value == "decode":
        pool_env_value = os.getenv("SGLANG_PD_HIDDEN_RECV_POOL_TOKENS")
        if pool_env_value is None:
            pool_env_value = os.getenv("SGLANG_PD_HIDDEN_POOL_TOKENS")
    hidden_pool_size = max(
        0,
        int(pool_env_value if pool_env_value is not None else str(default_pool_rows)),
    )

    if mode_value == "prefill" and pp_size > 1:
        pp_start, pp_end = get_pp_indices(
            model_config.num_hidden_layers,
            pp_rank,
            pp_size,
        )
        owns_layer = any(pp_start <= layer_id < pp_end for layer_id in target_layer_ids)
        if not owns_layer:
            hidden_pool_size = 0

    return DSparkDisaggMetadataConfig(
        hidden_pool_size=hidden_pool_size,
        hidden_size=hidden_size,
        hidden_device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
        hidden_states_dtype=model_config.dtype,
        target_layer_ids=tuple(target_layer_ids),
    )


def resolve_hidden_bootstrap_plan(
    *,
    req: Any,
    metadata: dict,
    decode_prefix_len: int,
    pp_rank: int,
    model_config: Any,
    model_runner: Any,
    metadata_buffers: Any,
    prefill_radix_enabled: bool,
) -> Tuple[Optional[PDHiddenBootstrapPlan], Optional[str]]:
    hidden_start = int(metadata.get("hidden_start", 0))
    hidden_len = int(metadata.get("hidden_len", len(req.origin_input_ids)))
    if hidden_start != int(decode_prefix_len):
        return None, (
            "DSpark hidden metadata must align with decode radix prefix: "
            f"hidden_start={hidden_start}, decode_prefix_len={decode_prefix_len}, "
            f"rid={req.rid}"
        )

    decode_radix_enabled = bool(metadata.get("decode_radix_cache_enabled", False))
    if prefill_radix_enabled != decode_radix_enabled:
        return None, (
            "DSpark hidden PD requires matching prefill/decode radix cache "
            f"policies: prefill={prefill_radix_enabled}, "
            f"decode={decode_radix_enabled}, rid={req.rid}"
        )

    pp_slices = metadata.get("pp_slices") or {}
    local_pp_slice = pp_slices.get(str(pp_rank)) if pp_slices else None
    if pp_slices and local_pp_slice is None:
        return None, (
            "DSpark hidden metadata is missing PP slice for prefill rank: "
            f"pp_rank={pp_rank}, available_pp_slices={sorted(pp_slices.keys())}"
        )

    local_layer_ids = (
        [int(x) for x in local_pp_slice.get("layer_ids", [])]
        if local_pp_slice
        else (
            []
            if pp_slices
            else [int(x) for x in metadata.get("target_layer_ids", [])]
        )
    )
    local_slice_len = (
        int(local_pp_slice.get("slice_len", 0))
        if local_pp_slice
        else len(local_layer_ids) * int(model_config.hidden_size)
    )
    if not local_layer_ids:
        return (
            PDHiddenBootstrapPlan(
                hidden_start=hidden_start,
                hidden_len=hidden_len,
                streaming_hidden=bool(metadata.get("streaming_hidden", False)),
                dst_indices=[],
                local_layer_ids=[],
                local_slice_len=0,
                source_window_rows=0,
                pool=None,
            ),
            None,
        )

    expected_hidden_size = len(local_layer_ids) * int(model_config.hidden_size)
    if local_slice_len != expected_hidden_size:
        return None, (
            "DSpark hidden size mismatch on prefill: "
            f"layers={local_layer_ids}, expected={expected_hidden_size}, "
            f"metadata={local_slice_len}, pp_rank={pp_rank}"
        )

    spec_aux_config = getattr(model_runner, "spec_aux_config", None)
    configured_layer_ids = getattr(spec_aux_config, "dflash_target_layer_ids", None)
    all_target_layer_ids = [
        int(x) for x in metadata.get("target_layer_ids", local_layer_ids)
    ]
    if (
        configured_layer_ids is not None
        and list(configured_layer_ids) != all_target_layer_ids
    ):
        return None, (
            "DSpark target layer mismatch between prefill config and decode "
            f"metadata: prefill={configured_layer_ids}, decode={all_target_layer_ids}"
        )

    streaming_hidden = bool(metadata.get("streaming_hidden", False))
    dst_indices = [
        int(x)
        for x in (
            local_pp_slice.get("dst_indices", [])
            if local_pp_slice
            else metadata.get("dst_indices", [])
        )
    ]
    dst_len_valid = (
        0 < len(dst_indices) <= hidden_len
        if streaming_hidden
        else hidden_len == len(dst_indices)
    )
    if (
        not dst_len_valid
        or hidden_start < 0
        or hidden_len < 0
        or hidden_start + hidden_len > len(req.origin_input_ids)
    ):
        return None, (
            "Invalid DSpark hidden metadata from decode: "
            f"hidden_start={hidden_start}, hidden_len={hidden_len}, "
            f"dst_indices={len(dst_indices)}, "
            f"prompt_len={len(req.origin_input_ids)}, pp_rank={pp_rank}"
        )

    pool = getattr(metadata_buffers, "pd_hidden_pool", None)
    if pool is None:
        return None, (
            "DSpark hidden metadata targets a prefill PP rank without a hidden row "
            f"pool: pp_rank={pp_rank}, local_layer_ids={local_layer_ids}"
        )

    source_window_rows = (
        min(hidden_len, len(dst_indices)) if streaming_hidden else hidden_len
    )
    if source_window_rows > pool.size:
        return None, (
            "DSpark hidden rows exceed prefill hidden pool capacity: "
            f"rid={req.rid}, hidden_len={hidden_len}, "
            f"required_rows={source_window_rows}, pool_size={pool.size}"
        )

    return (
        PDHiddenBootstrapPlan(
            hidden_start=hidden_start,
            hidden_len=hidden_len,
            streaming_hidden=streaming_hidden,
            dst_indices=dst_indices,
            local_layer_ids=local_layer_ids,
            local_slice_len=local_slice_len,
            source_window_rows=source_window_rows,
            pool=pool,
        ),
        None,
    )
