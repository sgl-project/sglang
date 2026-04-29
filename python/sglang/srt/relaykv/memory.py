from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from .planner import RelayKVPlan


_DTYPE_BYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e4m3fnuz": 1,
    "torch.float8_e5m2": 1,
    "torch.int8": 1,
    "torch.uint8": 1,
}


@dataclass(frozen=True)
class RelayKVMemoryEstimate:
    num_layers: Optional[int]
    head_dim: Optional[int]
    kv_dtype_bytes: Optional[int]
    kv_bytes_per_token: Optional[int]
    logical_kv_bytes: Optional[int]
    planned_resident_kv_bytes: Optional[int]
    planned_cold_kv_bytes: Optional[int]
    planned_resident_kv_mib: Optional[float]
    planned_cold_kv_mib: Optional[float]
    logical_kv_mib: Optional[float]
    kv_memory_estimate_reason: str

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


RELAYKV_SHADOW_LOG_MEMORY_KEYS = {
    "num_layers",
    "head_dim",
    "kv_dtype_bytes",
    "kv_bytes_per_token",
    "logical_kv_bytes",
    "planned_resident_kv_bytes",
    "planned_cold_kv_bytes",
    "planned_resident_kv_mib",
    "planned_cold_kv_mib",
    "logical_kv_mib",
}

RELAYKV_SHADOW_LOG_HOST_BACKUP_KEYS = {
    "host_backup_shadow",
    "host_backup_candidate_tokens",
    "host_backup_candidate_kv_bytes",
    "host_backup_candidate_kv_mib",
    "resident_anchor_ranges",
    "resident_recent_ranges",
    "cold_candidate_ranges",
    "host_backup_copy_target_ranges",
    "host_backup_copy_target_tokens",
    "host_backup_copy_target_reason",
    "host_backup_max_mib",
    "host_backup_budget_ok",
    "host_backup_would_copy",
    "host_backup_reason",
    "host_backup_dry_copy",
    "host_backup_dry_copy_guard_ok",
    "host_backup_dry_copy_would_run",
    "host_backup_dry_copy_reason",
    "kv_layout_observed",
    "kv_layout_object_type",
    "kv_layout_k_shape",
    "kv_layout_v_shape",
    "kv_layout_dtype",
    "kv_layout_device",
    "kv_layout_num_layers_observed",
    "kv_layout_reason",
    "kv_layout_range_mapping_supported",
    "kv_layout_range_mapping_reason",
    "kv_pool_mapping_observed",
    "kv_pool_mapping_reason",
    "kv_pool_mapping_object_type",
    "kv_pool_mapping_shape",
    "kv_pool_mapping_dtype",
    "kv_pool_mapping_device",
    "request_pool_indices_count",
    "request_pool_indices_preview_head",
    "request_pool_indices_preview_tail",
    "cold_range_pool_indices_preview",
    "cold_range_pool_indices_count",
    "cold_range_pool_mapping_supported",
    "cold_range_pool_mapping_reason",
    "mapping_valid_count",
    "mapping_zero_count",
    "mapping_invalid_count",
    "mapping_ready_for_copy",
    "mapping_readiness_reason",
    "prefill_pending_tokens",
    "prefill_complete_for_request",
    "host_backup_dry_copy_final_guard_ok",
    "host_backup_dry_copy_final_guard_reason",
}


@dataclass(frozen=True)
class RelayKVHostBackupShadowEstimate:
    host_backup_shadow: bool
    host_backup_candidate_tokens: int
    host_backup_candidate_kv_bytes: Optional[int]
    host_backup_candidate_kv_mib: Optional[float]
    resident_anchor_ranges: list[list[int]]
    resident_recent_ranges: list[list[int]]
    cold_candidate_ranges: list[list[int]]
    host_backup_copy_target_ranges: list[list[int]]
    host_backup_copy_target_tokens: int
    host_backup_copy_target_reason: str
    host_backup_max_mib: float
    host_backup_budget_ok: Optional[bool]
    host_backup_would_copy: bool
    host_backup_reason: str
    host_backup_dry_copy: bool
    host_backup_dry_copy_guard_ok: bool
    host_backup_dry_copy_would_run: bool
    host_backup_dry_copy_reason: str

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayKVLayoutObservation:
    kv_layout_observed: bool
    kv_layout_object_type: Optional[str]
    kv_layout_k_shape: Optional[list[int]]
    kv_layout_v_shape: Optional[list[int]]
    kv_layout_dtype: Optional[str]
    kv_layout_device: Optional[str]
    kv_layout_num_layers_observed: Optional[int]
    kv_layout_reason: str
    kv_layout_range_mapping_supported: bool
    kv_layout_range_mapping_reason: str

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayKVPoolMappingObservation:
    kv_pool_mapping_observed: bool
    kv_pool_mapping_reason: str
    kv_pool_mapping_object_type: Optional[str]
    kv_pool_mapping_shape: Optional[list[int]]
    kv_pool_mapping_dtype: Optional[str]
    kv_pool_mapping_device: Optional[str]
    request_pool_indices_count: int
    request_pool_indices_preview_head: list[int]
    request_pool_indices_preview_tail: list[int]
    cold_range_pool_indices_preview: list[int]
    cold_range_pool_indices_count: int
    cold_range_pool_mapping_supported: bool
    cold_range_pool_mapping_reason: str
    mapping_valid_count: int
    mapping_zero_count: int
    mapping_invalid_count: int
    mapping_ready_for_copy: bool
    mapping_readiness_reason: str
    prefill_pending_tokens: Optional[int]
    prefill_complete_for_request: Optional[bool]
    host_backup_dry_copy_final_guard_ok: bool
    host_backup_dry_copy_final_guard_reason: str

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype_bytes(dtype: Any) -> Optional[int]:
    if dtype is None:
        return None
    return _DTYPE_BYTES.get(str(dtype))


def _to_mib(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return round(num_bytes / (1024 * 1024), 3)


def _range_tokens(ranges: list[list[int]]) -> int:
    total = 0
    for start, end in ranges:
        total += max(end - start, 0)
    return total


def _preview_values(tensor: Any, limit: int) -> list[int]:
    if tensor is None:
        return []
    if getattr(tensor, "numel", lambda: 0)() == 0:
        return []
    return [int(x) for x in tensor[:limit].tolist()]


def _tensor_shape_list(tensor: Any) -> Optional[list[int]]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _extract_buffer_sample(buffer: Any) -> tuple[Optional[Any], Optional[int]]:
    if buffer is None:
        return None, None
    if isinstance(buffer, list):
        if not buffer:
            return None, 0
        sample = buffer[0]
        if hasattr(sample, "shape"):
            return sample, len(buffer)
        return None, len(buffer)
    if hasattr(buffer, "shape"):
        return buffer, None
    return None, None


def _resolve_kv_layout_object(token_to_kv_pool_allocator: Any) -> tuple[Optional[Any], str]:
    if token_to_kv_pool_allocator is None:
        return None, "kv_cache_object_not_found"
    if hasattr(token_to_kv_pool_allocator, "get_kvcache"):
        kvcache = token_to_kv_pool_allocator.get_kvcache()
    else:
        kvcache = getattr(token_to_kv_pool_allocator, "_kvcache", None)
    if kvcache is None:
        return None, "kv_cache_object_not_found"
    if hasattr(kvcache, "full_kv_pool"):
        return getattr(kvcache, "full_kv_pool"), "ok_swa_full_kv_pool"
    return kvcache, "ok"


def estimate_kv_memory_from_metadata(
    *,
    seq_len: int,
    planned_resident_tokens: int,
    planned_cold_tokens: int,
    num_layers: Optional[int],
    num_key_value_heads: Optional[int],
    head_dim: Optional[int],
    kv_dtype_bytes: Optional[int],
) -> RelayKVMemoryEstimate:
    if (
        num_layers is None
        or num_key_value_heads is None
        or head_dim is None
        or kv_dtype_bytes is None
    ):
        return RelayKVMemoryEstimate(
            num_layers=num_layers,
            head_dim=head_dim,
            kv_dtype_bytes=kv_dtype_bytes,
            kv_bytes_per_token=None,
            logical_kv_bytes=None,
            planned_resident_kv_bytes=None,
            planned_cold_kv_bytes=None,
            planned_resident_kv_mib=None,
            planned_cold_kv_mib=None,
            logical_kv_mib=None,
            kv_memory_estimate_reason="insufficient metadata for KV memory estimate",
        )

    kv_bytes_per_token = int(
        2 * num_key_value_heads * head_dim * kv_dtype_bytes * num_layers
    )
    logical_kv_bytes = int(seq_len * kv_bytes_per_token)
    planned_resident_kv_bytes = int(planned_resident_tokens * kv_bytes_per_token)
    planned_cold_kv_bytes = int(planned_cold_tokens * kv_bytes_per_token)

    return RelayKVMemoryEstimate(
        num_layers=int(num_layers),
        head_dim=int(head_dim),
        kv_dtype_bytes=int(kv_dtype_bytes),
        kv_bytes_per_token=kv_bytes_per_token,
        logical_kv_bytes=logical_kv_bytes,
        planned_resident_kv_bytes=planned_resident_kv_bytes,
        planned_cold_kv_bytes=planned_cold_kv_bytes,
        planned_resident_kv_mib=_to_mib(planned_resident_kv_bytes),
        planned_cold_kv_mib=_to_mib(planned_cold_kv_bytes),
        logical_kv_mib=_to_mib(logical_kv_bytes),
        kv_memory_estimate_reason="ok",
    )


def estimate_kv_memory_for_plan(
    plan: RelayKVPlan,
    *,
    model_config: Any,
    kv_dtype: Any,
) -> RelayKVMemoryEstimate:
    num_layers = getattr(model_config, "num_hidden_layers", None)
    num_kv_heads = getattr(model_config, "num_key_value_heads", None)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(model_config, "hidden_size", None)
        num_attention_heads = getattr(model_config, "num_attention_heads", None)
        if hidden_size is not None and num_attention_heads:
            head_dim = hidden_size // num_attention_heads

    return estimate_kv_memory_from_metadata(
        seq_len=plan.seq_len,
        planned_resident_tokens=plan.planned_resident_tokens,
        planned_cold_tokens=plan.planned_cold_tokens,
        num_layers=num_layers,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype_bytes=_dtype_bytes(kv_dtype),
    )


def estimate_host_backup_shadow_for_plan(
    plan: RelayKVPlan,
    *,
    memory_estimate: RelayKVMemoryEstimate,
    host_backup_shadow: bool,
    host_backup_max_mib: float,
    host_backup_dry_copy: bool,
) -> RelayKVHostBackupShadowEstimate:
    if not host_backup_shadow:
        return RelayKVHostBackupShadowEstimate(
            host_backup_shadow=False,
            host_backup_candidate_tokens=0,
            host_backup_candidate_kv_bytes=0,
            host_backup_candidate_kv_mib=0.0,
            resident_anchor_ranges=plan.resident_anchor_ranges,
            resident_recent_ranges=plan.resident_recent_ranges,
            cold_candidate_ranges=plan.cold_candidate_ranges,
            host_backup_copy_target_ranges=[],
            host_backup_copy_target_tokens=0,
            host_backup_copy_target_reason="host_backup_shadow_disabled",
            host_backup_max_mib=host_backup_max_mib,
            host_backup_budget_ok=True,
            host_backup_would_copy=False,
            host_backup_reason="host_backup_shadow_disabled",
            host_backup_dry_copy=host_backup_dry_copy,
            host_backup_dry_copy_guard_ok=False,
            host_backup_dry_copy_would_run=False,
            host_backup_dry_copy_reason="dry_copy_disabled",
        )

    candidate_kv_bytes = memory_estimate.planned_cold_kv_bytes
    candidate_kv_mib = memory_estimate.planned_cold_kv_mib
    copy_target_ranges = plan.cold_candidate_ranges
    copy_target_tokens = _range_tokens(copy_target_ranges)
    if copy_target_tokens != plan.planned_cold_tokens:
        copy_target_reason = (
            "metadata_only_no_tensor_copy_range_token_mismatch"
        )
    else:
        copy_target_reason = "metadata_only_no_tensor_copy"
    if candidate_kv_mib is None:
        budget_ok = None
        reason = "metadata_only_no_tensor_copy_insufficient_memory_metadata"
    elif host_backup_max_mib == 0.0:
        budget_ok = True
        reason = "metadata_only_no_tensor_copy"
    else:
        budget_ok = candidate_kv_mib <= host_backup_max_mib
        reason = "metadata_only_no_tensor_copy"

    dry_copy_guard_ok = bool(
        host_backup_shadow
        and host_backup_dry_copy
        and budget_ok is True
        and copy_target_tokens > 0
    )
    if not host_backup_dry_copy:
        dry_copy_reason = "dry_copy_disabled"
    elif budget_ok is False:
        dry_copy_reason = "host_backup_budget_exceeded"
    elif copy_target_tokens <= 0:
        dry_copy_reason = "no_copy_target_tokens"
    elif budget_ok is None:
        dry_copy_reason = "insufficient_memory_metadata"
    else:
        dry_copy_reason = "guard_only_no_tensor_copy"

    return RelayKVHostBackupShadowEstimate(
        host_backup_shadow=True,
        host_backup_candidate_tokens=plan.planned_cold_tokens,
        host_backup_candidate_kv_bytes=candidate_kv_bytes,
        host_backup_candidate_kv_mib=candidate_kv_mib,
        resident_anchor_ranges=plan.resident_anchor_ranges,
        resident_recent_ranges=plan.resident_recent_ranges,
        cold_candidate_ranges=plan.cold_candidate_ranges,
        host_backup_copy_target_ranges=copy_target_ranges,
        host_backup_copy_target_tokens=copy_target_tokens,
        host_backup_copy_target_reason=copy_target_reason,
        host_backup_max_mib=host_backup_max_mib,
        host_backup_budget_ok=budget_ok,
        host_backup_would_copy=False,
        host_backup_reason=reason,
        host_backup_dry_copy=host_backup_dry_copy,
        host_backup_dry_copy_guard_ok=dry_copy_guard_ok,
        host_backup_dry_copy_would_run=dry_copy_guard_ok,
        host_backup_dry_copy_reason=dry_copy_reason,
    )


def observe_kv_layout_for_host_backup(
    *,
    token_to_kv_pool_allocator: Any,
    req_to_token_pool: Any,
    request_id: Optional[str],
    seq_len: int,
    copy_target_ranges: list[list[int]],
) -> RelayKVLayoutObservation:
    kv_object, base_reason = _resolve_kv_layout_object(token_to_kv_pool_allocator)
    if kv_object is None:
        return RelayKVLayoutObservation(
            kv_layout_observed=False,
            kv_layout_object_type=None,
            kv_layout_k_shape=None,
            kv_layout_v_shape=None,
            kv_layout_dtype=None,
            kv_layout_device=None,
            kv_layout_num_layers_observed=None,
            kv_layout_reason=base_reason,
            kv_layout_range_mapping_supported=False,
            kv_layout_range_mapping_reason="kv_cache_object_not_found",
        )

    k_sample, k_layers = _extract_buffer_sample(getattr(kv_object, "k_buffer", None))
    v_sample, v_layers = _extract_buffer_sample(getattr(kv_object, "v_buffer", None))
    observe_reason = base_reason

    if k_sample is None and v_sample is None:
        kv_sample, kv_layers = _extract_buffer_sample(getattr(kv_object, "kv_buffer", None))
        if kv_sample is None:
            return RelayKVLayoutObservation(
                kv_layout_observed=False,
                kv_layout_object_type=type(kv_object).__name__,
                kv_layout_k_shape=None,
                kv_layout_v_shape=None,
                kv_layout_dtype=str(getattr(kv_object, "dtype", None))
                if getattr(kv_object, "dtype", None) is not None
                else None,
                kv_layout_device=str(getattr(kv_object, "device", None))
                if getattr(kv_object, "device", None) is not None
                else None,
                kv_layout_num_layers_observed=kv_layers,
                kv_layout_reason="kv_layout_buffer_attrs_not_found",
                kv_layout_range_mapping_supported=False,
                kv_layout_range_mapping_reason="kv_layout_not_observed",
            )
        k_shape = _tensor_shape_list(kv_sample)
        return RelayKVLayoutObservation(
            kv_layout_observed=True,
            kv_layout_object_type=type(kv_object).__name__,
            kv_layout_k_shape=k_shape,
            kv_layout_v_shape=None,
            kv_layout_dtype=str(getattr(kv_sample, "dtype", getattr(kv_object, "dtype", None))),
            kv_layout_device=str(
                getattr(kv_sample, "device", getattr(kv_object, "device", None))
            ),
            kv_layout_num_layers_observed=kv_layers,
            kv_layout_reason="combined_kv_buffer_only",
            kv_layout_range_mapping_supported=bool(
                request_id and seq_len > 0 and copy_target_ranges
                and getattr(token_to_kv_pool_allocator, "page_size", None) == 1
                and req_to_token_pool is not None
            ),
            kv_layout_range_mapping_reason=(
                "page_size_1_metadata_only"
                if (
                    request_id
                    and seq_len > 0
                    and copy_target_ranges
                    and getattr(token_to_kv_pool_allocator, "page_size", None) == 1
                    and req_to_token_pool is not None
                )
                else "combined_kv_buffer_range_mapping_unknown"
            ),
        )

    sample = k_sample if k_sample is not None else v_sample
    page_size = getattr(token_to_kv_pool_allocator, "page_size", None)
    range_mapping_supported = bool(
        request_id
        and seq_len > 0
        and copy_target_ranges
        and page_size == 1
        and req_to_token_pool is not None
    )
    if not request_id:
        range_mapping_reason = "request_id_missing"
    elif seq_len <= 0:
        range_mapping_reason = "seq_len_missing"
    elif not copy_target_ranges:
        range_mapping_reason = "copy_target_ranges_empty"
    elif page_size != 1:
        range_mapping_reason = "page_size_not_token_aligned"
    elif req_to_token_pool is None:
        range_mapping_reason = "req_to_token_pool_not_found"
    else:
        range_mapping_reason = "page_size_1_metadata_only"

    return RelayKVLayoutObservation(
        kv_layout_observed=True,
        kv_layout_object_type=type(kv_object).__name__,
        kv_layout_k_shape=_tensor_shape_list(k_sample),
        kv_layout_v_shape=_tensor_shape_list(v_sample),
        kv_layout_dtype=str(getattr(sample, "dtype", getattr(kv_object, "dtype", None))),
        kv_layout_device=str(
            getattr(sample, "device", getattr(kv_object, "device", None))
        ),
        kv_layout_num_layers_observed=k_layers or v_layers,
        kv_layout_reason=observe_reason,
        kv_layout_range_mapping_supported=range_mapping_supported,
        kv_layout_range_mapping_reason=range_mapping_reason,
    )


def observe_request_kv_pool_mapping(
    *,
    req_to_token_pool: Any,
    request_pool_idx: Optional[int],
    seq_len: int,
    observed_token_count: Optional[int],
    cold_candidate_ranges: list[list[int]],
    preview_limit: int = 8,
) -> RelayKVPoolMappingObservation:
    if req_to_token_pool is None:
        return RelayKVPoolMappingObservation(
            kv_pool_mapping_observed=False,
            kv_pool_mapping_reason="req_to_token_pool_not_found",
            kv_pool_mapping_object_type=None,
            kv_pool_mapping_shape=None,
            kv_pool_mapping_dtype=None,
            kv_pool_mapping_device=None,
            request_pool_indices_count=0,
            request_pool_indices_preview_head=[],
            request_pool_indices_preview_tail=[],
            cold_range_pool_indices_preview=[],
            cold_range_pool_indices_count=0,
            cold_range_pool_mapping_supported=False,
            cold_range_pool_mapping_reason="req_to_token_pool_not_found",
            mapping_valid_count=0,
            mapping_zero_count=0,
            mapping_invalid_count=0,
            mapping_ready_for_copy=False,
            mapping_readiness_reason="req_to_token_pool_not_found",
            prefill_pending_tokens=None,
            prefill_complete_for_request=None,
            host_backup_dry_copy_final_guard_ok=False,
            host_backup_dry_copy_final_guard_reason="req_to_token_pool_not_found",
        )
    if request_pool_idx is None:
        return RelayKVPoolMappingObservation(
            kv_pool_mapping_observed=False,
            kv_pool_mapping_reason="request_pool_idx_missing",
            kv_pool_mapping_object_type=type(req_to_token_pool).__name__,
            kv_pool_mapping_shape=None,
            kv_pool_mapping_dtype=None,
            kv_pool_mapping_device=None,
            request_pool_indices_count=0,
            request_pool_indices_preview_head=[],
            request_pool_indices_preview_tail=[],
            cold_range_pool_indices_preview=[],
            cold_range_pool_indices_count=0,
            cold_range_pool_mapping_supported=False,
            cold_range_pool_mapping_reason="request_pool_idx_missing",
            mapping_valid_count=0,
            mapping_zero_count=0,
            mapping_invalid_count=0,
            mapping_ready_for_copy=False,
            mapping_readiness_reason="request_pool_idx_missing",
            prefill_pending_tokens=None,
            prefill_complete_for_request=None,
            host_backup_dry_copy_final_guard_ok=False,
            host_backup_dry_copy_final_guard_reason="request_pool_idx_missing",
        )
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    if req_to_token is None:
        return RelayKVPoolMappingObservation(
            kv_pool_mapping_observed=False,
            kv_pool_mapping_reason="req_to_token_tensor_not_found",
            kv_pool_mapping_object_type=type(req_to_token_pool).__name__,
            kv_pool_mapping_shape=None,
            kv_pool_mapping_dtype=None,
            kv_pool_mapping_device=None,
            request_pool_indices_count=0,
            request_pool_indices_preview_head=[],
            request_pool_indices_preview_tail=[],
            cold_range_pool_indices_preview=[],
            cold_range_pool_indices_count=0,
            cold_range_pool_mapping_supported=False,
            cold_range_pool_mapping_reason="req_to_token_tensor_not_found",
            mapping_valid_count=0,
            mapping_zero_count=0,
            mapping_invalid_count=0,
            mapping_ready_for_copy=False,
            mapping_readiness_reason="req_to_token_tensor_not_found",
            prefill_pending_tokens=None,
            prefill_complete_for_request=None,
            host_backup_dry_copy_final_guard_ok=False,
            host_backup_dry_copy_final_guard_reason="req_to_token_tensor_not_found",
        )

    mapping_shape = _tensor_shape_list(req_to_token)
    mapping_dtype = str(getattr(req_to_token, "dtype", None))
    mapping_device = str(getattr(req_to_token, "device", None))
    if len(req_to_token.shape) < 2:
        return RelayKVPoolMappingObservation(
            kv_pool_mapping_observed=False,
            kv_pool_mapping_reason="req_to_token_tensor_rank_unsupported",
            kv_pool_mapping_object_type=type(req_to_token_pool).__name__,
            kv_pool_mapping_shape=mapping_shape,
            kv_pool_mapping_dtype=mapping_dtype,
            kv_pool_mapping_device=mapping_device,
            request_pool_indices_count=0,
            request_pool_indices_preview_head=[],
            request_pool_indices_preview_tail=[],
            cold_range_pool_indices_preview=[],
            cold_range_pool_indices_count=0,
            cold_range_pool_mapping_supported=False,
            cold_range_pool_mapping_reason="req_to_token_tensor_rank_unsupported",
            mapping_valid_count=0,
            mapping_zero_count=0,
            mapping_invalid_count=0,
            mapping_ready_for_copy=False,
            mapping_readiness_reason="req_to_token_tensor_rank_unsupported",
            prefill_pending_tokens=None,
            prefill_complete_for_request=None,
            host_backup_dry_copy_final_guard_ok=False,
            host_backup_dry_copy_final_guard_reason="req_to_token_tensor_rank_unsupported",
        )
    if request_pool_idx < 0 or request_pool_idx >= int(req_to_token.shape[0]):
        return RelayKVPoolMappingObservation(
            kv_pool_mapping_observed=False,
            kv_pool_mapping_reason="request_pool_idx_out_of_bounds",
            kv_pool_mapping_object_type=type(req_to_token_pool).__name__,
            kv_pool_mapping_shape=mapping_shape,
            kv_pool_mapping_dtype=mapping_dtype,
            kv_pool_mapping_device=mapping_device,
            request_pool_indices_count=0,
            request_pool_indices_preview_head=[],
            request_pool_indices_preview_tail=[],
            cold_range_pool_indices_preview=[],
            cold_range_pool_indices_count=0,
            cold_range_pool_mapping_supported=False,
            cold_range_pool_mapping_reason="request_pool_idx_out_of_bounds",
            mapping_valid_count=0,
            mapping_zero_count=0,
            mapping_invalid_count=0,
            mapping_ready_for_copy=False,
            mapping_readiness_reason="request_pool_idx_out_of_bounds",
            prefill_pending_tokens=None,
            prefill_complete_for_request=None,
            host_backup_dry_copy_final_guard_ok=False,
            host_backup_dry_copy_final_guard_reason="request_pool_idx_out_of_bounds",
        )

    total_seq_len = max(int(seq_len), 0)
    if observed_token_count is None:
        observed_count = min(total_seq_len, int(req_to_token.shape[1]))
        pending_tokens = None
        prefill_complete = None
    else:
        observed_count = max(min(int(observed_token_count), int(req_to_token.shape[1])), 0)
        pending_tokens = max(total_seq_len - observed_count, 0)
        prefill_complete = pending_tokens == 0

    request_pool_indices = req_to_token[request_pool_idx, :observed_count]
    request_pool_indices_count = int(request_pool_indices.numel())
    request_pool_indices_preview_head = _preview_values(request_pool_indices, preview_limit)
    request_pool_indices_preview_tail = _preview_values(
        request_pool_indices[-preview_limit:], preview_limit
    )

    cold_preview_segments = []
    cold_range_pool_indices_count = 0
    cold_range_mapping_supported = True
    mapping_valid_count = 0
    mapping_zero_count = 0
    mapping_invalid_count = 0
    if not cold_candidate_ranges:
        cold_range_mapping_reason = "cold_candidate_ranges_empty"
        cold_range_mapping_supported = False
    else:
        cold_range_mapping_reason = "ok"
        for start, end in cold_candidate_ranges:
            if start < 0 or end < start:
                cold_range_mapping_supported = False
                cold_range_mapping_reason = "cold_candidate_range_invalid"
                break
            if end > total_seq_len:
                cold_range_mapping_supported = False
                cold_range_mapping_reason = "cold_candidate_range_exceeds_request_seqlen"
                break
            range_total = max(end - start, 0)
            cold_range_pool_indices_count += range_total
            observed_start = min(start, observed_count)
            observed_end = min(end, observed_count)
            observed_range = request_pool_indices[observed_start:observed_end]
            if observed_range.numel() > 0:
                mapping_zero_count += int((observed_range == 0).sum().item())
                mapping_invalid_count += int((observed_range < 0).sum().item())
                mapping_valid_count += int((observed_range > 0).sum().item())
            if end > observed_count:
                mapping_invalid_count += end - max(start, observed_count)
                cold_range_mapping_supported = False
                cold_range_mapping_reason = "prefill_incomplete_for_cold_range"
            if len(cold_preview_segments) < preview_limit:
                cold_preview_segments.append(observed_range)
        if cold_range_mapping_supported and cold_range_pool_indices_count <= 0:
            cold_range_mapping_supported = False
            cold_range_mapping_reason = "cold_candidate_ranges_empty_after_clip"

    if cold_preview_segments:
        cold_preview_tensor = cold_preview_segments[0]
        for segment in cold_preview_segments[1:]:
            remaining = preview_limit - int(cold_preview_tensor.numel())
            if remaining <= 0:
                break
            cold_preview_tensor = cold_preview_tensor.new_tensor(
                cold_preview_tensor.tolist() + segment[:remaining].tolist()
            )
        cold_range_pool_indices_preview = _preview_values(cold_preview_tensor, preview_limit)
    else:
        cold_range_pool_indices_preview = []

    if mapping_zero_count > 0:
        mapping_ready_for_copy = False
        mapping_readiness_reason = "mapping_contains_zero_entries"
    elif mapping_invalid_count > 0:
        mapping_ready_for_copy = False
        if pending_tokens and pending_tokens > 0:
            mapping_readiness_reason = "prefill_incomplete_pending_tokens"
        else:
            mapping_readiness_reason = "mapping_contains_invalid_entries"
    elif mapping_valid_count == cold_range_pool_indices_count and mapping_zero_count == 0:
        mapping_ready_for_copy = True
        mapping_readiness_reason = "ready_for_copy_metadata_only"
    else:
        mapping_ready_for_copy = False
        mapping_readiness_reason = "mapping_count_mismatch"

    if prefill_complete is False:
        final_guard_ok = False
        final_guard_reason = "prefill_not_complete"
    elif not cold_range_mapping_supported:
        final_guard_ok = False
        final_guard_reason = cold_range_mapping_reason
    elif not mapping_ready_for_copy:
        final_guard_ok = False
        final_guard_reason = mapping_readiness_reason
    else:
        final_guard_ok = True
        final_guard_reason = "ready_for_execution_metadata_only"

    return RelayKVPoolMappingObservation(
        kv_pool_mapping_observed=True,
        kv_pool_mapping_reason="ok",
        kv_pool_mapping_object_type=type(req_to_token_pool).__name__,
        kv_pool_mapping_shape=mapping_shape,
        kv_pool_mapping_dtype=mapping_dtype,
        kv_pool_mapping_device=mapping_device,
        request_pool_indices_count=request_pool_indices_count,
        request_pool_indices_preview_head=request_pool_indices_preview_head,
        request_pool_indices_preview_tail=request_pool_indices_preview_tail,
        cold_range_pool_indices_preview=cold_range_pool_indices_preview,
        cold_range_pool_indices_count=cold_range_pool_indices_count,
        cold_range_pool_mapping_supported=cold_range_mapping_supported,
        cold_range_pool_mapping_reason=cold_range_mapping_reason,
        mapping_valid_count=mapping_valid_count,
        mapping_zero_count=mapping_zero_count,
        mapping_invalid_count=mapping_invalid_count,
        mapping_ready_for_copy=mapping_ready_for_copy,
        mapping_readiness_reason=mapping_readiness_reason,
        prefill_pending_tokens=pending_tokens,
        prefill_complete_for_request=prefill_complete,
        host_backup_dry_copy_final_guard_ok=final_guard_ok,
        host_backup_dry_copy_final_guard_reason=final_guard_reason,
    )


def validate_shadow_log_schema(payload: dict[str, Any]) -> None:
    missing = RELAYKV_SHADOW_LOG_MEMORY_KEYS - payload.keys()
    if missing:
        raise AssertionError(f"Missing RelayKV shadow log keys: {sorted(missing)}")
    missing_host_backup = RELAYKV_SHADOW_LOG_HOST_BACKUP_KEYS - payload.keys()
    if missing_host_backup:
        raise AssertionError(
            f"Missing RelayKV host-backup log keys: {sorted(missing_host_backup)}"
        )
