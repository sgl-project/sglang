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


def _dtype_bytes(dtype: Any) -> Optional[int]:
    if dtype is None:
        return None
    return _DTYPE_BYTES.get(str(dtype))


def _to_mib(num_bytes: Optional[int]) -> Optional[float]:
    if num_bytes is None:
        return None
    return round(num_bytes / (1024 * 1024), 3)


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

    dtype_bytes = _dtype_bytes(kv_dtype)

    if (
        num_layers is None
        or num_kv_heads is None
        or head_dim is None
        or dtype_bytes is None
    ):
        return RelayKVMemoryEstimate(
            num_layers=num_layers,
            head_dim=head_dim,
            kv_dtype_bytes=dtype_bytes,
            kv_bytes_per_token=None,
            logical_kv_bytes=None,
            planned_resident_kv_bytes=None,
            planned_cold_kv_bytes=None,
            planned_resident_kv_mib=None,
            planned_cold_kv_mib=None,
            logical_kv_mib=None,
            kv_memory_estimate_reason=(
                "insufficient metadata for KV memory estimate"
            ),
        )

    kv_bytes_per_token = int(2 * num_kv_heads * head_dim * dtype_bytes * num_layers)
    logical_kv_bytes = int(plan.seq_len * kv_bytes_per_token)
    planned_resident_kv_bytes = int(plan.planned_resident_tokens * kv_bytes_per_token)
    planned_cold_kv_bytes = int(plan.planned_cold_tokens * kv_bytes_per_token)

    return RelayKVMemoryEstimate(
        num_layers=int(num_layers),
        head_dim=int(head_dim),
        kv_dtype_bytes=int(dtype_bytes),
        kv_bytes_per_token=kv_bytes_per_token,
        logical_kv_bytes=logical_kv_bytes,
        planned_resident_kv_bytes=planned_resident_kv_bytes,
        planned_cold_kv_bytes=planned_cold_kv_bytes,
        planned_resident_kv_mib=_to_mib(planned_resident_kv_bytes),
        planned_cold_kv_mib=_to_mib(planned_cold_kv_bytes),
        logical_kv_mib=_to_mib(logical_kv_bytes),
        kv_memory_estimate_reason="ok",
    )
