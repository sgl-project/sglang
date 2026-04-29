from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RelayKVModelProfile:
    model_arch: Optional[str]
    attention_type: str
    relaykv_profile_supported: bool
    reason: str
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_attention_arch_name(attention_arch: Any) -> Optional[str]:
    if attention_arch is None:
        return None
    return getattr(attention_arch, "name", str(attention_arch))


def infer_model_profile(
    hf_config: Any, attention_arch: Any = None
) -> RelayKVModelProfile:
    """Infer a conservative RelayKV shadow-planning profile.

    MVP-0.3 keeps this intentionally shallow:
    - Qwen2.5-style standard MHA/GQA full attention is considered supported
    - unknown layouts remain shadow-ok but conservative
    - unsupported layouts only trigger warnings/logging, never behavioral changes
    """

    architectures = getattr(hf_config, "architectures", None) or []
    model_arch = architectures[0] if architectures else None
    n_heads = getattr(hf_config, "num_attention_heads", None)
    n_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    attention_arch_name = _normalize_attention_arch_name(attention_arch)

    if n_heads is None:
        return RelayKVModelProfile(
            model_arch=model_arch,
            attention_type="unknown",
            relaykv_profile_supported=False,
            reason="missing num_attention_heads; shadow-only logging remains conservative",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )

    if n_kv_heads is None or n_kv_heads == n_heads:
        attention_type = "mha"
    elif 1 < n_kv_heads < n_heads:
        attention_type = "gqa"
    elif n_kv_heads == 1:
        attention_type = "mqa"
    else:
        attention_type = "unknown"

    if attention_arch_name is not None and attention_arch_name != "MHA":
        return RelayKVModelProfile(
            model_arch=model_arch,
            attention_type=attention_type,
            relaykv_profile_supported=False,
            reason=f"attention_arch={attention_arch_name} is outside MVP-0.3 standard full-attention scope",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )

    if model_arch and model_arch.startswith("Qwen2") and attention_type in ("mha", "gqa"):
        return RelayKVModelProfile(
            model_arch=model_arch,
            attention_type=attention_type,
            relaykv_profile_supported=True,
            reason="Qwen2.5-style standard full attention is supported for shadow planning",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )

    if attention_type in ("mha", "gqa"):
        return RelayKVModelProfile(
            model_arch=model_arch,
            attention_type=attention_type,
            relaykv_profile_supported=True,
            reason="standard MHA/GQA full attention inferred; shadow-only logging is allowed",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )

    if attention_type == "unknown":
        return RelayKVModelProfile(
            model_arch=model_arch,
            attention_type=attention_type,
            relaykv_profile_supported=False,
            reason="unknown attention layout; shadow-only logging remains conservative",
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
        )

    return RelayKVModelProfile(
        model_arch=model_arch,
        attention_type=attention_type,
        relaykv_profile_supported=False,
        reason=f"{attention_type} is outside MVP-0.3 supported MHA/GQA scope",
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
    )
