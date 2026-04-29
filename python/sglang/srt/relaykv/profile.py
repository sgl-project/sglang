from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RelayKVModelProfile:
    architecture: Optional[str]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    is_mha_or_gqa_candidate: bool
    note: str


def infer_model_profile(hf_config: Any) -> RelayKVModelProfile:
    """Infer a conservative model profile from a HuggingFace-style config.

    MVP-0 only uses this for logging/guardrails. It does not decide tensor layout.
    """

    architectures = getattr(hf_config, "architectures", None) or []
    architecture = architectures[0] if architectures else None
    n_heads = getattr(hf_config, "num_attention_heads", None)
    n_kv_heads = getattr(hf_config, "num_key_value_heads", None)

    # If num_key_value_heads is absent, many HF configs imply MHA.
    if n_heads is not None and n_kv_heads is None:
        is_candidate = True
        note = "num_key_value_heads missing; treating as MHA-style candidate"
    elif n_heads is not None and n_kv_heads is not None and n_kv_heads <= n_heads:
        is_candidate = True
        note = "MHA/GQA-style candidate"
    else:
        is_candidate = False
        note = "unsupported or unknown attention layout for RelayKV MVP-0"

    return RelayKVModelProfile(
        architecture=architecture,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        is_mha_or_gqa_candidate=is_candidate,
        note=note,
    )
