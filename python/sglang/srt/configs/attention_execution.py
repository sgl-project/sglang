from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def attention_dcp_size(
    *,
    backend: str,
    model_config: ModelConfig,
    dcp_size: int,
    is_draft_worker: bool,
    is_dspark: bool,
    is_hip: bool,
) -> int:
    """Execution width, shared by attention construction and KV planning."""
    if backend == "triton":
        return 1 if is_hip and is_draft_worker and is_dspark else dcp_size

    from sglang.srt.configs.model_config import AttentionArch, mla_attention_dcp_size

    if model_config.attention_arch == AttentionArch.MLA:
        return mla_attention_dcp_size(
            model_config.hf_config, dcp_size=dcp_size, is_nextn=is_draft_worker
        )
    return 1
