from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.dllm.config import DllmConfig
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def get_dllm_causal_attention(
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    dllm_config: DllmConfig | None,
    default_causal: bool,
) -> bool:
    """Return DLLM causal masking while preserving non-DLLM backend defaults."""
    if dllm_config is None or layer.attn_type != AttentionType.ENCODER_ONLY:
        return default_causal
    if layer.is_cross_attention:
        return False
    if not dllm_config.causal_context:
        return False
    return (
        not forward_batch.forward_mode.is_dllm_extend()
        or forward_batch.dllm_causal_kv_update
    )
