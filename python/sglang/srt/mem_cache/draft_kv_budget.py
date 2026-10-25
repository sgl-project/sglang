from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.runtime_context import get_model, get_spec

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def resolve_dflash_draft_cell_size(
    *,
    draft_model_config: ModelConfig,
    draft_num_layers: int,
    attn_tp_size: int,
) -> int | None:
    """Bytes/token the DFLASH draft KV pool will cost the target's pool budget.

    Resolved from the draft's own attention geometry and the KV dtype the draft
    worker will actually resolve. Returns None if anything is unresolvable,
    leaving callers on layer-count scaling.
    """
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
    from sglang.srt.speculative.dflash_utils import dflash_draft_cell_size_per_token

    try:
        _, draft_kv_cache_dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=get_model().kv_cache_dtype,
            speculative_draft_kv_cache_dtype=(
                get_spec().speculative_draft_kv_cache_dtype
            ),
            model=None,
            model_dtype=draft_model_config.dtype,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend=(
                get_spec().speculative_draft_attention_backend
            ),
        )
        return dflash_draft_cell_size_per_token(
            draft_model_config=draft_model_config,
            draft_num_layers=draft_num_layers,
            draft_kv_cache_dtype=draft_kv_cache_dtype,
            tp_size=attn_tp_size,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not resolve DFLASH draft KV bytes/token (%s); falling back to "
            "layer-count scaling for the KV pool budget.",
            e,
        )
        return None
