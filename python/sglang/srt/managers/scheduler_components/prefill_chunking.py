"""How prefill is chunked: the per-rank chunk size and the alignment every
chunk start keeps."""

from __future__ import annotations

import logging
import math
from typing import Optional

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import mamba_prefill_chunk_grid
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)


def resolve_chunked_prefill_size(model_config) -> Optional[int]:
    """The chunk size, or None when prefill is not chunked."""
    chunked_prefill_size = get_schedule().chunked_prefill_size
    uses_transformers_backend = (
        get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
    )
    if (
        chunked_prefill_size is not None
        and chunked_prefill_size > 0
        and model_config.is_multimodal
        and uses_transformers_backend
    ):
        logger.warning(
            "Chunked prefill is disabled for multimodal models with the "
            "Transformers backend to avoid partial multimodal chunk mismatches."
        )
        chunked_prefill_size = None
    elif chunked_prefill_size is not None and chunked_prefill_size <= 0:
        chunked_prefill_size = None
    return chunked_prefill_size


def resolve_truncation_align_size(
    *,
    hf_config,
    chunked_prefill_size: Optional[int],
    disaggregation_mode: DisaggregationMode,
) -> Optional[int]:
    """The alignment of chunk starts, or None; raises when a chunk is shorter."""
    truncation_align_size = deterministic_truncation_align_size()
    truncation_align_size = with_dsa_kpool_align(truncation_align_size, hf_config)
    truncation_align_size = with_mamba_chunk_align(truncation_align_size)
    check_truncation_align_fits_chunk(
        truncation_align_size, chunked_prefill_size, disaggregation_mode
    )
    return truncation_align_size


def deterministic_truncation_align_size() -> Optional[int]:
    """The chunk alignment deterministic inference needs from the prefill
    attention backend, or None."""
    if not get_exec().deterministic.enable_deterministic_inference:
        return None

    # Only flashinfer needs chunks on its prefill split tiles; triton tiles
    # keys at absolute positions (align_window_kv_to_tiles).
    backend_sizes = {
        "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
    }
    prefill_backend, _ = attention_backends()
    env_var, default_size = backend_sizes.get(prefill_backend, (None, None))
    return get_int_env_var(env_var, default_size) if env_var else None


def with_dsa_kpool_align(
    truncation_align_size: Optional[int], hf_config
) -> Optional[int]:
    """Kpool compress-write asserts chunked extends start on pool boundaries.
    Use the LCM to preserve any existing deterministic-inference alignment."""
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        is_deepseek_dsa,
    )

    if not is_deepseek_dsa(hf_config):
        return truncation_align_size

    dsa_index_kpool = get_dsa_index_kpool(hf_config)
    if dsa_index_kpool <= 1:
        return truncation_align_size

    if truncation_align_size is None:
        truncation_align_size = dsa_index_kpool
    else:
        truncation_align_size = math.lcm(truncation_align_size, dsa_index_kpool)
    return truncation_align_size


def with_mamba_chunk_align(truncation_align_size: Optional[int]) -> Optional[int]:
    """Under DCP a mamba checkpoint may only be donated where the cached
    prefix sits on the kernel chunk grid, so a chunk that stops anywhere
    else costs the request every donation it would have made after that.
    Without DCP the prefix-relative rule already donates, so leave the
    existing alignment alone. Use the LCM to preserve any alignment that
    is already set."""
    chunk_size = mamba_prefill_chunk_grid()
    if chunk_size <= 1:
        return truncation_align_size

    if truncation_align_size is None:
        truncation_align_size = chunk_size
    else:
        truncation_align_size = math.lcm(truncation_align_size, chunk_size)
    return truncation_align_size


def check_truncation_align_fits_chunk(
    truncation_align_size: Optional[int],
    chunked_prefill_size: Optional[int],
    disaggregation_mode: DisaggregationMode,
) -> None:
    """A chunk shorter than the alignment truncates to zero tokens, so a prompt
    longer than the chunk would wait in the queue forever."""
    align = truncation_align_size
    chunk = chunked_prefill_size
    if align is None or chunk is None or chunk >= align:
        return
    if disaggregation_mode == DisaggregationMode.DECODE:
        # A decode server never prefills a prompt.
        return
    attn_dp_size = get_parallel().attn_dp_size
    dp_note = (
        f" (--chunked-prefill-size is split across {attn_dp_size} DP attention ranks)"
        if attn_dp_size > 1
        else ""
    )
    raise ValueError(
        f"Chunked prefill is aligned to {align} tokens, but each rank's "
        f"chunked prefill size is {chunk}{dp_note}; a prompt longer than "
        f"{chunk} tokens could never be scheduled. Raise --chunked-prefill-size "
        f"to at least {align * attn_dp_size}, or, with the flashinfer backend "
        f"under deterministic inference, lower "
        f"SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE."
    )
