"""Watermark algorithm initialization and ordinary Gumbel sampling lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.runtime_context import get_exec
from sglang.srt.watermarking.synthid.setup import init_synthid_watermark

from .core import WatermarkState

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def init_watermark_engine(
    *, sampler: Sampler, request_pool, device, vocab_size: int
) -> Optional[WatermarkState]:
    features = get_exec().features
    synthid_configured = get_exec().kernel.synthid_watermark_key_file is not None
    if not features.enable_watermark and not synthid_configured:
        return None
    algorithm = features.watermark_algorithm or (
        "synthid" if synthid_configured else "gumbel"
    )
    if algorithm == "synthid":
        (
            sampler.synthid_watermark,
            sampler.use_ordinary_synthid_sampling,
            sampler.use_fused_synthid_sampling,
        ) = init_synthid_watermark(
            num_slots=request_pool.req_to_token.shape[0],
            device=device,
            tp_sync_group=sampler.tp_sync_group,
        )
        return None
    if algorithm != "gumbel":
        raise ValueError("Unknown watermark algorithm in the published runtime")
    state = WatermarkState(
        max_num_reqs=request_pool.req_to_token.shape[0],
        context_window=features.watermark_context_window,
        max_contexts_per_req=request_pool.req_to_token.shape[1],
        vocab_size=vocab_size,
        key=features.watermark_key,
        key_b=features.watermark_key_b,
        mixing_probability=features.watermark_mixing_probability,
        max_probability=features.watermark_max_probability,
        default_enabled=features.watermark_default_enabled,
        enforce_all=features.watermark_enforce_all,
        device=device,
    )
    sampler.watermark_state = state
    return state


def prepare_watermark_sample(
    *,
    state: Optional[WatermarkState],
    logits_output: LogitsProcessorOutput,
    batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    if state is None or not batch.sampling_info.has_watermark_candidates:
        return None
    logits = logits_output.next_token_logits
    logprob_logits = logits.clone() if batch.return_logprob else None
    indices = batch.req_pool_indices[: logits.shape[0]]
    state.init_from_prompt(
        indices, batch.watermark_prompt_tail_ids, batch.watermark_context_hash_history
    )
    state.force(logits, indices, batch.sampling_info)
    return logprob_logits


def commit_watermark_sample(
    *, state: Optional[WatermarkState], tokens: torch.Tensor, batch: ForwardBatch
) -> None:
    if state is not None and batch.sampling_info.has_watermark_candidates:
        state.append(batch.req_pool_indices[: tokens.shape[0]], tokens)
