"""Dispatch selected-token logprobs after DSpark chain verification."""

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import _deterministic_inference_enabled
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_cuda

__all__ = ["dspark_selected_logprobs"]


def dspark_selected_logprobs(
    batch,
    logits: torch.Tensor,
    accept_tokens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return ``[bs, width]`` logprobs, or ``None`` to use the shared path."""
    bs, width = accept_tokens.shape

    # DFlash shares the chain branch. The switch and deterministic mode
    # keep the original computation.
    if (
        batch.spec_algorithm != SpeculativeAlgorithm.DSPARK
        or not envs.SGLANG_ENABLE_FAST_SPEC_LOGPROBS.get()
        or _deterministic_inference_enabled()
    ):
        return None

    # top-k / per-token-id consumers read full materialized rows, which the
    # fused operator deliberately never builds.
    needs_materialized_rows = (
        batch.top_logprobs_nums and any(k > 0 for k in batch.top_logprobs_nums)
    ) or (
        batch.token_ids_logprobs
        and any(t is not None for t in batch.token_ids_logprobs)
    )
    if needs_materialized_rows:
        return None

    # The kernel needs contiguous FP32 logits on NVIDIA CUDA and one row
    # per chain position. Tensor.is_cuda alone also admits ROCm.
    if not (
        is_cuda()
        and logits.ndim == 2
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and logits.shape[0] == bs * width > 0
    ):
        return None

    # Greedy / original-logprob requests skip scaling. Each request otherwise
    # uses the same temperature for all its chain rows.
    temperatures = None
    if not (
        batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()
    ):
        temperatures = batch.sampling_info.temperatures
        if (
            temperatures is None
            or temperatures.shape != (bs, 1)
            or temperatures.device != logits.device
            or temperatures.dtype != torch.float32
            or not temperatures.is_contiguous()
        ):
            return None

    # Load Triton only when a batch takes the fast path.
    from sglang.kernels.ops.sampling.selected_token_logprobs import (
        selected_token_logprobs,
    )

    return selected_token_logprobs(
        logits, accept_tokens.reshape(-1), temperatures, width
    )
