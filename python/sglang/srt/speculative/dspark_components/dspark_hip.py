"""ROCm glue of DSparkWorkerV2 on gfx950: with the MiniMax draft, the target verify reduces
its logits to their row argmax without their TP gather, and only the steps that read the
logits gather them; a decode step queues its draft before its host seq lens land
(overlap_utils_hip). dspark_worker_v2 binds it under _is_hip."""

from __future__ import annotations

import functools
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import logits_processor_hip
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.srt.utils import is_gfx95_supported


def defers_seq_lens_cpu(worker) -> bool:
    """Whether a decode step can queue its draft before the host seq lens land: the
    draft reads them only on the GPU, and the host bound for the verify window is the
    reserved length."""
    return (
        worker._hosts_draft
        and not worker._draft_is_moe
        and not get_parallel().attn_dp_enabled
        and not worker.draft_model_runner.attn_backend.needs_cpu_seq_lens
    )


def defer_verify_logits_gather(worker) -> None:
    """Leave the target verify's logits vocab-sharded when the non-compact eager accept is
    their only reader."""
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.models.dspark_minimax import DSparkMiniMaxDraftModel

    if (
        not is_gfx95_supported()
        # measured with MiniMax-M3.1 only
        or not isinstance(worker.draft_model, DSparkMiniMaxDraftModel)
        or worker._verify_epilogue is not None
        or worker._verify_planner.is_compact_mode
    ):
        return
    processors = [
        m
        for m in worker.target_worker.model_runner.model.modules()
        if isinstance(m, LogitsProcessor)
    ]
    if len(processors) != 1 or not logits_processor_hip.can_verify_argmax_only(
        processors[0]
    ):
        return
    (processor,) = processors
    processor.verify_argmax_only = True
    worker._verify_executor.gather_deferred_logits = functools.partial(
        logits_processor_hip.gather_deferred_verify_logits, processor
    )


def verify_needs_logits(
    worker, batch, sampling_info, simulate_bonus_sampling_info
) -> bool:
    """Whether this verify step reads the target logits beyond their row argmax; decided
    alike on every rank, since one rank gathering alone would hang the collective."""
    if (
        batch.has_grammar
        or batch.return_logprob
        or simulate_bonus_sampling_info is not None
        or _observers_read_target_logits()
    ):
        return True
    if sampling_info is None:
        return False
    if (
        sampling_info.sampling_mask_batch_indices is not None
        or not verify_logits_adjustments_are_noop(sampling_info)
    ):
        return True
    # simulated acceptance makes the ROCm accept greedy (TargetVerifyExecutor)
    return worker._simulate_acc_len <= 0 and not sampling_info.is_all_greedy


def take_target_argmax(
    logits_output, verify_ids_2d: torch.Tensor
) -> Optional[torch.Tensor]:
    """The deferred verify's row argmax for the accept: None when the logits were gathered,
    since they take the grammar mask and adjustments that the argmax predates."""
    argmax = logits_output.tp_logits_argmax
    # graph-owned, or this step's own: either way not the scheduler's to keep
    logits_output.tp_local_logits = logits_output.tp_logits_argmax = None
    if argmax is None or logits_output.next_token_logits is not None:
        return None
    return argmax[: verify_ids_2d.numel()]


def _observers_read_target_logits() -> bool:
    # DsparkStepObservers.observe_verify_step reads the logits for these recorders
    return bool(
        envs.SGLANG_DSPARK_STS_COLLECT_PATH.get()
        or envs.SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH.get()
        or envs.SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL.get() > 0
        or envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.get()
    )
