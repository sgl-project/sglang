from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len_from_batch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _late_layer_dp_counts(
    forward_batch: ForwardBatch, local_late_rows: int
) -> Optional[List[int]]:
    # Bounded replay shrinks an extending rank's late layers to its tail, but the
    # TP-MoE gather / combine there spans every DP rank, so each rank needs all
    # ranks' late-layer row counts. Eager forwards only: every rank takes this
    # branch together, and graph replays never reach the late-layer switch.
    early = forward_batch.global_num_tokens_padded_cpu
    if early is None:
        early = forward_batch.global_num_tokens_cpu
    if early is None or torch.cuda.is_current_stream_capturing():
        return None
    device = forward_batch.global_num_tokens_gpu.device
    local = torch.tensor([local_late_rows], dtype=torch.int64, device=device)
    counts = torch.empty(len(early), dtype=torch.int64, device=device)
    get_parallel().tp_group.all_gather_into_tensor(counts, local)
    counts = counts.tolist()
    return None if counts == list(early) else counts


def _enter_late_layer_dp(forward_batch: ForwardBatch, counts: List[int]) -> tuple:
    fb = forward_batch
    saved = (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
    )
    fb.global_num_tokens_cpu = counts
    fb.global_num_tokens_gpu = torch.tensor(
        counts, dtype=torch.int64, device=fb.global_num_tokens_gpu.device
    )
    fb.global_num_tokens_padded_cpu = counts
    # The MoE gatherv reads its sizes from the logprob counts when present.
    fb.global_num_tokens_for_logprob_cpu = counts
    fb.global_dp_buffer_len = sum(counts)
    # Tail rows are real rows only, so the late layers gather variable lengths.
    fb.dp_padding_mode = DpPaddingMode.SUM_LEN
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    set_dp_buffer_len_from_batch(fb)
    return saved


def _exit_late_layer_dp(forward_batch: ForwardBatch, saved: tuple) -> None:
    fb = forward_batch
    (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
    ) = saved
    set_dp_buffer_len_from_batch(fb)
