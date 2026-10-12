from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len_from_batch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def late_layer_dp_counts(
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
    if len(early) == 1:
        # Local-only counts (MegaMoE with a DP LM head): no peer geometry to share.
        counts = [local_late_rows]
    else:
        # Tail row counts are host-known; the CPU group avoids a device sync.
        counts = torch.empty(len(early), dtype=torch.int64)
        all_gather_single(
            counts,
            torch.tensor([local_late_rows], dtype=torch.int64),
            group=get_parallel().tp_group.cpu_group,
        )
        counts = counts.tolist()
    return None if counts == list(early) else counts


def enter_late_layer_dp(forward_batch: ForwardBatch, counts: List[int]) -> tuple:
    fb = forward_batch
    saved = (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_num_tokens_for_logprob_gpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
        fb.num_token_non_padded,
    )
    fb.global_num_tokens_cpu = counts
    fb.global_num_tokens_gpu = torch.tensor(
        counts, dtype=torch.int64, device=fb.global_num_tokens_gpu.device
    )
    fb.global_num_tokens_padded_cpu = counts
    # The MoE gatherv reads its sizes from the logprob counts when present.
    fb.global_num_tokens_for_logprob_cpu = counts
    fb.global_num_tokens_for_logprob_gpu = fb.global_num_tokens_gpu
    fb.global_dp_buffer_len = sum(counts)
    # Tail rows are real rows only, so the late layers gather variable lengths.
    fb.dp_padding_mode = DpPaddingMode.SUM_LEN
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    # With attention TP=1 the invariant scalar also describes local MoE rows.
    # Preserve an idle peer's zero bound even when it carries fabricated rows.
    local_rows = counts[get_parallel().attn_dp_rank if len(counts) > 1 else 0]
    real_rows = min(local_rows, fb.global_num_token_non_padded_cpu)
    fb.global_num_token_non_padded_cpu = real_rows
    if fb.global_num_token_non_padded is not None:
        fb.global_num_token_non_padded = fb.global_num_token_non_padded.new_tensor(
            real_rows
        )
        fb.num_token_non_padded = fb.global_num_token_non_padded
    set_dp_buffer_len_from_batch(fb)
    return saved


def exit_late_layer_dp(forward_batch: ForwardBatch, saved: tuple) -> None:
    fb = forward_batch
    (
        fb.global_num_tokens_cpu,
        fb.global_num_tokens_gpu,
        fb.global_num_tokens_padded_cpu,
        fb.global_num_tokens_for_logprob_cpu,
        fb.global_num_tokens_for_logprob_gpu,
        fb.global_dp_buffer_len,
        fb.dp_padding_mode,
        fb.dp_local_start_pos,
        fb.dp_local_num_tokens,
        fb.global_num_token_non_padded,
        fb.global_num_token_non_padded_cpu,
        fb.num_token_non_padded,
    ) = saved
    set_dp_buffer_len_from_batch(fb)
