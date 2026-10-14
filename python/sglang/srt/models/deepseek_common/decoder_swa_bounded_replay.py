from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len_from_batch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def late_layer_dp_counts(
    forward_batch: ForwardBatch, local_late_rows: int, *, in_graph: bool
) -> Optional[List[int]]:
    # The TP-MoE gather / combine in the late layers spans every DP rank, so each
    # rank needs all ranks' tail row counts.
    early = forward_batch.global_num_tokens_padded_cpu
    if early is None:
        early = forward_batch.global_num_tokens_cpu
    if early is None:
        return None
    if len(early) == 1:
        # Local-only counts (MegaMoE with a DP LM head): no peer geometry to share.
        counts = [local_late_rows]
    elif in_graph:
        # Prefill graphs pad every DP rank to one shape, so the tail block is
        # rank-uniform and the capture bakes it in; replays never run this code.
        counts = [local_late_rows] * len(early)
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
    # Device ops only: this runs inside prefill graph capture, where a host copy
    # would be recorded against pageable memory.
    gpu = fb.global_num_tokens_gpu
    fb.global_num_tokens_gpu = (
        torch.full_like(gpu, counts[0])
        if len(set(counts)) == 1
        else gpu.new_tensor(counts)
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
    fb.global_num_token_non_padded_cpu = min(
        local_rows, fb.global_num_token_non_padded_cpu
    )
    if fb.global_num_token_non_padded is not None:
        # Clamp the runner's static scalar so a graph replay rebounds it per batch.
        fb.global_num_token_non_padded = torch.clamp(
            fb.global_num_token_non_padded, max=local_rows
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
