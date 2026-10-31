from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len_from_batch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def late_layer_dp_counts(
    forward_batch: ForwardBatch,
    local_late_rows: int,
    cp_metadata=None,
    *,
    in_graph: bool,
) -> Optional[List[int]]:
    # The TP-MoE gather / combine in the late layers spans every DP rank, so each
    # rank needs all ranks' tail row counts.
    early = forward_batch.global_num_tokens_padded_cpu
    if early is None:
        early = forward_batch.global_num_tokens_cpu
    if early is None:
        return None
    # DP slots contain the complete CP rank-major buffer, including collective
    # padding, not this process's logical tail rows. CP partners publish the same
    # slot size even when their logical tails have different lengths.
    refresh_cp = cp_metadata is not None
    if cp_metadata is not None:
        local_late_rows = sum(cp_metadata.max_rank_len)
    if len(early) == 1:
        # Local-only counts (MegaMoE with a DP LM head): no peer geometry to share.
        counts = [local_late_rows]
    elif in_graph:
        # The capture bakes the tail block and replays never run this code. Every
        # rank shares one bucket whenever the graph holds a DP gather.
        counts = [local_late_rows] * len(early)
    elif forward_batch.global_late_layer_num_tokens_cpu is not None:
        # Exchanged in ForwardBatch.prepare_mlp_sync_batch, which a peer replaying
        # the prefill graph reaches too; a gather here would wait for it forever.
        # That exchange publishes logical tail rows, so it is skipped under
        # attention CP, where a DP slot is the padded rank-major CP buffer.
        assert cp_metadata is None
        counts = forward_batch.global_late_layer_num_tokens_cpu
        assert counts[get_parallel().attn_dp_rank] == local_late_rows, (
            counts,
            local_late_rows,
        )
    else:
        # Without the exchange (no extend in the batch, or attention CP, whose
        # slot width is only known here) every rank takes this path; the CPU
        # group avoids a device sync.
        parallel = get_parallel()
        gathered = torch.empty((parallel.tp_size, 2), dtype=torch.int64)
        all_gather_single(
            gathered.flatten(),
            torch.tensor([local_late_rows, int(refresh_cp)], dtype=torch.int64),
            group=parallel.tp_group.cpu_group,
        )
        # TP order is [DP, CP, attention TP]. A DP slot is replicated by
        # CP/attention-TP partners; gathering into a DP-sized tensor overruns it.
        replicas = gathered[:, 0].reshape(len(early), -1)
        if not torch.all(replicas == replicas[:, :1]):
            raise ValueError("late-layer DP slot sizes disagree across CP/TP partners")
        counts = replicas[:, 0].tolist()
        refresh_cp = bool(gathered[:, 1].any())
    # Even an equally sized CP tail can change its logical lengths/row mapping.
    # Refresh the bounds and routing IDs at the handoff in that case as well.
    # This decision controls subsequent collectives, not just local metadata.
    # An active short CP prefill can retain its early size while idle peers have
    # no CP metadata at all. Publish the refresh bit with the counts so those
    # peers also enter SUM_LEN and (for VL) the token-ID gather.
    return None if not refresh_cp and counts == list(early) else counts


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
    if len(set(counts)) == 1:
        fb.global_num_tokens_gpu = torch.full_like(gpu, counts[0])
    else:
        fb.global_num_tokens_gpu = async_h2d(counts, dtype=gpu.dtype, device=gpu.device)
    fb.global_num_tokens_padded_cpu = counts
    # The MoE gatherv reads its sizes from the logprob counts when present.
    fb.global_num_tokens_for_logprob_cpu = counts
    fb.global_num_tokens_for_logprob_gpu = fb.global_num_tokens_gpu
    fb.global_dp_buffer_len = sum(counts)
    # CP slots include equal-shard padding; DP slots may have different lengths.
    fb.dp_padding_mode = DpPaddingMode.SUM_LEN
    fb.dp_local_start_pos = fb.dp_local_num_tokens = None
    # The global bound describes the logical sequence, while a CP-local MoE
    # routes only its shard's real prefix. Never use the physical DP slot length
    # as either logical bound. Preserve an idle peer's zero bound.
    local_rows = counts[get_parallel().attn_dp_rank if len(counts) > 1 else 0]
    cp = fb.attn_cp_metadata
    local_real_rows = None
    if cp is not None:
        local_rows = cp.total_seq_lens
        logical = cp.per_rank_logical_token or cp.per_rank_actual_token
        local_real_rows = logical[get_parallel().attn_cp_rank]
    real_rows = min(local_rows, fb.global_num_token_non_padded_cpu)
    fb.global_num_token_non_padded_cpu = real_rows
    if fb.global_num_token_non_padded is not None:
        # Clamp the runner's static scalar so a graph replay rebounds it per batch.
        fb.global_num_token_non_padded = torch.clamp(
            fb.global_num_token_non_padded, max=local_rows
        )
        fb.num_token_non_padded = fb.global_num_token_non_padded.new_tensor(
            min(local_real_rows, real_rows)
            if local_real_rows is not None
            else real_rows
        )
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
