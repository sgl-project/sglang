from copy import copy
from typing import Any, Optional

import msgspec
import torch


class FoldedExtend(msgspec.Struct, frozen=True):
    """An extend batch whose prefix hits start at their replay start."""

    batch: Any  # ScheduleBatch copy fed to ForwardBatch.init_new
    keep_rows: torch.Tensor  # [original rows] int64, folded-row index of each
    scheduled_row: torch.Tensor  # [folded rows] int64, original row, -1 if replayed
    row_floor: torch.Tensor  # [folded rows] int64
    compress_skip: torch.Tensor  # [bs] int32, leading replay rows per request
    replay_lens: list[int]  # [bs] leading replay rows per request
    num_rows: int


def fold_encoder_swa_replay(worker, batch) -> Optional[FoldedExtend]:
    """Prepend each hit's <=128 replay tokens to its extend instead of replaying."""
    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return None
    rows = _reset_windows_and_collect_hits(batch=batch, window=window)
    if not rows:
        return None
    return _fold_batch(batch=batch, runner=runner, rows=rows)


def apply_folded_extend(folded: FoldedExtend, forward_batch) -> None:
    forward_batch.encoder_swa_row_floor = folded.row_floor
    forward_batch.encoder_swa_compress_skip = folded.compress_skip
    forward_batch.encoder_swa_compress_rows = folded.keep_rows


def drop_folded_rows(*, logits_output, folded: FoldedExtend) -> None:
    """Hand downstream consumers (the DSpark draft) only the original extend rows."""
    hidden = logits_output.hidden_states
    token_indices = logits_output.hidden_states_token_indices
    if token_indices is not None:
        # Decoder-SWA tail rows index the folded extend; keep those on original rows.
        keep = _tail_rows_on_original(
            folded=folded, num_tail_rows=token_indices.shape[0]
        ).to(token_indices.device, non_blocking=True)
        logits_output.hidden_states_token_indices = folded.scheduled_row[
            token_indices[keep]
        ]
        if hidden is not None:
            logits_output.hidden_states = hidden[keep]
    elif hidden is not None and hidden.shape[0] >= folded.num_rows:
        # MLP-sync padding rows, if any, follow the real rows and are dropped too.
        logits_output.hidden_states = hidden[folded.keep_rows]


def _tail_rows_on_original(*, folded: FoldedExtend, num_tail_rows: int):
    # Each request's tail is its last SWA_WINDOW folded rows; replay rows lead it.
    from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

    keep, off = [], 0
    for n, r in zip(folded.batch.extend_lens, folded.replay_lens):
        tail = min(SWA_WINDOW, n)
        keep.append(torch.arange(off + max(0, tail - (n - r)), off + tail))
        off += tail
    assert off == num_tail_rows, (off, num_tail_rows)
    return torch.cat(keep)


def run_encoder_swa_replay(worker, batch):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return
    rows = _reset_windows_and_collect_hits(batch=batch, window=window)
    if not rows:
        return
    # One forward replays every hitting request; each request's window is
    # floored at its own start, so the result matches per-request replays.
    replay = _build_replay_batch(batch=batch, runner=runner, rows=rows)
    fb = ForwardBatch.init_new(
        replay,
        runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    fb.encoder_swa_replay = True
    runner.forward(fb)


def _reset_windows_and_collect_hits(*, batch, window):
    rows = []
    for i, reset in enumerate(batch.encoder_swa_reset):
        if not reset:
            continue
        window.reset(batch.req_pool_indices[i : i + 1])
        end = batch.prefix_lens[i]
        if not end:
            continue
        if end % 2:
            raise ValueError(
                "encoder SWA replay requires an even cached-prefix boundary"
            )
        rows.append((i, max(0, end - 128), end))
    return rows


def _build_replay_batch(*, batch, runner, rows):
    idx = [i for i, _, _ in rows]
    starts = [s for _, s, _ in rows]
    ends = [e for _, _, e in rows]
    lens = [e - s for s, e in zip(starts, ends)]
    reqs = [batch.reqs[i] for i in idx]
    slots = batch.req_pool_indices[idx]
    req_to_token = runner.req_to_token_pool.req_to_token

    replay = copy(batch)
    replay.reqs = reqs
    replay.input_ids = torch.tensor(
        [
            t
            for r, s, e in zip(reqs, starts, ends)
            for t in r.full_untruncated_fill_ids[s:e]
        ],
        dtype=torch.int64,
        device=runner.device,
    )
    replay.prefill_input_ids_cpu = None
    replay.req_pool_indices = slots
    replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[idx]
    replay.prefix_lens = starts
    replay.extend_lens = lens
    replay.extend_num_tokens = sum(lens)
    replay.seq_lens = torch.tensor(ends, dtype=torch.int64, device=runner.device)
    replay.seq_lens_cpu = torch.tensor(ends, dtype=torch.int64)
    replay.seq_lens_sum = sum(ends)
    replay.orig_seq_lens = replay.seq_lens
    replay.out_cache_loc = torch.cat(
        [req_to_token[slots[k], s:e] for k, (s, e) in enumerate(zip(starts, ends))]
    ).long()
    replay.return_logprob = False
    replay.top_logprobs_nums = None
    replay.token_ids_logprobs = None
    replay.extend_logprob_start_lens = lens
    replay.extend_input_logprob_token_ids = None
    replay.is_prefill_only = True
    replay.spec_info = None
    replay.sampling_info = None
    replay.has_grammar = False
    replay.multimodal_inputs = [None] * len(reqs)
    replay.engram_history = _engram_history(reqs=reqs, starts=starts, runner=runner)
    _set_execution_counts(batch=replay, scheduled=batch)
    return replay


def _engram_history(*, reqs, starts, runner):
    hasher = runner.model.model.engram_hasher
    if hasher is None:
        return None
    n = hasher.max_ngram_size - 1
    hist = []
    for r, s in zip(reqs, starts):
        ids = list(r.full_untruncated_fill_ids[max(0, s - n) : s])
        hist.append([0] * (n - len(ids)) + ids)
    return torch.tensor(hist, dtype=torch.int32, device=runner.device)


def _fold_batch(*, batch, runner, rows) -> FoldedExtend:
    bs = len(batch.reqs)
    device = runner.device
    replay = [0] * bs
    for i, s, e in rows:
        replay[i] = e - s
    ext = list(batch.extend_lens)
    pre = [p - r for p, r in zip(batch.prefix_lens, replay)]
    new_ext = [x + r for x, r in zip(ext, replay)]
    req_to_token = runner.req_to_token_pool.req_to_token
    ids, locs, keep, cpu_ids = [], [], [], []
    off = fold_off = 0
    for i, req in enumerate(batch.reqs):
        r, n = replay[i], ext[i]
        if r:
            head = list(req.full_untruncated_fill_ids[pre[i] : pre[i] + r])
            ids.append(torch.tensor(head, dtype=batch.input_ids.dtype, device=device))
            locs.append(
                req_to_token[batch.req_pool_indices[i], pre[i] : pre[i] + r].long()
            )
            cpu_ids.append(torch.tensor(head, dtype=torch.int64))
        ids.append(batch.input_ids[off : off + n])
        locs.append(batch.out_cache_loc[off : off + n].long())
        if batch.prefill_input_ids_cpu is not None:
            cpu_ids.append(batch.prefill_input_ids_cpu[off : off + n].to(torch.int64))
        keep.append(torch.arange(fold_off + r, fold_off + r + n))
        off += n
        fold_off += r + n

    folded = copy(batch)
    folded.input_ids = torch.cat(ids)
    folded.out_cache_loc = torch.cat(locs)
    folded.prefill_input_ids_cpu = (
        torch.cat(cpu_ids) if batch.prefill_input_ids_cpu is not None else None
    )
    folded.prefix_lens = pre
    folded.extend_lens = new_ext
    folded.extend_num_tokens = fold_off
    if batch.extend_logprob_start_lens is not None:
        folded.extend_logprob_start_lens = [
            x + r for x, r in zip(batch.extend_logprob_start_lens, replay)
        ]
    if batch.engram_history is not None:
        folded.engram_history = _engram_history(
            reqs=batch.reqs, starts=pre, runner=runner
        )
    _set_execution_counts(batch=folded, scheduled=batch)
    floor = torch.tensor(
        [p if r else 0 for p, r in zip(pre, replay)], dtype=torch.int64
    )
    row_floor = torch.repeat_interleave(
        floor, torch.tensor(new_ext), output_size=fold_off
    )
    keep_rows = torch.cat(keep)
    scheduled_row = torch.full((fold_off,), -1, dtype=torch.int64)
    scheduled_row[keep_rows] = torch.arange(keep_rows.shape[0])
    return FoldedExtend(
        batch=folded,
        keep_rows=keep_rows.to(device, non_blocking=True),
        scheduled_row=scheduled_row.to(device, non_blocking=True),
        row_floor=row_floor.to(device, non_blocking=True),
        compress_skip=torch.tensor(replay, dtype=torch.int32).to(
            device, non_blocking=True
        ),
        replay_lens=replay,
        num_rows=fold_off,
    )


def _set_execution_counts(*, batch, scheduled) -> None:
    """MLP-sync counts of a batch copied from `scheduled` with different rows."""
    if scheduled.global_num_tokens is None:
        return
    # The flag rejects attention DP, so the gather holds only this rank's count.
    if (
        len(scheduled.global_num_tokens) != 1
        or scheduled.global_num_tokens[0] != scheduled.extend_num_tokens
    ):
        raise ValueError(
            "encoder SWA replay expects one MLP-sync count equal to the scheduled "
            f"extend ({scheduled.extend_num_tokens}), got {scheduled.global_num_tokens}"
        )
    batch.global_num_tokens = [batch.extend_num_tokens]
    # The scheduler's formula; the draft counts stay with the scheduled batch.
    batch.global_num_tokens_for_logprob = [
        sum(
            max(n - start, 1)
            for n, start in zip(batch.extend_lens, batch.extend_logprob_start_lens)
        )
    ]
