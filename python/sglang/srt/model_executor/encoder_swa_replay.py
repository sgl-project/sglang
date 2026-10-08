from copy import copy

import torch


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
        [t for r, s, e in zip(reqs, starts, ends) for t in r.full_untruncated_fill_ids[s:e]],
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
