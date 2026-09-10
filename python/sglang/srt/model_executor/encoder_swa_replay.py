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
    for i, reset in enumerate(batch.encoder_swa_reset):
        if not reset:
            continue
        slot = batch.req_pool_indices[i : i + 1]
        window.reset(slot)
        end = batch.prefix_lens[i]
        if not end:
            continue
        if end % 2:
            raise ValueError(
                "encoder SWA replay requires an even cached-prefix boundary"
            )
        start = max(0, end - 128)
        req = batch.reqs[i]
        replay = copy(batch)
        replay.reqs = [req]
        replay.input_ids = torch.tensor(
            list(req.full_untruncated_fill_ids[start:end]),
            dtype=torch.int64,
            device=runner.device,
        )
        replay.prefill_input_ids_cpu = None
        replay.req_pool_indices = slot
        replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[i : i + 1]
        replay.prefix_lens = [start]
        replay.extend_lens = [end - start]
        replay.extend_num_tokens = end - start
        replay.seq_lens = torch.tensor([end], dtype=torch.int64, device=runner.device)
        replay.seq_lens_cpu = torch.tensor([end], dtype=torch.int64)
        replay.seq_lens_sum = end
        replay.orig_seq_lens = replay.seq_lens
        replay.out_cache_loc = runner.req_to_token_pool.req_to_token[
            slot[0], start:end
        ].long()
        replay.return_logprob = False
        replay.top_logprobs_nums = None
        replay.token_ids_logprobs = None
        replay.extend_logprob_start_lens = [end - start]
        replay.extend_input_logprob_token_ids = None
        replay.is_prefill_only = True
        replay.spec_info = None
        replay.sampling_info = None
        replay.has_grammar = False
        replay.multimodal_inputs = [None]
        replay.ne_history = None
        hasher = runner.model.model.engram_hasher
        if hasher is not None:
            n = hasher.max_ngram_size - 1
            ids = list(req.full_untruncated_fill_ids[max(0, start - n) : start])
            replay.ne_history = torch.tensor(
                [[0] * (n - len(ids)) + ids],
                dtype=torch.int32,
                device=runner.device,
            )
        fb = ForwardBatch.init_new(
            replay,
            runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        fb.encoder_swa_replay = True
        runner.forward(fb)
