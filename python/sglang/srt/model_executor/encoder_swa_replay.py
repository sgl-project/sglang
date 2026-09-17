from copy import copy

import torch

from sglang.srt.environ import envs


def run_encoder_swa_replay(worker, batch):
    from sglang.srt.runtime_context import get_schedule

    runner = worker.model_runner
    window = runner.token_to_kv_pool.request_window
    if window is None or not batch.forward_mode.is_extend_without_speculative():
        return

    reset_indices = [i for i, reset in enumerate(batch.encoder_swa_reset) if reset]
    if not reset_indices:
        return
    replay_indices = [i for i in reset_indices if batch.prefix_lens[i]]
    if any(batch.prefix_lens[i] % 2 for i in replay_indices):
        raise ValueError("encoder SWA replay requires an even cached-prefix boundary")
    max_batch_size = envs.SGLANG_ENCODER_SWA_REPLAY_MAX_BATCH_SIZE.get()
    if max_batch_size < 1:
        raise ValueError("encoder SWA replay max batch size must be positive")
    if max_batch_size > 1 and len(replay_indices) > 1:
        _validate_batch_runtime(runner)
    window.reset(batch.req_pool_indices[reset_indices])
    if not replay_indices:
        return

    # RequestWindow reserves chunked_prefill_size query rows in addition to
    # per-request history rows. A short suffix can admit many long-prefix
    # requests, so bound replay tokens independently of the normal extend.
    token_budget = max(128, get_schedule().chunked_prefill_size or 0)
    indices = []
    num_tokens = 0
    for i in replay_indices:
        length = min(batch.prefix_lens[i], 128)
        if indices and (
            len(indices) >= max_batch_size or num_tokens + length > token_budget
        ):
            _run_replay_batch(runner, batch, indices)
            indices = []
            num_tokens = 0
        indices.append(i)
        num_tokens += length
    if indices:
        _run_replay_batch(runner, batch, indices)


def _run_replay_batch(runner, batch, indices):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    replay = copy(batch)
    replay.reqs = [batch.reqs[i] for i in indices]
    ends = [batch.prefix_lens[i] for i in indices]
    replay.prefix_lens = [max(0, end - 128) for end in ends]
    replay.extend_lens = [end - start for start, end in zip(replay.prefix_lens, ends)]
    replay.extend_num_tokens = sum(replay.extend_lens)
    replay.input_ids = torch.tensor(
        [
            token
            for req, start, end in zip(replay.reqs, replay.prefix_lens, ends)
            for token in req.full_untruncated_fill_ids[start:end]
        ],
        dtype=torch.int64,
        device=runner.device,
    )
    replay.prefill_input_ids_cpu = None
    replay.req_pool_indices = batch.req_pool_indices[indices]
    replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[indices]
    replay.seq_lens = torch.tensor(ends, dtype=torch.int64, device=runner.device)
    replay.seq_lens_cpu = torch.tensor(ends, dtype=torch.int64)
    replay.seq_lens_sum = sum(ends)
    replay.orig_seq_lens = replay.seq_lens

    # Gather each request's cached tail without reading GPU slot scalars on CPU.
    positions = torch.tensor(
        [
            pos
            for start, end in zip(replay.prefix_lens, ends)
            for pos in range(start, end)
        ],
        dtype=torch.int64,
        device=runner.device,
    )
    slots = torch.repeat_interleave(
        replay.req_pool_indices,
        torch.tensor(replay.extend_lens, dtype=torch.int64, device=runner.device),
        output_size=replay.extend_num_tokens,
    )
    replay.out_cache_loc = runner.req_to_token_pool.req_to_token[
        slots, positions
    ].long()
    replay.return_logprob = False
    replay.top_logprobs_nums = None
    replay.token_ids_logprobs = None
    replay.extend_logprob_start_lens = replay.extend_lens
    replay.extend_input_logprob_token_ids = None
    replay.is_prefill_only = True
    replay.spec_info = None
    replay.sampling_info = None
    replay.has_grammar = False
    replay.multimodal_inputs = [None] * len(indices)
    replay.engram_history = None
    hasher = runner.model.model.engram_hasher
    if hasher is not None:
        n = hasher.max_ngram_size - 1
        histories = []
        for req, start in zip(replay.reqs, replay.prefix_lens):
            ids = list(req.full_untruncated_fill_ids[max(0, start - n) : start])
            histories.append([0] * (n - len(ids)) + ids)
        replay.engram_history = torch.tensor(
            histories, dtype=torch.int32, device=runner.device
        )
    fb = ForwardBatch.init_new(
        replay,
        runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )
    fb.encoder_swa_replay = True
    runner.forward(fb)


def _validate_batch_runtime(runner):
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.runtime_context import get_forward, get_parallel, get_platform

    parallel = get_parallel()
    if (
        not get_platform().is_sm90
        or any(
            getattr(parallel, name) != 1
            for name in ("attn_dp_size", "attn_cp_size", "attn_dcp_size", "pp_size")
        )
        or get_forward().sp_active
        or getattr(parallel, "enable_layernorm_sp", False)
        or getattr(parallel, "enable_attn_tp_input_scattered", False)
        or (
            runner.prefill_cuda_graph_runner is not None
            and not isinstance(runner.prefill_cuda_graph_runner, EagerRunner)
        )
        or not get_moe_a2a_backend().is_none()
        or runner.attn_backend.trtllm_attn
        or any(
            layer.self_attn.compress_ratio not in (0, 1, 2)
            for layer in runner.model.model.layers
        )
    ):
        raise NotImplementedError(
            "Grouped encoder SWA replay currently requires SM90 FlashMLA, "
            "PP=DP=CP=DCP=1, no sequence sharding or MoE all-to-all, "
            "ratio-1/2 compressed caches, and eager prefill"
        )
