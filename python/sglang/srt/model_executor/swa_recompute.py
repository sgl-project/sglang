"""Rebuild the trailing SWA window of a cached DeepSeek-V4 prefix by replaying
its last ``swa_recompute_len`` tokens. The replay reads compressed and indexer
entries from the cache and writes only SWA rows and the 4x compressor state."""

from copy import copy
from typing import List

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch


class SWARecomputeOutput(msgspec.Struct):
    # Extend-shaped view of the replayed span: prefix_lens, extend_lens, out_cache_loc.
    batch: ScheduleBatch
    logits_output: LogitsProcessorOutput


def _replay_batch(batch: ScheduleBatch, i: int, start: int, runner) -> ScheduleBatch:
    end = batch.prefix_lens[i]
    req = batch.reqs[i]
    slot = batch.req_pool_indices[i : i + 1]
    device = runner.device
    replay = copy(batch)
    replay.reqs = [req]
    replay.input_ids = torch.tensor(
        list(req.full_untruncated_fill_ids[start:end]),
        dtype=torch.int64,
        device=device,
    )
    replay.prefill_input_ids_cpu = None
    replay.req_pool_indices = slot
    replay.req_pool_indices_cpu = batch.req_pool_indices_cpu[i : i + 1]
    replay.prefix_lens = [start]
    replay.extend_lens = [end - start]
    replay.extend_num_tokens = end - start
    replay.seq_lens = torch.tensor([end], dtype=torch.int64, device=device)
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
    replay.engram_history = None
    replay.swa_recompute_starts = None
    return replay


def run_swa_recompute(
    worker, batch: ScheduleBatch, capture_hidden_mode
) -> List[SWARecomputeOutput]:
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )

    runner = worker.model_runner
    outputs = []
    for i, start in enumerate(batch.swa_recompute_starts):
        if start is None:
            continue
        replay = _replay_batch(batch, i, start, runner)
        forward_batch = ForwardBatch.init_new(
            replay,
            runner,
            capture_hidden_mode=capture_hidden_mode or CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        forward_batch.swa_recompute = True
        out = runner.forward(forward_batch)
        outputs.append(
            SWARecomputeOutput(batch=replay, logits_output=out.logits_output)
        )
    return outputs
