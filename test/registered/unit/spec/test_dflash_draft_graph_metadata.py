from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.dflash_worker_v2 import (
    _inherit_draft_graph_sync_metadata,
)


def _forward_batch(*, width: int = 8) -> ForwardBatch:
    batch_size = 4
    num_tokens = batch_size * width
    return ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=batch_size,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int32),
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
        seq_lens_sum=batch_size,
        spec_info=SimpleNamespace(
            num_tokens_per_req=width,
            num_tokens_for_logprob_per_req=1,
        ),
    )


def _schedule_batch(
    *,
    can_run_graph: bool,
    global_num_tokens=None,
    draft_global_num_tokens=None,
):
    global_num_tokens_for_logprob = (
        None if global_num_tokens is None else list(global_num_tokens)
    )
    draft_global_num_tokens_for_logprob = (
        None if draft_global_num_tokens is None else list(draft_global_num_tokens)
    )
    return SimpleNamespace(
        can_run_decode_cuda_graph=can_run_graph,
        can_run_dp_draft_cuda_graph=can_run_graph,
        is_extend_in_batch=False,
        dp_spec_prefill_coordination_applied=False,
        global_num_tokens=global_num_tokens,
        global_num_tokens_for_logprob=global_num_tokens_for_logprob,
        draft_global_num_tokens=draft_global_num_tokens,
        draft_global_num_tokens_for_logprob=draft_global_num_tokens_for_logprob,
    )


def test_dflash_draft_inherits_graph_vote_without_sync_metadata():
    forward_batch = _forward_batch()
    batch = _schedule_batch(can_run_graph=True)

    _inherit_draft_graph_sync_metadata(forward_batch, batch, torch.device("cpu"))

    assert forward_batch.can_run_decode_cuda_graph
    assert forward_batch.can_run_dp_draft_cuda_graph
    assert forward_batch.global_num_tokens_cpu is None


def test_dflash_draft_scales_gathered_buffer_metadata_to_block_width():
    forward_batch = _forward_batch(width=8)
    batch = _schedule_batch(can_run_graph=True, global_num_tokens=[4])

    _inherit_draft_graph_sync_metadata(forward_batch, batch, torch.device("cpu"))

    assert forward_batch.original_global_num_tokens_cpu == [4]
    assert forward_batch.global_num_tokens_cpu == [32]
    assert forward_batch.global_num_tokens_for_logprob_cpu == [4]
    assert forward_batch.global_num_tokens_gpu.tolist() == [32]
    assert forward_batch.can_run_decode_cuda_graph


def test_dflash_draft_uses_draft_dp_counts_and_preserves_negative_vote():
    forward_batch = _forward_batch(width=8)
    batch = _schedule_batch(
        can_run_graph=False,
        global_num_tokens=[3, 5],
        draft_global_num_tokens=[4, 4],
    )

    _inherit_draft_graph_sync_metadata(forward_batch, batch, torch.device("cpu"))

    assert forward_batch.original_global_num_tokens_cpu == [4, 4]
    assert forward_batch.global_num_tokens_cpu == [32, 32]
    assert not forward_batch.can_run_decode_cuda_graph
    assert not forward_batch.can_run_dp_draft_cuda_graph
