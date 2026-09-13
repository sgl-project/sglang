"""K-pool dispatch contracts without launching GPU kernels."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa import paged_mqa_logits
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as kpool
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.mark.parametrize("mode", ["decode", "target_verify"])
def test_hip_paged_logits_backend_preserves_expanded_rows(mode):
    # A speculative batch has per-query lengths and can have padded query rows.
    rows, padded_rows, heads = 3, 4, 32
    seq_lens = torch.tensor([2049, 2050, 2051], dtype=torch.int32)
    pooled_lens = seq_lens // 4
    page_table = torch.zeros(rows, 33, dtype=torch.int32)
    pooled_table = torch.zeros(rows, 9, dtype=torch.int32)
    cache = torch.empty(9, 64 * 132, dtype=torch.uint8)
    q = torch.empty(padded_rows, heads, 128, dtype=torch.float8_e4m3fn)
    weights = torch.empty(padded_rows, heads, 1)
    logits = torch.empty(rows, 9 * 64)
    selected = torch.empty(padded_rows, 2051, dtype=torch.int32)
    forward_mode = SimpleNamespace(
        is_target_verify=lambda: mode == "target_verify",
        is_draft_extend_v2=lambda: mode == "draft_extend_v2",
    )
    metadata = SimpleNamespace(
        get_page_table_64=lambda: page_table,
        get_seqlens_expanded=lambda: seq_lens,
        get_seqlens_int32=lambda: seq_lens,
    )
    decode_metadata = Mock(
        return_value=(pooled_lens, pooled_lens[:, None], pooled_table, None)
    )
    select = Mock(return_value=selected)
    indexer = SimpleNamespace(
        _get_index_k_read_buffer=lambda pool, layer_id: cache,
        _should_use_tilelang_paged_mqa_logits=lambda q: False,
        _get_kpool_decode_metadata=decode_metadata,
        _kpool_fused_topk_mapping=lambda metadata: (None, None, None),
        _topk_from_kpool_logits=select,
    )
    with (
        patch.object(kpool, "is_hip", return_value=True),
        patch.object(
            kpool, "get_token_to_kv_pool", return_value=SimpleNamespace(page_size=64)
        ),
        patch.object(
            paged_mqa_logits, "aiter_paged_mqa_logits", return_value=logits
        ) as aiter,
    ):
        result = kpool.IndexerKPool._get_topk_paged(
            indexer, SimpleNamespace(forward_mode=forward_mode), 0, q, weights, metadata
        )
    assert result is selected
    assert decode_metadata.call_args.kwargs["build_schedule_metadata"] is False
    aiter.assert_called_once()
    args, kwargs = aiter.call_args
    # The shared AITER wrapper, unlike DeepGEMM, adds its own next-N axis.
    assert args[0].shape == (rows, heads, 128)
    assert args[1].shape == (9, 64, 1, 132)
    assert args[2].shape == (rows, heads)
    torch.testing.assert_close(args[3], pooled_lens)
    assert args[4] is pooled_table
    assert args[5] == 9 * 64
    assert kwargs == {"preshuffle": True, "kv_block_size": 64}
    assert select.call_args.args[0] is logits
    torch.testing.assert_close(select.call_args.kwargs["seq_lens"], seq_lens)
    assert select.call_args.kwargs["out_rows"] == padded_rows


def test_hip_ragged_logits_keep_per_row_bounds_and_clean_padding():
    q, keys, scales, weights = (torch.empty(n) for n in (4, 5, 6, 7))
    starts = torch.tensor([0, 10], dtype=torch.int32)
    ends = torch.tensor([5, 18], dtype=torch.int32)
    logits = torch.empty(2, 24)
    aiter = Mock(return_value=logits)
    with (
        patch.object(kpool, "is_hip", return_value=True),
        patch.dict(
            sys.modules,
            {"aiter.ops.triton.fp8_mqa_logits": SimpleNamespace(fp8_mqa_logits=aiter)},
        ),
    ):
        result = kpool.IndexerKPool._ragged_mqa_logits(
            q, keys, scales, weights, starts, ends
        )
    assert result is logits
    aiter.assert_called_once_with(
        q, keys, scales, weights, starts, ends, clean_logits=True
    )


@pytest.mark.parametrize("mode", ["decode", "draft_extend_v2"])
def test_hip_write_plan_updates_request_metadata(mode):
    tokens = 1 if mode == "decode" else 6
    plan = kpool_plan._alloc_kpool_write_plan_buffers(
        max_bs=2,
        num_draft_tokens=tokens,
        pool_size=4,
        device=torch.device("cpu"),
        is_verify=mode != "decode",
        is_v2=mode == "draft_extend_v2",
    )
    write_start = torch.tensor([3, 30037], dtype=torch.int32)
    req_indices = torch.tensor([7, 11], dtype=torch.int64)
    page_table = torch.zeros(2 * tokens, 512, dtype=torch.int32)
    accepted = torch.tensor([2, 5], dtype=torch.int32)
    forward_mode = SimpleNamespace(
        is_decode_or_idle=lambda: mode == "decode",
        is_target_verify=lambda: mode == "target_verify",
        is_draft_extend_v2=lambda: mode == "draft_extend_v2",
    )
    with (
        patch.object(kpool_plan, "is_cuda", return_value=False),
        patch.object(kpool_plan, "is_hip", return_value=True),
        patch.object(kpool_plan, "update_kpool_write_plan_cuda_graph") as update,
        patch.object(kpool_plan, "_compute_pool_schedule_metadata") as schedule,
    ):
        kpool_plan.update_kpool_write_plan(
            SimpleNamespace(kpool_write_plan=plan),
            write_start=write_start,
            req_pool_indices=req_indices,
            real_page_table=page_table,
            pool_size=4,
            real_page_size=64,
            num_draft_tokens=tokens,
            forward_mode=forward_mode,
            slots_per_page=64,
            effective_n_per_batch=accepted,
        )
    # HIP needs request/write-position updates, but no captured DeepGEMM schedule.
    schedule.assert_not_called()
    update.assert_called_once()
    kwargs = update.call_args.kwargs
    assert kwargs["write_start"] is write_start
    assert kwargs["req_pool_indices"] is req_indices
    assert kwargs["real_page_table"] is page_table
    assert kwargs["req_out"] is plan.req
    assert kwargs["write_start_out"] is plan.write_start
    assert kwargs["write_loc_out"] is plan.write_loc
    assert kwargs["seqlens_per_q_out"] is plan.seqlens_per_q
    assert kwargs["pool_seqlens_per_q_out"] is plan.pool_seqlens_per_q
    if mode == "draft_extend_v2":
        torch.testing.assert_close(plan.effective_n_per_batch, accepted)
