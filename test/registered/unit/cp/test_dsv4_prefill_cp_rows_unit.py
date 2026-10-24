"""CPU checks for DeepSeek V4 prefill CP interleave rows: the request of each of a
rank's local rows given to the indexer, and the bounded-replay tail put back into
the rank's extend rows."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveCPStrategy,
    interleave_rows_per_request,
)
from sglang.srt.models.deepseek_v4 import _cp_tail_to_extend_rows
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ExtendMode:
    def is_idle(self):
        return False

    def is_decode(self):
        return False

    def is_target_verify(self):
        return False

    def is_extend(self):
        return True


def _local_rows_request(extend_lens, req_pool_indices, rank, cp_size):
    """The request of each row the interleave shard gives this rank."""
    global_req = torch.repeat_interleave(
        req_pool_indices, torch.tensor(extend_lens, dtype=torch.int64)
    )
    with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
        return InterleaveCPStrategy(cp_size=cp_size)._interleave_shard(global_req)


def _run_indexer_cp(extend_lens, req_pool_indices, rank, cp_size, tail=None):
    total = sum(extend_lens)
    counts = interleave_rows_per_request(extend_lens, rank, cp_size)
    num_local = sum(counts)
    pad = 3  # the rank's padding rows come after its real rows
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(late_layer_tail=tail)
    backend._low_ratio_index_topk = MagicMock()
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        attn_cp_metadata=SimpleNamespace(total_seq_lens=total),
        extend_seq_lens_cpu=extend_lens,
        extend_seq_lens=torch.tensor(extend_lens),
        req_pool_indices=req_pool_indices,
        positions=torch.arange(total),
    )
    layer = SimpleNamespace(compressor=None, indexer=object())
    with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
        backend._forward_low_ratio_sources_cp(
            layer=layer,
            x=torch.zeros(num_local + pad, 8),
            q_lora=torch.zeros(num_local + pad, 4),
            positions=torch.arange(num_local + pad),
            forward_batch=forward_batch,
            run_compressor=False,
            run_indexer=True,
        )
    backend._low_ratio_index_topk.assert_called_once()
    args, kwargs = backend._low_ratio_index_topk.call_args
    return args, kwargs, counts


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("extend_lens", [[10], [7, 1, 13], [3, 16, 2, 9]])
def test_indexer_rows_name_their_request(cp_size, extend_lens):
    req_pool_indices = torch.arange(len(extend_lens), dtype=torch.int64) * 5 + 2
    for rank in range(cp_size):
        args, kwargs, counts = _run_indexer_cp(
            extend_lens, req_pool_indices, rank, cp_size
        )
        _, x, q_lora, req, pos, _ = args
        num_local = sum(counts)
        assert x.shape[0] == q_lora.shape[0] == pos.shape[0] == num_local
        assert kwargs["rows_per_request"] == counts
        assert req.dtype == torch.int64
        assert torch.equal(
            req, _local_rows_request(extend_lens, req_pool_indices, rank, cp_size)
        )


def test_indexer_rows_follow_the_tail_lengths():
    # Under bounded replay the rank's rows are its tail rows, tail.local_lens_cpu each.
    req_pool_indices = torch.tensor([4, 9, 1], dtype=torch.int64)
    tail = SimpleNamespace(
        local_lens_cpu=[2, 0, 3],
        req_global=torch.zeros(0, dtype=torch.int64),
        pos_global=torch.zeros(0, dtype=torch.int64),
    )
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(late_layer_tail=tail)
    backend._low_ratio_index_topk = MagicMock()
    forward_batch = SimpleNamespace(
        forward_mode=_ExtendMode(),
        attn_cp_metadata=SimpleNamespace(total_seq_lens=40),
        req_pool_indices=req_pool_indices,
    )
    with get_parallel().override(attn_cp_rank=1, attn_cp_size=4):
        backend._forward_low_ratio_sources_cp(
            layer=SimpleNamespace(compressor=None, indexer=object()),
            x=torch.zeros(6, 8),
            q_lora=torch.zeros(6, 4),
            positions=torch.arange(6),
            forward_batch=forward_batch,
            run_compressor=False,
            run_indexer=True,
        )
    args, kwargs = backend._low_ratio_index_topk.call_args
    assert torch.equal(args[3], torch.tensor([4, 4, 1, 1, 1]))
    assert kwargs["rows_per_request"] == [2, 0, 3]


def _tail(token_indices, pad_rows):
    n = len(token_indices)
    return LateLayerTail(
        token_indices=torch.tensor(token_indices, dtype=torch.int64),
        positions=torch.zeros(n, dtype=torch.int64),
        extend_seq_lens=torch.zeros(1, dtype=torch.int64),
        extend_seq_lens_cpu=[n],
        swa_out_cache_loc=torch.zeros(n, dtype=torch.int64),
        pad_rows=pad_rows,
    )


@pytest.mark.parametrize("shape", [(4,), (4, 3)])
def test_cp_tail_rows_return_to_extend_rows(shape):
    # Local extend of 9 rows; the tail is rows 2, 3 (request 0) and 7, 8 (request 1),
    # followed by 2 padding rows.
    tail = _tail([2, 3, 7, 8], pad_rows=2)
    rows = torch.arange(1, 6 * torch.Size(shape[1:]).numel() + 1, dtype=torch.float32)
    rows = rows.reshape(6, *shape[1:]).to(torch.bfloat16)
    full = _cp_tail_to_extend_rows(tail=tail, rows=rows, num_tokens=9)
    assert full.shape == (9, *shape[1:])
    assert full.dtype == rows.dtype
    assert torch.equal(full[tail.token_indices], rows[:4])
    outside = torch.tensor([0, 1, 4, 5, 6])
    assert torch.count_nonzero(full[outside]) == 0


def test_cp_tail_rows_are_zero_not_uninitialized():
    tail = _tail([5], pad_rows=0)
    rows = torch.full((1, 2), float("nan"))
    full = _cp_tail_to_extend_rows(tail=tail, rows=rows, num_tokens=6)
    assert torch.isnan(full[5]).all()
    assert torch.equal(full[:5], torch.zeros(5, 2))
