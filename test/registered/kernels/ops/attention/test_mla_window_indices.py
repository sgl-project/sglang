"""mla_window_sparse_indices: per query row, the flat paged-cache rows of the
window keys [p - window_left, p] that one DCP rank owns, against a Python
reference of the cyclic DCP layout (position pos on rank pos % N at local slot
req_to_token[pos] // N, page through v2p when given). The union over ranks
must be exactly the window, each key once."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.mla_window_indices import (
    mla_window_sparse_indices,
    mla_window_top_k,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64


def _layout(seq_lens, dcp, with_v2p, seed):
    """req_to_token of page-aligned requests in a shuffled pool, so slot % dcp
    cycles with position, and a shuffled virtual->physical page table."""
    g = torch.Generator().manual_seed(seed)
    cycle = PAGE * dcp
    runs = [-(-s // cycle) for s in seq_lens]
    order = torch.randperm(sum(runs) + 1, generator=g)[: len(seq_lens)] + 1
    req_to_token = torch.full((len(seq_lens) + 3, max(seq_lens) + 16), -1, dtype=torch.int32)
    pool_idx = torch.randperm(req_to_token.shape[0], generator=g)[: len(seq_lens)]
    base = 0
    for i, s in enumerate(seq_lens):
        base = int(order[i]) * cycle * 64
        req_to_token[pool_idx[i], :s] = torch.arange(base, base + s, dtype=torch.int32)
    num_pages = int(req_to_token.max()) // dcp // PAGE + 1
    v2p = torch.randperm(num_pages, generator=g).to(torch.int64) if with_v2p else None
    return req_to_token, pool_idx, v2p


def _reference(req_to_token, pool_idx, positions, q_len, window_left, dcp, rank, v2p):
    rows = []
    for r, p in enumerate(positions.tolist()):
        req = int(pool_idx[r // q_len])
        keys = []
        for pos in range(max(0, p - window_left), p + 1):
            if pos % dcp != rank:
                continue
            slot = int(req_to_token[req, pos]) // dcp
            page = slot // PAGE
            if v2p is not None:
                page = int(v2p[page])
            keys.append(page * PAGE + slot % PAGE)
        rows.append(keys)
    return rows


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("window_left", [4095, 127])
@pytest.mark.parametrize("with_v2p", [False, True])
def test_window_indices(dcp, window_left, with_v2p):
    q_len = 8
    ctx = [0, 5, window_left - 3, window_left, window_left + 1, 3 * window_left + 77]
    seq_lens = [c + q_len for c in ctx]
    req_to_token, pool_idx, v2p = _layout(seq_lens, dcp, with_v2p, seed=dcp + window_left)
    positions = torch.tensor(
        [c + i for c in ctx for i in range(q_len)], dtype=torch.int64
    )
    top_k = mla_window_top_k(window_left, dcp)
    union = [[] for _ in positions]
    for rank in range(dcp):
        indices, counts = mla_window_sparse_indices(
            req_to_token=req_to_token.cuda(),
            req_pool_indices=pool_idx.cuda(),
            positions=positions.cuda(),
            window_left=window_left,
            dcp_size=dcp,
            dcp_rank=rank,
            page_size=PAGE,
            v2p=None if v2p is None else v2p.cuda(),
        )
        assert indices.shape == (positions.numel(), top_k)
        ref = _reference(req_to_token, pool_idx, positions, q_len, window_left, dcp, rank, v2p)
        indices, counts = indices.cpu(), counts.cpu()
        for r, keys in enumerate(ref):
            assert int(counts[r]) == len(keys), (rank, r)
            assert indices[r, : len(keys)].tolist() == keys, (rank, r)
            assert (indices[r, len(keys) :] == -1).all(), (rank, r)
            union[r] += [(rank, k) for k in keys]
    for r, p in enumerate(positions.tolist()):
        assert len(union[r]) == min(p, window_left) + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
