"""MiniMax sparse-attention decode sizes its split-K grids (and, on gfx950, the score
tile) from the batch size. With --enable-deterministic-inference they are sized for a
fixed batch size instead, so a request's output must not depend on its batch: decode
one request alone and among other requests, and compare its rows bit for bit.
"""

import random
import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.common import utils as sparse_utils
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
NUM_REQS = 64
MAX_KV_LEN = 8192
BATCH_SIZES = [2, 3, 4, 5, 8, 16, 17, 32, 64]


@pytest.fixture(autouse=True)
def _deterministic_mode():
    with patch.object(sparse_utils, "_is_deterministic_inference", return_value=True):
        yield


def _pool(num_q_heads, num_kv_heads, seed):
    """K/V and a req_to_token row for NUM_REQS requests; batches pick rows by slot id."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_shape = (NUM_REQS * MAX_KV_LEN, num_kv_heads, HEAD_DIM)
    k, v = (
        torch.randn(kv_shape, generator=g, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    q = torch.randn(
        NUM_REQS,
        num_q_heads,
        HEAD_DIM,
        generator=g,
        device="cuda",
        dtype=torch.bfloat16,
    )
    req_to_token = torch.arange(
        NUM_REQS * MAX_KV_LEN, dtype=torch.int32, device="cuda"
    ).view(NUM_REQS, MAX_KV_LEN)
    rng = random.Random(seed)
    seq_lens = [rng.randint(TOPK * BLOCK_SIZE, MAX_KV_LEN) for _ in range(NUM_REQS)]
    return (
        q,
        k,
        v,
        req_to_token,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
    )


def _batches(seed):
    """(target position, slot ids) for batches of every size, the target request 0 at
    a random position among random batchmates."""
    rng = random.Random(seed)
    for bs in BATCH_SIZES:
        mates = rng.sample(range(1, NUM_REQS), bs - 1)
        pos = rng.randrange(bs)
        yield pos, mates[:pos] + [0] + mates[pos:]


# MiniMax-M3's sparse-layer indexer at TP1 / TP2 / TP8: index heads per KV head, scores
# only (its sparse layers disable the index-value output).
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 4), (2, 2), (8, 1)])
def test_flash_decode_with_topk_idx_batch_invariant(num_q_heads, num_kv_heads):
    q, k, v, req_to_token, seq_lens = _pool(num_q_heads, num_kv_heads, seed=0)

    def run(slots):
        ids = torch.tensor(slots, dtype=torch.int64, device="cuda")
        lens = seq_lens[ids]
        _, topk_idx, _ = flash_decode_with_topk_idx(
            q[ids],
            None,
            k,
            v,
            req_to_token,
            lens,
            int(lens.max().item()),
            ids,
            BLOCK_SIZE,
            TOPK,
            0,
            1,
            disable_index_value=True,
        )
        return topk_idx

    alone = run([0])
    for pos, slots in _batches(seed=1):
        assert torch.equal(run(slots)[:, pos], alone[:, 0]), f"batch of {len(slots)}"


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(64, 4), (32, 2), (16, 1)])
def test_flash_decode_with_gqa_share_sparse_batch_invariant(num_q_heads, num_kv_heads):
    q, k, v, req_to_token, seq_lens = _pool(num_q_heads, num_kv_heads, seed=2)
    rng = random.Random(3)
    # TOPK distinct blocks per request and kv head, inside the request's context
    blocks = torch.tensor(
        [
            [
                sorted(rng.sample(range(-(-int(n) // BLOCK_SIZE)), TOPK))
                for n in seq_lens.tolist()
            ]
            for _ in range(num_kv_heads)
        ],
        dtype=torch.int32,
        device="cuda",
    )

    def run(slots):
        ids = torch.tensor(slots, dtype=torch.int64, device="cuda")
        return flash_decode_with_gqa_share_sparse(
            q[ids],
            None,
            k,
            v,
            req_to_token,
            seq_lens[ids],
            ids,
            BLOCK_SIZE,
            blocks[:, ids].contiguous(),
        )

    alone = run([0])
    for pos, slots in _batches(seed=4):
        assert torch.equal(run(slots)[pos], alone[0]), f"batch of {len(slots)}"


def test_grid_batch_size():
    assert sparse_utils.grid_batch_size(1) == sparse_utils.DETERMINISTIC_GRID_BATCH_SIZE
    assert (
        sparse_utils.grid_batch_size(64) == sparse_utils.DETERMINISTIC_GRID_BATCH_SIZE
    )
    with patch.object(sparse_utils, "_is_deterministic_inference", return_value=False):
        assert sparse_utils.grid_batch_size(1) == 1
        assert sparse_utils.grid_batch_size(64) == 64


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
