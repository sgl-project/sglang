"""Tests for the fused LiteTopK DSA indexer top-k (``fused_mqa_topk_prefill``).

The fused kernel never materializes the indexer logits, so every check compares
its indices with an fp32 oracle of the indexer score: the selection must be the
exact top-k up to ties at the k-th score, in the RAGGED indexer contract
(``key - start + offset``) or the PAGED one (the KV slot
``page_table[request, key - start]``), ``-1`` padded. Parity with the dense
transforms through the indexer is tested in test_dsa_fused_mqa_topk_dispatch.py.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.fused_mqa_topk import (
    FUSED_MQA_TOPK_MAX_KEYS,
    FUSED_MQA_TOPK_MIN_KEYS,
    can_use_fused_mqa_topk_prefill,
    fused_mqa_topk_prefill,
    is_fused_mqa_topk_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dsa_fused_mqa_topk import (
    assert_topk_matches_reference,
    make_page_table,
    make_ragged_batch,
    paged_to_relative,
    reference_scores,
    row_requests,
)

register_cuda_ci(est_time=200, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_fused_mqa_topk_supported(), reason="requires CUDA SM100"
)

TOPK = 2048
OPERANDS = ("q_fp8", "k_fp8", "k_scale", "weights", "starts", "ends")


def _fused(batch, **kwargs):
    return fused_mqa_topk_prefill(*(batch[n] for n in OPERANDS), TOPK, **kwargs)


# The ends of the op's key range, a mid-length gate, and both sides of the switch
# from the streaming to the staged selection at 196608 keys.
@pytest.mark.parametrize(
    "keys,rows",
    [
        (FUSED_MQA_TOPK_MIN_KEYS, 40),
        (65536, 24),
        (196608, 9),
        (196609, 9),
        (FUSED_MQA_TOPK_MAX_KEYS, 8),
    ],
)
def test_single_request_is_exact_on_every_kernel_path(keys, rows):
    batch = make_ragged_batch([keys], [rows], seed=keys)
    assert_topk_matches_reference(_fused(batch), batch, reference_scores(batch))


def test_packed_batch_under_both_transforms():
    # A cached-prefix request, a long one, a fresh request whose rows see fewer
    # keys than top-k, and a short continuation. Offsets differ from the row
    # starts, requests own disjoint shuffled pages and the table's row stride
    # exceeds its width, so a dropped or absolute rebase, another request's
    # table row, or the width as row stride names a key the oracle rejects.
    seq_lens, extend_lens = [2500, 40000, 1200, 20000], [256, 384, 1200, 128]
    batch = make_ragged_batch(seq_lens, extend_lens, seed=5)
    scores = reference_scores(batch)
    offsets = batch["starts"] + torch.randint_like(batch["starts"], 1000)
    ragged = _fused(batch, out_offsets=offsets)
    assert_topk_matches_reference(ragged, batch, scores, out_offsets=offsets)
    page_table, row_batch = make_page_table(seq_lens, seed=5), row_requests(extend_lens)
    paged = _fused(batch, page_table=page_table, row_batch=row_batch)
    relative = paged_to_relative(paged, page_table, seq_lens, row_batch)
    assert_topk_matches_reference(relative, batch, scores)


def test_graph_replays_reuse_the_stream_workspace():
    """Replaying one capture stays exact: after the first call on a stream the op
    neither allocates nor syncs, and every launch leaves its claim counter zeroed."""
    static = make_ragged_batch([65536], [64], seed=1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = _fused(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _fused(static, out=out)
    for seed in (2, 3):
        batch = make_ragged_batch([65536], [64], seed=seed)
        for name, tensor in static.items():
            tensor.copy_(batch[name])
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert_topk_matches_reference(out, batch, reference_scores(batch))


def test_uncovered_operands_are_declined():
    batch = make_ragged_batch([FUSED_MQA_TOPK_MIN_KEYS + 1], [16], seed=0)
    q, k, k_scale, w = (batch[n] for n in OPERANDS[:4])
    assert can_use_fused_mqa_topk_prefill(q, k, k_scale, w, TOPK)
    few = FUSED_MQA_TOPK_MIN_KEYS - 1
    too_many = torch.zeros(
        FUSED_MQA_TOPK_MAX_KEYS + 1, 128, dtype=k.dtype, device="cuda"
    )
    cases = {
        "heads": (q[:, :16].contiguous(), k, k_scale, w[:, :16].contiguous(), TOPK),
        "head_dim": (
            q[..., :64].contiguous(),
            k[:, :64].contiguous(),
            k_scale,
            w,
            TOPK,
        ),
        "k_dtype": (q, k.view(torch.uint8), k_scale, w, TOPK),
        # The kernel is instantiated for top-2048 only.
        "topk": (q, k, k_scale, w, 512),
        # The vendored kernel has no path for fewer keys.
        "few_keys": (q, k[:few], k_scale[:few], w, TOPK),
        "keys": (q, too_many, k_scale.new_ones(too_many.shape[0]), w, TOPK),
        "misaligned": (q, k[1:], k_scale[1:], w, TOPK),
        "strided": (q[::2], k, k_scale, w[::2], TOPK),
    }
    for case, operands in cases.items():
        assert not can_use_fused_mqa_topk_prefill(*operands), case
    # Called anyway, the op rejects too few keys instead of launching.
    with pytest.raises(RuntimeError, match="keys must be in"):
        _fused(dict(batch, k_fp8=k[:few], k_scale=k_scale[:few]))


def test_more_rows_than_one_scheduler_stride_are_exact():
    """Past 4096 8-row blocks (32768 rows) a grid-stride block walk wraps; every row
    must still get its own top-k, also from the exact rescan of an early block."""
    batch = make_ragged_batch([65536], [4096], seed=65536)
    # A subnormal head weight sends the row's block to the exact rescan.
    batch["weights"][0, 0] = 1e-39
    tiled = dict(batch)
    for name in ("q_fp8", "weights", "starts", "ends"):
        tiled[name] = batch[name].repeat(9, *[1] * (batch[name].dim() - 1))
    out = _fused(tiled)
    scores = reference_scores(batch)
    for part in out.split(4096):
        assert_topk_matches_reference(part, batch, scores)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
