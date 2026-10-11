"""Correctness for the experimental direct-candidate BF16 scorer."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.candidate_bf16_mqa import (
    candidate_bf16_mqa_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(q, keys, weights, blocks, lens):
    positions = (
        blocks.long()[:, :, None] * 8 + torch.arange(8, device=q.device)
    ).flatten(1)
    valid = (positions >= 0) & (positions < keys.shape[0]) & (positions < lens[:, None])
    out = torch.full(positions.shape, -torch.inf, device=q.device)
    # Bound reference memory, independently of the tested Triton tile size.
    if keys.shape[0] and positions.shape[1]:
        for lo in range(0, q.shape[0], 4):
            p = positions[lo : lo + 4]
            k = keys[p.clamp(0, keys.shape[0] - 1)]
            dots = torch.einsum("rhd,rnd->rhn", q[lo : lo + 4], k)
            out[lo : lo + 4] = (
                (dots.relu() * weights[lo : lo + 4, :, None]).sum(1).float()
            )
    return out.masked_fill(~valid, -torch.inf), positions, valid


def inputs(rows, length, count, *, strided=False, dyadic=False):
    torch.manual_seed(431 + rows + length)
    factor = 2 if strided else 1
    q = torch.randn(rows, 32, 128 * factor, device="cuda", dtype=torch.bfloat16)[
        ..., ::factor
    ]
    k = torch.randn(length, 128 * factor, device="cuda", dtype=torch.bfloat16)[
        :, ::factor
    ]
    w = torch.randn(rows, 32 * factor, device="cuda", dtype=torch.bfloat16)[:, ::factor]
    if dyadic:
        # Exact binary products/sums avoid masking a numeric discrepancy with tolerance.
        q.copy_(q.round().clamp(-2, 2) / 8)
        k.copy_(k.round().clamp(-2, 2) / 8)
        w.copy_(w.round().clamp(-2, 2) / 8)
    blocks = torch.randint(
        -1,
        max(2, (length + 7) // 8 + 2),
        (rows, count * factor),
        device="cuda",
        dtype=torch.int64,
    )[:, ::factor]
    lens = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    if rows > 1:
        lens[0] = 0
        lens[1] = max(0, length - 3)
    return q, k, w, blocks, lens


@pytest.mark.parametrize("length,count", [(0, 3), (1, 1), (7, 3), (65, 9), (511, 67)])
@pytest.mark.parametrize("strided", [False, True])
def test_exact_dyadic_boundaries(length, count, strided):
    args = inputs(3, length, count, strided=strided, dyadic=True)
    actual = candidate_bf16_mqa_logits(*args)
    expected, _, _ = oracle(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows,count", [(0, 3), (3, 0)])
def test_empty_output(rows, count):
    assert candidate_bf16_mqa_logits(*inputs(rows, 9, count)).shape == (rows, count * 8)


@pytest.mark.parametrize(
    "rows,length,count", [(1, 16385, 2048), (17, 32768, 2048), (128, 65536, 2048)]
)
def test_real_candidate_width(rows, length, count):
    # Positive weights avoid cancellation in the error bound and match an
    # additional common regime; signed weights are covered by the exact cases.
    args = inputs(rows, length, count)
    args[2].abs_()
    actual = candidate_bf16_mqa_logits(*args)
    expected, _, valid = oracle(*args)
    assert torch.equal(torch.isneginf(actual), ~valid)
    # BF16 rounding points are preserved, but GEMM reduction order may differ.
    # Bound score error; do not silently demand identical indices across ties.
    torch.testing.assert_close(
        actual[valid], expected[valid], rtol=1 / 128, atol=1 / 128
    )
    take = min(512, actual.shape[1])
    vals, indices = actual.topk(take, dim=1)
    ref_cutoff = expected.topk(take, dim=1).values[:, -1:]
    ref_selected = expected.gather(1, indices)
    finite = torch.isfinite(vals) & torch.isfinite(ref_cutoff)
    tolerance = (
        expected.abs().masked_fill(~valid, 0).amax(1, keepdim=True) / 64 + 1 / 64
    )
    assert torch.all((ref_selected + tolerance >= ref_cutoff) | ~finite)


@pytest.mark.parametrize("ties", [False, True])
def test_topk_and_attention_exact(ties):
    # A simple exact scorer with >512 unique candidates, shuffled block order.
    q = torch.zeros(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    q[:, 0, 0] = 1
    keys = torch.zeros(1024, 128, device="cuda", dtype=torch.bfloat16)
    keys[:, 0] = 1 if ties else torch.arange(1024, device="cuda").to(torch.bfloat16)
    weights = torch.ones(3, 32, device="cuda", dtype=torch.bfloat16)
    blocks = torch.stack([torch.randperm(128, device="cuda") for _ in range(3)])
    lens = torch.tensor([0, 513, 1021], device="cuda", dtype=torch.int64)
    args = q, keys, weights, blocks, lens
    actual = candidate_bf16_mqa_logits(*args)
    expected, positions, _ = oracle(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    av, ai = actual.topk(512, dim=1, sorted=False)
    ev, ei = expected.topk(512, dim=1, sorted=False)
    assert torch.equal(ai, ei)  # Same scores and same downstream Torch selector.
    selected = positions.gather(1, ai).masked_fill(~torch.isfinite(av), -1)
    assert torch.all((selected < lens[:, None]) | (selected == -1))
    # Check the selected positions produce the same downstream attention output.
    value = torch.randn(1024, 16, device="cuda")
    logits = torch.randn(3, 1024, device="cuda")

    def attention(index, values):
        p = positions.gather(1, index)
        finite = torch.isfinite(values)
        score = logits.gather(1, p).masked_fill(~finite, -torch.inf)
        probability = torch.softmax(score, dim=1).nan_to_num()
        return (probability[..., None] * value[p]).sum(1)

    torch.testing.assert_close(attention(ai, av), attention(ei, ev), rtol=0, atol=0)


def test_cuda_graph_replay_and_invalid_blocks():
    args = inputs(3, 511, 67, dyadic=True)
    for _ in range(3):
        candidate_bf16_mqa_logits(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = candidate_bf16_mqa_logits(*args)
    args[3].fill_(-1)
    graph.replay()
    assert torch.isneginf(actual).all()
    args[3].zero_()
    args[4].fill_(5)
    graph.replay()
    torch.testing.assert_close(actual, oracle(*args)[0], rtol=0, atol=0)


def test_reject_non_bf16_weights():
    args = list(inputs(1, 8, 1))
    args[2] = args[2].float()
    with pytest.raises(ValueError, match="BF16"):
        candidate_bf16_mqa_logits(*args)


def test_random_signed_weights_cancellation():
    args = inputs(5, 513, 67, strided=True)
    q, keys, weights, _, _ = args
    actual = candidate_bf16_mqa_logits(*args)
    expected, positions, valid = oracle(*args)
    assert torch.equal(torch.isneginf(actual), ~valid)
    dots = torch.einsum("rhd,rnd->rhn", q, keys[positions.clamp(0, 512)])
    terms = (dots.relu() * weights[:, :, None]).float()
    # Relative error in the final sum is unsuitable near cancellation. Bound
    # error against the absolute head contributions instead, with BF16 epsilon.
    tolerance = terms.abs().sum(1) / 64 + 1e-5
    assert torch.all((actual[valid] - expected[valid]).abs() <= tolerance[valid])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
