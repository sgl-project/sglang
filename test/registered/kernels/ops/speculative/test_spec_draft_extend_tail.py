# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL``: the topk-1 draft-extend tail in two launches.

``draft_extend_tail`` must equal the stock statements bit for bit (values,
dtypes, shapes, strides):

    topk_index = torch.argmax(logits, dim=-1, keepdim=True)
    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    seed = capture[select_index]
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.kernels.ops.speculative.row_argmax import draft_extend_tail  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

VOCAB = 154880
WIDTH = 4
SEED_WIDTH = 2048


def _stock(logits, capture, select_index):
    index = torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.ones_like(index, dtype=torch.float32)
    seed = capture[select_index] if capture is not None else None
    return probs, index, seed


def _same(a, b, what):
    if a is None or b is None:
        assert a is None and b is None, what
        return
    assert (a.dtype, a.shape, a.stride()) == (b.dtype, b.shape, b.stride()), what
    assert torch.equal(a, b), what


def _logits(bs, case, gen):
    x = torch.randn(bs, VOCAB, generator=gen, device="cuda")
    for r in range(bs):
        kind = (case + r) % 8
        if kind == 1:  # tie at the first and last column
            x[r, 0] = x[r, -1] = x[r].max() + 1
        elif kind == 2:  # tie at random columns
            cols = torch.randint(0, VOCAB, (3,), generator=gen, device="cuda")
            x[r, cols] = x[r].max() + 1
        elif kind == 3:
            x[r, VOCAB // 3] = float("inf")
        elif kind == 4:
            x[r] = float("-inf")
        elif kind == 5:
            x[r, VOCAB - 7] = float("nan")
        elif kind == 6:
            x[r, 5] = float("nan")
            x[r, 9] = float("nan")
        elif kind == 7:
            x[r] = x[r].round()  # many near-ties, exact equal values
    return x


def _select_index(bs, gen, synthetic_len=None):
    if synthetic_len is None:
        accept = torch.randint(1, WIDTH + 1, (bs,), generator=gen, device="cuda")
    else:
        accept = torch.full((bs,), synthetic_len, device="cuda")
    return torch.arange(bs, device="cuda") * WIDTH + accept - 1


@pytest.mark.parametrize("bs", [1, 2, 8])
@pytest.mark.parametrize("with_seed", [False, True])
@pytest.mark.parametrize("synthetic_len", [None, 4])
def test_matches_stock(bs, with_seed, synthetic_len):
    gen = torch.Generator(device="cuda").manual_seed(bs * 31 + (synthetic_len or 0))
    capture = (
        torch.randint(
            -1, 1 << 20, (bs * WIDTH + 5, SEED_WIDTH), generator=gen, device="cuda"
        ).to(torch.int32)
        if with_seed
        else None
    )
    for case in range(8):
        logits = _logits(bs, case, gen)
        select = _select_index(bs, gen, synthetic_len)
        ref = _stock(logits, capture, select)
        got = draft_extend_tail(logits, capture, select if with_seed else None)
        for name, a, b in zip(("topk_p", "topk_index", "seed"), got, ref):
            _same(a, b, (case, name))


@pytest.mark.parametrize("bs", [1, 2, 8])
def test_graph_replay(bs):
    gen = torch.Generator(device="cuda").manual_seed(500 + bs)
    capture = torch.randint(
        -1, 1 << 20, (bs * WIDTH, SEED_WIDTH), generator=gen, device="cuda"
    ).to(torch.int32)
    logits = _logits(bs, 0, gen)
    select = _select_index(bs, gen)
    draft_extend_tail(logits, capture, select)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = draft_extend_tail(logits, capture, select)
    for case in range(8):
        logits.copy_(_logits(bs, case, gen))
        select.copy_(_select_index(bs, gen))
        capture.random_(-1, 1 << 20, generator=gen)
        graph.replay()
        torch.cuda.synchronize()
        ref = _stock(logits, capture, select)
        for name, a, b in zip(("topk_p", "topk_index", "seed"), got, ref):
            _same(a, b, (case, name))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
