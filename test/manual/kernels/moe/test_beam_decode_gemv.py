"""Manual correctness test for the SM103 single-token Beam MoE decode GEMV.

Run on one SM103 GPU (GB300): ``python test/manual/kernels/moe/test_beam_decode_gemv.py``.

Compares the weighted per-expert outputs with an independent FP64 reference that
keeps the kernel's BF16 rounding points (gate/up, activation, weighted down
projection), eagerly and through CUDA-graph replays whose inputs change between
replays.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.beam_decode_gemv import beam_decode_gemv

# Beam's experts, hidden size and TP4 intermediate size.
E, H, I = 256, 5120, 640
TOL = dict(rtol=1.6e-2, atol=1e-3)


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    # Interleaved gate/up rows: row 2j is gate j and row 2j + 1 is up j.
    w13 = (torch.randn(E, 2 * I, H, device="cuda") / H**0.5).bfloat16()
    w2 = (torch.randn(E, H, I, device="cuda") / I**0.5).bfloat16()
    return w13, w2


def _reference(h, ids, tw, w13, w2):
    out = []
    for r in range(8):
        e = int(ids[0, r])
        y = (w13[e].double() @ h[0].double()).bfloat16().float()
        gate, up = y[0::2], y[1::2]
        act = (gate / (1 + torch.exp(-gate)) * up).bfloat16().double()
        out.append(((w2[e].double() @ act).float() * tw[0, r]).bfloat16())
    return torch.stack(out)[None]


def _refill(seed, h, ids, tw):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h.copy_(torch.randn(1, H, device="cuda", generator=g))
    ids.copy_(torch.randperm(E, device="cuda", generator=g)[:8].view(1, 8))
    tw.copy_(torch.softmax(torch.randn(1, 8, device="cuda", generator=g), -1))


def test_matches_fp64_reference_under_graph_replay(weights):
    w13, w2 = weights
    h = torch.empty(1, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(1, 8, device="cuda", dtype=torch.int64)
    tw = torch.empty(1, 8, device="cuda")
    _refill(0, h, ids, tw)
    expected = _reference(h, ids, tw, w13, w2)
    torch.testing.assert_close(beam_decode_gemv(h, w13, w2, tw, ids), expected, **TOL)
    torch.testing.assert_close(
        beam_decode_gemv(h, w13, w2, tw, ids.int()), expected, **TOL
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = beam_decode_gemv(h, w13, w2, tw, ids)
    for seed in range(1, 6):
        _refill(seed, h, ids, tw)
        graph.replay()
        torch.testing.assert_close(out, _reference(h, ids, tw, w13, w2), **TOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
