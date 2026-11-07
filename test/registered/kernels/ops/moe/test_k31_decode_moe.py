"""Kimi-K3.1 decode MoE kernels (``k31_decode_moe``): FC2 rows against an fp64
reference on random MXFP4 weights, the packed top-k handoff, and CUDA graph
replays."""

import functools
import sys

import pytest
import torch

from sglang.kernels.arch import current_arch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E, H, I, TOPK = 896, 3584, 384, 16
# Mean error relative to the reference RMS. The kernel and flashinfer both land
# near 1.1e-3 on these inputs (MXFP8 activations and intermediate).
MEAN_REL_TOL = 3e-3


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = current_arch()
    return arch.is_sm(10, 0) or arch.is_sm(10, 3)


pytestmark = pytest.mark.skipif(not _supported(), reason="SM100 / SM103 kernels")

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float64)


def _dequant_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] (element 2j in the low nibble), E8M0 [..., K/32] -> f64 [..., K]."""
    lut = _E2M1.to(packed.device)
    v = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1)
    s = torch.pow(2.0, scales.double() - 127.0)
    return (v.flatten(-2).unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2)


def _dequant_mxfp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    sc = torch.pow(2.0, s.double() - 127.0)
    return (q.double().unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2)


def _quant_dequant_mxfp8(a: torch.Tensor) -> torch.Tensor:
    """Blocks of 32: UE8M0 scale 2^ceil(log2(amax / 448)), E4M3 round to nearest."""
    blk = a.unflatten(-1, (-1, 32))
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blk / torch.pow(2.0, e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.double() * torch.pow(2.0, e)).flatten(-2)


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * (25.0 * torch.tanh(up / 25.0))
    )


@functools.cache
def _weights():
    """Random checkpoint-layout MXFP4 weights (w13 rows: [gate; up]) and the
    flashinfer_mxfp4 layout the engine prepares from them."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    # scales around 2^-7 keep the FC outputs O(1)
    s13 = torch.randint(
        117, 122, (E, 2 * I, H // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    s2 = torch.randint(
        118, 123, (E, H, I // 32), dtype=torch.uint8, device="cuda", generator=g
    )

    # as Mxfp4MoEMethod.process_weights_after_loading: (up_i, gate_i) row pairs,
    # the 128-row epilogue permute, then the scale interleave
    pair = torch.empty(2 * I, dtype=torch.long)
    pair[0::2] = torch.arange(I) + I
    pair[1::2] = torch.arange(I)
    pw13, ps13 = w13[:, pair.cuda()], s13[:, pair.cuda()]
    cache = {}
    p13 = get_w2_permute_indices_with_cache(cache, pw13[0], 128).cuda()
    q13 = get_w2_permute_indices_with_cache(
        cache, ps13[0], 128, num_elts_per_sf=16
    ).cuda()
    p2 = get_w2_permute_indices_with_cache(cache, w2[0], 128).cuda()
    q2 = get_w2_permute_indices_with_cache(cache, s2[0], 128, num_elts_per_sf=16).cuda()
    fw13, fw2 = pw13[:, p13].contiguous(), w2[:, p2].contiguous()
    fs13, fs2 = torch.empty_like(ps13), torch.empty_like(s2)
    for e in range(E):
        fs13[e] = nvfp4_block_scale_interleave(ps13[e][q13].contiguous()).reshape(
            2 * I, H // 32
        )
        fs2[e] = nvfp4_block_scale_interleave(s2[e][q2].contiguous()).reshape(
            H, I // 32
        )
    return (w13, s13, w2, s2), (fw13, fs13, fw2, fs2)


def _inputs(num_tokens: int, kind: str, seed: int):
    """MXFP8 activations and a top-16 routing. "hot": every token picks among 24
    experts, so an expert spans several 32-token chunks."""
    from flashinfer import mxfp8_quantize

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, H, generator=g).to(torch.bfloat16).cuda()
    xq, xs = mxfp8_quantize(x, False)
    xs = xs.view(torch.uint8).reshape(num_tokens, H // 32)
    scores = torch.rand(num_tokens, E, generator=g)
    if kind == "hot":
        scores[:, :24] += 1.0
    ids = scores.topk(TOPK, dim=1).indices.to(torch.int32).cuda()
    w = torch.softmax(torch.randn(num_tokens, TOPK, generator=g), dim=1).cuda()
    return xq, xs, ids, w


def _reference_rows(xq, xs, ids) -> torch.Tensor:
    """FC2 output of every (token, k) slot, unweighted, [T * 16, H] f64."""
    (w13, s13, w2, s2), _ = _weights()
    x = _dequant_mxfp8(xq, xs)
    flat = ids.flatten().long()
    act = torch.zeros(flat.numel(), I, dtype=torch.float64, device="cuda")
    y = torch.zeros(flat.numel(), H, dtype=torch.float64, device="cuda")
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero().flatten()
        h = x[rows // TOPK] @ _dequant_fp4(w13[e], s13[e]).T
        act[rows] = _situ(h[:, :I], h[:, I:])
    act = _quant_dequant_mxfp8(act)
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero().flatten()
        y[rows] = act[rows] @ _dequant_fp4(w2[e], s2[e]).T
    return y


def _run(xq, xs, ids, w, packed: bool):
    from sglang.kernels.ops.moe.k31_decode_moe import k31_moe_mxfp4

    _, (fw13, fs13, fw2, fs2) = _weights()
    if packed:
        wbits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        ids = ((ids << 16) | wbits).contiguous()
    return k31_moe_mxfp4(xq, xs, ids, fw13, fs13, fw2, fs2, packed=packed)


def _mean_rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref).abs().mean() / ref.pow(2).mean().sqrt()).item()


def _check(gemm2_out, expanded_idx, w, ref_rows) -> None:
    num_slots = ref_rows.shape[0]
    assert torch.equal(
        torch.sort(expanded_idx.long())[0], torch.arange(num_slots, device="cuda")
    ), "expanded_idx is not a permutation"
    rows = gemm2_out[expanded_idx.long()]
    assert _mean_rel(rows, ref_rows) < MEAN_REL_TOL
    # the engine's finalize: sum_k w[t, k] * row (t, k)
    out = (rows.double().view(-1, TOPK, H) * w.double().unsqueeze(-1)).sum(1)
    ref_out = (ref_rows.view(-1, TOPK, H) * w.double().unsqueeze(-1)).sum(1)
    assert _mean_rel(out, ref_out) < MEAN_REL_TOL


@pytest.mark.parametrize("kind", ["uniform", "hot"])
@pytest.mark.parametrize("num_tokens", [1, 30, 96, 256])
@pytest.mark.parametrize("packed", [False, True])
def test_k31_decode_moe(num_tokens: int, kind: str, packed: bool) -> None:
    xq, xs, ids, w = _inputs(num_tokens, kind, seed=num_tokens)
    gemm2_out, expanded_idx, weights = _run(xq, xs, ids, w, packed)
    torch.cuda.synchronize()
    if packed:
        assert torch.equal(weights, w.to(torch.bfloat16))
    _check(gemm2_out, expanded_idx, w, _reference_rows(xq, xs, ids))


def test_k31_decode_moe_graph_replays() -> None:
    """Many replays of a graph that chains several calls of different sizes: a
    lost handoff between the kernels' roles shows up as a hang or a wrong row."""
    cases = [
        _inputs(t, kind, seed=100 + t)
        for t, kind in [(6, "uniform"), (36, "hot"), (240, "uniform")]
    ]
    for case in cases:  # build the module and warm the allocator outside the capture
        _run(*case, packed=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [_run(*case, packed=False) for case in cases]
    for _ in range(200):
        graph.replay()
    torch.cuda.synchronize()
    for (xq, xs, ids, w), (gemm2_out, expanded_idx, _) in zip(cases, outs):
        _check(gemm2_out, expanded_idx, w, _reference_rows(xq, xs, ids))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
