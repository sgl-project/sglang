"""Correctness tests for the Inkling gate JIT kernels.

Covers the v2 standalone top-k+renorm kernel, the gate GEMV kernel, the fused
GEMV+gate kernel (including CUDA-graph replay safety of its ticket), and the
`sigmoid_gate_topk_renorm` dispatch that routes the production shape to the
JIT kernel.

The reference emulates the triton kernel's selection bit-exactly (uint64 sort
keys: fp32 selection score in the high bits, `256 - expert` in the low 16), so
index comparisons are exact, not tolerance-based.
"""

import os

import pytest
import torch

from sglang.jit_kernel.inkling_gate_topk_renorm import (
    inkling_gate_gemv,
    inkling_gate_gemv_fused,
    inkling_gate_topk_renorm,
    inkling_gate_topk_renorm_v2,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils.sigmoid_gate_topk_renorm import (
    sigmoid_gate_topk_renorm,
)

HIDDEN = 6144
N_ROUTED = 256
N_SHARED = 2
N_TOTAL = 258
N_PADDED = 264
TOPK = 6
ROUTE_SCALE = 8.0

_is_ci = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Cover the launch-shape boundaries: sliced GEMV (<=2), smem-sliced (3..7),
# warp-per-token (>=8), fused cap (64), multi-block v2 grids.
TOKENS_FULL = [1, 2, 3, 5, 8, 17, 64, 127, 512, 4096]
TOKENS_CI = [1, 3, 8, 64]
TOKENS = TOKENS_CI if _is_ci else TOKENS_FULL

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA-only JIT kernels",
)


def _make_inputs(m: int, seed: int = 0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    x = (torch.randn((m, HIDDEN), device=dev) * 0.05).to(torch.bfloat16)
    weight = (torch.randn((N_PADDED, HIDDEN), device=dev) * 0.02).to(torch.bfloat16)
    weight[N_TOTAL:].zero_()
    bias = torch.randn((N_ROUTED,), device=dev) * 0.1
    global_scale = torch.tensor([1.25], device=dev)
    logits = torch.mm(x, weight.T, out_dtype=torch.float32)[:, :N_TOTAL]
    return x, weight, bias, global_scale, logits


def _ref_gate(logits: torch.Tensor, bias: torch.Tensor, global_scale: torch.Tensor):
    """Bit-exact emulation of the triton kernel's selection + renorm."""
    raw = logits[:, :N_ROUTED].float()
    sel = torch.sigmoid(raw) + bias[None, :]
    bits = sel.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = bits ^ torch.where(bits & 0x80000000 != 0, 0xFFFFFFFF, 0x80000000)
    idx = torch.arange(N_ROUTED, device=logits.device, dtype=torch.int64)
    packed = (key << 16) | (N_ROUTED - idx)[None, :]
    top = torch.sort(packed, dim=1, descending=True).values[:, :TOPK]
    indices = (N_ROUTED - (top & 0xFFFF)).to(torch.int32)

    sel_raw = raw.gather(dim=1, index=indices.long())
    active = torch.cat(
        [torch.sigmoid(sel_raw), torch.sigmoid(logits[:, N_ROUTED:N_TOTAL].float())],
        dim=1,
    )
    weights = active / active.sum(dim=1, keepdim=True)
    weights = weights * (ROUTE_SCALE * global_scale.item())
    return weights[:, :TOPK].contiguous(), indices, weights[:, TOPK:].contiguous()


def _unpack(packed: torch.Tensor):
    indices = (packed >> 16).to(torch.int32)
    weights = (packed & 0xFFFF).to(torch.int32).to(torch.uint16).view(torch.bfloat16)
    return weights.float(), indices


def _assert_gate_output(out, logits, bias, global_scale, packed_mode: bool):
    routed_w, indices, shared_w, packed = out
    ref_w, ref_idx, ref_sh = _ref_gate(logits, bias, global_scale)
    if packed_mode:
        routed_w, indices = _unpack(packed)
        watol = 2e-2  # bf16-rounded weights
    else:
        watol = 1e-5
    torch.testing.assert_close(indices, ref_idx, atol=0, rtol=0)
    torch.testing.assert_close(routed_w, ref_w, atol=watol, rtol=1e-3)
    torch.testing.assert_close(shared_w, ref_sh, atol=1e-5, rtol=1e-3)


@requires_cuda
@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("warps", [0] if _is_ci else [0, 1, 2, 4, 8])
def test_v2_matches_reference(m: int, packed: bool, warps: int):
    _, _, bias, global_scale, logits = _make_inputs(m)
    out = inkling_gate_topk_renorm_v2(
        logits,
        bias,
        global_scale,
        ROUTE_SCALE,
        return_packed=packed,
        enable_pdl=False,
        warps_per_block=warps,
    )
    _assert_gate_output(out, logits, bias, global_scale, packed)


@requires_cuda
@pytest.mark.parametrize("m", TOKENS_CI)
def test_v1_matches_reference(m: int):
    _, _, bias, global_scale, logits = _make_inputs(m)
    routed_w, shared_w, indices = inkling_gate_topk_renorm(
        logits, bias, global_scale, ROUTE_SCALE
    )
    out = (routed_w, indices.to(torch.int32), shared_w, None)
    _assert_gate_output(out, logits, bias, global_scale, packed_mode=False)


@requires_cuda
@pytest.mark.parametrize("m", TOKENS)
def test_gemv_matches_cublas(m: int):
    x, weight, _, _, logits_ref = _make_inputs(m)
    logits = inkling_gate_gemv(x, weight, enable_pdl=False)
    assert logits.shape == (m, N_TOTAL)
    torch.testing.assert_close(logits, logits_ref, atol=1e-4, rtol=1e-4)


@requires_cuda
@pytest.mark.parametrize("m", [t for t in TOKENS if t <= 64])
@pytest.mark.parametrize("packed", [False, True])
def test_gemv_fused_matches_pair(m: int, packed: bool):
    """The fused kernel shares the GEMV + epilogue code paths with the split
    pair, so their outputs must match bitwise."""
    x, weight, bias, global_scale, _ = _make_inputs(m)
    logits = inkling_gate_gemv(x, weight, enable_pdl=False)
    pair = inkling_gate_topk_renorm_v2(
        logits, bias, global_scale, ROUTE_SCALE, return_packed=packed
    )
    fused = inkling_gate_gemv_fused(
        x, weight, bias, global_scale, ROUTE_SCALE, return_packed=packed
    )
    _assert_gate_output(fused, logits, bias, global_scale, packed)
    for a, b in zip(pair, fused):
        if a is not None:
            assert torch.equal(a, b)


@requires_cuda
def test_gemv_fused_graph_replay():
    """Ticket self-reset: the fused kernel must produce correct results across
    CUDA-graph replays without any workspace re-initialization."""
    x, weight, bias, global_scale, _ = _make_inputs(16)
    run = lambda: inkling_gate_gemv_fused(  # noqa: E731
        x, weight, bias, global_scale, ROUTE_SCALE, return_packed=False
    )
    eager = run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = [run() for _ in range(3)]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    for out in captured:
        for a, b in zip(out, eager):
            if a is not None:
                assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("m", TOKENS_CI)
@pytest.mark.parametrize("packed", [False, True])
def test_dispatch_matches_triton(m: int, packed: bool):
    """sigmoid_gate_topk_renorm must give identical selections whether the JIT
    dispatch is on or off (production shape, padded-slice layout)."""
    _, _, bias, global_scale, logits = _make_inputs(m)

    def run():
        return sigmoid_gate_topk_renorm(
            logits,
            TOPK,
            N_SHARED,
            ROUTE_SCALE,
            global_scale,
            bias,
            return_packed_topk=packed,
        )

    with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
        triton_out = run()
    with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(True):
        jit_out = run()

    def parts(out):
        routed_w, indices, shared_w, pk = out
        if pk is not None:  # packed weights are bf16-rounded: compare unpacked
            routed_w, indices = _unpack(pk)
        return routed_w, indices, shared_w

    for (aw, ai, ash), (bw, bi, bsh) in [(parts(triton_out), parts(jit_out))]:
        torch.testing.assert_close(
            ai.to(torch.int64), bi.to(torch.int64), atol=0, rtol=0
        )
        watol = 2e-2 if packed else 1e-5
        torch.testing.assert_close(aw, bw, atol=watol, rtol=1e-3)
        torch.testing.assert_close(ash, bsh, atol=1e-5, rtol=1e-3)


@requires_cuda
def test_zero_tokens():
    _, _, bias, global_scale, logits = _make_inputs(0)
    routed_w, indices, shared_w, packed = inkling_gate_topk_renorm_v2(
        logits, bias, global_scale, ROUTE_SCALE
    )
    assert routed_w.shape == (0, TOPK) and shared_w.shape == (0, N_SHARED)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
