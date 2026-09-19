"""Correctness tests for the M=1 skinny fp8 PTPC GEMV (gfx950).

The kernel computes ``out[n] = sum_k(a[k] * w[n, k]) * x_scale * w_scale[n]``
over aiter (16,16)-preshuffled fp8 weights with a bf16 output. It replaces
``aiter.gemm_a8w8_bpreshuffle`` for the dispatch-gated shapes, so the accuracy
contract is: per-output error vs the fp32 reference never worse than the
aiter kernel's on the same inputs.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

# M3 qkv shapes first (N 1280/1536 at TP8, 2304/2560 at TP4), then an
# off-gate generic shape the kernel still supports (N % 16, K % 2048).
SHAPES = [(1280, 6144), (1536, 6144), (2304, 6144), (2560, 6144), (64, 2048)]


def _gate_available() -> bool:
    try:
        from sglang.kernels.ops.gemm.skinny_ptpc_gemv import (
            skinny_ptpc_gemv_supported,
        )

        return skinny_ptpc_gemv_supported(1, 1280, 6144)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gate_available(), reason="skinny_ptpc_gemv targets gfx950 MFMA fp8"
)


def _make_inputs(n, k, seed, scale=1.0):
    torch.manual_seed(seed)
    dev = "cuda"
    w32 = torch.randn(n, k, device=dev) * 0.02 * scale
    w_scale = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
    wq = (w32 / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    aq = (
        (torch.randn(1, k, device=dev) * scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    x_scale = torch.rand(1, 1, device=dev).float().clamp(min=1e-6)
    return aq, wq, x_scale, w_scale.float()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(shape, seed):
    from aiter.ops.shuffle import shuffle_weight

    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    n, k = shape
    aq, wq, x_scale, w_scale = _make_inputs(n, k, seed)
    w_shuf = shuffle_weight(wq, (16, 16))
    ref = (aq.float() @ wq.float().T) * x_scale * w_scale.T
    got = skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)
    assert got.shape == (1, n) and got.dtype == torch.bfloat16
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"{shape} seed={seed}: rel_err={rel:.2e}"


@pytest.mark.parametrize("shape", SHAPES[:4])
@pytest.mark.parametrize("seed", range(5))
def test_never_worse_than_aiter(shape, seed):
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    n, k = shape
    aq, wq, x_scale, w_scale = _make_inputs(n, k, seed, scale=10.0 ** (seed - 2))
    w_shuf = shuffle_weight(wq, (16, 16))
    ref = (aq.float() @ wq.float().T) * x_scale * w_scale.T
    base = aiter.gemm_a8w8_bpreshuffle(
        aq, w_shuf, x_scale, w_scale, None, torch.bfloat16
    )
    got = skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)
    e_base = (base.float() - ref).abs().max().item()
    e_got = (got.float() - ref).abs().max().item()
    assert e_got <= e_base + 1e-12, f"{shape} seed={seed}: {e_got} > {e_base}"


def test_zero_activation():
    from aiter.ops.shuffle import shuffle_weight

    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    n, k = 1280, 6144
    aq, wq, x_scale, w_scale = _make_inputs(n, k, 0)
    aq.zero_()
    w_shuf = shuffle_weight(wq, (16, 16))
    got = skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)
    assert torch.all(got == 0)


def test_cuda_graph_replay():
    from aiter.ops.shuffle import shuffle_weight

    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv

    n, k = 1536, 6144
    aq, wq, x_scale, w_scale = _make_inputs(n, k, 0)
    w_shuf = shuffle_weight(wq, (16, 16))
    ref = skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = skinny_ptpc_gemv(aq, w_shuf, x_scale, w_scale)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_dispatch_gate():
    from sglang.kernels.ops.gemm.skinny_ptpc_gemv import skinny_ptpc_gemv_supported

    assert skinny_ptpc_gemv_supported(1, 1280, 6144)
    assert skinny_ptpc_gemv_supported(1, 1536, 6144)
    assert skinny_ptpc_gemv_supported(1, 2304, 6144)
    assert not skinny_ptpc_gemv_supported(2, 1280, 6144)  # batched
    assert not skinny_ptpc_gemv_supported(1, 6144, 2048)  # wide-N shape
    assert not skinny_ptpc_gemv_supported(1, 1280, 4096)  # off-gate K


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
