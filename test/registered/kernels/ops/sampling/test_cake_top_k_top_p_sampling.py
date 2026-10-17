"""Cake fused top-k-first sampling and top-k slab through sglang.kernels.

Checks for the Cake sampling adapters distributed by FlashInfer: the registry
resolves the explicit FlashInfer backend; the facade result is bitwise
identical to calling FlashInfer directly (explicit Philox seed/offset and a
seeded generator); repeated calls are deterministic; every sampled id lies in
the top-k-then-top-p support computed in torch; and the stage-1 slab holds the
exact top-k set. Skips (with the reason) when the installed FlashInfer lacks
the Cake module or the GPU is not a 9.x / 10.x / 11.x / 12.x part.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import sampling as cake_sampling
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.sampling.cake import (
    cake_top_k_probs_to_slab,
    cake_top_k_top_p_sampling_from_probs_top_k_first,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SAMPLE_OP = "sampling.top_k_top_p_sampling_from_probs_top_k_first"
SLAB_OP = "sampling.top_k_probs_to_slab"


@pytest.mark.parametrize("op", [SAMPLE_OP, SLAB_OP])
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.sampling:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_sampling.FI_MODULE, cake_sampling.FI_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks flashinfer.cake_sampling")
    cc = torch.cuda.get_device_capability()
    if cc[0] not in cake_sampling.ARCH_MAJORS:
        pytest.skip(f"Cake sampling is built for CC 9.x-12.x, device is {cc}")


def _probs(batch: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(batch, vocab, device="cuda", generator=g)
    return torch.softmax(logits, dim=-1).contiguous()


def _top_k_first_support(row: torch.Tensor, k: int, p: float) -> set:
    """Indices kept by top-k (ties toward the lower index) then top-p.

    Mirrors the documented Cake semantics: the support is the first ``k``
    entries of ``lexsort(-prob, index)``; top-p keeps the shortest prefix
    whose exclusive mass is below ``p * mass(top-k)``. A relative slack of
    1e-6 on the threshold absorbs the FP32-vs-fixed-point rounding at the
    boundary (this is a membership check, not a bitwise one).
    """
    vals, idx = torch.sort(row, descending=True, stable=True)
    vals = vals[:k].double()
    idx = idx[:k]
    mass = vals.sum()
    excl = torch.cumsum(vals, dim=0) - vals
    kept = excl < p * mass * (1.0 + 1e-6)
    kept[0] = True
    return set(idx[kept].tolist())


@pytest.mark.parametrize("vocab", [32000, 128256])
@pytest.mark.parametrize("k,p", [(1, 1.0), (16, 0.5), (64, 0.9), (1024, 0.95)])
def test_matches_flashinfer_and_support(vocab, k, p):
    _skip_unless_supported()
    batch = 4
    probs = _probs(batch, vocab, seed=1000 + vocab + k)
    assert cake_sampling.supports_top_k_top_p_sampling_top_k_first(probs, k)
    out = cake_top_k_top_p_sampling_from_probs_top_k_first(
        probs, k, p, philox_seed=7, philox_offset=8
    )
    from flashinfer.cake_sampling import top_k_top_p_sampling_from_probs as fi_direct

    out_fi = fi_direct(probs, k, p, philox_seed=7, philox_offset=8)
    out_again = cake_top_k_top_p_sampling_from_probs_top_k_first(
        probs, k, p, philox_seed=7, philox_offset=8
    )
    torch.cuda.synchronize()
    assert out.dtype == torch.int32 and out.shape == (batch,)
    assert torch.equal(out, out_fi)
    assert torch.equal(out, out_again)
    for r in range(batch):
        support = _top_k_first_support(probs[r], k, p)
        assert int(out[r]) in support, f"row {r}: sample outside top-k-first support"


def test_per_row_params_and_generator_match_flashinfer():
    _skip_unless_supported()
    batch, vocab = 8, 32000
    probs = _probs(batch, vocab, seed=42)
    top_k = torch.tensor(
        [1, 4, 16, 64, 256, 1024, 8, 2], device="cuda", dtype=torch.int32
    )
    top_p = torch.tensor(
        [1.0, 0.5, 0.9, 0.95, 0.8, 0.99, 0.3, 0.6], device="cuda", dtype=torch.float32
    )
    assert cake_sampling.supports_top_k_top_p_sampling_top_k_first(
        probs, top_k, top_k_max=1024
    )
    g1 = torch.Generator(device="cuda").manual_seed(123)
    out = cake_top_k_top_p_sampling_from_probs_top_k_first(
        probs, top_k, top_p, top_k_max=1024, generator=g1
    )
    from flashinfer.cake_sampling import top_k_top_p_sampling_from_probs as fi_direct

    g2 = torch.Generator(device="cuda").manual_seed(123)
    out_fi = fi_direct(probs, top_k, top_p, top_k_max=1024, generator=g2)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert g1.get_state().equal(g2.get_state())
    for r in range(batch):
        support = _top_k_first_support(probs[r], int(top_k[r]), float(top_p[r]))
        assert int(out[r]) in support, f"row {r}: sample outside top-k-first support"


@pytest.mark.parametrize("k", [1, 32, 1024])
def test_slab_matches_flashinfer_and_torch_topk(k):
    _skip_unless_supported()
    batch, vocab = 3, 65536
    probs = _probs(batch, vocab, seed=77 + k)
    vals, idx, cnt = cake_top_k_probs_to_slab(probs, k)
    from flashinfer.cake_sampling import top_k_probs_to_slab as fi_direct

    vals_fi, idx_fi, cnt_fi = fi_direct(probs, k)
    torch.cuda.synchronize()
    assert vals.shape == (batch, cake_sampling.SLAB) and vals.dtype == torch.float32
    assert idx.shape == (batch, cake_sampling.SLAB) and idx.dtype == torch.int32
    assert cnt.shape == (batch,) and cnt.dtype == torch.int32
    assert torch.equal(cnt, cnt_fi)
    for r in range(batch):
        n = int(cnt[r])
        assert n == k
        assert torch.equal(vals[r, :n], vals_fi[r, :n])
        assert torch.equal(idx[r, :n], idx_fi[r, :n])
        ref_vals, ref_idx = torch.sort(probs[r], descending=True, stable=True)
        got_vals = probs[r][idx[r, :n].long()]
        assert torch.equal(torch.sort(got_vals, descending=True).values, ref_vals[:k])
        assert torch.equal(vals[r, :n].sort(descending=True).values, ref_vals[:k])
        if k < vocab and ref_vals[k - 1] != ref_vals[k]:
            assert set(idx[r, :n].tolist()) == set(ref_idx[:k].tolist())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
