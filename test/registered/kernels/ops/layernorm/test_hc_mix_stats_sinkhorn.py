"""The fused split-K mixing + Sinkhorn (CUDA Triton path) against the unfused kernels and
an FP64 reference; a row's result must not depend on the batch."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA Triton path",
)

HC, ITERS, RMS_EPS, HC_EPS = 4, 20, 1e-6, 1e-6
K, MIX = HC * 5120, (2 + HC) * HC


def _params(seed=0, k=K):
    gen = torch.Generator("cuda").manual_seed(seed)
    w = torch.randn(MIX, k, device="cuda", generator=gen) * 0.01
    scale = torch.rand(3, device="cuda", generator=gen) + 0.5
    base = torch.randn(MIX, device="cuda", generator=gen) * 0.1
    return w, scale, base


def _x(m, seed=1, k=K):
    gen = torch.Generator("cuda").manual_seed(seed)
    return (torch.randn(m, k, device="cuda", generator=gen) * 2).bfloat16()


def _fused(x, w, scale, base, decode=False):
    return mhc.hc_mix_stats_sinkhorn(
        x, w, scale, base, HC, ITERS, RMS_EPS, HC_EPS, decode=decode
    )


def _fused_as(device_sm, x, w, scale, base, decode):
    """The fused path as an SM<device_sm> GPU runs it, whatever the host GPU; also
    returns the slice reducer it launched and its slice count."""
    platform = SimpleNamespace(
        device_sm=device_sm, is_blackwell=mhc.get_platform().is_blackwell
    )
    pick, picked = mhc._hc_mix_reducer_for, []

    def spy(num_slices, **kwargs):
        kernel, extra = pick(num_slices, **kwargs)
        picked.append((kernel, num_slices))
        return kernel, extra

    with (
        mock.patch.object(mhc, "get_platform", return_value=platform),
        mock.patch.object(mhc, "_hc_mix_reducer_for", spy),
    ):
        out = _fused(x, w, scale, base, decode)
    assert len(picked) == 1
    return out, picked[0]


def _sinkhorn(mixes, scale, base):
    return mhc._hc_split_sinkhorn_torch(mixes[:, None], scale, base, HC, ITERS, HC_EPS)


def test_slice_count_depends_on_mode_not_rows():
    for device_sm, prefill, decode in ((120, 80, 160), (100, 80, 80), (121, 80, 80)):
        platform = SimpleNamespace(device_sm=device_sm)
        with mock.patch.object(mhc, "get_platform", return_value=platform):
            assert mhc._num_slices_for(K) == prefill
            assert mhc._num_slices_for(K, decode=True) == decode
            # K that 160 does not divide keeps the existing choice.
            assert mhc._num_slices_for(5120, decode=True) == 80


# K=20480 is DeepSeek-V4.1's (on SM120 160 slices for decode, 80 for prefill and in
# hc_mix_stats); the others select 80, 64 and 1 slices. Each case runs as SM120
# (vectorized reduce) and as SM100 (serial reduce) on any host GPU.
@pytest.mark.parametrize("device_sm", [100, 120])
@pytest.mark.parametrize(
    "m, k, decode",
    [
        (1, K, True),
        (6, K, True),
        (48, K, True),
        (300, K, True),
        (6, K, False),
        (300, K, False),
        (6, 5120, True),
        (48, 4096, False),
        (6, 192, True),
    ],
)
def test_matches_unfused_and_fp64(m, k, decode, device_sm):
    w, scale, base = _params(k=k)
    x = _x(m, k=k)
    actual, (reducer, slices) = _fused_as(device_sm, x, w, scale, base, decode)
    assert reducer is (
        mhc._hc_mix_reduce_sinkhorn_vec_kernel
        if device_sm == 120
        else mhc._hc_mix_reduce_sinkhorn_kernel
    )
    if k == K:
        assert slices == (160 if decode and device_sm == 120 else 80)
    # hc_mix_stats sums its K slices serially before the Sinkhorn.
    unfused = _sinkhorn(mhc.hc_mix_stats(x, w, RMS_EPS), scale, base)
    # FP64 mixing statistics; the reference Sinkhorn itself runs in FP32.
    xd = x.double()
    rsqrt = torch.rsqrt(xd.norm(dim=-1, keepdim=True).square() / k + RMS_EPS)
    reference = _sinkhorn((xd @ w.double().T * rsqrt).float(), scale, base)
    for a, u, r in zip(actual, unfused, reference):
        a = a.reshape(m, -1)
        torch.testing.assert_close(a, u.reshape(m, -1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(a, r.reshape(m, -1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("decode", [False, True])
def test_batch_invariant_and_deterministic(decode):
    w, scale, base = _params()
    x = _x(300)
    full = _fused(x, w, scale, base, decode)
    for a, b in zip(full, _fused(x, w, scale, base, decode)):
        assert torch.equal(a, b)
    for rows in ([0], [299], list(range(8)), list(range(0, 300, 7))):
        idx = torch.tensor(rows, device="cuda")
        sub = _fused(x[idx].contiguous(), w, scale, base, decode)
        for a, b in zip(sub, full):
            assert torch.equal(a, b[idx]), rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
