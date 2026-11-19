"""SM120 mHC tuning keeps import, device and arithmetic boundaries intact."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="mHC SM120 regression tests require CUDA",
)


def _require_sm120():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120, not SM121 or datacenter Blackwell")


def _inputs(m, k, seed):
    torch.manual_seed(seed)
    weight = torch.randn(24, k, device="cuda", dtype=torch.float32) / k**0.5
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
    base = torch.randn(24, device="cuda") * 0.1
    return x, weight, scale, base


def _fused(x, weight, scale, base):
    return mhc.hc_mix_stats_sinkhorn(x, weight, scale, base, 4, 20, 1e-20, 1e-6)


def _assert_bitwise(got, reference):
    assert torch.equal(
        got.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
    )


def test_module_body_does_not_query_device_capability():
    # Preload dependencies: platform discovery has its own CUDA initialization.
    # Reload isolates mHC's module body and catches a reintroduced eager query,
    # including a call through the cached is_sm120 helper.
    script = """
import importlib
from unittest.mock import patch
import torch
import sglang.kernels.ops.layernorm.mhc as mhc

with (
    patch('sglang.srt.utils.is_sm120', side_effect=AssertionError('eager is_sm120')),
    patch('torch.cuda.get_device_capability', side_effect=AssertionError('eager capability query')),
):
    importlib.reload(mhc)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_tile_selection_uses_the_input_device_and_excludes_sm121(monkeypatch):
    capabilities = {0: (12, 0), 1: (12, 1), 2: (10, 0), 3: (9, 0)}
    query = Mock(side_effect=lambda device: capabilities[device.index])
    monkeypatch.setattr(mhc, "_is_hip", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", query)
    select = mhc._hc_mix_stats_sinkhorn_block_k
    select.cache_clear()
    try:
        assert select(torch.device("cpu")) == 64
        query.assert_not_called()
        for index, expected in enumerate((32, 64, 64, 64)):
            assert select(torch.device("cuda", index)) == expected
        assert select(torch.device("cuda", 0)) == 32
        assert query.call_count == 4
        select.cache_clear()
        monkeypatch.setattr(mhc, "_is_hip", True)
        assert select(torch.device("cuda", 0)) == 64
        assert query.call_count == 4
    finally:
        select.cache_clear()


@pytest.mark.parametrize("seed", [7, 11, 19])
@pytest.mark.parametrize("k", [2560, 20480])
def test_fused_accuracy_and_batch_invariance(seed, k):
    _require_sm120()
    x, weight, scale, base = _inputs(4096, k, seed)
    fused_reference = _fused(x, weight, scale, base)
    with patch.object(mhc, "_hc_mix_stats_sinkhorn_block_k", return_value=64):
        baseline = _fused(x, weight, scale, base)
    for got, expected in zip(fused_reference, baseline):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=2e-6)
    for m in (1, 8, 9, 16, 32, 2048, 2049):
        for got, reference in zip(_fused(x[:m], weight, scale, base), fused_reference):
            _assert_bitwise(got, reference[:m])
    # Checking only the first eight rows misses data-dependent rounding changes.
    # Route every reference row through the actual small-batch launch.
    for start in range(0, x.shape[0], 8):
        stop = start + 8
        for got, reference in zip(
            _fused(x[start:stop], weight, scale, base), fused_reference
        ):
            _assert_bitwise(got, reference[start:stop])


def test_bf16x3_is_independent_of_partial_tile_width():
    _require_sm120()
    x, weight, scale, base = _inputs(4096, 20480, 7)
    parts = mhc.split_bf16_hc_weight(weight)

    def run():
        return mhc.hc_mix_stats_sinkhorn_bf16x3(x, parts, scale, base, 20, 1e-20, 1e-6)

    reference = run()
    with patch.object(mhc, "_HC_MIX_BLOCK_K", 32):
        tuned = run()
    for got, expected in zip(tuned, reference):
        _assert_bitwise(got, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
