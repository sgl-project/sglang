import inspect
import sys

import pytest
import torch
import torch.nn.functional as F

import sglang.srt.layers.hc_mix_triton as hc_mix_triton
from sglang.srt.layers.hc_mix_triton import (
    _FUSED_MIX_MAX_ROWS,
    _SM120_FUSED_MIX_MAX_ROWS,
    _get_hc_mix_config,
    _hc_mix_persistent_kernel,
    _hc_mix_persistent_kernel_sm120,
    _select_hc_mix_config,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [17, _SM120_FUSED_MIX_MAX_ROWS])
def test_sm120_fused_hc_mix_matches_reference(dtype, num_tokens):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120-specific row range")
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


@pytest.mark.parametrize("num_tokens", [1, 16, _SM120_FUSED_MIX_MAX_ROWS])
def test_sm120_fused_hc_mix_graph_replay_stability(num_tokens):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120-specific launch geometry")
    x, w_down, w_up = _make_inputs(num_tokens, torch.bfloat16)
    for _ in range(3):
        fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    graph.replay()
    torch.cuda.synchronize()
    expected = out.clone()
    for _ in range(10_000):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out, expected, rtol=0, atol=torch.finfo(torch.bfloat16).eps
    )


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_SM120_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


def test_fused_hc_mix_deterministic_inference_fallback(monkeypatch):
    x, w_down, w_up = _make_inputs(1, torch.bfloat16)
    monkeypatch.setattr(hc_mix_triton, "_deterministic_inference_cached", True)
    assert not fused_hc_mix_supported(x, w_down, w_up)


@pytest.mark.parametrize(
    ("num_rows", "max_rows", "rows_pad", "num_ctas", "block_k", "num_warps"),
    [
        (1, 1, 16, 80, 256, 4),
        (2, 16, 16, 80, 256, 4),
        (16, 16, 16, 80, 256, 4),
        (17, 64, 64, 80, 128, 8),
        (64, 64, 64, 80, 128, 8),
    ],
)
def test_sm120_hc_mix_dispatch_buckets(
    num_rows, max_rows, rows_pad, num_ctas, block_k, num_warps
):
    config = _select_hc_mix_config(num_rows, (12, 0), 188)
    assert config is not None
    assert config.max_rows == max_rows
    assert config.rows_pad == rows_pad
    assert config.num_ctas == num_ctas
    assert config.block_k == block_k
    assert config.num_warps == num_warps


def test_hc_mix_dispatch_keeps_other_devices_unchanged():
    config = _select_hc_mix_config(16, (10, 0), 148)
    assert config is not None
    assert config.rows_pad == _FUSED_MIX_MAX_ROWS
    assert config.num_ctas == 148
    assert config.block_k == 256
    assert config.num_warps == 8
    assert _select_hc_mix_config(17, (10, 0), 148) is None
    assert _select_hc_mix_config(65, (12, 0), 188) is None


def test_hc_mix_constexpr_dims_are_sm120_scoped():
    runtime_signature = inspect.signature(_hc_mix_persistent_kernel.fn)
    sm120_signature = inspect.signature(_hc_mix_persistent_kernel_sm120.fn)
    for name in ("K", "LOWRANK", "HS"):
        assert runtime_signature.parameters[name].annotation is inspect.Parameter.empty
        assert (
            sm120_signature.parameters[name].annotation is not inspect.Parameter.empty
        )


def test_hc_mix_device_config_is_cached(monkeypatch):
    calls = []

    class Props:
        major = 12
        minor = 0
        multi_processor_count = 188

    def get_device_properties(device):
        calls.append(device)
        return Props()

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    _get_hc_mix_config.cache_clear()
    try:
        device = torch.device("cuda:0")
        first = _get_hc_mix_config(1, device)
        second = _get_hc_mix_config(1, device)
        assert first is second
        assert calls == [device]
    finally:
        _get_hc_mix_config.cache_clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
