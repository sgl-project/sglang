import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

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


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


_CONCURRENT_STREAMS_SCRIPT = textwrap.dedent(
    """
    import torch
    from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix

    HC, HS, LOWRANK, ROWS, NUM_STREAMS, TRIALS = {hc}, {hs}, {lowrank}, 16, {streams}, 8
    torch.manual_seed(0)
    xs = [
        torch.randn(ROWS, HC * HS, dtype=torch.bfloat16, device="cuda")
        for _ in range(NUM_STREAMS)
    ]
    w_down = torch.randn(LOWRANK, HC * HS, dtype=torch.bfloat16, device="cuda") * 0.02
    w_up = torch.randn(HC * HS, LOWRANK, dtype=torch.bfloat16, device="cuda") * 0.02
    serial = [fused_hc_mix(x, w_down, w_up, HC, HS) for x in xs]
    torch.cuda.synchronize()

    streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
    for trial in range(TRIALS):
        # Hold every stream behind a delayed gate so the launches overlap.
        gate = torch.cuda.Event()
        gate_stream = torch.cuda.Stream()
        with torch.cuda.stream(gate_stream):
            torch.cuda._sleep(50_000_000)
            gate.record()
        outs = []
        for stream, x in zip(streams, xs):
            stream.wait_event(gate)
            with torch.cuda.stream(stream):
                outs.append(fused_hc_mix(x, w_down, w_up, HC, HS))
        torch.cuda.synchronize()
        for out, ref in zip(outs, serial):
            torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-3)
    print("CONCURRENT_STREAMS_OK")
    """
)


@pytest.mark.parametrize("num_streams", [2, 8])
def test_fused_hc_mix_concurrent_streams_complete(num_streams):
    """Launches from several CUDA streams must finish instead of hanging.

    Runs in a subprocess because a hung kernel cannot be interrupted in-process.
    """
    script = _CONCURRENT_STREAMS_SCRIPT.format(
        hc=HC_COUNT, hs=HIDDEN_SIZE, lowrank=LOWRANK, streams=num_streams
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"fused_hc_mix hung with {num_streams} concurrent streams: {e}")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "CONCURRENT_STREAMS_OK" in result.stdout


def test_fused_hc_mix_cuda_graph_replays_match_eager():
    x, w_down, w_up = _make_inputs(4, torch.bfloat16)
    expected = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=5e-3)
        # An eager call between replays still sees counters in their initial state.
        again = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
        torch.testing.assert_close(again, expected, rtol=1e-2, atol=5e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
