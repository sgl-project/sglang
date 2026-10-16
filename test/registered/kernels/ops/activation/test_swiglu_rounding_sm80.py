"""Direct SM80 rounded-SwiGLU qualification; does not change the runtime gate.

Run on A100 from the repo root with the existing CUDA/Torch environment::

    PYTHONPATH=python python -m pytest -q -s \
        test/registered/kernels/ops/activation/test_swiglu_rounding_sm80.py
    PYTHONPATH=python python \
        test/registered/kernels/ops/activation/test_swiglu_rounding_sm80.py --benchmark

Shapes below are full [tokens, gate+up] input shapes. Finite outputs must
match bits (including signed zero); NaN payload/sign are not a contract, but
NaN locations and infinity signs must match. No tolerance silently accepts ULP
differences. Non-finite inputs are tested separately from normal finite inputs.
The benchmark runs only after correctness passes; both paths include activation
rounding, optional clamping, and the final write to a caller-provided output.
"""

import argparse
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F


INPUT_SHAPES = [(6, 640), (6, 576), (1, 5120)]
DTYPES = [torch.bfloat16, torch.float16]
LIMITS = [0.0, 10.0]


def _require_sm80():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA SM80")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 0):
        pytest.skip(f"requires actual SM80, got {capability}")
    return capability


@pytest.fixture(autouse=True)
def require_actual_sm80():
    _require_sm80()


def _fused(x, out=None, *, clamp_limit=0.0):
    # Import only after checking the real device. Deliberately bypass the
    # SM90-only swiglu_limit_func wrapper, not its rounding semantics.
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
    )

    return silu_and_mul_with_activation_rounding(x, out, clamp_limit=clamp_limit)


def _torch_chain(x, out=None, *, clamp_limit=0.0):
    gate, up = x.chunk(2, dim=-1)
    if clamp_limit > 0:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    # Do NOT promote to float or fuse this expression: eager F.silu writes a
    # BF16/FP16 intermediate before the multiplication, as the runtime does.
    value = F.silu(gate) * up
    if out is None:
        return value
    out.copy_(value)
    return out


def _bits(x):
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _ordered_bits(x):
    bits = _bits(x)
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)


def _assert_exact(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    both_nan = torch.isnan(actual) & torch.isnan(expected)
    mismatch = (_bits(actual) != _bits(expected)) & ~both_nan
    count = int(mismatch.sum().item())
    if not count:
        return
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    ulps = (_ordered_bits(actual) - _ordered_bits(expected)).abs()
    max_ulp = int(ulps[finite].max().item()) if bool(finite.any()) else None
    first = mismatch.nonzero()[0].tolist()
    index = tuple(first)
    pytest.fail(
        f"{label}: bit mismatches={count}/{actual.numel()}, "
        f"max_finite_ULP={max_ulp}, first_index={first}, "
        f"actual={actual[index].item()}, expected={expected[index].item()}, "
        f"actual_bits=0x{int(_bits(actual)[index]):04x}, "
        f"expected_bits=0x{int(_bits(expected)[index]):04x}; "
        "NaN payload/sign excluded, signed zero included"
    )


def _input(shape, dtype, case):
    generator = torch.Generator(device="cuda").manual_seed(20260416)
    if case == "random":
        return torch.randn(shape, dtype=dtype, device="cuda", generator=generator) * 12
    finfo = torch.finfo(dtype)
    values = torch.tensor(
        [-finfo.max, -1000, -100, -90, -20, -10, -1, -finfo.tiny,
         -0.0, 0.0, finfo.tiny, 1, 10, 20, 90, 100, 1000, finfo.max],
        dtype=dtype, device="cuda",
    )
    if case == "boundary":
        centers = torch.tensor([-10.0, 0.0, 10.0], dtype=dtype, device="cuda")
        values = torch.cat(
            [torch.nextafter(centers, torch.full_like(centers, -float("inf"))),
             centers,
             torch.nextafter(centers, torch.full_like(centers, float("inf")))]
        )
    rows, width = shape
    n = rows * (width // 2)
    indices = torch.arange(n, device="cuda")
    # Different coprime walks expose every gate/up extreme combination rather
    # than coupling large gates only to large up values.
    gate = values[indices % values.numel()].view(rows, width // 2)
    up = values[(indices // values.numel() + indices * 5 + 1) % values.numel()]
    return torch.cat((gate, up.view_as(gate)), dim=-1).contiguous()


def test_gpu_capability():
    assert _require_sm80() == (8, 0)
    print(f"device={torch.cuda.get_device_name()} capability=(8, 0) "
          f"torch={torch.__version__} cuda={torch.version.cuda}")


@pytest.mark.parametrize("shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("case", ["random", "extreme", "boundary"])
def test_swiglu_finite_exact(shape, dtype, limit, case):
    x = _input(shape, dtype, case)
    original = x.clone()
    expected = _torch_chain(x, clamp_limit=limit)
    label = f"shape={shape} {dtype} limit={limit} {case}"
    _assert_exact(_fused(x, clamp_limit=limit), expected, label + " allocated")
    out = torch.empty_like(expected)
    assert _fused(x, out, clamp_limit=limit) is out
    _assert_exact(out, expected, label + " out")
    assert torch.equal(_bits(x), _bits(original)), "input unexpectedly mutated"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite-bits", "nan-inf"])
def test_swiglu_all_16bit_patterns(dtype, limit, nonfinite):
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(dtype)
    gate = values.view(-1, 32)
    up = values.roll(17953).view_as(gate)
    if not nonfinite:
        # Preserve all finite encodings, including signed zeros/subnormals.
        # Non-finites are independently qualified by the nan-inf case.
        gate = torch.where(torch.isfinite(gate), gate, torch.ones_like(gate))
        up = torch.where(torch.isfinite(up), up, torch.ones_like(up))
    x = torch.cat((gate, up), dim=-1)
    _assert_exact(
        _fused(x, clamp_limit=limit), _torch_chain(x, clamp_limit=limit),
        f"all patterns {dtype} limit={limit} nonfinite={nonfinite}",
    )


@pytest.mark.parametrize("shape", INPUT_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("limit", LIMITS)
def test_swiglu_cuda_graph(shape, dtype, limit):
    x = _input(shape, dtype, "random")
    out = _fused(x, clamp_limit=limit)  # JIT and warmup before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _fused(x, out, clamp_limit=limit)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _fused(x, out, clamp_limit=limit)
    for _ in range(2):
        x.neg_()  # Replay must read current storage, not captured constants.
        graph.replay()
        _assert_exact(out, _torch_chain(x, clamp_limit=limit),
                      f"graph {shape} {dtype} limit={limit}")


def _event_us(fn, iterations, repeats):
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(samples)


def benchmark(iterations=200, repeats=7, include_fp16=False):
    _require_sm80()
    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    print("All times median CUDA-event us/call; graph = one full chain per replay.")
    print("input_shape dtype limit mode torch_us fused_us speedup")
    for dtype in (DTYPES if include_fp16 else [torch.bfloat16]):
        for shape in INPUT_SHAPES:
            for limit in LIMITS:
                x = _input(shape, dtype, "random")
                reference = _torch_chain(x, clamp_limit=limit)
                outputs = [torch.empty_like(reference), torch.empty_like(reference)]
                funcs = [lambda: _torch_chain(x, outputs[0], clamp_limit=limit),
                         lambda: _fused(x, outputs[1], clamp_limit=limit)]
                graphs = []
                for fn, out in zip(funcs, outputs):
                    for _ in range(10):
                        fn()
                    torch.cuda.synchronize()
                    _assert_exact(out, reference, "benchmark warmup")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    graph.replay()
                    _assert_exact(out, reference, "benchmark graph")
                    graphs.append(graph)
                for mode, calls in [("events", funcs), ("cudagraph", [g.replay for g in graphs])]:
                    times = [_event_us(fn, iterations, repeats) for fn in calls]
                    print(f"{shape} {dtype} {limit:g} {mode} "
                          f"{times[0]:.4f} {times[1]:.4f} {times[0] / times[1]:.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--include-fp16", action="store_true", help="also benchmark FP16")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args, pytest_args = parser.parse_known_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        print("SKIP: requires an actual NVIDIA SM80 GPU; no benchmark was run")
        sys.exit(0)
    status = pytest.main([__file__, "-q", "-s", *pytest_args])
    if status == 0 and args.benchmark:
        benchmark(args.iterations, args.repeats, args.include_fp16)
    sys.exit(status)
