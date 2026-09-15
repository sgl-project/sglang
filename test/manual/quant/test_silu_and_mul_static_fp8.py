"""SM89 SiLU-and-mul + static per-tensor FP8 operator/consumer regressions.

No checkpoint is required. From the repository root, with CUDA test dependencies:
    SGLANG_TEST_REQUIRE_STATIC_FP8=1 python -m pytest -q \
        test/manual/quant/test_silu_and_mul_static_fp8.py

Unsupported hardware skips by default; require mode fails instead. Covers exact
bytes, 441 rounding cases including direct-conversion counterexamples, streams,
CUDA Graph replay, real GEMM consumption and row-scale storage reuse.
"""

import json
import os
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _inspect(scales, before, intermediate, quantized, count: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    scale = tl.load(scales + offsets, offsets < count, other=1)
    value = 20.0 * (1.0 / scale)
    value = tl.clamp(value, -448.0, 448.0)
    half = value.to(tl.float16, fp_downcast_rounding="rtz").to(tl.float32)
    q = value.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(before + offsets, value, offsets < count)
    tl.store(intermediate + offsets, half, offsets < count)
    tl.store(quantized + offsets, q, offsets < count)


def rounding_cases():
    levels = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (levels[:-1] + levels[1:]) / 2
    halves = midpoints.half()
    targets = torch.stack(
        [
            torch.nextafter(halves, torch.full_like(halves, -float("inf"))),
            halves,
            torch.nextafter(halves, torch.full_like(halves, float("inf"))),
        ],
        dim=1,
    ).float()
    centers = (20.0 / targets).contiguous().view(torch.int32)
    bits = centers[:, :, None] + torch.arange(-64, 65, dtype=torch.int32)
    scales = bits.contiguous().view(torch.float32).cuda().flatten()
    before = torch.empty_like(scales)
    half = torch.empty_like(scales)
    quantized = torch.empty_like(scales, dtype=torch.uint8)
    compiled = _inspect[(triton.cdiv(scales.numel(), 256),)](
        scales, before, half, quantized, scales.numel()
    )
    # Detect the tempting but incompatible direct FP32->FP8 replacement.
    native = before.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.any(native != quantized), (
        "Dataset must distinguish direct FP8 conversion"
    )
    b, h, s, q, direct = [
        x.cpu().reshape(126, -1) for x in (before, half, scales, quantized, native)
    ]
    cases = []
    for index, midpoint in enumerate(midpoints):
        for side, mask in (
            ("below", h[index] < midpoint),
            ("tie", h[index] == midpoint),
            ("above", h[index] > midpoint),
        ):
            candidates = torch.where(mask)[0]
            assert candidates.numel(), (index, side)
            distance = (b[index, candidates] - midpoint).abs()
            chosen = candidates[distance.argmin()]
            # The FP32 side and the post-RTZ side are deliberately separate.
            cases.append(
                dict(
                    midpoint=float(midpoint),
                    side=side,
                    before=float(b[index, chosen]),
                    half=float(h[index, chosen]),
                    scale=float(s[index, chosen]),
                    byte=int(q[index, chosen]),
                )
            )
        # Nearest-to-midpoint selection can discard every double-rounding
        # counterexample. Keep one explicitly wherever this midpoint has one.
        different = torch.where(direct[index] != q[index])[0]
        if different.numel():
            chosen = different[0]
            cases.append(
                dict(
                    midpoint=float(midpoint),
                    side="direct_conversion_differs",
                    before=float(b[index, chosen]),
                    half=float(h[index, chosen]),
                    scale=float(s[index, chosen]),
                    byte=int(q[index, chosen]),
                )
            )
    # Check the returned dataset, not just the search pool. CPU conversion is
    # deliberately independent of Triton's SM89 FP16 intermediate lowering.
    selected = torch.tensor([case["before"] for case in cases], dtype=torch.float32)
    direct_bytes = selected.to(torch.float8_e4m3fn).view(torch.uint8)
    reference_bytes = torch.tensor([case["byte"] for case in cases], dtype=torch.uint8)
    assert torch.any(direct_bytes != reference_bytes), (
        "Returned cases must distinguish direct FP32-to-FP8 conversion"
    )
    return cases, compiled.asm["ptx"]


def assert_current_stream(launch, dtype, stream, trace_path):
    # Warm up the producer before profiling. Updates, the producer, and dependent
    # copies must execute on the same side stream.
    x = torch.zeros(17, 512, device="cuda", dtype=dtype)
    scale = torch.ones((), device="cuda")
    q, rows = launch(x, scale)
    observed_q, observed_rows = torch.empty_like(q), torch.empty_like(rows)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        with torch.cuda.stream(stream):
            x.fill_(2)
            scale.fill_(0.02)
            q, rows = launch(x, scale)
            observed_q.copy_(q)
            observed_rows.copy_(rows)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    fused = [e for e in kernels if "silu_and_mul_static_fp8_kernel" in e["name"]]
    assert len(fused) == 1, "Trace must contain the real fused producer"
    assert len(kernels) >= 3, "Trace must include updates and producer"
    streams = {e["args"]["stream"] for e in kernels}
    assert len(streams) == 1, f"Current-stream violation: GPU kernels used {streams}"
    expected_q, expected_rows = launch(x, scale)
    assert torch.equal(observed_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(observed_rows, expected_rows)


@contextmanager
def observe_gemm():
    from sglang.srt.layers.quantization import fp8_utils

    calls = []

    def wrap(name, original):
        def call(*args, **kwargs):
            calls.append(
                dict(
                    backend=name,
                    input_shape=list(args[0].shape),
                    a_scale_ptr=args[2].data_ptr(),
                    options={k: str(v) for k, v in kwargs.items()},
                )
            )
            return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for name in ("fp8_scaled_mm", "triton_scaled_mm"):
            stack.enter_context(
                patch.object(fp8_utils, name, wrap(name, getattr(fp8_utils, name)))
            )
        yield calls


def assert_row_reuse(call, rows):
    assert call["a_scale_ptr"] == rows.data_ptr(), "row scales were not reused by GEMM"


@pytest.fixture(scope="module", autouse=True)
def qualified_environment():
    from sglang.srt.layers.quantization.silu_fp8_fusion import (
        silu_static_fp8_supported,
    )

    supported = torch.cuda.is_available()
    if supported:
        supported = torch.cuda.get_device_capability() == (8, 9)
    supported = supported and silu_static_fp8_supported()
    if not supported:
        if os.environ.get("SGLANG_TEST_REQUIRE_STATIC_FP8") == "1":
            pytest.fail("This run requires the qualified SM89 environment")
        pytest.skip("Requires the qualified SM89 static-FP8 environment")


def check(x, scale):
    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8,
    )
    from sglang.srt.layers.activation import silu_and_mul
    from sglang.srt.layers.quantization.fp8_utils import static_quant_fp8

    before = x.clone()
    q, rows = silu_and_mul_static_fp8(x, scale)
    assert q.shape == (x.shape[0], x.shape[1] // 2)
    assert q.dtype == torch.float8_e4m3fn
    assert rows.shape == (x.shape[0], 1)
    if x.shape[0]:
        expected, scales = static_quant_fp8(silu_and_mul(x), scale, repeat_scale=True)
        assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(rows, scales)
    assert torch.equal(x.view(torch.uint8), before.view(torch.uint8))
    return q, rows


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(0, 16), (1, 24), (3, 256), (129, 512), (17, 12288)])
@pytest.mark.parametrize("scalar", [0.0001, 0.02, 0.0287388414144516, 1.0])
def test_byte_exact(dtype, shape, scalar):
    m, k = shape
    generator = torch.Generator(device="cuda").manual_seed(9271)
    x = torch.randn(m, 2 * k, dtype=dtype, device="cuda", generator=generator) * 5
    check(x, torch.tensor(scalar, device="cuda", dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rounding_midpoints_and_special_values(dtype, tmp_path):

    cases, ptx = rounding_cases()
    (tmp_path / "rounding-cases.json").write_text(json.dumps(cases, indent=2))
    (tmp_path / "rounding-diagnostic.ptx").write_text(ptx)
    for sign in (-1, 1):
        x = torch.cat(
            (
                torch.full((1, 16), 20.0, device="cuda", dtype=dtype),
                torch.full((1, 16), float(sign), device="cuda", dtype=dtype),
            ),
            dim=1,
        )
        for case in cases:
            q, _ = check(
                x, torch.tensor(case["scale"], device="cuda", dtype=torch.float32)
            )
            # The diagnostic must also predict the actual unfused/fused bytes.
            expected = case["byte"] if sign > 0 else case["byte"] ^ 128
            assert torch.all(q.view(torch.uint8) == expected), case
    edge = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-7, -1e-7, 1000.0],
        device="cuda",
        dtype=dtype,
    )
    check(edge.repeat(64).view(1, 512), torch.tensor(0.02, device="cuda"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_replay_updates_input_and_scale(dtype):
    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8,
    )

    x = torch.randn(17, 512, dtype=dtype, device="cuda")
    scale = torch.tensor(0.02, device="cuda")
    check(x, scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_and_mul_static_fp8(x, scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q, rows = silu_and_mul_static_fp8(x, scale)
    for scalar in (0.01, 0.04):
        x.mul_(0.75)
        scale.fill_(scalar)
        graph.replay()
        torch.cuda.synchronize()
        expected, erows = check(x, scale)
        assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(rows, erows)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_independent_streams_and_opcheck(dtype, tmp_path):

    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8,
    )

    for index in range(2):
        assert_current_stream(
            silu_and_mul_static_fp8,
            dtype,
            torch.cuda.Stream(),
            tmp_path / f"stream-{index}.json",
        )

    def wrong_stream(x, scale):
        with torch.cuda.stream(torch.cuda.default_stream()):
            return silu_and_mul_static_fp8(x, scale)

    # The assertion must catch an actual wrong-stream launch even if its
    # values happen to be correct. No random race or sleep is required.
    with pytest.raises(AssertionError, match="Current-stream violation"):
        assert_current_stream(
            wrong_stream, dtype, torch.cuda.Stream(), tmp_path / "wrong-stream.json"
        )
    x = torch.randn(7, 512, device="cuda", dtype=dtype)
    scale = torch.tensor(0.02, device="cuda")
    q = torch.empty(7, 256, device="cuda", dtype=torch.uint8)
    rows = torch.empty(7, 1, device="cuda")
    result = torch.library.opcheck(
        torch.ops.sglang.silu_and_mul_static_fp8_out.default, (x, scale, q, rows)
    )
    assert all(value == "SUCCESS" for value in result.values())


def test_invalid_and_unaligned_inputs():
    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8,
        silu_and_mul_static_fp8_module,
    )

    scale = torch.tensor(0.02, device="cuda")
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    unaligned = torch.empty(65, device="cuda", dtype=x.dtype)[1:].view(2, 32)
    assert unaligned.is_contiguous()
    for invalid in (unaligned, x[:, ::2], x.float(), x.cpu(), x.view(64), x[:, :0]):
        with pytest.raises(ValueError):
            silu_and_mul_static_fp8(invalid, scale)
    with pytest.raises(ValueError):
        silu_and_mul_static_fp8(x, scale.double())
    # The raw entry point must reject misalignment before launching a kernel.
    q = torch.empty(2, 16, device="cuda", dtype=torch.uint8)
    rows = torch.empty(2, 1, device="cuda")
    with pytest.raises(Exception, match="16-byte aligned"):
        silu_and_mul_static_fp8_module(x.dtype).run(
            unaligned, q, scale.reshape(1), rows
        )
    bad_q = torch.empty(33, device="cuda", dtype=torch.uint8)[1:].view(2, 16)
    with pytest.raises(Exception, match="8-byte aligned"):
        silu_and_mul_static_fp8_module(x.dtype).run(x, bad_q, scale.reshape(1), rows)
    check(x, scale)  # CUDA context remains healthy after rejected inputs.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 7, 129])
@pytest.mark.parametrize("backend", ["auto", "cutlass", "triton"])
def test_real_modelopt_consumer_and_legacy_contract(
    dtype, m, backend, tmp_path, monkeypatch
):

    from sglang.srt.environ import envs
    from sglang.srt.layers.activation import silu_and_mul
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.modelopt_fp8_input import ModelOptFp8Input
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    # Construct only the state consumed by apply; no fake GEMM or checkpoint.
    method = object.__new__(ModelOptFp8LinearMethod)
    method._static_fp8_sm89 = True
    method.use_marlin = False
    method.use_sm120_fp8 = False
    method.cutlass_fp8_supported = True
    if backend != "auto":
        monkeypatch.setattr(
            fp8_utils, "use_triton_w8a8_fp8_kernel", backend == "triton"
        )
        monkeypatch.setenv("SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE", "0")
        assert not envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get()
    weight = (
        torch.randn(64, 256, device="cuda", dtype=dtype).to(torch.float8_e4m3fn).t()
    )
    scale = torch.tensor(0.02, device="cuda")
    layer = SimpleNamespace(
        weight=weight,
        input_scale=scale,
        orig_dtype=dtype,
        weight_scale=torch.full((64, 1), 0.01, device="cuda"),
        use_flashinfer_bmm=False,
    )
    x = torch.randn(m, 512, device="cuda", dtype=dtype)
    activated = silu_and_mul(x)
    with observe_gemm() as baseline_calls:
        expected = method.apply(layer, activated)
    assert len(baseline_calls) == 1
    if backend != "auto":
        assert baseline_calls[0]["backend"] == (
            "fp8_scaled_mm" if backend == "cutlass" else "triton_scaled_mm"
        )
    q, rows = check(x, scale)
    value = ModelOptFp8Input(q, scale, dtype, rows)
    with (
        observe_gemm() as calls,
        patch(
            "sglang.srt.layers.quantization.fp8_utils.static_quant_fp8",
            side_effect=AssertionError("pre-quantized input was quantized again"),
        ),
    ):
        for payload in (value, (q, scale), (q, scale, dtype)):
            assert torch.equal(method.apply(layer, payload), expected)
            assert calls[-1]["backend"] == baseline_calls[0]["backend"]
            assert calls[-1]["options"] == baseline_calls[0]["options"]
            if payload is value:
                assert_row_reuse(calls[-1], rows)
        # Dropping row scales preserves numerical results but must fail the
        # storage-reuse assertion for the fused producer.
        assert torch.equal(
            method.apply(layer, ModelOptFp8Input(q, scale, dtype)), expected
        )
        with pytest.raises(AssertionError, match="row scales were not reused"):
            assert_row_reuse(calls[-1], rows)
    (tmp_path / "gemm-calls.json").write_text(
        json.dumps(dict(baseline=baseline_calls, candidate=calls), indent=2)
    )
    for bad in (
        q,
        (q,),
        (q, scale.clone()),
        (q, scale, torch.float32),
        ModelOptFp8Input(q[:, :0], scale, dtype, rows),
        ModelOptFp8Input(q, scale, dtype, rows.flatten()),
    ):
        with pytest.raises((ValueError, TypeError)):
            method.apply(layer, bad)
    with patch.object(method, "_static_fp8_sm89", False):
        with pytest.raises(TypeError, match="SM89"):
            method.apply(layer, value)
