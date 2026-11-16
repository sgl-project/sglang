"""BF16 NT backend parity and independent CUDA graph scratch lifetimes."""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.sm120_bf16 import (
    Sm120Bf16Linear,
    linear_bf16_cublaslt,
    linear_bf16_quack,
)
from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
    assert_special_values_match,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip("SM120 required", allow_module_level=True)


def inputs(m, n, k):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=x.dtype) / k**0.5
    return x, w


@pytest.mark.parametrize("m", [224, 448])
@pytest.mark.parametrize(
    "n,k", [(24576, 4096), (4096, 12288), (6144, 4096), (4096, 4096)]
)
@torch.inference_mode()
def test_precision_and_updated_operands(m, n, k):
    backend = linear_bf16_quack if n == 24576 else linear_bf16_cublaslt
    x, w = inputs(m, n, k)
    for scale in (1.0, -0.5):
        x.mul_(scale)
        w.neg_()
        actual = backend(x, w)
        assert actual.shape == (m, n) and actual.dtype == torch.bfloat16
        assert_error_no_worse_than_reference(
            reference_fp64=F.linear(x.double(), w.double()),
            baseline=F.linear(x, w),
            candidate=actual,
        )


@pytest.mark.parametrize(
    "backend,n", [(linear_bf16_quack, 24576), (linear_bf16_cublaslt, 6144)]
)
@torch.inference_mode()
def test_nan_inf_and_zero_propagation(backend, n):
    x, w = inputs(224, n, 4096)
    x[0, 0] = float("inf")
    x[1, 0] = -float("inf")
    x[2].zero_()
    x[3, 3] = float("nan")
    w[0, 1] = float("nan")
    assert_special_values_match(
        baseline=F.linear, candidate=backend, cases={"special_values": (x, w)}
    )


@pytest.mark.parametrize(
    "backend,n", [(linear_bf16_quack, 24576), (linear_bf16_cublaslt, 6144)]
)
@torch.inference_mode()
def test_graphs_captured_on_one_stream_replay_independently(backend, n):
    from flashinfer.autotuner import AutoTuner, autotune

    capture_stream = torch.cuda.Stream()
    states = []
    scratch = []
    tactics = []
    tuner = AutoTuner.get()
    choose_one = tuner.choose_one

    def record_tactic(*args, **kwargs):
        inputs = kwargs["inputs"] if "inputs" in kwargs else args[3]
        scratch.append(inputs[5].data_ptr())
        runner, tactic = choose_one(*args, **kwargs)
        tactics.append(tactic)
        return runner, tactic

    # The same capture stream must not cause two graphs to share mutable scratch.
    with torch.cuda.stream(capture_stream):
        for _ in range(2):
            x, w = inputs(448, n, 4096)
            with autotune(True):
                backend(x, w)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with patch.object(tuner, "choose_one", record_tactic):
                with torch.cuda.graph(graph, stream=capture_stream):
                    out = backend(x, w)
            states.append((torch.cuda.Stream(), graph, x, w, out))
    if backend is linear_bf16_cublaslt:
        assert len(scratch) == 2 and len(set(scratch)) == 2
        assert len(tactics) == 2 and all(tactic != -1 for tactic in tactics)
    for step in range(3):
        for index, (stream, graph, x, w, out) in enumerate(states):
            stream.wait_stream(capture_stream)
            with torch.cuda.stream(stream):
                x.fill_((index + 1) * (step + 1) / 16)
                w.neg_()
                graph.replay()
        for stream, graph, x, w, out in states:
            stream.synchronize()
            torch.testing.assert_close(
                out.float(), F.linear(x.float(), w.float()), rtol=0.02, atol=0.03
            )


@torch.inference_mode()
def test_unsupported_layouts_and_shapes_keep_native_dispatch():
    x, w = inputs(224, 4096, 4096)
    op = Sm120Bf16Linear()
    for backend in (linear_bf16_quack, linear_bf16_cublaslt):
        assert op(x, w, x.new_zeros(4096), backend=backend) is None
        assert op(x[:, ::2], w[:, ::2], backend=backend) is None
        assert op(x.float(), w.float(), backend=backend) is None
        assert op(x[:0], w, backend=backend) is None
        assert op(x, w[:0], backend=backend) is None
        assert (
            op(x[:, :4095].contiguous(), w[:, :4095].contiguous(), backend=backend)
            is None
        )
        assert op(x, w[:4095], backend=backend) is None
    assert not op._warmed


@pytest.mark.parametrize("m", [128, 223, 256])
@pytest.mark.parametrize("backend", [linear_bf16_quack, linear_bf16_cublaslt])
@torch.inference_mode()
def test_explicit_backends_support_non_cosmos_shapes(m, backend):
    x, w = inputs(m, 512, 256)
    x = x.unsqueeze(0)
    actual = Sm120Bf16Linear()(x, w, backend=backend)
    assert actual is not None and actual.shape == (1, m, 512)
    assert_error_no_worse_than_reference(
        reference_fp64=F.linear(x.double(), w.double()),
        baseline=F.linear(x, w),
        candidate=actual,
    )


@torch.inference_mode()
def test_backend_warmup_is_independent_for_same_shape():
    x, w = inputs(128, 512, 256)
    op = Sm120Bf16Linear()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        F.linear(x, w)
        stream.synchronize()
        for backend in (linear_bf16_quack, linear_bf16_cublaslt):
            cold = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cold, stream=stream):
                assert op(x, w, backend=backend) is None
                fallback = F.linear(x, w)
            cold.replay()
            stream.synchronize()
            assert torch.equal(fallback, F.linear(x, w))
            assert op(x, w, backend=backend) is not None
            stream.synchronize()
            warm = torch.cuda.CUDAGraph()
            with torch.cuda.graph(warm, stream=stream):
                out = op(x, w, backend=backend)
            assert out is not None
            x.mul_(0.5)
            warm.replay()
            stream.synchronize()
            torch.testing.assert_close(
                out.float(), F.linear(x.float(), w.float()), rtol=0.02, atol=0.03
            )


@torch.inference_mode()
def test_cublaslt_profiles_exact_m_when_shapes_are_interleaved():
    # cuBLASLt tactic indices refer to per-shape algorithm lists. Bucketed
    # tuning reused a different M's index and doubled QKV latency after warmup.
    from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

    runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
    get_tactics = runner.get_valid_tactics
    profiled_m = set()

    def record_profile(inputs, profile):
        profiled_m.add(inputs[0].shape[0])
        return get_tactics(inputs, profile)

    op = Sm120Bf16Linear()
    with patch.object(runner, "get_valid_tactics", side_effect=record_profile):
        for m in (129, 193, 129):
            x, w = inputs(m, 768, 256)
            out = op(x, w, backend=linear_bf16_cublaslt)
            assert_error_no_worse_than_reference(
                reference_fp64=F.linear(x.double(), w.double()),
                baseline=F.linear(x, w),
                candidate=out,
            )
    assert profiled_m == {129, 193}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
