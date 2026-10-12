"""Signature-cached Triton launches (``fast_launch``) behind the diffusion norm wrappers.

A cached launch must reproduce the JIT launch bit-for-bit, must not serve an operand
its compiled kernel was not specialized for, must not sync inside graph capture, and
must stay out of torch.compile tracing.
"""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.common import fast_launch
from sglang.kernels.ops.diffusion.norm import native_bf16_rmsnorm_triton as rms
from sglang.kernels.ops.diffusion.norm import zimage_qk_rmsnorm_triton as qk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EPS = 1e-6
CASES = ["rmsnorm_scale", "rmsnorm_tanh_residual", "qk_rmsnorm"]


class _NoJit:
    def __getitem__(self, grid):
        raise AssertionError("JIT launch")


@pytest.fixture(autouse=True)
def fresh_launches(monkeypatch):
    monkeypatch.setattr(fast_launch, "_ENABLED", True)
    monkeypatch.setattr(rms, "_RMSNORM_SCALE_LAUNCH", fast_launch.SignatureLaunch())
    monkeypatch.setattr(
        rms, "_RMSNORM_TANH_RESIDUAL_LAUNCH", fast_launch.SignatureLaunch()
    )
    monkeypatch.setattr(qk, "_QK_RMSNORM_LAUNCH", fast_launch.SignatureLaunch())
    torch.manual_seed(0)


def _bf16(*shape, offset=0):
    """Random bf16 tensor whose storage starts ``offset`` elements into its buffer."""
    numel = math.prod(shape)
    buffer = torch.randn(numel + offset, device="cuda", dtype=torch.bfloat16)
    return buffer[offset:].view(shape)


def _call(case, offset=0):
    """(wrapper, kernel module, kernel name, arguments) with ``x`` offset by elements."""
    if case == "rmsnorm_scale":
        x = _bf16(1, 1088, 3840, offset=offset)
        args = (x, _bf16(3840), _bf16(1, 1, 3840), EPS)
        return rms.rmsnorm_scale, rms, "_rmsnorm_scale_kernel", args
    if case == "rmsnorm_tanh_residual":
        x = _bf16(2, 77, 2560, offset=offset)
        args = (x, _bf16(2, 1, 2560), _bf16(2, 77, 2560), _bf16(2560), EPS)
        return rms.rmsnorm_tanh_residual, rms, "_rmsnorm_tanh_residual_kernel", args
    # The q slice of a fused qkv projection: strided tokens, packed heads.
    qkv = _bf16(1, 1088, 3 * 30 * 128, offset=offset)
    x = qkv[..., : 30 * 128].unflatten(-1, (30, 128))
    return (
        qk.zimage_qk_rmsnorm_native,
        qk,
        "_qk_rmsnorm_native_kernel",
        (x, _bf16(128), EPS),
    )


def _jit(monkeypatch, fn, args):
    monkeypatch.setattr(fast_launch, "_ENABLED", False)
    out = fn(*args)
    monkeypatch.setattr(fast_launch, "_ENABLED", True)
    return out


@torch.no_grad()
@pytest.mark.parametrize("case", CASES)
def test_cached_launch_matches_jit(monkeypatch, case):
    fn, module, kernel, args = _call(case)
    assert fn(*args) is not None  # JIT launch, recorded
    fresh = _call(case)[3]  # same signature, new values
    expected = _jit(monkeypatch, fn, fresh)

    monkeypatch.setattr(module, kernel, _NoJit())
    assert torch.equal(fn(*fresh), expected)


@torch.no_grad()
@pytest.mark.parametrize("case", CASES)
def test_misaligned_operand_skips_aligned_launch(monkeypatch, case):
    # Triton specializes on 16-byte pointer alignment, so a launch compiled for
    # an aligned x must not run on a misaligned one.
    fn, module, kernel, args = _call(case)
    fn(*args)
    misaligned = _call(case, offset=1)[3]
    assert misaligned[0].data_ptr() % 16 != 0

    monkeypatch.setattr(module, kernel, _NoJit())
    with pytest.raises(AssertionError, match="JIT launch"):
        fn(*misaligned)


@torch.no_grad()
def test_capture_replays_and_records_verified_kernel(monkeypatch):
    fn, module, kernel, recorded = _call("rmsnorm_scale")
    fn(*recorded)  # verifies the kernel's launcher
    unseen = (_bf16(2, 64, 3840), recorded[1], _bf16(2, 1, 3840), EPS)
    _jit(monkeypatch, fn, unseen)  # compile it, so capture does not

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_recorded = fn(*recorded)
        out_unseen = fn(*unseen)  # JIT launch, recorded without a sync
    for t in (recorded[0], unseen[0]):
        t.copy_(torch.randn_like(t))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(out_recorded, _jit(monkeypatch, fn, recorded))
    expected = _jit(monkeypatch, fn, unseen)
    assert torch.equal(out_unseen, expected)
    monkeypatch.setattr(module, kernel, _NoJit())
    assert torch.equal(fn(*unseen), expected)


@torch.no_grad()
def test_capture_skips_unverified_kernel(monkeypatch):
    fn, module, kernel, args = _call("rmsnorm_scale")
    _jit(monkeypatch, fn, args)  # compile it, so capture does not

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args)  # verifying would sync, so nothing is recorded
    graph.replay()
    torch.cuda.synchronize()

    monkeypatch.setattr(module, kernel, _NoJit())
    with pytest.raises(AssertionError, match="JIT launch"):
        fn(*args)


@torch.no_grad()
@pytest.mark.parametrize("case", CASES)
def test_wrapper_traces_whole_under_compile(monkeypatch, case):
    fn, _, _, args = _call(case)
    expected = _jit(monkeypatch, fn, args)
    fn(*args)  # recorded; tracing must still take the JIT path
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert torch.equal(compiled(*args), expected)


def test_launcher_is_checked_once_per_kernel():
    calls = []

    class _Exact:
        def __getitem__(self, grid):
            def runner(out, *rest):
                calls.append(grid)
                out.zero_()

            return runner

    launches = fast_launch.SignatureLaunch()
    out = torch.zeros(8, device="cuda")
    launches.record(("a",), _Exact(), (1,), (out, 3))
    launches.record(("b",), _Exact(), (2,), (out, 3))
    assert len(calls) == 1
    assert launches.get(("a",)) is not None and launches.get(("b",)) is not None


def test_launcher_that_does_not_reproduce_is_not_kept():
    class _Wrong:
        def __getitem__(self, grid):
            return lambda out, *rest: out.fill_(1)

    launches = fast_launch.SignatureLaunch()
    out = torch.zeros(8, device="cuda")
    launches.record(("a",), _Wrong(), (1,), (out, 3))
    launches.record(("b",), _Wrong(), (1,), (out, 3))
    assert launches.get(("a",)) is None and launches.get(("b",)) is None
    assert fast_launch._ENABLED


def test_launcher_error_disables_fast_launch():
    class _OtherConvention:
        def __getitem__(self, grid):
            def runner(*args):
                raise TypeError("function takes exactly 1 argument")

            return runner

    x = torch.zeros(8, device="cuda")
    fast_launch.SignatureLaunch().record(("sig",), _OtherConvention(), (1,), (x, 3))
    assert fast_launch.launch_signature(x) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
