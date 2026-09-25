"""Correctness and graph tests for Qwen4-Exp QSA MQA scoring."""

import inspect
import sys

import pytest
import torch

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.kernels.ops.attention.qsa import mqa as triton_mqa_module
from sglang.kernels.ops.attention.qsa.mqa import (
    triton_qsa_mqa_decode,
    triton_qsa_mqa_prefill,
)
from sglang.srt.layers.attention.qsa import mqa as mqa_module
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
    torch_qsa_mqa_decode,
    torch_qsa_mqa_prefill,
)

HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
TOPK = 512


def _require_sm120():
    _require_cuda()
    if not mqa_module.is_sm120_supported() or mqa_module.is_sm121():
        pytest.skip("the production Triton dispatch is specific to SM120")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _assert_logits(actual, expected):
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-2, rtol=2e-2)


def _assert_topk_sets(actual, expected, starts, ends, topk=TOPK):
    """Require exact sets unless the reference cutoff is numerically tied."""

    finite = torch.isfinite(expected)
    max_error = (actual[finite] - expected[finite]).abs().max()
    for row in range(expected.shape[0]):
        start = int(starts[row])
        end = int(ends[row])
        width = min(topk, end - start)
        if width <= 0:
            continue
        actual_idx = torch.topk(actual[row, start:end], width).indices + start
        expected_values, expected_local = torch.topk(expected[row, start:end], width)
        expected_idx = expected_local + start
        if set(actual_idx.tolist()) == set(expected_idx.tolist()):
            continue
        cutoff = expected_values[-1]
        tied = (expected[row, start:end] - cutoff).abs() <= 2 * max_error
        stable = torch.nonzero((expected[row, start:end] > cutoff) & ~tied).flatten()
        stable_expected = set((stable + start).tolist())
        assert stable_expected.issubset(set(actual_idx.tolist()))


def _make_prefill_case():
    torch.manual_seed(41)
    lengths = torch.tensor([1, 127, 513, 2049, 8192], dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    sequence_ids = torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.long)
    starts = offsets[:-1].index_select(0, sequence_ids).cuda()
    ends = offsets[1:].index_select(0, sequence_ids).cuda()
    q = torch.randn(
        sequence_ids.numel(), HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(int(offsets[-1]), 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return q, k, starts, ends


def _make_decode_case():
    torch.manual_seed(42)
    lengths = torch.tensor(
        [1, 513, 2047, 8193, 32768], device="cuda", dtype=torch.int32
    )
    max_len = int(lengths.max())
    max_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE
    batch = lengths.numel()
    page_table = torch.arange(
        batch * max_pages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_pages)
    cache = torch.randn(
        batch * max_pages,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q, torch.zeros_like(q)], dim=1)
    return q, cache, page_table, lengths, max_len


def test_qsa_mqa_triton_ragged_prefill_logits_and_topk(monkeypatch):
    _require_cuda()
    case = _make_prefill_case()
    expected = torch_qsa_mqa_prefill(*case)
    actual = triton_qsa_mqa_prefill(*case)
    _assert_logits(actual, expected)
    _assert_topk_sets(actual, expected, case[2], case[3])
    monkeypatch.setenv("SGLANG_QSA_MQA_BACKEND", "triton")
    _assert_logits(qsa_mqa_prefill(*case), expected)


def test_qsa_mqa_triton_ragged_decode_logits_topk_and_graph():
    _require_cuda()
    case = _make_decode_case()
    expected = torch_qsa_mqa_decode(*case)
    actual = triton_qsa_mqa_decode(*case)
    _assert_logits(actual, expected)
    starts = torch.zeros_like(case[3])
    _assert_topk_sets(actual, expected, starts, case[3])

    for _ in range(3):
        triton_qsa_mqa_decode(*case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = triton_qsa_mqa_decode(*case)
    graph.replay()
    torch.cuda.synchronize()
    first = replay_output.clone()
    case[0].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(first, replay_output)
    scaled_expected = torch_qsa_mqa_decode(*case)
    _assert_logits(replay_output, scaled_expected)


def test_qsa_mqa_sm120_auto_uses_triton(monkeypatch):
    _require_sm120()
    case = _make_decode_case()
    monkeypatch.delenv("SGLANG_QSA_MQA_BACKEND", raising=False)
    expected = triton_qsa_mqa_decode(*case)
    actual = qsa_mqa_decode(*case)
    assert torch.equal(actual, expected)


def test_qsa_mqa_sm120_does_not_fall_back_from_triton(monkeypatch):
    _require_cuda()
    case = _make_prefill_case()
    monkeypatch.delenv("SGLANG_QSA_MQA_BACKEND", raising=False)
    monkeypatch.setattr(mqa_module, "is_sm120_supported", lambda: True)

    def fail(*args, **kwargs):
        raise RuntimeError("triton launch failed")

    monkeypatch.setattr(mqa_module, "triton_qsa_mqa_prefill", fail)
    with pytest.raises(RuntimeError, match="triton launch failed"):
        qsa_mqa_prefill(*case)


def test_qsa_mqa_auto_dispatch_uses_sm120_predicate(monkeypatch):
    monkeypatch.delenv("SGLANG_QSA_MQA_BACKEND", raising=False)
    monkeypatch.setattr(mqa_module, "is_sm120_supported", lambda: True)
    assert mqa_module._resolve_mqa_backend(is_cuda=True) == "triton"
    monkeypatch.setattr(mqa_module, "is_sm120_supported", lambda: False)
    expected = "tilelang" if mqa_module.HAS_TILELANG else "torch"
    assert mqa_module._resolve_mqa_backend(is_cuda=True) == expected


def test_qsa_mqa_tilelang_failure_is_not_silently_downgraded(monkeypatch):
    class CudaQuery:
        is_cuda = True

    monkeypatch.setenv("SGLANG_QSA_MQA_BACKEND", "tilelang")
    monkeypatch.setattr(mqa_module, "HAS_TILELANG", True)

    def fail():
        raise RuntimeError("tilelang launch failed")

    with pytest.raises(RuntimeError, match="tilelang launch failed"):
        mqa_module._run_tilelang_or_torch(CudaQuery(), fail, lambda: "torch")


def test_qsa_mqa_tilelang_initial_compile_holds_serialization_lock(monkeypatch):
    class CudaQuery:
        is_cuda = True

    class RecordingLock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, exc_type, exc, traceback):
            self.held = False

    lock = RecordingLock()
    monkeypatch.setenv("SGLANG_QSA_MQA_BACKEND", "tilelang")
    monkeypatch.setattr(mqa_module, "HAS_TILELANG", True)
    monkeypatch.setattr(mqa_module, "_tilelang_backend_lock", lock)
    monkeypatch.setattr(mqa_module, "_tilelang_backend_ready", False)

    def tilelang_call():
        assert lock.held
        return "tilelang"

    assert (
        mqa_module._run_tilelang_or_torch(CudaQuery(), tilelang_call, lambda: "torch")
        == "tilelang"
    )
    assert mqa_module._tilelang_backend_ready


def test_qsa_mqa_kernels_are_registered_for_sm120():
    for op, target in (
        (
            "attention.qsa_mqa_decode",
            "sglang.kernels.ops.attention.qsa.mqa:triton_qsa_mqa_decode",
        ),
        (
            "attention.qsa_mqa_prefill",
            "sglang.kernels.ops.attention.qsa.mqa:triton_qsa_mqa_prefill",
        ),
    ):
        spec = registry.get_backend(op, KernelBackend.TRITON)
        assert spec.target == target
        assert len(spec.capabilities) == 1
        capability = next(iter(spec.capabilities))
        assert capability.min_cuda_arch == (12, 0)
        assert capability.max_cuda_arch == (12, 0)


def test_qsa_mqa_strides_are_runtime_arguments():
    for kernel in (
        triton_mqa_module._qsa_mqa_prefill_kernel,
        triton_mqa_module._qsa_mqa_decode_kernel,
    ):
        signature = inspect.signature(kernel.fn)
        for name, parameter in signature.parameters.items():
            if name.startswith("stride_"):
                assert parameter.annotation is inspect.Parameter.empty, name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
