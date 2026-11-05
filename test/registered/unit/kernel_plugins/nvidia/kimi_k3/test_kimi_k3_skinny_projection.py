# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-only routing contracts; these do not establish GPU kernel correctness."""

from contextlib import nullcontext

import pytest
import torch

from sglang.srt.kernel_plugins.nvidia.kimi_k3 import skinny_projection as skinny
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MetadataTensor:
    def __init__(
        self,
        shape,
        *,
        dtype=torch.bfloat16,
        device="cuda:0",
        stride=None,
        pointer=4096,
    ):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self._stride = stride or (shape[1], 1)
        self._pointer = pointer
        self.written = False

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def is_contiguous(self):
        return self._stride == (self.shape[1], 1)

    def data_ptr(self):
        return self._pointer

    def new_zeros(self, shape):
        return _MetadataTensor(shape)

    def new_empty(self, shape):
        return _MetadataTensor(shape)


class _Fallback:
    def __init__(self):
        self.calls = []

    def run(self, x, out):
        self.calls.append((x, out))
        out.written = True
        return out


def _prepare(monkeypatch, *, supported=True):
    calls = []

    def tiny(x, weight, *, out, max_m):
        calls.append((x, weight, out, max_m))
        out.written = True
        return out

    def can_use(n, k, *, max_m):
        assert (n, k, max_m) == (64, 512, 2)
        return supported

    monkeypatch.setattr(skinny, "_load_backend", lambda: (can_use, tiny))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fallback = _Fallback()
    projection = skinny.SkinnyProjection(_MetadataTensor((64, 512)), fallback)
    assert projection.prepare() is projection
    return projection, fallback, calls


def test_b1_b2_are_warmed_then_write_the_callers_output(monkeypatch):
    projection, fallback, calls = _prepare(monkeypatch)
    assert [call[0].shape[0] for call in calls] == [1, 2]
    assert all(call[3] == 2 for call in calls)
    projection.prepare()
    assert len(calls) == 2

    # A serving call must use the prepared backend, even during capture.
    monkeypatch.setattr(
        skinny, "_load_backend", lambda: pytest.fail("serving must not import/JIT")
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for batch in (1, 2):
        x = _MetadataTensor((batch, 512), stride=(528, 1))
        out = _MetadataTensor((batch, 64))
        assert projection.run(x, out) is out
        assert out.written
    assert len(calls) == 4
    assert not fallback.calls


def test_unsupported_batch_or_tensor_layout_uses_the_existing_fallback(monkeypatch):
    projection, fallback, calls = _prepare(monkeypatch)
    invalid_calls = [
        (_MetadataTensor((3, 512)), _MetadataTensor((3, 64))),
        (_MetadataTensor((1, 512), stride=(513, 1)), _MetadataTensor((1, 64))),
        (_MetadataTensor((1, 512), pointer=4098), _MetadataTensor((1, 64))),
        (_MetadataTensor((1, 512), dtype=torch.float32), _MetadataTensor((1, 64))),
        (_MetadataTensor((1, 512), device="cuda:1"), _MetadataTensor((1, 64))),
        (_MetadataTensor((1, 512)), _MetadataTensor((1, 64), stride=(128, 2))),
    ]
    for x, out in invalid_calls:
        assert projection.run(x, out) is out
        assert out.written
    assert fallback.calls == invalid_calls
    assert len(calls) == 2  # Only startup warmup, no new kernel work.


def test_backend_rejection_stays_on_fallback(monkeypatch):
    projection, fallback, calls = _prepare(monkeypatch, supported=False)
    x, out = _MetadataTensor((1, 512)), _MetadataTensor((1, 64))
    assert projection.run(x, out) is out
    assert fallback.calls == [(x, out)]
    assert not calls


def test_cpu_fallback_preserves_real_output_without_importing_gpu_backend(monkeypatch):
    monkeypatch.setattr(
        skinny, "_load_backend", lambda: pytest.fail("CPU fallback must stay lazy")
    )
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    class Fallback:
        def run(self, x, out):
            torch.mm(x, weight.T, out=out)
            return out

    projection = skinny.SkinnyProjection(weight, Fallback())
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    out = torch.empty((2, 4), dtype=torch.bfloat16)
    assert projection.run(x, out) is out  # Also safe before explicit preparation.
    projection.prepare()
    assert projection.run(x, out) is out
    torch.testing.assert_close(out, x @ weight.T)


def test_preparation_during_capture_is_rejected_before_backend_import(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        skinny, "_load_backend", lambda: pytest.fail("capture must not import/JIT")
    )
    projection = skinny.SkinnyProjection(_MetadataTensor((64, 512)), _Fallback())
    with pytest.raises(RuntimeError, match="before CUDA capture"):
        projection.prepare()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
