"""CPU tests for the residue fold warmup registry."""

import contextlib

import pytest

from sglang.srt.layers.quantization.residue_nvfp4 import warmup as W
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


@pytest.fixture(autouse=True)
def clean_registry():
    saved_shapes = set(W._FOLD_SHAPES)
    saved_flag = W._WARMED_UP
    W._FOLD_SHAPES.clear()
    W._WARMED_UP = False
    yield
    W._FOLD_SHAPES.clear()
    W._FOLD_SHAPES.update(saved_shapes)
    W._WARMED_UP = saved_flag


def test_register_and_dedup():
    W.register_fold_shape(4096, 4096, 4096, True)
    W.register_fold_shape(4096, 4096, 4096, True)
    W.register_fold_shape(2048, 4096, 5120, False)
    assert W.registered_fold_shapes() == frozenset(
        {(4096, 4096, 4096, True), (2048, 4096, 5120, False)}
    )


def test_warmup_noops_without_registered_shapes():
    # Must not touch CUDA or import the fold package on a plain checkpoint.
    assert W.maybe_warmup_residue_fold() == 0


def test_observer_noops_without_registered_shapes():
    ctx = W.observe_residue_fold_compiles("test")
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_warmup_is_idempotent(monkeypatch):
    calls = []

    def fake_warmup(major, minor, shapes=None):
        calls.append((major, minor, tuple(shapes)))
        return 3

    import sglang.kernels.ops.gemm.residue_fold as rf

    monkeypatch.setattr(rf, "warmup", fake_warmup)
    monkeypatch.setattr(W, "_precompile_tuner_candidates", lambda major, shapes: 0)

    class _Cap:
        @staticmethod
        def get_device_capability():
            return (10, 0)

    monkeypatch.setattr(W.torch, "cuda", _Cap)

    W.register_fold_shape(4096, 4096, 4096, True)
    assert W.maybe_warmup_residue_fold() == 3
    assert W.maybe_warmup_residue_fold() == 0  # second call is a no-op
    assert len(calls) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
