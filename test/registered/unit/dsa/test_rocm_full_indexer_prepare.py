import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.hip_gfx950 import indexer_prepare
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _arguments(rows: int, heads: int = 32):
    tensor = torch.empty(rows)
    arguments = [tensor] * 11
    arguments[4] = torch.empty(heads)
    return tuple(arguments)


def test_full_indexer_prepare_falls_back_for_unprofiled_rows():
    assert indexer_prepare.full_indexer_prepare(*_arguments(0), eps=1e-6) is None
    assert indexer_prepare.full_indexer_prepare(*_arguments(129), eps=1e-6) is None


def test_full_indexer_prepare_dispatches_supported_decode_rows(monkeypatch):
    calls = []

    def run_small(*args, **kwargs):
        calls.append(("small", args[0].shape[0], kwargs))
        return "q", "weights"

    def run_large(*args, **kwargs):
        calls.append(("large", args[0].shape[0], kwargs))
        return "q", "weights"

    package = "sglang.kernels.ops.attention.dsa.hip_gfx950"
    monkeypatch.setitem(
        sys.modules,
        f"{package}.indexer_prepare_m1",
        SimpleNamespace(indexer_prepare=run_small),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package}.indexer_prepare_m4",
        SimpleNamespace(indexer_prepare=run_small),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package}.indexer_prepare_m128",
        SimpleNamespace(indexer_prepare=run_large),
    )

    assert indexer_prepare.full_indexer_prepare(*_arguments(1), eps=1e-6) == (
        "q",
        "weights",
    )
    for rows in (2, 4, 10, 40, 64, 96, 128):
        assert indexer_prepare.full_indexer_prepare(*_arguments(rows), eps=1e-6) == (
            "q",
            "weights",
        )
    assert indexer_prepare.full_indexer_prepare(
        *_arguments(64, heads=16), eps=1e-6
    ) == ("q", "weights")
    assert calls == [
        ("small", 1, {"eps": 1e-6}),
        ("small", 2, {"eps": 1e-6}),
        ("small", 4, {"eps": 1e-6}),
        ("small", 10, {"eps": 1e-6}),
        ("small", 40, {"eps": 1e-6}),
        ("large", 64, {"eps": 1e-6}),
        ("large", 96, {"eps": 1e-6}),
        ("large", 128, {"eps": 1e-6}),
        ("small", 64, {"eps": 1e-6}),
    ]


def test_full_indexer_prepare_rejects_old_triton(monkeypatch):
    def import_old_triton(name):
        assert name == "triton"
        return SimpleNamespace(__version__="3.4.0")

    indexer_prepare.is_full_indexer_prepare_available.cache_clear()
    try:
        monkeypatch.setattr(indexer_prepare, "import_module", import_old_triton)
        assert not indexer_prepare.is_full_indexer_prepare_available()
    finally:
        indexer_prepare.is_full_indexer_prepare_available.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
