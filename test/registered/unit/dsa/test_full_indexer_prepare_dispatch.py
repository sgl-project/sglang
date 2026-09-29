"""CPU tests for full-indexer eligibility and kernel dispatch."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.hip_gfx950 import indexer_prepare
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _arguments(
    rows: int,
    heads: int = 32,
    hidden_size: int = 6144,
    q_lora_rank: int = 2048,
    page_size: int = 64,
    rope_dim: int = 64,
):
    tensor = torch.empty(0)
    arguments = [tensor] * 11
    arguments[0] = torch.empty(rows, hidden_size)
    arguments[1] = torch.empty(rows, q_lora_rank)
    arguments[4] = torch.empty(heads)
    arguments[7] = torch.empty(1, rope_dim)
    arguments[10] = torch.empty(1, page_size, 132)
    return tuple(arguments)


def test_full_indexer_prepare_falls_back_for_unprofiled_rows():
    assert indexer_prepare.full_indexer_prepare(*_arguments(0), eps=1e-6) is None
    assert indexer_prepare.full_indexer_prepare(*_arguments(129), eps=1e-6) is None


@pytest.mark.parametrize(
    "hidden_size,heads,q_lora_rank",
    (
        (1536, 16, 1024),
        (3072, 32, 1536),
        (4608, 48, 2048),
        (7680, 64, 2560),
    ),
)
def test_full_indexer_prepare_supports_general_geometries(
    hidden_size, heads, q_lora_rank
):
    assert indexer_prepare.is_full_indexer_prepare_geometry_supported(
        hidden_size, heads, q_lora_rank
    )


@pytest.mark.parametrize("rope_dim", (2, 32, 64, 96, 128))
def test_full_indexer_prepare_supports_general_rope_dimensions(rope_dim):
    assert indexer_prepare.is_full_indexer_prepare_layout_supported(
        128, rope_dim, 128, "ue8m0"
    )


@pytest.mark.parametrize(
    "head_dim,rope_dim,block_size,scale_fmt",
    (
        (64, 64, 128, "ue8m0"),
        (128, 0, 128, "ue8m0"),
        (128, 63, 128, "ue8m0"),
        (128, 130, 128, "ue8m0"),
        (128, 64, 64, "ue8m0"),
        (128, 64, 128, "tensor"),
    ),
)
def test_full_indexer_prepare_rejects_unsupported_layouts(
    head_dim, rope_dim, block_size, scale_fmt
):
    assert not indexer_prepare.is_full_indexer_prepare_layout_supported(
        head_dim, rope_dim, block_size, scale_fmt
    )


@pytest.mark.parametrize(
    "hidden_size,heads,q_lora_rank,page_size",
    (
        (2048, 32, 2048, 64),
        (6144, 24, 2048, 64),
        (6144, 32, 1280, 64),
        (6144, 32, 2048, 40),
    ),
)
def test_full_indexer_prepare_rejects_unsupported_geometries(
    hidden_size, heads, q_lora_rank, page_size
):
    assert (
        indexer_prepare.full_indexer_prepare(
            *_arguments(
                1,
                heads=heads,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                page_size=page_size,
            ),
            eps=1e-6,
        )
        is None
    )


def test_full_indexer_prepare_dispatches_supported_decode_rows(monkeypatch):
    calls = []

    def run_small(*args, **kwargs):
        calls.append(("small", args[0].shape[0], kwargs))
        return "q", "weights"

    def run_large(*args, **kwargs):
        calls.append(("large", args[0].shape[0], kwargs))
        return "q", "weights"

    package = "sglang.kernels.ops.attention.dsa.hip_gfx950.gluon"
    monkeypatch.setitem(
        sys.modules,
        f"{package}.generic",
        SimpleNamespace(indexer_prepare=run_small),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package}.large_m",
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
    assert indexer_prepare.full_indexer_prepare(
        *_arguments(64, hidden_size=3072, q_lora_rank=1536), eps=1e-6
    ) == ("q", "weights")
    assert indexer_prepare.full_indexer_prepare(
        *_arguments(64, hidden_size=4608), eps=1e-6
    ) == ("q", "weights")
    assert indexer_prepare.full_indexer_prepare(
        *_arguments(10, rope_dim=32),
        eps=1e-6,
        rope_dim=32,
        is_neox_style=True,
    ) == ("q", "weights")
    assert indexer_prepare.full_indexer_prepare(
        *_arguments(64, rope_dim=128),
        eps=1e-6,
        rope_dim=128,
        is_neox_style=True,
    ) == ("q", "weights")
    assert [(kind, rows) for kind, rows, _ in calls] == [
        ("small", 1),
        ("small", 2),
        ("small", 4),
        ("small", 10),
        ("small", 40),
        ("large", 64),
        ("large", 96),
        ("large", 128),
        ("small", 64),
        ("large", 64),
        ("small", 64),
        ("small", 10),
        ("large", 64),
    ]
    for _, _, kwargs in calls[:-2]:
        assert kwargs == {
            "eps": 1e-6,
            "rope_dim": 64,
            "is_neox_style": False,
        }
    assert calls[-2][2] == {
        "eps": 1e-6,
        "rope_dim": 32,
        "is_neox_style": True,
    }
    assert calls[-1][2] == {
        "eps": 1e-6,
        "rope_dim": 128,
        "is_neox_style": True,
    }


@pytest.mark.parametrize("rope_dim", (0, 63, 130))
def test_full_indexer_prepare_rejects_invalid_rope_dimensions(rope_dim):
    cos_sin_width = max(rope_dim, 1)
    assert (
        indexer_prepare.full_indexer_prepare(
            *_arguments(1, rope_dim=cos_sin_width), eps=1e-6, rope_dim=rope_dim
        )
        is None
    )


def test_full_indexer_prepare_rejects_mismatched_rope_table():
    assert (
        indexer_prepare.full_indexer_prepare(
            *_arguments(1, rope_dim=64), eps=1e-6, rope_dim=32
        )
        is None
    )


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
