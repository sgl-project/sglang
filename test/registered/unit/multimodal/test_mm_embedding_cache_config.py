import sys

import pytest

from sglang.srt.mem_cache.kv_cache_builder import (
    resolve_mm_embedding_cache_size_mb,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _K3Model:
    auto_mm_embedding_cache_size_mb = 4096


class _DefaultModel:
    pass


def test_kimi_k3_uses_larger_lazy_embedding_cache(monkeypatch):
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_builder.get_model_architecture",
        lambda _model_config: (_K3Model, "TestK3"),
    )

    assert resolve_mm_embedding_cache_size_mb(object()) == 4096


def test_other_models_keep_historical_default(monkeypatch):
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.kv_cache_builder.get_model_architecture",
        lambda _model_config: (_DefaultModel, "TestDefault"),
    )

    assert resolve_mm_embedding_cache_size_mb(object()) == 100


@pytest.mark.parametrize("value", [0, 256])
def test_explicit_embedding_cache_size_wins(monkeypatch, value):
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", str(value))

    assert resolve_mm_embedding_cache_size_mb(object()) == value


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
