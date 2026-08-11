import sys
from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.kv_cache_builder import (
    resolve_mm_embedding_cache_size_mb,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _model_config(architecture: str):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
    )


def test_kimi_k3_uses_larger_lazy_embedding_cache(monkeypatch):
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)

    assert (
        resolve_mm_embedding_cache_size_mb(
            _model_config("KimiK3ForConditionalGeneration")
        )
        == 1024
    )


def test_other_models_keep_historical_default(monkeypatch):
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)

    assert (
        resolve_mm_embedding_cache_size_mb(
            _model_config("Qwen3VLForConditionalGeneration")
        )
        == 100
    )


@pytest.mark.parametrize("value", [0, 256])
def test_explicit_embedding_cache_size_wins(monkeypatch, value):
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", str(value))

    assert (
        resolve_mm_embedding_cache_size_mb(
            _model_config("KimiK3ForConditionalGeneration")
        )
        == value
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
