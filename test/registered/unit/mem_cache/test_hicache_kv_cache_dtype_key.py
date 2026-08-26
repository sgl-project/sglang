"""Unit tests for kv_cache_dtype inclusion in HiCache storage keys.

Verifies that HiCacheStorageConfig.kv_cache_dtype is threaded into the
key derivation of each backend so that runs with different --kv-cache-dtype
do not silently reuse each other's cache entries.

Issue: https://github.com/sgl-project/sglang/issues/33268
"""

import os
import tempfile
import pytest

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
)


def _make_config(
    kv_cache_dtype=None,
    tp_rank=0,
    tp_size=8,
    is_mla_model=True,
    model_name="DeepSeek-V4-Flash",
    pp_size=1,
    pp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=model_name,
        kv_cache_dtype=kv_cache_dtype,
    )


class TestHiCacheDtypeKeyCollision:
    """Verify that different kv_cache_dtype values produce different storage keys."""

    def test_different_dtype_produces_different_suffix(self):
        """Two configs differing only in kv_cache_dtype must have different suffixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_bf16 = HiCacheFile(_make_config("torch.bfloat16"), file_path=tmpdir)
            cache_fp8 = HiCacheFile(_make_config("torch.float8_e4m3fn"), file_path=tmpdir)

            assert cache_bf16.config_suffix != cache_fp8.config_suffix, (
                f"Suffix collision: bf16={cache_bf16.config_suffix}, "
                f"fp8={cache_fp8.config_suffix}"
            )
            assert "dtype_torch.bfloat16" in cache_bf16.config_suffix
            assert "dtype_torch.float8_e4m3fn" in cache_fp8.config_suffix

    def test_dtype_in_suffixed_key(self):
        """The suffixed key must contain the dtype segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config("torch.float8_e4m3fn"), file_path=tmpdir)
            key = cache._get_suffixed_key("page_001")
            assert "dtype_torch.float8_e4m3fn" in key, f"dtype missing from key: {key}"

    def test_none_dtype_backward_compatible(self):
        """When kv_cache_dtype is None, suffix must not contain dtype.

        This ensures backward compatibility: existing cache files created
        before the fix remain accessible with the old key format.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config(None), file_path=tmpdir)
            assert "dtype" not in cache.config_suffix, (
                f"dtype leaked into suffix when kv_cache_dtype is None: "
                f"{cache.config_suffix}"
            )
            assert cache.config_suffix == "_DeepSeek-V4-Flash", (
                f"Unexpected suffix for None dtype: {cache.config_suffix}"
            )

    def test_same_dtype_produces_same_suffix(self):
        """Two configs with the same kv_cache_dtype must produce the same suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = HiCacheFile(_make_config("torch.bfloat16"), file_path=tmpdir)
            cache2 = HiCacheFile(_make_config("torch.bfloat16"), file_path=tmpdir)
            assert cache1.config_suffix == cache2.config_suffix

    def test_non_mla_model_dtype_in_suffix(self):
        """Non-MLA models should also include dtype in the suffix."""
        cfg_bf16 = _make_config("torch.bfloat16", is_mla_model=False, tp_size=4,
                                model_name="Qwen2.5-7B")
        cfg_fp8 = _make_config("torch.float8_e5m2", is_mla_model=False, tp_size=4,
                               model_name="Qwen2.5-7B")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_bf16 = HiCacheFile(cfg_bf16, file_path=tmpdir)
            cache_fp8 = HiCacheFile(cfg_fp8, file_path=tmpdir)
            assert cache_bf16.config_suffix != cache_fp8.config_suffix
            assert "dtype_torch.bfloat16" in cache_bf16.config_suffix
            assert "dtype_torch.float8_e5m2" in cache_fp8.config_suffix

    def test_dtype_suffix_with_pp(self):
        """Dtype suffix should coexist with PP suffix."""
        cfg = _make_config("torch.bfloat16", pp_size=2, pp_rank=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(cfg, file_path=tmpdir)
            assert "dtype_torch.bfloat16" in cache.config_suffix
            assert "_2_1" in cache.config_suffix  # pp_size_pp_rank

    def test_dtype_suffix_with_cp(self):
        """Dtype suffix should coexist with CP suffix."""
        cfg = _make_config("torch.float8_e4m3fn", attn_cp_size=4, attn_cp_rank=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(cfg, file_path=tmpdir)
            assert "dtype_torch.float8_e4m3fn" in cache.config_suffix
            assert "_cp2_4" in cache.config_suffix

    def test_dtype_suffix_order_after_cp(self):
        """Dtype suffix should appear after the CP suffix, not before."""
        cfg = _make_config("torch.bfloat16", attn_cp_size=2, attn_cp_rank=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(cfg, file_path=tmpdir)
            cp_pos = cache.config_suffix.find("_cp")
            dtype_pos = cache.config_suffix.find("_dtype_")
            assert cp_pos >= 0 and dtype_pos >= 0
            assert dtype_pos > cp_pos, (
                f"dtype suffix should come after CP suffix: "
                f"{cache.config_suffix}"
            )

    def test_empty_string_dtype_treated_as_set(self):
        """An empty string dtype is not None — suffix should be added.

        This is an edge case: if someone passes kv_cache_dtype="" it should
        still produce a suffix ("_dtype_"), because the field is set.
        The guard is `is not None`, not a truthiness check.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config(""), file_path=tmpdir)
            assert "_dtype_" in cache.config_suffix

    def test_different_model_names_no_false_collision(self):
        """Different models with same dtype should have different suffixes."""
        cfg1 = _make_config("torch.bfloat16", model_name="Model-A")
        cfg2 = _make_config("torch.bfloat16", model_name="Model-B")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = HiCacheFile(cfg1, file_path=tmpdir)
            cache2 = HiCacheFile(cfg2, file_path=tmpdir)
            assert cache1.config_suffix != cache2.config_suffix
            # Both should have the dtype suffix
            assert "dtype_torch.bfloat16" in cache1.config_suffix
            assert "dtype_torch.bfloat16" in cache2.config_suffix

    def test_component_key_includes_dtype(self):
        """_get_component_key should also carry the dtype suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config("torch.float8_e4m3fn"), file_path=tmpdir)
            key = cache._get_component_key("page_001", "kv")
            assert "dtype_torch.float8_e4m3fn" in key

    def test_none_dtype_component_key_backward_compatible(self):
        """_get_component_key with None dtype should not have dtype suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HiCacheFile(_make_config(None), file_path=tmpdir)
            key = cache._get_component_key("page_001", "kv")
            assert "dtype" not in key


class TestHiCacheStorageConfigField:
    """Verify the HiCacheStorageConfig dataclass field behavior."""

    def test_default_kv_cache_dtype_is_none(self):
        """The field should default to None for backward compatibility."""
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=False,
            model_name="test",
        )
        assert cfg.kv_cache_dtype is None

    def test_kv_cache_dtype_can_be_set(self):
        """The field should be settable via constructor."""
        cfg = HiCacheStorageConfig(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=False,
            model_name="test",
            kv_cache_dtype="torch.bfloat16",
        )
        assert cfg.kv_cache_dtype == "torch.bfloat16"

    def test_existing_fields_unaffected(self):
        """Adding kv_cache_dtype should not break existing fields."""
        cfg = HiCacheStorageConfig(
            tp_rank=2, tp_size=4, pp_rank=1, pp_size=2,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=True,
            enable_storage_metrics=True, is_page_first_layout=True,
            model_name="test-model",
            kv_cache_dtype="torch.float8_e4m3fn",
        )
        assert cfg.tp_rank == 2
        assert cfg.tp_size == 4
        assert cfg.pp_rank == 1
        assert cfg.pp_size == 2
        assert cfg.is_mla_model is True
        assert cfg.enable_storage_metrics is True
        assert cfg.is_page_first_layout is True
        assert cfg.model_name == "test-model"
        assert cfg.kv_cache_dtype == "torch.float8_e4m3fn"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
