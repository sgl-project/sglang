import json

import pytest
import torch

from sglang.srt.layers.tilelang_gemm_wrapper import runtime
from sglang.srt.layers.tilelang_gemm_wrapper.configs import (
    DEFAULT_M_VALUES,
    SelectedConfigStore,
    generate_candidate_configs,
    validate_search_policy,
    write_selected_config_file,
)


def test_tilelang_gemm_default_m_values_are_unique():
    assert DEFAULT_M_VALUES == sorted(set(DEFAULT_M_VALUES))


def test_tilelang_gemm_autotune_search_policy_counts():
    assert len(generate_candidate_configs(1, 4096, 1024, search_policy="full")) == 832
    assert (
        len(generate_candidate_configs(1, 4096, 1024, search_policy="family_pruned"))
        == 128
    )
    assert (
        len(generate_candidate_configs(1, 4096, 1024, search_policy="fast_sm90")) == 4
    )

    splitk_decode = generate_candidate_configs(
        1, 1024, 3072, search_policy="family_pruned"
    )
    assert len(splitk_decode) == 288
    assert {config["kernel_type"] for config in splitk_decode} == {"splitK_swapAB"}

    fast_prefill = generate_candidate_configs(
        128, 1024, 3072, search_policy="fast_sm90"
    )
    assert len(fast_prefill) == 12
    assert {config["kernel_type"] for config in fast_prefill} == {"base"}


def test_tilelang_gemm_autotune_search_policy_validation():
    assert validate_search_policy("fast_sm90") == "fast_sm90"
    with pytest.raises(ValueError, match="autotune search_policy"):
        validate_search_policy("unknown")


def test_tilelang_gemm_selected_config_export_roundtrip(tmp_path):
    config = generate_candidate_configs(1, 4096, 1024, search_policy="fast_sm90")[0]
    path = tmp_path / "selected.json"
    write_selected_config_file(
        str(path),
        [config],
        metadata={"autotune_search_policy": "fast_sm90"},
    )

    payload = json.loads(path.read_text())
    assert payload["metadata"]["autotune_search_policy"] == "fast_sm90"

    store = SelectedConfigStore.from_file(str(path))
    assert store.get_exact(1, 4096, 1024)["kernel_type"] == "swapAB"


def test_tilelang_gemm_selected_config_skips_incompatible_m_family():
    decode_config = generate_candidate_configs(
        1, 4096, 1024, search_policy="fast_sm90"
    )[0]
    assert decode_config["kernel_type"] == "swapAB"

    store = SelectedConfigStore()
    store.add(decode_config)

    selected = store.select(64, 4096, 1024)
    assert selected["kernel_type"] == "base"
    assert "tuned_M" not in selected


def test_tilelang_gemm_exact_compatible_rejects_incompatible_m_family():
    decode_config = generate_candidate_configs(
        1, 4096, 1024, search_policy="fast_sm90"
    )[0]
    incompatible_exact = {**decode_config, "M": 64}

    store = SelectedConfigStore()
    store.add(incompatible_exact)

    assert store.get_exact(64, 4096, 1024)["kernel_type"] == "swapAB"
    assert store.get_exact_compatible(64, 4096, 1024) is None


def test_tilelang_gemm_exact_compatible_rejects_illegal_config_fields():
    config = generate_candidate_configs(128, 4096, 1024, search_policy="fast_sm90")[0]
    store = SelectedConfigStore()
    store.add({**config, "block_K": 64})

    assert store.get_exact(128, 4096, 1024)["block_K"] == 64
    assert store.get_exact_compatible(128, 4096, 1024) is None


def test_tilelang_gemm_selected_config_uses_nearest_compatible_m_family():
    decode_config = generate_candidate_configs(
        1, 4096, 1024, search_policy="fast_sm90"
    )[0]
    prefill_config = generate_candidate_configs(
        128, 4096, 1024, search_policy="fast_sm90"
    )[0]

    store = SelectedConfigStore()
    store.add(decode_config)
    store.add(prefill_config)

    selected = store.select(64, 4096, 1024)
    assert selected["kernel_type"] == "base"
    assert selected["tuned_M"] == 128


def test_tilelang_gemm_clear_cache_resets_loaded_configs(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_TILELANG_GEMM_CONFIG_PATH", raising=False)
    runtime.clear_cache()
    config = generate_candidate_configs(1, 4096, 1024, search_policy="fast_sm90")[0]
    path = tmp_path / "selected.json"
    write_selected_config_file(str(path), [config])

    try:
        runtime.load_selected_configs(str(path))
        assert runtime.has_selected_config(1, 4096, 1024)

        runtime.clear_cache()

        assert not runtime.has_selected_config(1, 4096, 1024)
        assert runtime.list_available_configs() == []
    finally:
        runtime.clear_cache()


def test_tilelang_gemm_partial_buffer_cache_grows_without_m_key():
    runtime.clear_runtime_cache()
    try:
        small = runtime._get_partial_buffer(
            "splitK", 2, 4, 16, torch.device("cpu"), "float32"
        )
        assert small.shape == (2, 4, 16)
        assert small.is_contiguous()
        assert len(runtime._PARTIAL_BUFFER_CACHE) == 1

        large = runtime._get_partial_buffer(
            "splitK", 2, 8, 16, torch.device("cpu"), "float32"
        )
        assert large.shape == (2, 8, 16)
        assert large.is_contiguous()
        assert len(runtime._PARTIAL_BUFFER_CACHE) == 1

        small_again = runtime._get_partial_buffer(
            "splitK", 2, 4, 16, torch.device("cpu"), "float32"
        )
        assert small_again.shape == (2, 4, 16)
        assert small_again.is_contiguous()
        assert small_again.data_ptr() == large.data_ptr()
        assert len(runtime._PARTIAL_BUFFER_CACHE) == 1
    finally:
        runtime.clear_runtime_cache()
