import json

from sglang.srt.layers.tilelang_gemm_wrapper.configs import (
    SelectedConfigStore,
    generate_candidate_configs,
    write_selected_config_file,
)


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
