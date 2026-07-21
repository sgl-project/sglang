"""Source-level contracts for the optional FlexKV integration.

FlexKV is not installed in the CPU test image, so these checks intentionally
avoid importing its runtime module while still protecting SGLang-side wiring.
"""

from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
CONNECTOR_SOURCE = (
    REPO_ROOT / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py"
).read_text(encoding="utf-8")
HYBRID_CACHE_SOURCE = (
    REPO_ROOT
    / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_hybrid_radix_cache.py"
).read_text(encoding="utf-8")


def test_dsv4_state_sidecars_default_on_with_swa_only_opt_out():
    assert '"swa_multi_group",\n            None,' in CONNECTOR_SOURCE
    assert "if swa_multi_group is not False:" in CONNECTOR_SOURCE
    assert "registering SWA without" in CONNECTOR_SOURCE


def test_cp_layerwise_metadata_uses_effective_tp_coordinates():
    assert "self.rank_info.effective_tp_rank" in CONNECTOR_SOURCE
    assert "self.model_config.effective_tp_size_per_node" in CONNECTOR_SOURCE


def test_swa_and_state_use_dedicated_registration_channel():
    assert "swa_layer_groups=swa_layer_groups" in CONNECTOR_SOURCE
    assert "swa_handles_per_group=swa_handles_per_group" in CONNECTOR_SOURCE
    assert "swa_slot_mappings=[swa_slot_mapping]" in CONNECTOR_SOURCE


def test_dsv4_pool_and_swa_mapping_guards():
    assert 'getattr(kvcache, "c4_kv_pool", None)' in CONNECTOR_SOURCE
    assert 'getattr(kvcache, "c128_kv_pool", None)' in CONNECTOR_SOURCE
    assert "if swa_indices.numel() == 0:" in CONNECTOR_SOURCE


def test_hybrid_cache_does_not_delegate_dunder_attributes():
    assert 'if name.startswith("__"):' in HYBRID_CACHE_SOURCE
