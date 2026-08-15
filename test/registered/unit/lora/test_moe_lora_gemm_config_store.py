"""CPU coverage for the MoE LoRA base-GEMM M-bucketed config store.

The load contract is fail-closed: any missing, malformed, or
version-mismatched table returns ``None`` and the providers keep their
built-in heuristics — behavior without config files must be byte-identical
to a build that has none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
    GemmConfigTable,
    config_file_name,
    load_config_table,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-c-test-cpu")

_GEOMETRY = dict(num_local_experts=32, n_gemm1=1536, n_gemm2=7168, k=7168)
_VALID = {
    "version": {"cutedsl": "4.2.0", "generated_on": "GB300-152SM"},
    "tiles": [
        {"token_width": 64, "persistent_clusters": 128},
        {"token_width": 128, "persistent_clusters": 152},
    ],
    "buckets": {
        "8": {"token_width": 8},
        "16": {"token_width": 64},
        "96": {"token_width": 128},
        "192": {"token_width": 64},
    },
}


def _write(tmp_path: Path, payload, *, provider_key: str = "cutedsl_bf16") -> Path:
    name = config_file_name(provider_key, device_name="MOCK GPU", **_GEOMETRY)
    (tmp_path / "base_gemm").mkdir(exist_ok=True)
    path = tmp_path / "base_gemm" / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def _load(tmp_path: Path, **overrides):
    kwargs = dict(_GEOMETRY, device_name="MOCK GPU", **overrides)
    with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
        return load_config_table("cutedsl_bf16", **kwargs)


def test_config_file_name_encodes_geometry_and_device() -> None:
    assert config_file_name(
        "deepgemm_bf16", device_name="NVIDIA GB300", **_GEOMETRY
    ) == (
        "provider=deepgemm_bf16,E=32,N1=1536,N2=7168,K=7168,"
        "device_name=NVIDIA_GB300,dtype=bf16.json"
    )


def test_missing_file_falls_back_to_none(tmp_path: Path) -> None:
    assert _load(tmp_path) is None


def test_valid_table_loads_via_env_override(tmp_path: Path) -> None:
    _write(tmp_path, _VALID)
    table = _load(tmp_path)
    assert table is not None
    assert table.buckets[96] == {"token_width": 128}
    assert tuple((t.token_width, t.persistent_clusters) for t in table.tiles) == (
        (64, 128),
        (128, 152),
    )
    assert table.version["generated_on"] == "GB300-152SM"


def test_nearest_m_pick() -> None:
    table = GemmConfigTable(
        buckets={
            8: {"token_width": 8},
            16: {"token_width": 64},
            96: {"token_width": 128},
            192: {"token_width": 64},
        }
    )
    assert table.pick(1)["token_width"] == 8
    assert table.pick(8)["token_width"] == 8
    # 13 snaps to bucket 16, 150 to bucket 192: assertions check the chosen
    # bucket's payload, not the bucket key.
    assert table.pick(13) is table.buckets[16]
    assert table.pick(90) is table.buckets[96]
    assert table.pick(150) is table.buckets[192]
    assert table.pick(100_000) is table.buckets[192]


@pytest.mark.parametrize(
    "payload",
    (
        "{not json",
        {"buckets": {}},
        {"buckets": {"eight": {"token_width": 8}}},
        {"buckets": {"8": {"token_width": "wide"}}},
        {"buckets": {"8": {}}},
        {"buckets": {"-8": {"token_width": 8}}},
        {"buckets": {"8": {"token_width": -8}}},
        {
            "tiles": [
                {"token_width": 64, "persistent_clusters": 128},
                {"token_width": 64, "persistent_clusters": 152},
            ],
            "buckets": {"8": {"token_width": 64}},
        },
        {"tiles": [{"token_width": 64}], "buckets": {"8": {"token_width": 64}}},
    ),
    ids=(
        "invalid-json",
        "empty-buckets",
        "non-int-bucket-key",
        "non-int-payload",
        "empty-payload",
        "negative-bucket-key",
        "negative-payload",
        "duplicate-tile-width",
        "tile-missing-field",
    ),
)
def test_malformed_files_are_rejected(tmp_path: Path, payload) -> None:
    _write(tmp_path, payload)
    assert _load(tmp_path) is None


def test_version_mismatch_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, _VALID)
    assert _load(tmp_path, expected_versions={"cutedsl": "9.9.9"}) is None


def test_matching_and_absent_versions_are_accepted(tmp_path: Path) -> None:
    _write(tmp_path, _VALID)
    assert _load(tmp_path, expected_versions={"cutedsl": "4.2.0"}) is not None
    # Keys the file does not record are provenance-only, never a rejection.
    stripped = dict(_VALID, version={"generated_on": "GB300-152SM"})
    other = tmp_path / "stripped"
    other.mkdir()
    _write(other, stripped)
    assert _load(other, expected_versions={"cutedsl": "4.2.0"}) is not None
