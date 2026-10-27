"""Tuned GEMM tables load from the env override dir, fall back per file to the
packaged tables, and pick the nearest token bucket; every shipped table loads."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe.base_gemm_provider import gemm_config_store
from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
    config_file_name,
    load_config_table,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

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


def _write(
    tmp_path: Path, payload, *, provider_key: str = "cutedsl_bf16_masked"
) -> Path:
    name = config_file_name(provider_key, device_name="MOCK GPU", **_GEOMETRY)
    (tmp_path / "base_gemm").mkdir(exist_ok=True)
    path = tmp_path / "base_gemm" / name
    path.write_text(json.dumps(payload))
    return path


def _load(tmp_path: Path):
    with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
        return load_config_table(
            "cutedsl_bf16_masked", device_name="MOCK GPU", **_GEOMETRY
        )


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
    # The nearest-M pick runs on the parsed int bucket keys.
    assert table.pick(1)["token_width"] == 8
    assert table.pick(8)["token_width"] == 8
    assert table.pick(13) is table.buckets[16]
    assert table.pick(90) is table.buckets[96]
    assert table.pick(150) is table.buckets[192]
    assert table.pick(100_000) is table.buckets[192]


def test_an_override_dir_without_the_file_inherits_the_packaged_one(
    tmp_path: Path, monkeypatch
) -> None:
    # The env-override dir is a tuner output dir carrying plans/tiles only;
    # without per-file fallback it silently disables every packaged table.
    package, override = tmp_path / "pkg", tmp_path / "override"
    override.mkdir()
    package.mkdir()
    _write(package, _VALID)
    monkeypatch.setattr(gemm_config_store, "_PACKAGE_CONFIG_DIR", str(package))
    table = _load(override)
    assert table is not None
    assert table.buckets[96] == {"token_width": 128}


def test_every_shipped_table_loads() -> None:
    shipped = sorted(
        (Path(gemm_config_store._PACKAGE_CONFIG_DIR) / "base_gemm").glob("*.json")
    )
    assert shipped
    pattern = re.compile(
        r"provider=(\w+),E=(\d+),N1=(\d+),N2=(\d+),K=(\d+),device_name=(.+)\.json"
    )
    with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(None):
        for path in shipped:
            key, experts, n1, n2, k, device = pattern.fullmatch(path.name).groups()
            table = load_config_table(
                key,
                num_local_experts=int(experts),
                n_gemm1=int(n1),
                n_gemm2=int(n2),
                k=int(k),
                device_name=device,
            )
            assert table is not None, path.name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
