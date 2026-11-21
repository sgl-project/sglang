"""XPU perf-baseline selection: each Arc Pro SKU must load its own baseline file."""

import json

import pytest

from sglang.multimodal_gen.test.server import testcase_configs as cfg

GIB = 1024**3


@pytest.mark.parametrize(
    "device_name, total_gib, expected",
    [
        ("Intel(R) Arc(TM) Pro B70 Graphics", 32, "xpu_b70"),
        ("Intel(R) Arc(TM) Pro B60 Graphics", 24, "xpu_b60"),
        ("Intel(R) Graphics [0xe223]", 32, "xpu_b70"),
        ("Intel(R) Graphics [0xe211]", 24, "xpu_b60"),
    ],
)
def test_xpu_device_selects_baseline(monkeypatch, device_name, total_gib, expected):
    monkeypatch.delenv(cfg.PERF_BASELINE_PLATFORM_ENV, raising=False)
    monkeypatch.setattr(cfg.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(cfg.current_platform, "get_device_name", lambda *a: device_name)
    monkeypatch.setattr(
        cfg.current_platform, "get_device_total_memory", lambda *a: total_gib * GIB
    )
    assert cfg.get_perf_baseline_platform() == expected


@pytest.mark.parametrize(
    "override, expected",
    [
        ("xpu_b70", "xpu_b70"),
        ("B70", "xpu_b70"),
        ("xpu_b60", "xpu_b60"),
        ("bmg", "xpu_b60"),
    ],
)
def test_env_override_wins(monkeypatch, override, expected):
    monkeypatch.setenv(cfg.PERF_BASELINE_PLATFORM_ENV, override)
    monkeypatch.setattr(cfg.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(
        cfg.current_platform,
        "get_device_name",
        lambda *a: "Intel(R) Arc(TM) Pro B60 Graphics",
    )
    assert cfg.get_perf_baseline_platform() == expected


def test_xpu_baselines_cover_same_scenarios():
    b60 = json.loads(cfg.get_perf_baseline_path("xpu_b60").read_text())["scenarios"]
    b70 = json.loads(cfg.get_perf_baseline_path("xpu_b70").read_text())["scenarios"]
    assert set(b70) == set(b60)
    for name, sc in b70.items():
        assert cfg.ScenarioConfig.from_dict(sc).expected_e2e_ms > 0, name
