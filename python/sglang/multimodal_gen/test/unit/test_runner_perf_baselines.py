import json
from dataclasses import replace

import pytest

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.test.server.test_server_utils import PerformanceValidator
from sglang.multimodal_gen.test.server.testcase_configs import (
    BaselineConfig,
    PerformanceSummary,
    get_perf_baseline_path,
)


@pytest.mark.parametrize(
    "runner",
    ["b200-fin03-4-4567", "unknown", ""],
)
def test_default_runner_baseline(monkeypatch, runner):
    monkeypatch.setenv("RUNNER_NAME", runner)
    config = BaselineConfig.load(get_perf_baseline_path("b200"))
    assert config.scenarios["flux2_modelopt_nvfp4_t2i"].expected_e2e_ms == 1990.84
    assert (
        config.scenarios["qwen_image_2512_modelopt_nvfp4_t2i"].expected_e2e_ms
        == 9650.06
    )


@pytest.mark.parametrize("phase", ["load", "runtime"])
def test_turbowan_host_budget_has_a_bounded_environment_allowance(monkeypatch, phase):
    monkeypatch.setattr(current_platform, "is_hip", lambda: False)
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    config = BaselineConfig.load(get_perf_baseline_path("5090"))
    scenario = config.scenarios["turbo_wan2_1_t2v_1.3b"]
    assert config.tolerances.host_anon == 0.02
    assert scenario.host_anon_tolerance is None
    validator = PerformanceValidator(scenario, config.tolerances, config.step_fractions)
    expected = (
        scenario.load_peak_host_anon_mb
        if phase == "load"
        else scenario.runtime_peak_host_anon_mb
    )
    limit = expected + max(expected * config.tolerances.host_anon, 256.0)
    summary = PerformanceSummary(0, 0, 0, {}, [], {}, {})
    summary.load_peak_host_anon_mb = scenario.load_peak_host_anon_mb
    summary.runtime_peak_host_anon_mb = scenario.runtime_peak_host_anon_mb
    if phase == "load":
        summary.load_peak_host_anon_mb = limit - 1
    else:
        summary.runtime_peak_host_anon_mb = limit - 1
    validator.validate_peak_host_anon(
        summary, scenario.load_peak_host_anon_mb, scenario.runtime_peak_host_anon_mb
    )
    if phase == "load":
        summary.load_peak_host_anon_mb = limit + 1
    else:
        summary.runtime_peak_host_anon_mb = limit + 1
    with pytest.raises(AssertionError, match="Peak Host Anon"):
        validator.validate_peak_host_anon(
            summary, scenario.load_peak_host_anon_mb, scenario.runtime_peak_host_anon_mb
        )


def test_consumer_host_budget_keeps_the_profile_tolerance(monkeypatch):
    monkeypatch.setattr(current_platform, "is_hip", lambda: False)
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    config = BaselineConfig.load(get_perf_baseline_path("5090"))
    scenario = config.scenarios["minimax_h3_t2va_consumer_budget_1gpu_5090"]
    assert scenario.host_anon_tolerance is None
    validator = PerformanceValidator(scenario, config.tolerances, config.step_fractions)
    summary = PerformanceSummary(0, 0, 0, {}, [], {}, {})
    summary.load_peak_host_anon_mb = scenario.load_peak_host_anon_mb * 1.02 + 1
    summary.runtime_peak_host_anon_mb = scenario.runtime_peak_host_anon_mb
    with pytest.raises(AssertionError, match="Load Peak Host Anon"):
        validator.validate_peak_host_anon(
            summary, scenario.load_peak_host_anon_mb, scenario.runtime_peak_host_anon_mb
        )


@pytest.mark.parametrize(
    "runner,qwen",
    [
        ("b200-di01-4567", 16126.87),
        ("b200-cirrascale1-0123", 17742.04),
        ("b200-cirrascale1-4567", 17742.04),
        ("b200-cirrascale2", 17346.65),
        ("b200-cirrascale3-0123", 17894.49),
        ("b200-cirrascale3-4567", 17346.65),
        ("b200-cirrascale4-0123", 17894.49),
        ("b200-cirrascale4-4567", 17346.65),
    ],
)
def test_runner_override_preserves_other_metrics(monkeypatch, runner, qwen):
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    default = BaselineConfig.load(get_perf_baseline_path("b200"))
    h100_default = BaselineConfig.load(get_perf_baseline_path("h100"))
    monkeypatch.setenv("RUNNER_NAME", runner)
    pool = BaselineConfig.load(get_perf_baseline_path("b200"))
    expected = {
        "qwen_image_2512_modelopt_nvfp4_t2i": qwen,
    }
    for name, scenario in default.scenarios.items():
        assert pool.scenarios[name] == replace(
            scenario, expected_e2e_ms=expected.get(name, scenario.expected_e2e_ms)
        )
    assert pool.tolerances == default.tolerances
    assert pool.step_fractions == default.step_fractions
    assert BaselineConfig.load(get_perf_baseline_path("h100")) == h100_default


@pytest.mark.parametrize(
    "runner",
    [
        "b200-di01-4567",
        "b200-fin03-4-4567",
        "b200-cirrascale1-0123",
        "b200-cirrascale1-4567",
        "b200-cirrascale2",
        "b200-cirrascale3-0123",
        "b200-cirrascale3-4567",
        "b200-cirrascale4-0123",
        "b200-cirrascale4-4567",
    ],
)
def test_runner_baseline_enforces_e2e_boundary(monkeypatch, runner):
    monkeypatch.setattr(current_platform, "is_hip", lambda: False)
    monkeypatch.setenv("RUNNER_NAME", runner)
    monkeypatch.setenv("SGLANG_GEN_BASELINE", "0")
    monkeypatch.delenv("SGLANG_E2E_TOLERANCE", raising=False)
    config = BaselineConfig.load(get_perf_baseline_path("b200"))
    for name in ("flux2_modelopt_nvfp4_t2i", "qwen_image_2512_modelopt_nvfp4_t2i"):
        scenario = config.scenarios[name]
        validator = PerformanceValidator(
            scenario, config.tolerances, config.step_fractions
        )
        limit = scenario.expected_e2e_ms * (1 + config.tolerances.e2e)
        validator.validate_e2e(PerformanceSummary(limit - 1, 0, 0, {}, [], {}, {}))
        with pytest.raises(AssertionError, match="E2E Latency"):
            validator.validate_e2e(PerformanceSummary(limit + 1, 0, 0, {}, [], {}, {}))


@pytest.mark.parametrize(
    "runner,prefix",
    [
        ("h100-computable-d-gpu-0", "h100-computable-d-"),
        ("h100-computable-d-gpu-7", "h100-computable-d-"),
        ("h100-computable-e-gpu-0", "h100-computable-e-"),
        ("h100-computable-e-gpu-7", "h100-computable-e-"),
    ],
)
def test_h100_runner_override_only_raises_listed_metrics(monkeypatch, runner, prefix):
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    path = get_perf_baseline_path("h100")
    default = BaselineConfig.load(path)
    with path.open(encoding="utf-8") as fh:
        overrides = json.load(fh)["runner_overrides"][prefix]
    monkeypatch.setenv("RUNNER_NAME", runner)
    pool = BaselineConfig.load(path)
    assert overrides
    for name, scenario in default.scenarios.items():
        expected = {}
        for key, value in overrides.get(name, {}).items():
            if isinstance(value, dict):
                # per-stage/per-step maps merge; unlisted entries keep their default
                default_map = getattr(scenario, key)
                cast = int if key == "denoise_step_ms" else str
                merged = dict(default_map)
                for sub, sub_value in value.items():
                    assert sub_value > default_map[cast(sub)], (name, key, sub)
                    merged[cast(sub)] = sub_value
                expected[key] = merged
            else:
                assert value > getattr(scenario, key), (name, key)
                expected[key] = value
        assert pool.scenarios[name] == replace(scenario, **expected)
    assert pool.tolerances == default.tolerances
    assert pool.step_fractions == default.step_fractions


@pytest.mark.parametrize(
    "runner", ["h100-novita-host1-gpu-0", "h100-computable-dx-gpu-0", "unknown", ""]
)
def test_h100_default_runner_baseline(monkeypatch, runner):
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    default = BaselineConfig.load(get_perf_baseline_path("h100"))
    monkeypatch.setenv("RUNNER_NAME", runner)
    assert BaselineConfig.load(get_perf_baseline_path("h100")) == default
