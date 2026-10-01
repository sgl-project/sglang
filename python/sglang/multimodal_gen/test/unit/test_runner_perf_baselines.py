from dataclasses import replace

import pytest

from sglang.multimodal_gen.test.server.test_server_utils import PerformanceValidator
from sglang.multimodal_gen.test.server.testcase_configs import (
    BaselineConfig,
    PerformanceSummary,
    get_perf_baseline_path,
)


@pytest.mark.parametrize(
    "runner",
    ["b200-fin03-4-4567", "b200-cirrascale2", "b200-cirrascale4-0123", "unknown", ""],
)
def test_default_runner_baseline(monkeypatch, runner):
    monkeypatch.setenv("RUNNER_NAME", runner)
    config = BaselineConfig.load(get_perf_baseline_path("b200"))
    assert config.scenarios["flux1_modelopt_nvfp4_t2i"].expected_e2e_ms == 836.71
    assert (
        config.scenarios["qwen_image_2512_modelopt_nvfp4_t2i"].expected_e2e_ms
        == 9650.06
    )


@pytest.mark.parametrize(
    "runner,flux,qwen",
    [
        ("b200-di01-4567", 1334.16, 16126.87),
        ("b200-cirrascale1-0123", 1574.32, 17742.04),
        ("b200-cirrascale3-0123", 1470.24, 17894.49),
        ("b200-cirrascale3-4567", 1471.19, 17346.65),
    ],
)
def test_runner_override_preserves_other_metrics(monkeypatch, runner, flux, qwen):
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    default = BaselineConfig.load(get_perf_baseline_path("b200"))
    h100_default = BaselineConfig.load(get_perf_baseline_path("h100"))
    monkeypatch.setenv("RUNNER_NAME", runner)
    pool = BaselineConfig.load(get_perf_baseline_path("b200"))
    expected = {
        "flux1_modelopt_nvfp4_t2i": flux,
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
        "b200-cirrascale3-0123",
        "b200-cirrascale3-4567",
    ],
)
def test_runner_baseline_enforces_e2e_boundary(monkeypatch, runner):
    monkeypatch.setenv("RUNNER_NAME", runner)
    monkeypatch.setenv("SGLANG_GEN_BASELINE", "0")
    monkeypatch.delenv("SGLANG_E2E_TOLERANCE", raising=False)
    config = BaselineConfig.load(get_perf_baseline_path("b200"))
    for name in ("flux1_modelopt_nvfp4_t2i", "qwen_image_2512_modelopt_nvfp4_t2i"):
        scenario = config.scenarios[name]
        validator = PerformanceValidator(
            scenario, config.tolerances, config.step_fractions
        )
        limit = scenario.expected_e2e_ms * (1 + config.tolerances.e2e)
        validator.validate_e2e(PerformanceSummary(limit - 1, 0, 0, {}, [], {}, {}))
        with pytest.raises(AssertionError, match="E2E Latency"):
            validator.validate_e2e(PerformanceSummary(limit + 1, 0, 0, {}, [], {}, {}))
