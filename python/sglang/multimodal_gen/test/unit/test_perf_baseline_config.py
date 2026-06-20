import json

import pytest

from sglang.multimodal_gen.test.server.testcase_configs import (
    BaselineConfig,
    ToleranceConfig,
)


def test_scenario_tolerance_overrides_are_loaded(tmp_path):
    baseline_path = tmp_path / "perf_baselines.json"
    baseline_path.write_text(
        json.dumps(
            {
                "tolerances": {
                    "pr_test": {
                        "e2e": 0.25,
                        "denoise_stage": 0.25,
                        "non_denoise_stage": 0.8,
                        "denoise_step": 0.3,
                        "denoise_agg": 0.2,
                    }
                },
                "sampling": {"step_fractions": [0.0, 1.0]},
                "scenarios": {
                    "layerwise_offload": {
                        "stages_ms": {"DenoisingStage": 1478.72},
                        "denoise_step_ms": {"6": 166.38},
                        "expected_e2e_ms": 1976.64,
                        "expected_avg_denoise_ms": 163.76,
                        "expected_median_denoise_ms": 188.61,
                        "estimated_full_test_time_s": 122.0,
                        "tolerance_overrides": {
                            "e2e": 0.5,
                            "denoise_stage": 0.5,
                            "denoise_step": 0.5,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = BaselineConfig.load(baseline_path)
    scenario = config.scenarios["layerwise_offload"]

    assert scenario.tolerance_overrides == {
        "e2e": 0.5,
        "denoise_stage": 0.5,
        "denoise_step": 0.5,
    }
    assert config.tolerances.e2e == 0.25

    overridden = config.tolerances.with_overrides(scenario.tolerance_overrides)

    assert overridden.e2e == 0.5
    assert overridden.denoise_stage == 0.5
    assert overridden.denoise_step == 0.5
    assert overridden.non_denoise_stage == config.tolerances.non_denoise_stage
    assert config.tolerances.e2e == 0.25


def test_tolerance_overrides_reject_unknown_fields():
    tolerances = ToleranceConfig(
        e2e=0.25,
        denoise_stage=0.25,
        non_denoise_stage=0.8,
        denoise_step=0.3,
        denoise_agg=0.2,
    )

    with pytest.raises(ValueError, match="Unknown tolerance override"):
        tolerances.with_overrides({"unknown": 0.5})
