"""Mode functional smoke tests: five real-SDK estimator configurations.

Every case runs the actual AIS SDK (no estimator or mode-selection mocks):
the selected mode is read back through the SDK diagnostics and each phase
predicts a finite positive latency wherever the supplied data supports it.
"""

import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo
from sglang_simulator.time_predictor.aisimulate import (
    AISimulateTimePredictor,
    batch_to_fpm_metrics,
)
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from test_simulation_sglang_serving import ASSETS, SIM_CONFIGS

MODEL_ASSET = str(ASSETS / "qwen3-8b")
_SOL_CONFIG = json.loads(SIM_CONFIGS["aisimulate_sol"].read_text())


def _model() -> ModelInfo:
    return ModelInfo(model_path=MODEL_ASSET)


def _hw():
    # Resolve the accelerator through the factory, like the serving tests.
    with patch.object(ConfigManager, "_raw_config", _SOL_CONFIG):
        return ConfigManager.get_accelerator_info()


def _sched_config() -> SchedulerConfig:
    return SchedulerConfig(
        data_type=DataType.BF16,
        kv_cache_data_type=DataType.BF16,
        **_SOL_CONFIG["scheduler"],
    )


def _predictor(config=None, **kwargs) -> AISimulateTimePredictor:
    return AISimulateTimePredictor(_model(), _hw(), config or _sched_config(), **kwargs)


def batch(n=128, kv=0, mode="extend"):
    return ScheduleBatch([ScheduleRequest(n, kv, mode)], mode)


@pytest.fixture
def sol():
    """Predictors registered through this fixture are closed after the test."""
    made = []

    def make(predictor):
        made.append(predictor)
        return predictor

    yield make
    for predictor in made:
        predictor._fpm.close()


def _assert_mode(predictor, mode, correction_enabled=None):
    provenance = predictor.get_metrics()["provenance"]
    assert provenance["selected_estimation_mode"] == mode
    if correction_enabled is not None:
        correction = (provenance["config"].get("estimator_config") or {}).get(
            "correction", {}
        )
        assert correction.get("enabled") is correction_enabled


def _assert_positive(predictor, *batches):
    for b in batches:
        estimate = predictor.predict_infer_time(b)
        assert math.isfinite(estimate) and estimate > 0


# --- synthetic observations consumed through tune_observations -----------------


def _calibration_rows():
    """Six deterministic pure-prefill and six pure-decode iterations in the
    SDK's native observations format, covering each phase beyond the SDK's
    per-bucket minimum. Mixed iterations are covered by the runner test."""
    rows = []
    for i, n in enumerate((128, 256, 512, 1024, 2048, 4096)):
        rows.append(
            [dict(batch_to_fpm_metrics(batch(n)), wall_time=0.002 + i * 0.0011)]
        )
    for i, kv in enumerate((255, 511, 1023, 2047, 4095, 8191)):
        rows.append(
            [
                dict(
                    batch_to_fpm_metrics(batch(1, kv, "decode")),
                    wall_time=0.001 + i * 0.0007,
                )
            ]
        )
    return rows


def _write_observations(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_calibration_rows()))
    return path


def _factory_predictor(tmp_path, monkeypatch, predictor_config):
    """Build a predictor through the config factory from a config file in
    tmp_path; a relative tune_observations path resolves against that file."""
    config = {"predictor": predictor_config, "scheduler": _SOL_CONFIG["scheduler"]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(path))
    ConfigManager.reset_config_cache()
    predictor = ConfigManager.get_inference_time_predictor(
        _model(), _hw(), _sched_config()
    )
    ConfigManager.reset_config_cache()
    return predictor


# --- the five mode configurations ----------------------------------------------


_SOL_CASES = [
    pytest.param(
        # estimation_mode is omitted: the default engine is what is tested.
        {"name": "aisimulate", "database_mode": "SOL"},
        "op_level",
        False,
        id="default",
    ),
    pytest.param(
        {
            "name": "aisimulate",
            "estimation_mode": "op_level",
            "database_mode": "SOL",
            "estimator_config": {"correction": {"enabled": True}},
            "tune_observations": "calibration/TP0.ais_observations.json",
        },
        "op_level",
        True,
        id="corrected",
    ),
    pytest.param(
        {
            "name": "aisimulate",
            "estimation_mode": "fpm_regression",
            "database_mode": "SOL",
            "tune_observations": "calibration/TP0.ais_observations.json",
        },
        "fpm_regression",
        False,
        id="regression",
    ),
    pytest.param(
        {"name": "aisimulate", "estimation_mode": "auto", "database_mode": "SOL"},
        "op_level",
        False,
        id="auto",
    ),
]


@pytest.mark.parametrize("predictor_config,mode,correction_enabled", _SOL_CASES)
def test_sol_mode_configurations(
    sol, tmp_path, monkeypatch, predictor_config, mode, correction_enabled
):
    """The four SOL configurations load through the config factory, consume
    the relative observation file when one is declared, and estimate every
    phase the data covers. The diagnostics prove the file was actually read:
    the SDK retains observations and reports ready correction buckets."""
    _write_observations(tmp_path / "calibration" / "TP0.ais_observations.json")
    predictor = sol(_factory_predictor(tmp_path, monkeypatch, predictor_config))
    _assert_mode(predictor, mode, correction_enabled)
    metrics = predictor.get_metrics()
    assert metrics["readiness"] == "ready"
    if "tune_observations" in predictor_config:
        assert metrics["retained_observations"] > 0
    if correction_enabled:
        assert metrics["correction_ready_buckets"] > 0
    _assert_positive(predictor, batch(512), batch(1, 1023, "decode"))


# --- fpm_interpolation over a synthetic real-schema profile ---------------------


SYNTHETIC_FPM_VERSION = "0.13.0-synthetic"
# aisimulate 0.13.0 pins this model-config digest for DeepSeek-V4.1-Flash in
# its own shipped manifests (aisimulate_core/systems/dsv41_fpm_hf.json); the
# FPM loader requires it on every parquet row.
_DSV41_MODEL_CONFIG_SHA256 = (
    "22c3912140adeb60ecbd3c7a9b54e62997adf65ad4c69440f8082cf29f5ef0d4"
)
_SYNTH_PREFILL_TOKENS = (128, 512, 2048, 4096)
_SYNTH_DECODE_KV_TOKENS = (512, 2048, 8192)


def _synthetic_prefill_latency_ms(tokens):
    return 1.0 + 0.004 * tokens + 1.2e-6 * tokens * tokens


def _synthetic_decode_latency_ms(kv_tokens):
    return 0.3 + 1.5e-4 * kv_tokens


def _write_synthetic_fpm_profile(root):
    """Generate a minimal SDK v7 profile with synthetic latencies, not GPU GT."""
    import aisimulate_core
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = root / "data/b300_sxm/sglang" / SYNTHETIC_FPM_VERSION
    data_dir.mkdir(parents=True)
    shutil.copy2(
        Path(aisimulate_core.__file__).parent / "systems" / "b300_sxm.yaml",
        root / "b300_sxm.yaml",
    )
    identity = {
        "model_path": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system": "b300_sxm",
        "backend": "sglang",
        "backend_version": SYNTHETIC_FPM_VERSION,
        "weight_quantization": "fp8_block",
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "w4a8_mxfp4_mxfp8_trtllm",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "tp": 2,
        "moe_tp": 2,
        "pp": 1,
        "dp": 1,
        "moe_ep": 1,
        "cp": 1,
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
        "model_config_sha256": _DSV41_MODEL_CONFIG_SHA256,
        "execution_profile": "full",
        "engram_residency": "hbm_tp_sharded",
        "input_modality": "text",
    }

    def row(workload, batch_size, prefill_tokens, kv_tokens, latency_ms):
        return {
            **identity,
            "cell_id": hashlib.sha256(
                f"{workload}-{batch_size}-{prefill_tokens}-{kv_tokens}".encode()
            ).hexdigest(),
            "workload_kind": workload,
            "batch_size": batch_size,
            "total_prefill_tokens": prefill_tokens,
            "total_kv_read_tokens": kv_tokens,
            "partition_policy": "balanced_v1",
            "kv_seed_regime": "n/a" if workload == "prefill" else "real_kv",
            "latency_ms": latency_ms,
        }

    rows = [
        row("prefill", 1, tokens, 0, _synthetic_prefill_latency_ms(tokens))
        for tokens in _SYNTH_PREFILL_TOKENS
    ] + [
        row("decode", 1, 0, kv, _synthetic_decode_latency_ms(kv))
        for kv in _SYNTH_DECODE_KV_TOKENS
    ]
    parquet_path = data_dir / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    # measurement_policy is a loader-validated enum; the free-form note keeps
    # the file from being mistaken for collected measurements.
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 7,
        "admission": "serving",
        "backend": "sglang",
        "backend_version": SYNTHETIC_FPM_VERSION,
        "system": "b300_sxm",
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "provenance_note": (
            "synthetic test fixture: analytic latency formulas for CI coverage; "
            "not measurement ground truth"
        ),
        "configuration_selector": dict(identity),
        "row_count": len(rows),
        "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    }
    (data_dir / "fpm_forward_perf.metadata.json").write_text(json.dumps(metadata))
    return root


def _interpolation_predictor(root):
    """Adapter predictor identity-matched to the synthetic profile root."""
    cfg = _sched_config()
    cfg.tp_size = 2
    cfg.backend_version = SYNTHETIC_FPM_VERSION
    cfg.data_type = DataType.FP8
    cfg.kv_cache_data_type = DataType.FP8
    cfg.moe_quant_mode_override = "w4a8_mxfp4_mxfp8_trtllm"
    cfg.comm_quant_mode_override = "half"
    return AISimulateTimePredictor(
        NS(model_path="deepseek-ai/DeepSeek-V4.1-Flash"),
        NS(name="b300_sxm"),
        cfg,
        database_mode="SOL",
        estimation_mode="fpm_interpolation",
        systems_paths=[str(root)],
    )


def test_interpolation_synthetic_profile(sol, tmp_path):
    """One valid interpolation over a real-schema synthetic profile: a real
    SDK load and query, no external files or downloads needed."""
    root = _write_synthetic_fpm_profile(tmp_path / "synthetic_fpm")
    predictor = sol(_interpolation_predictor(root))
    _assert_mode(predictor, "fpm_interpolation")
    _assert_positive(predictor, batch(512), batch(1, 1023, "decode"))
