"""Host bridge tests using the actual SDK; tuning data are synthetic API fixtures."""

import ast
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from sglang_simulator.simulation.sglang.batch_snapshot import capture_batch_entry
from sglang_simulator.simulation.sglang.utils import resolve_scheduler_config
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor.aisimulate import (
    AISimulateTimePredictor,
    batch_to_fpm_metrics,
)
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from test_aisimulate_predictor import _hw, _model, _sched_config


def batch(n=128, kv=0, mode="extend"):
    return ScheduleBatch([ScheduleRequest(n, kv, mode)], mode)


def direct_config(mode, correction=False):
    return dict(
        model=_model().model_path,
        system="a100_sxm",
        backend="sglang",
        backend_version="0.5.10",
        tp=1,
        worker_type="aggregated",
        database_mode="SOL",
        estimation_mode=mode,
        fallback_policy="deny",
        gemm_quant_mode="bfloat16",
        moe_quant_mode="bfloat16",
        fmha_quant_mode="bfloat16",
        kvcache_quant_mode="bfloat16",
        comm_quant_mode="half",
        estimator_config={}
        if mode == "fpm_regression"
        else {"correction": {"enabled": correction}},
    )


@pytest.mark.parametrize(
    "mode,corrected",
    [
        ("op_level", False),
        ("op_level", True),
        ("fpm_regression", False),
        ("auto", False),
    ],
)
def test_native_parity_and_observation_replay(tmp_path, mode, corrected):
    config = direct_config(mode, corrected)
    direct = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(**config)
    )
    observations = None
    if corrected:
        q = batch_to_fpm_metrics(batch())
        raw = direct.estimate_forward_pass_time_ms(q)
        rows = [[dict(q, wall_time=raw / 1000 * 1.25)] for _ in range(6)]
    elif mode == "fpm_regression":
        assert (
            direct.estimate_forward_pass_time_ms(batch_to_fpm_metrics(batch())) is None
        )
        rows = [
            [dict(batch_to_fpm_metrics(batch(n)), wall_time=0.002 + i * 0.0011)]
            for i, n in enumerate([128, 192, 256, 384, 512, 768, 1024, 1536, 2048])
        ]
    else:
        rows = None
    if rows is not None:
        direct.tune_with_fpms(rows)
        observations = tmp_path / "iterations.json"
        observations.write_text(json.dumps(rows))
    predictor = AISimulateTimePredictor(
        _model(),
        _hw(),
        _sched_config(),
        database_mode="SOL",
        estimation_mode=mode,
        estimator_config=config["estimator_config"],
        tune_observations=observations,
    )
    try:
        for n in [128, 256, 512]:
            assert predictor.predict_infer_time(batch(n)) == pytest.approx(
                direct.estimate_forward_pass_time_ms(batch_to_fpm_metrics(batch(n)))
                / 1000
            )
        if corrected:
            assert predictor.predict_infer_time(batch()) == pytest.approx(
                raw / 1000 * 1.25, rel=0.01
            )
        assert (
            predictor.get_metrics()["provenance"]["selected_estimation_mode"]
            == direct.diagnostics()["provenance"]["selected_estimation_mode"]
        )
    finally:
        direct.close()
        predictor._fpm.close()


def test_unready_and_legacy_options_are_not_silently_ignored():
    predictor = AISimulateTimePredictor(
        _model(),
        _hw(),
        _sched_config(),
        database_mode="SOL",
        estimation_mode="fpm_regression",
    )
    try:
        with pytest.raises(RuntimeError, match="no ready estimate"):
            predictor.predict_infer_time(batch())
    finally:
        predictor._fpm.close()
    with pytest.raises(ValueError, match="legacy scaling"):
        AISimulateTimePredictor(
            _model(),
            _hw(),
            _sched_config(),
            estimation_mode="op_level",
            prefill_scale_factor=2,
        )
    with pytest.raises(ValueError, match="Set estimation_mode"):
        AISimulateTimePredictor(_model(), _hw(), _sched_config(), estimator_config={})


def test_snapshot_prefix_mixed_and_single_token():
    p, d = NS(prefix_len=2047, extend_len=1), NS(prefix_len=0, extend_len=99)
    entry = NS(
        reqs=[p, d],
        forward_mode=NS(name="MIXED"),
        decoding_reqs=[d],
        seq_lens_cpu=[2048, 513],
        prefix_lens=[2047, 512],
        extend_lens=[1, 1],
    )
    result = batch_to_fpm_metrics(capture_batch_entry(entry))["scheduled_requests"]
    assert result == dict(
        num_prefill_requests=1,
        sum_prefill_tokens=1,
        sum_prefill_kv_tokens=2047,
        var_prefill_length=0.0,
        num_decode_requests=1,
        sum_decode_kv_tokens=512,
        var_decode_kv_tokens=0.0,
    )
    entry.forward_mode = NS(name="EXTEND")
    entry.reqs = [p]
    entry.prefix_lens = [2047]
    entry.extend_lens = [1]
    assert not capture_batch_entry(entry).is_decode()


def _actual_wrapper(original, predictor):
    # Compile the actual closure body, with its external state injected.
    # Full scheduler/runner behavior is covered by test_aisimulate_native_runner.
    source = (
        Path(__file__).parents[1]
        / "src/sglang_simulator/simulation/sglang/scheduler.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "wrapped_run_batch"
    )
    hook = NS(
        INFERENCE_PREDICTOR=predictor,
        TOTAL_PREDICTOR_TIME_COST=0,
        SIM_MODE=SimulationMode.OFFLINE,
    )
    state = NS(inc_iteration=Mock(), set_current_inference_dur=Mock())
    env = dict(
        original_run_batch=original,
        get_obj_from_args=lambda name, *args, **kw: args[0],
        capture_batch_entry=capture_batch_entry,
        C_SchedulerHook=hook,
        StateManager=state,
        SimulationMode=SimulationMode,
        time=time,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), env
    )
    return env["wrapped_run_batch"], hook


@pytest.mark.parametrize("mode", [None, "TARGET_VERIFY"])
def test_actual_wrapper_refuses_before_bound(mode):
    bound = Mock()
    fn, _ = _actual_wrapper(bound, NS(requires_batch_snapshot=True))
    with pytest.raises(ValueError, match="forward mode"):
        fn(None, NS(reqs=[NS()], forward_mode=NS(name=mode) if mode else None))
    bound.assert_not_called()


def test_actual_wrapper_uses_entry_state_after_bound_mutation():
    entry = NS(reqs=[NS(extend_len=1, prefix_len=2047)], forward_mode=NS(name="EXTEND"))

    def bound(self, b):
        b.reqs.clear()
        b.forward_mode = None
        return type("GenerationBatchResult", (), {})()

    predict = Mock(return_value=0.001)
    fn, hook = _actual_wrapper(
        bound, NS(requires_batch_snapshot=True, predict_infer_time=predict)
    )
    fn(None, entry)
    captured = predict.call_args.args[0]
    assert (
        batch_to_fpm_metrics(captured)["scheduled_requests"]["sum_prefill_kv_tokens"]
        == 2047
    )
    assert hook.SIMULATION_BATCH is captured


def test_effective_topology_cannot_be_hidden_by_config_override():
    args = NS(resolved_dict=lambda: {"attn_dp_size": 2})
    with pytest.raises(ValueError, match="attn_dp_size"):
        resolve_scheduler_config(
            args, NS(), native=True, scheduler_overrides={"dp_size": 1}
        )


def test_interpolation_matches_native_and_preserves_domain_error():
    root = os.environ.get("SGLANG_SIMULATOR_TEST_FPM_SYSTEMS_ROOT")
    if not root:
        pytest.skip(
            "Set SGLANG_SIMULATOR_TEST_FPM_SYSTEMS_ROOT to the official B300 TP2 profile"
        )
    cfg = _sched_config()
    cfg.tp_size = 2
    cfg.backend_version = "dev-c4ca651192e5-tp267c0e788b4b2"
    from sglang_simulator.spec import DataType

    cfg.data_type = DataType.FP8
    cfg.kv_cache_data_type = DataType.FP8
    cfg.moe_quant_mode_override = "w4a8_mxfp4_mxfp8_trtllm"
    cfg.comm_quant_mode_override = "half"
    model = NS(model_path="deepseek-ai/DeepSeek-V4.1-Flash")
    hw = NS(name="b300_sxm")
    options = dict(
        model=model.model_path,
        system=hw.name,
        backend="sglang",
        backend_version=cfg.backend_version,
        tp=2,
        moe_tp_size=2,
        moe_ep_size=1,
        worker_type="aggregated",
        database_mode="SOL",
        estimation_mode="fpm_interpolation",
        systems_paths=(root,),
        fallback_policy="deny",
        gemm_quant_mode="fp8_block",
        moe_quant_mode=cfg.moe_quant_mode_override,
        fmha_quant_mode="fp8",
        kvcache_quant_mode="fp8",
        comm_quant_mode="half",
        estimator_config={"correction": {"enabled": False}},
    )
    direct = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(**options)
    )
    predictor = AISimulateTimePredictor(
        model,
        hw,
        cfg,
        database_mode="SOL",
        estimation_mode="fpm_interpolation",
        systems_paths=[root],
        estimator_config=options["estimator_config"],
    )
    try:
        assert predictor.predict_infer_time(batch()) == pytest.approx(
            direct.estimate_forward_pass_time_ms(batch_to_fpm_metrics(batch())) / 1000
        )
        with pytest.raises(PerfDataNotAvailableError):
            predictor.predict_infer_time(batch(4000))
    finally:
        direct.close()
        predictor._fpm.close()


def test_native_factory_resolves_calibration_path(tmp_path, monkeypatch):
    from sglang_simulator.simulation.manager.config import ConfigManager

    rows = [
        [dict(batch_to_fpm_metrics(batch(n)), wall_time=0.002 + i * 0.0011)]
        for i, n in enumerate([128, 192, 256, 384, 512, 768, 1024, 1536, 2048])
    ]
    (tmp_path / "observations.json").write_text(json.dumps(rows))
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "predictor": {
                    "name": "aisimulate",
                    "estimation_mode": "fpm_regression",
                    "database_mode": "SOL",
                    "tune_observations": "observations.json",
                }
            }
        )
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(path))
    ConfigManager.reset_config_cache()
    predictor = ConfigManager.get_inference_time_predictor(
        _model(), _hw(), _sched_config()
    )
    try:
        assert isinstance(predictor, AISimulateTimePredictor)
        assert predictor.predict_infer_time(batch(512)) > 0
    finally:
        predictor._fpm.close()
        ConfigManager.reset_config_cache()


def test_native_does_not_inherit_inexact_legacy_fp16_mapping():
    from sglang_simulator.spec import DataType

    config = _sched_config()
    config.data_type = DataType.FP16
    with pytest.raises(ValueError, match="faithful FP16"):
        AISimulateTimePredictor(_model(), _hw(), config, estimation_mode="op_level")
