"""Real-SDK predictor tests for the AISimulate time predictor.

These tests run against the installed AISimulate SDK and cover:

- predictor construction and prefill/decode estimation through the SOL and
  SILICON performance databases;
- a prefix-cache representative scenario: a cached prefill reusing KV must be
  faster than the cold prefill of the same total length;
- the SGLang-side regression surface: the uneven-batch correction coefficient,
  the 0.4 threshold that decides when the coefficient reaches AISimulate, the
  prefill/decode phase scale factors, the prefill minimum-latency floor, and
  the millisecond-to-second unit boundary. These compare the adapter against
  the real SDK with fixed RuntimeConfig inputs instead of re-implementing the
  production formula;
- the predictor configuration registry: the ``aisimulate`` predictor name
  resolves through ``ConfigManager``, the legacy ``aiconfigurator`` name
  fails with a migration error, and unknown names list the supported set.

The serving, offline-blocking, and cache-hit-ratio tests exercise the same
predictor end to end through the simulator server.
"""

import json
from pathlib import Path

import pytest
from aisimulate.sdk.config import RuntimeConfig
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import AcceleratorInfo, DataType, ModelInfo
from sglang_simulator.time_predictor import AISimulateTimePredictor
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest

MODEL_ASSET = str(Path(__file__).parent / "assets" / "qwen3-8b")

# The backend slot the AISimulate performance databases publish for the
# a100_sxm/sglang system; matches examples/sim_configs/aisimulate_*.json.
AISIMULATE_BACKEND_VERSION = "0.5.10"

ISL = 1024
PREFIX_LEN = 896
EXTEND_LEN = ISL - PREFIX_LEN


def _model() -> ModelInfo:
    return ModelInfo(
        model_path=MODEL_ASSET,
        attention_arch="MHA",
        context_len=40960,
        hidden_size=4096,
        head_dim=128,
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
        vocab_size=151936,
        torch_dtype="bfloat16",
    )


def _hw() -> AcceleratorInfo:
    return AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
    )


def _sched_config() -> SchedulerConfig:
    return SchedulerConfig(
        data_type=DataType.BF16,
        kv_cache_data_type=DataType.BF16,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        backend_name="sglang",
        backend_version=AISIMULATE_BACKEND_VERSION,
    )


def _make_predictor(database_mode: str) -> AISimulateTimePredictor:
    return AISimulateTimePredictor(
        model=_model(),
        hw=_hw(),
        config=_sched_config(),
        database_mode=database_mode,
    )


@pytest.mark.parametrize("database_mode", ["SOL", "SILICON"])
def test_real_sdk_predicts_prefill_and_decode(database_mode):
    predictor = _make_predictor(database_mode)

    prefill_batch = ScheduleBatch([ScheduleRequest(extend_length=ISL)])
    decode_batch = ScheduleBatch([ScheduleRequest(extend_length=1, past_kv_length=ISL)])

    prefill_latency = predictor.predict_infer_time(prefill_batch)
    decode_latency = predictor.predict_infer_time(decode_batch)

    assert prefill_latency > 0, prefill_latency
    assert decode_latency > 0, decode_latency
    # A full prefill of ISL tokens must cost more than a single decode step.
    assert prefill_latency > decode_latency

    prefill_dict = predictor.predict_infer_latency_dict(prefill_batch)
    decode_dict = predictor.predict_infer_latency_dict(decode_batch)
    assert prefill_dict and sum(prefill_dict.values()) > 0
    assert decode_dict and sum(decode_dict.values()) > 0


@pytest.mark.parametrize("database_mode", ["SOL", "SILICON"])
def test_real_sdk_prefix_cached_prefill_is_faster_than_cold(database_mode):
    """Cache-representative scenario with the real estimator database.

    A prefill that reuses PREFIX_LEN cached KV tokens and extends only the
    remaining suffix must be cheaper than the cold prefill of the same total
    input length, in both database modes.
    """
    predictor = _make_predictor(database_mode)

    cold = ScheduleBatch([ScheduleRequest(extend_length=ISL, past_kv_length=0)])
    cached = ScheduleBatch(
        [ScheduleRequest(extend_length=EXTEND_LEN, past_kv_length=PREFIX_LEN)]
    )

    cold_latency = predictor.predict_infer_time(cold)
    cached_latency = predictor.predict_infer_time(cached)

    assert cold_latency > 0, cold_latency
    assert cached_latency > 0, cached_latency
    assert cached_latency < cold_latency, (cached_latency, cold_latency)


def test_config_registry_resolves_aisimulate_predictor(tmp_path, monkeypatch):
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
                },
                "predictor": {"name": "aisimulate", "database_mode": "SOL"},
                "scheduler": {
                    "tp_size": 1,
                    "ep_size": 1,
                    "backend_name": "sglang",
                    "backend_version": AISIMULATE_BACKEND_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    predictor = ConfigManager.get_inference_time_predictor(
        _model(), _hw(), _sched_config()
    )

    assert isinstance(predictor, AISimulateTimePredictor)
    latency = predictor.predict_infer_time(
        ScheduleBatch([ScheduleRequest(extend_length=ISL)])
    )
    assert latency > 0, latency

    ConfigManager.reset_config_cache()


def test_uneven_batch_coefficient_values():
    """The uneven-batch compensation coefficient keeps its closed-form values."""
    predictor = _make_predictor("SOL")
    balanced = [ScheduleRequest(extend_length=512), ScheduleRequest(extend_length=512)]
    uneven = [ScheduleRequest(extend_length=128), ScheduleRequest(extend_length=896)]
    assert predictor.ctx_attn_flops_ratio_with_avg(balanced) == pytest.approx(1.0)
    assert predictor.ctx_attn_flops_ratio_with_avg(uneven) == pytest.approx(1.5625)
    # A single request is balanced by definition.
    assert (
        predictor.ctx_attn_flops_ratio_with_avg([ScheduleRequest(extend_length=ISL)])
        == 1.0
    )


@pytest.mark.parametrize(
    "extend_past,ratio,isl,prefix",
    [
        # ratio 5/4 above the threshold: the coefficient reaches AISimulate.
        ([(256, 0), (768, 0)], 1.25, 512, 0),
        # ratio exactly 0.4: the >= threshold still applies the coefficient.
        ([(896, 0), (128, 3968)], 0.4, 2496, 1984),
        # ratio 19/48 below the threshold: AISimulate default scale is kept.
        ([(896, 0), (128, 4096)], 19 / 48, 2560, 2048),
    ],
)
def test_prefill_threshold_matches_real_sdk_reference(extend_past, ratio, isl, prefix):
    """The 0.4 threshold maps to the exact RuntimeConfig passed to the SDK.

    The adapter output is compared with ``InferenceSession.run_static`` on the
    fixed RuntimeConfig the adapter is expected to build: with the correction
    scale when the coefficient is at or above 0.4, without it otherwise.
    """
    predictor = _make_predictor("SOL")
    batch = ScheduleBatch(
        [ScheduleRequest(extend_length=e, past_kv_length=p) for e, p in extend_past]
    )
    assert predictor.ctx_attn_flops_ratio_with_avg(batch.reqs) == pytest.approx(ratio)

    adapter = predictor.predict_infer_latency_dict(batch)
    plain = RuntimeConfig(batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1)
    scaled = RuntimeConfig(
        batch_size=batch.batch_size,
        isl=isl,
        prefix=prefix,
        osl=1,
        seq_imbalance_correction_scale=ratio,
    )
    reference_plain = predictor._session.run_static(
        plain, mode="static_ctx"
    ).get_context_latency_dict()
    reference_scaled = predictor._session.run_static(
        scaled, mode="static_ctx"
    ).get_context_latency_dict()

    if ratio >= 0.4:
        assert adapter == reference_scaled
        # The scale must be observable so the assertion is not degenerate.
        assert adapter != reference_plain
    else:
        assert adapter == reference_plain


def test_phase_scales_min_latency_and_unit_boundary():
    """Phase scales, the prefill floor, and the ms-to-s boundary.

    Expected values come from a real-SDK baseline multiplied by fixed factors;
    the production estimation itself is not duplicated here.
    """
    base = _make_predictor("SOL")
    prefill = ScheduleBatch([ScheduleRequest(extend_length=ISL)])
    decode = ScheduleBatch([ScheduleRequest(extend_length=1, past_kv_length=ISL)])

    base_prefill = base.predict_infer_time(prefill)
    base_decode = base.predict_infer_time(decode)

    # The public API reports seconds while the SDK dict is in milliseconds.
    base_prefill_ms = sum(base.predict_infer_latency_dict(prefill).values())
    assert base_prefill == pytest.approx(base_prefill_ms / 1e3)

    scaled = AISimulateTimePredictor(
        model=_model(),
        hw=_hw(),
        config=_sched_config(),
        database_mode="SOL",
        prefill_scale_factor=2.0,
        decode_scale_factor=0.5,
    )
    assert scaled.predict_infer_time(prefill) == pytest.approx(2.0 * base_prefill)
    assert scaled.predict_infer_time(decode) == pytest.approx(0.5 * base_decode)

    # prefill_min_latency is a millisecond floor applied after the prefill
    # scale; decode is subject to neither the prefill scale nor the floor.
    floor_ms = 4.0 * base_prefill_ms
    floored = AISimulateTimePredictor(
        model=_model(),
        hw=_hw(),
        config=_sched_config(),
        database_mode="SOL",
        prefill_scale_factor=2.0,
        prefill_min_latency=floor_ms,
    )
    assert floored.predict_infer_time(prefill) == pytest.approx(floor_ms / 1e3)
    assert floored.predict_infer_time(decode) == pytest.approx(base_decode)


def test_config_registry_rejects_legacy_predictor_name(tmp_path, monkeypatch):
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
                },
                "predictor": {"name": "aiconfigurator", "database_mode": "SOL"},
                "scheduler": {
                    "tp_size": 1,
                    "ep_size": 1,
                    "backend_name": "sglang",
                    "backend_version": "0.5.9",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    with pytest.raises(ValueError, match="renamed 'aisimulate'"):
        ConfigManager.get_inference_time_predictor(_model(), _hw(), _sched_config())

    ConfigManager.reset_config_cache()


def test_config_registry_rejects_unknown_predictor_name(tmp_path, monkeypatch):
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
                },
                "predictor": {"name": "bogus_predictor"},
                "scheduler": {
                    "tp_size": 1,
                    "ep_size": 1,
                    "backend_name": "sglang",
                    "backend_version": AISIMULATE_BACKEND_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    with pytest.raises(ValueError, match="Supported: aisimulate, ml, replay"):
        ConfigManager.get_inference_time_predictor(_model(), _hw(), _sched_config())

    ConfigManager.reset_config_cache()
