"""Real-SDK predictor tests for the AISimulate time predictor.

These tests run against the installed AISimulate SDK and cover:

- predictor construction and prefill/decode estimation through the SOL and
  SILICON performance databases;
- a prefix-cache representative scenario: a cached prefill reusing KV must be
  faster than the cold prefill of the same total length;
- the predictor configuration registry: the ``aisimulate`` predictor name
  resolves through ``ConfigManager``, the legacy ``aiconfigurator`` name
  fails with a migration error, and unknown names list the supported set.

The serving, offline-blocking, and cache-hit-ratio tests exercise the same
predictor end to end through the simulator server.
"""

import json
from pathlib import Path

import pytest
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
