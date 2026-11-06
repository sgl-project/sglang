"""Real-SDK prefill, decode, and prefix-cache prediction on the unified ForwardPass estimator."""

import json
from unittest.mock import patch

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo
from sglang_simulator.time_predictor import AISimulateTimePredictor
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from test_simulation_sglang_serving import ASSETS, SIM_CONFIGS


@pytest.mark.parametrize("database_mode", ["SOL", "SILICON"])
def test_real_sdk_prefill_decode_and_prefix_cache(database_mode):
    config = json.loads(SIM_CONFIGS[f"aisimulate_{database_mode.lower()}"].read_text())
    with patch.object(ConfigManager, "_raw_config", config):
        hw = ConfigManager.get_accelerator_info()
    predictor = AISimulateTimePredictor(
        model=ModelInfo(model_path=str(ASSETS / "qwen3-8b")),
        hw=hw,
        config=SchedulerConfig(
            data_type=DataType.BF16,
            kv_cache_data_type=DataType.BF16,
            **config["scheduler"],
        ),
        database_mode=config["predictor"]["database_mode"],
    )
    cold = ScheduleBatch([ScheduleRequest(1024, 0, "extend")], "extend")
    cached = ScheduleBatch([ScheduleRequest(128, 896, "extend")], "extend")
    decode = ScheduleBatch([ScheduleRequest(1, 1024, "decode")], "decode")

    try:
        cold_latency = predictor.predict_infer_time(cold)
        cached_latency = predictor.predict_infer_time(cached)
        decode_latency = predictor.predict_infer_time(decode)

        assert 0 < cached_latency < cold_latency
        assert 0 < decode_latency < cold_latency
    finally:
        predictor._fpm.close()
