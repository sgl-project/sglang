from sglang_simulator.simulation.sglang.utils import resolve_model_info
from sglang_simulator.simulation.types import (
    SchedulerConfig,
)
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)

from sglang.srt.configs.model_config import ModelConfig


def test_time_predictor():
    model: ModelInfo = resolve_model_info(ModelConfig(model_path="Qwen/Qwen3-8B"))
    hw = AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
        inter_node_bandwidth_gb=25,
        intra_node_bandwidth_gb=300,
    )
    config = SchedulerConfig(backend_name="sglang", backend_version="0.5.9")
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            ScheduleRequest(512, 512),
            ScheduleRequest(1024, 0),
            ScheduleRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


if __name__ == "__main__":
    test_time_predictor()
