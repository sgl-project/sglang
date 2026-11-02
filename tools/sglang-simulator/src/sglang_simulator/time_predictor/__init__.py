from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sglang_simulator.time_predictor.replay import ReplayTimePredictor

__all__ = (
    "ScheduleRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "AIConfiguratorTimePredictor",
    "MLTimePredictor",
    "ReplayTimePredictor",
)


def __getattr__(name):
    if name == "AIConfiguratorTimePredictor":
        from sglang_simulator.time_predictor.aiconfigurator import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
