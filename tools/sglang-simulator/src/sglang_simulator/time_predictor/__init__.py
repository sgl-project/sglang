from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.replay import ReplayTimePredictor


def __getattr__(name):
    if name == "AIConfiguratorTimePredictor":
        from sglang_simulator.time_predictor.aiconfigurator import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor
    if name == "MLTimePredictor":
        from sglang_simulator.time_predictor.ml import MLTimePredictor

        return MLTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "AIConfiguratorTimePredictor",
    "InferTimePredictor",
    "MLTimePredictor",
    "ReplayTimePredictor",
    "ScheduleBatch",
    "ScheduleRequest",
)
