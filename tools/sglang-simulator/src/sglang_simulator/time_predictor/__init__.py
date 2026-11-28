from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.replay import ReplayTimePredictor


def __getattr__(name):
    if name == "AISimulateTimePredictor":
        from sglang_simulator.time_predictor.aisimulate import (
            AISimulateTimePredictor,
        )

        return AISimulateTimePredictor
    if name == "MLTimePredictor":
        from sglang_simulator.time_predictor.ml import MLTimePredictor

        return MLTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "AISimulateTimePredictor",
    "InferTimePredictor",
    "MLTimePredictor",
    "ReplayTimePredictor",
    "ScheduleBatch",
    "ScheduleRequest",
)
