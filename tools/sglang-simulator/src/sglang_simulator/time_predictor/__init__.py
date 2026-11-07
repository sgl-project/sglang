from importlib import import_module

from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    PredictorError,
    ScheduleBatch,
    ScheduleRequest,
)

_LAZY_CLASSES = {
    "AISimulateTimePredictor": "aisimulate",
    "MLTimePredictor": "ml",
    "ReplayTimePredictor": "replay",
    "SimOpusTimePredictor": "simopus",
}


def __getattr__(name: str):
    module = _LAZY_CLASSES.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(
        import_module(f"sglang_simulator.time_predictor.{module}"),
        name,
    )


__all__ = (
    "AISimulateTimePredictor",
    "SimOpusTimePredictor",
    "InferTimePredictor",
    "MLTimePredictor",
    "PredictorError",
    "ReplayTimePredictor",
    "ScheduleBatch",
    "ScheduleRequest",
)
