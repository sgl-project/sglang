from importlib import import_module

from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    PredictorError,
    ScheduleBatch,
    ScheduleRequest,
    validate_latency_seconds,
)

_LAZY_CLASSES = {
    "AIConfiguratorTimePredictor": "aiconfigurator",
    "MLTimePredictor": "ml",
    "ReplayTimePredictor": "replay",
    "InferCastTimePredictor": "infercast",
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
    "ScheduleRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "PredictorError",
    "validate_latency_seconds",
)
