# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
    from sglang.multimodal_gen.configs.sample import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

_LAZY_EXPORTS = {
    "DiffGenerator": (
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator",
        "DiffGenerator",
    ),
    "PipelineConfig": (
        "sglang.multimodal_gen.configs.pipeline_configs",
        "PipelineConfig",
    ),
    "SamplingParams": ("sglang.multimodal_gen.configs.sample", "SamplingParams"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


# Trigger multimodal CI tests
