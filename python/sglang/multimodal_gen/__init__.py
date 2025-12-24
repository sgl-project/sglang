# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#
# NOTE: Keep this module import-light. Importing diffusion configs and runtime
# can pull in optional dependencies (e.g., diffusers). We expose these symbols
# via lazy imports for better ergonomics and faster startup.

from __future__ import annotations

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]


def __getattr__(name: str):
    if name == "PipelineConfig":
        from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig

        return PipelineConfig
    if name == "SamplingParams":
        from sglang.multimodal_gen.configs.sample import SamplingParams

        return SamplingParams
    if name == "DiffGenerator":
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        return DiffGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
