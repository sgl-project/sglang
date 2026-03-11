# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import sys

# Native Windows runtime needs import compatibility for Triton-backed modules.
if sys.platform == "win32":
    from sglang._triton_stub import install as _install_triton_stub

    _install_triton_stub()
    del _install_triton_stub

from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

# Trigger multimodal CI tests
