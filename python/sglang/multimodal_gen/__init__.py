# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import platform

from sglang.multimodal_gen._triton_stub import install_triton_stub_if_needed

# Native Windows runtime needs import compatibility for Triton-backed modules.
if platform.system() == "Windows":
    install_triton_stub_if_needed()

from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

# Trigger multimodal CI tests
