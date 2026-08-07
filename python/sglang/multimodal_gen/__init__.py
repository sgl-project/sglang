# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import os

if os.environ.get("SGLANG_LIGHTWEIGHT_RUNTIME") != "1":
    from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
    from sglang.multimodal_gen.configs.sample import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    __all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]
else:
    __all__: list[str] = []

# Trigger multimodal CI tests
