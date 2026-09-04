# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from sglang.multimodal_gen.plugins import load_diffusion_plugins

# Load diffusion plugins before importing anything else from this package: hooks
# must be applied before the runtime binds module-level state, and before the
# platform singleton resolves. Every diffusion process imports this package,
# including spawned workers, so this covers them too.
load_diffusion_plugins()

from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]

# Trigger multimodal CI tests
