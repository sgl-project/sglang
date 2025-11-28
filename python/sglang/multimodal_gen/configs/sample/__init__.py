# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.sample.base import SamplingParams
from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)

__all__ = ["SamplingParams", "DiffusersGenericSamplingParams"]
