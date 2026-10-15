# SPDX-License-Identifier: Apache-2.0
import re
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class MingImageSamplingParams(SamplingParams):
    height: int = 2048
    width: int = 2048
    num_inference_steps: int = 12
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    generator_device: str = "cpu"

    @classmethod
    def default_image_output_format(cls):
        return "png"


@dataclass
class MingImageLayerSamplingParams(MingImageSamplingParams):
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 2.0
    prompt: str = ""
    num_layers: int = 4

    def build_request_extra(self):
        extra = super().build_request_extra()
        if type(self.num_layers) is not int or self.num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        match = re.search(
            r"(?:into\s+(\d+)\s+layers|number of layers:\s*(\d+))",
            self.prompt,
            re.IGNORECASE,
        )
        count = (
            int(next(value for value in match.groups() if value))
            if match
            else self.num_layers
        )
        if count < 1:
            raise ValueError("The prompt must request at least one layer")
        extra["ming_num_layers"] = count
        return extra

    @classmethod
    def image_request_extra_fields(cls):
        return frozenset({"num_layers"})

    @property
    def num_samples_per_request(self):
        return self.build_request_extra()["ming_num_layers"]
