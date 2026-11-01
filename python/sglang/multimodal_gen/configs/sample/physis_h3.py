# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def validate_physis_h3_sampling(params):
    if params.num_inference_steps != 4 or params.guidance_scale != 1:
        raise ValueError("Physis H3 uses four fixed steps and guidance_scale=1")
    if params.num_frames < 22 or (params.num_frames - 5) % 17:
        raise ValueError("Physis H3 num_frames must be 17n+5 and at least 22")
    if (
        params.width <= 0
        or params.height <= 0
        or params.width % 32
        or params.height % 32
    ):
        raise ValueError("Physis H3 width and height must be positive multiples of 32")
    if params.image_path or params.video_path or params.negative_prompt:
        raise ValueError("Physis H3 supports positive text-to-video only")
    if params.generator_device not in (None, "cuda"):
        raise ValueError("Physis H3 requires generator_device='cuda'")
    if (
        params.attention_backend_override is not None
        and params.attention_backend_override.lower() != "torch_sdpa"
    ):
        raise ValueError("Physis H3 requires torch_sdpa attention")
    if (
        params.enable_cache_dit
        or params.cache_dit_params is not None
        or params.enable_teacache
        or params.teacache_params is not None
        or params.enable_spectrum
        or params.spectrum_params is not None
        or params.skip_softmax_params is not None
    ):
        raise ValueError(
            "Physis H3 does not support request-level cache or sparse attention"
        )
    if params.flow_shift is not None:
        raise ValueError(
            "Physis H3 uses a fixed sigma schedule; flow_shift is unsupported"
        )
    if params.num_outputs_per_prompt != 1:
        raise ValueError("Physis H3 requires one output per prompt")
    if type(params.seed) is not int or not 0 <= params.seed < 2**31:
        raise ValueError("Physis H3 seed must be an integer in [0, 2**31)")


@dataclass
class PhysisH3SamplingParams(SamplingParams):
    height: int = 768
    width: int = 1344
    num_frames: int = 158
    fps: int = 24
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    negative_prompt: str | None = None
    adjust_frames: bool = False

    def _adjust(self, server_args):
        self.adjust_frames = False
        super()._adjust(server_args)
        validate_physis_h3_sampling(self)
