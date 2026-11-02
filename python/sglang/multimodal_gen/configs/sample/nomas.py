# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class NomasSamplingParams(SamplingParams):
    """Offline text-to-video. Each request owns an independent segment KV cache."""

    data_type: DataType = DataType.VIDEO
    height: int = 352
    width: int = 640
    num_frames: int = 40
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    generator_device: str = "cpu"
    # Nomas shards video tokens inside the model; the multi-GPU frame alignment
    # the framework applies for time-dim sharding would break event frame counts.
    adjust_frames: bool = False

    # Python API: captions for segments after the first (whose caption is prompt).
    # If omitted, repeat prompt for all segments. These are already prepared
    # captions; invoking an external PE/VLM is the caller's responsibility.
    continuation_prompts: list[str] | None = None
    # Event requests: [{"frames": positive_int, "caption": str}, ...].
    # Sum of frames must equal num_frames; mutually exclusive with continuations.
    segments: list[dict] | None = None
    # json enforces upstream segment/action schema; plain is a smoke-test shortcut.
    prompt_format: str = "plain"
    # Canonical normalized camera timeline: [(ceil(F/4)-1)*4+1,4,4].
    # Normalize the full trajectory before slicing; do not supply packed 4L poses.
    norm_extrinsics: list | None = None

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return super().video_request_extra_fields() | frozenset(
            {"segments", "continuation_prompts", "prompt_format", "norm_extrinsics"}
        )

    @classmethod
    def lower_video_request_kwargs(cls, request, kwargs):
        kwargs = super().lower_video_request_kwargs(request, dict(kwargs))
        # Both JSON and multipart video APIs default to a CUDA generator.
        # Nomas uses CPU RNG for upstream-compatible initial/cache noise;
        # enforce the same model-owned policy as the realtime adapter.
        kwargs["generator_device"] = "cpu"
        return kwargs
