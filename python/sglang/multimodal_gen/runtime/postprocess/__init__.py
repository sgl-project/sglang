# SPDX-License-Identifier: Apache-2.0
"""Frame interpolation support for SGLang diffusion pipelines."""

from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_frames,
)

__all__ = ["FrameInterpolator", "interpolate_video_frames"]
