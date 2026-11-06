# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for the SANA-Video 2.0 5B checkpoint."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sana_video import SanaVideoSamplingParams


@dataclass
class SanaVideo2SamplingParams(SanaVideoSamplingParams):
    height: int = 736
    width: int = 1280
    num_frames: int = 193
    fps: int = 24
    guidance_scale: float = 8.0
    motion_score: int = 10
    high_motion: bool = False

    def build_request_extra(self):
        return {
            **super().build_request_extra(),
            "motion_score": self.motion_score,
            "high_motion": self.high_motion,
        }

    @classmethod
    def video_request_extra_fields(cls):
        return super().video_request_extra_fields() | {"motion_score", "high_motion"}
