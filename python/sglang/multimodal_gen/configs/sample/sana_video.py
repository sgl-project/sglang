# SPDX-License-Identifier: Apache-2.0
"""Sampling defaults for SANA-Video 2B 480p."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaVideoSamplingParams(SamplingParams):
    data_type: DataType = DataType.VIDEO
    num_frames: int = 81
    fps: int = 16
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    height: int = 480
    width: int = 832
    max_sequence_length: int | None = 300
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
        "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
        "frames out of sync, inconsistent character shapes, temporal artifacts, "
        "jitter, and ghosting effects, creating a disorienting visual experience."
    )
