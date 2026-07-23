# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class QwenImageSamplingParams(SamplingParams):
    negative_prompt: str = " "
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 4.0
    num_inference_steps: int = 50


@dataclass
class QwenImage2512SamplingParams(QwenImageSamplingParams):
    negative_prompt: str = (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    )


@dataclass
class QwenImageEditPlusSamplingParams(QwenImageSamplingParams):
    # Denoising stage
    guidance_scale: float = 4.0
    # true_cfg_scale: float = 4.0
    num_inference_steps: int = 40


@dataclass
class QwenImageLayeredSamplingParams(QwenImageSamplingParams):
    # num_frames: int = 4
    height: int = 640
    width: int = 640
    prompt: str = " "
    negative_prompt: str = " "

    guidance_scale: float = 4.0
    num_inference_steps: int = 50
    cfg_normalize: bool = True
    use_en_prompt: bool = True
