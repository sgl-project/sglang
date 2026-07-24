# Adapted from: https://github.com/Robbyant/lingbot-video
# Reference (upstream): /vllm-workspace/lingbot-video/lingbot_video

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class LingBotVideoMoESamplingParams(SamplingParams):
    """Sampling parameters for the LingBot-Video MoE 30B T2V model.

    Defaults match the upstream parity configuration (480p, 81 frames, 40
    FlowUniPC steps, shift=3.0, guidance=6.0, fps=16).

    The DiT was trained on *structured JSON* captions (from a separate
    prompt-rewriter). The caption string must be supplied via the ``prompt``
    field; raw natural-language prompts are out-of-distribution and produce
    garbage. The negative prompt is likewise a structured JSON caption that
    the ``LingBotVideoBeforeDenoisingStage`` generates from
    ``DEFAULT_NEGATIVE_PROMPT``; leaving ``negative_prompt=None`` lets the
    pipeline supply its native negative.
    """

    # Video
    num_frames: int = 81
    height: int = 480
    width: int = 480
    fps: int = 16

    # Denoising
    num_inference_steps: int = 40
    guidance_scale: float = 6.0
    flow_shift: float = 3.0

    # The pipeline supplies the structured-JSON negative prompt natively.
    negative_prompt: str | None = None

    seed: int = 0
