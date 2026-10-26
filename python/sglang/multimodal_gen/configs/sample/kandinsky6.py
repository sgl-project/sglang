# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Kandinsky6TI2VASamplingParams(SamplingParams):
    """Sampling params for Kandinsky6 TI2VA (text[+image] -> video+audio).

    Defaults of the Diffusers ``Kandinsky6TI2VAPipeline`` (the undistilled
    ``Kandinsky-6.0-Pro-sft-5s-Diffusers`` checkpoint): 512x768 @ 24fps, 121
    frames ((121 - 1) // 4 + 1 == 31 latent frames at the video VAE's
    temporal_compression_ratio=4), guidance_scale=5.0, 50 inference steps.
    """

    # Video parameters (Kandinsky6 TI2VA default preset)
    height: int = 512
    width: int = 768
    num_frames: int = 121
    fps: int = 24

    # SamplingParams._adjust's multi-GPU frame-count rounding
    # (`orig_latent_num_frames` up to a multiple of `num_gpus`) is a second,
    # independent rounding rule on top of Kandinsky6LatentPreparationStage's
    # own `num_frames % 4 == 1` rounding (the diffusers reference's own
    # `latent_frames = (num_frames - 1) // 4 + 1` convention -- the single
    # source of truth for this model). The two compose for num_gpus > 1 and
    # land on a decoded frame count neither rule intended (e.g. the
    # documented default num_frames=121 becomes 125/129/125/125 decoded
    # frames at 2/3/4/8 GPUs instead of Diffusers' 121). Disabling it here
    # leaves Kandinsky6LatentPreparationStage's rounding as the only one
    # applied, regardless of num_gpus -- matching this model's precedent for
    # opting out of this same mechanism when it owns frame-count alignment
    # itself (see `enable_sequence_shard=True`, which disables it for the
    # same reason a few lines up in SamplingParams._adjust).
    adjust_frames: bool = False

    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 50

    negative_prompt: str = (
        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst "
        "quality, low quality, ugly, deformed, walking backwards"
    )


@dataclass
class Kandinsky6TI2VADistilledSamplingParams(Kandinsky6TI2VASamplingParams):
    """Sampling params for the pi-Flow distilled Kandinsky6 TI2VA checkpoint
    (``Kandinsky-6.0-Pro-distill-5s-Diffusers``).

    Distilled for 10 steps (model card; its ``scheduler_config.json`` leaves
    ``nfe`` unset): its ``PiflowScheduler`` runs one DiT call per step and has
    no classifier-free guidance, so any guidance_scale other than 1.0 is
    rejected by the denoising stage, and the negative prompt is never encoded
    either.
    """

    guidance_scale: float = 1.0
    num_inference_steps: int = 10


__all__ = ["Kandinsky6TI2VADistilledSamplingParams", "Kandinsky6TI2VASamplingParams"]
