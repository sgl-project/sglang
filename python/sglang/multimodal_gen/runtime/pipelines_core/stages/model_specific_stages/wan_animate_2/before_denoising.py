# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2 before-denoising stage.

Builds the per-clip conditioning container (``WanAnimate2ClipConditioning``) the DiT
reads via its ``clip_cond`` kwarg, plus the clip-invariant text/CLIP/VAE encodings.
Preprocessing helpers live in ``preprocess.py`` next to this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cv2
import msgspec
import numpy as np
import torch
from einops import rearrange

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_clip_conditioning import (
    WanAnimate2ClipConditioning,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_dpm_solver_multistep import (
    DPMSolverMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.audio import (
    extract_reference_video_audio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
    LetterboxInfo,
    get_padding_len,
    make_conditioning_mask,
    read_reference_video_frames,
    resize_by_area,
    validate_and_get_single_string,
    zigzag_padding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        Wan_Animate_2_14B_Config,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
        WanAnimate2ImageEncoderAdapter,
        WanAnimate2VaeAdapter,
    )

# Frames of the previous clip's decoded output that open the next clip as given (not denoised);
# the official pipelines/wan_animate_2_pipeline.py hard-codes num_frames_conditioning = 1.
_NUM_FRAMES_CONDITIONING = 1
# Wan VAE temporal compression: 4 pixel frames per latent frame after the first.
_VAE_TEMPORAL_COMPRESSION = 4
# Wan VAE latent channels.
_NUM_LATENT_CHANNELS = 16


def generation_token_grid(
    num_frames: int, latent_h: int, latent_w: int
) -> tuple[int, int, int]:
    """DiT token grid (f, h, w) of a clip of ``num_frames`` pixel frames. Wan VAE: 4k + 1 pixel
    frames -> k + 1 latent frames; one latent frame is prepended for the reference image;
    the DiT patchifies with patch_size (1, 2, 2), so each spatial side halves."""
    clip_latent_t = (num_frames - 1) // _VAE_TEMPORAL_COMPRESSION + 1
    return clip_latent_t + 1, latent_h // 2, latent_w // 2


def _generation_latent_shape(
    latent_h: int, latent_w: int, num_frames: int
) -> tuple[int, int, int, int]:
    """``[16, latent_t, latent_h, latent_w]`` of the latent the DiT denoises for a clip of
    ``num_frames`` pixel frames; latent_t comes from generation_token_grid (clip frames plus
    the leading reference-image slot)."""
    latent_t = generation_token_grid(num_frames, latent_h, latent_w)[0]
    return (_NUM_LATENT_CHANNELS, latent_t, latent_h, latent_w)


def get_sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    """Official Wan sampler grid: ``[sampling_steps]`` fp64 sigmas, uniform on [1, 0) with ``shift`` applied (sigma_0 == 1)."""
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


class _PerClipDenoisingMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Schedule metadata for one clip of the reference video."""

    # Position in the schedule, 0-based.
    clip_index: int
    # Start frame (inclusive) in the padded reference video.
    frame_start_index: int
    # End frame (exclusive) in the padded reference video.
    frame_end_index: int
    # Pixel-frame count, frame_end_index - frame_start_index, zigzag-padded frames included;
    # shorter than clip_len only on the last clip.
    num_frames: int
    # Leading frames taken from the previous clip's output: 0 for clip 0, else
    # num_frames_conditioning.
    num_frames_conditioning_for_clip: int


def build_schedule(
    reference_video_frames: list[np.ndarray],
    clip_len: int,
    num_frames_conditioning: int,
) -> list[_PerClipDenoisingMetadata]:
    """Prepare denoising metadata for all the clips of the reference video."""
    schedule: list[_PerClipDenoisingMetadata] = []
    start = 0
    clip_index = 0
    num_frames_reference_video = len(reference_video_frames)
    while True:
        if start + num_frames_conditioning >= num_frames_reference_video:
            break
        num_frames_conditioning_for_clip = 0 if start == 0 else num_frames_conditioning
        clip_len_current = clip_len
        if num_frames_reference_video - start < clip_len:  # last clip shrinks
            clip_len_current = num_frames_reference_video - start
        schedule.append(
            _PerClipDenoisingMetadata(
                clip_index=clip_index,
                frame_start_index=start,
                frame_end_index=start + clip_len_current,
                num_frames=clip_len_current,
                num_frames_conditioning_for_clip=num_frames_conditioning_for_clip,
            )
        )
        start += clip_len_current - num_frames_conditioning
        clip_index += 1
    if not schedule:
        raise ValueError(
            "Wan-Animate-2 reference video produced 0 clips; check "
            "clip_len/num_frames_conditioning and that the video decoded frames."
        )
    return schedule


class _WanAnimate2Inputs(msgspec.Struct, frozen=True, kw_only=True):
    """The per-request inputs read off the Req."""

    reference_image_path: str
    reference_video_path: str

    prompt: str
    prompt_ref: str
    negative_prompt: str

    width: int
    height: int
    clip_len: int
    num_frames_conditioning: int
    fps: int
    seed: int
    num_inference_steps: int
    guidance_scale: float
    enable_audio: bool

    @classmethod
    def from_request(cls, batch: Req) -> _WanAnimate2Inputs:
        """Read the Wan-Animate-2 input fields off the Req (all SamplingParam-backed)."""
        reference_image_path = validate_and_get_single_string(
            batch.image_path, "image_path"
        )
        reference_video_path = validate_and_get_single_string(
            batch.video_path, "video_path"
        )
        prompt = validate_and_get_single_string(batch.prompt, "prompt")
        prompt_ref = validate_and_get_single_string(batch.prompt_ref, "prompt_ref")
        # negative_prompt_embeddings is always computed; an unset negative prompt encodes "".
        negative_prompt = validate_and_get_single_string(
            batch.negative_prompt or "", "negative_prompt"
        )

        seed = batch.seed
        if isinstance(seed, list) and len(seed) != 1:
            raise ValueError(
                f"Wan-Animate-2 takes exactly one 'seed' per request, got {len(seed)}"
            )
        if isinstance(seed, list):
            seed = seed[0]

        return _WanAnimate2Inputs(
            reference_image_path=reference_image_path,
            reference_video_path=reference_video_path,
            prompt=prompt,
            prompt_ref=prompt_ref,
            negative_prompt=negative_prompt,
            width=int(batch.width),
            height=int(batch.height),
            clip_len=int(batch.clip_len),
            num_frames_conditioning=_NUM_FRAMES_CONDITIONING,
            fps=int(batch.fps),
            seed=int(seed),
            num_inference_steps=int(batch.num_inference_steps),
            guidance_scale=float(batch.guidance_scale),
            enable_audio=bool(batch.enable_audio),
        )


class WanAnimate2RequestState(msgspec.Struct, kw_only=True):
    """Clip-independent state, computed once, reused by every clip."""

    device: torch.device
    sp_size: int
    generator: torch.Generator
    inputs: _WanAnimate2Inputs

    # reference image
    letterbox_info: LetterboxInfo
    reference_image_height: int
    reference_image_width: int
    latent_h: int  # reference_image_height // 8
    latent_w: int  # reference_image_width // 8
    reference_image_condition: torch.Tensor  # [20, 1, latent_h, latent_w] bf16
    reference_image_embeddings: torch.Tensor  # [1, 257, 1280] bf16

    # text
    prompt_embeddings: torch.Tensor  # [L, 4096] bf16
    negative_prompt_embeddings: torch.Tensor  # [L, 4096] bf16
    prompt_ref_embeddings: torch.Tensor  # [L_prompt_ref, 4096] bf16

    # reference video, zigzag-padded to the clip schedule
    reference_video_frames: list[np.ndarray]  # resized [H, W, 3] uint8 RGB frames
    schedule: list[_PerClipDenoisingMetadata]
    # frame count before padding; the final video is trimmed to it
    num_reference_video_frames: int

    # Token grid of a full-length clip; every clip's attention layout is padded to it.
    full_clip_grid_sizes: tuple[int, int, int]

    # Reference-video audio, opt-in per request via enable_audio; None gives a silent mp4.
    audio: torch.Tensor | None = None  # [1, C, L] fp32 in [-1, 1]
    audio_sample_rate: int | None = None
    # Assembled by the denoising stage for the output stage, [T, H, W, C] uint8.
    decoded_frames: np.ndarray | None = None
    # Initial noise for clip 0, drawn by the before-denoising stage so that batch.latents is
    # the real denoising input; later clips draw from the generator in schedule order.
    clip_0_init_noise: torch.Tensor | None = None


# The one batch.extra entry Wan-Animate-2 uses; later stages read it back through
# request_state_from_batch.
WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY = "wan_animate_2_request_state"


def request_state_from_batch(batch: Req) -> WanAnimate2RequestState:
    state = batch.extra.get(WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY)
    if not isinstance(state, WanAnimate2RequestState):
        raise ValueError(
            f"batch.extra[{WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY!r}] is missing: "
            "WanAnimate2BeforeDenoisingStage must run before this stage."
        )
    return state


def _mirror_request_state_into_batch(
    batch: Req,
    request_state: WanAnimate2RequestState,
    *,
    scheduler: DPMSolverMultistepScheduler,
    flow_shift: float,
) -> None:
    """Populate the standard DenoisingStage input fields from the request state.

    The clip loop keeps reading ``request_state``; the embeddings, generator, sigma grid and
    timesteps exist for the shared stage verification, warmup and residency bookkeeping.
    ``batch.latents`` is clip 0's initial noise, drawn here (the seed's first draw) and
    consumed by ``build_clip_conditioning`` for clip 0, so the sequence of generator draws is
    the same as when the loop drew it: numerics are unchanged."""
    inputs = request_state.inputs
    batch.prompt_embeds = [request_state.prompt_embeddings]
    batch.negative_prompt_embeds = [request_state.negative_prompt_embeddings]
    batch.image_embeds = [request_state.reference_image_embeddings]
    batch.generator = request_state.generator

    sigmas = get_sampling_sigmas(inputs.num_inference_steps, flow_shift)
    scheduler.set_timesteps(sigmas=sigmas, device=request_state.device)
    batch.timesteps = scheduler.timesteps
    batch.sigmas = sigmas.tolist()

    batch.latents = _draw_clip_0_init_noise(request_state)
    batch.raw_latent_shape = batch.latents.shape


def _draw_clip_0_init_noise(request_state: WanAnimate2RequestState) -> torch.Tensor:
    """Draw clip 0's initial noise from the request generator (its first draw) and keep it on
    the request state for build_clip_conditioning; clips 1.. then take the following draws."""
    init_noise = torch.randn(
        *_generation_latent_shape(
            request_state.latent_h,
            request_state.latent_w,
            request_state.schedule[0].num_frames,
        ),
        dtype=torch.float32,
        device=request_state.device,
        generator=request_state.generator,
    )
    request_state.clip_0_init_noise = init_noise
    return init_noise


VaeEncoderFn = Callable[[list[torch.Tensor]], list[torch.Tensor]]
ImageEmbedderFn = Callable[[list[torch.Tensor]], torch.Tensor]


def build_clip_conditioning(
    request_state: WanAnimate2RequestState,
    clip_denoising_metadata: _PerClipDenoisingMetadata,
    prev_clip_conditioning_frames: torch.Tensor | None,
    *,
    vae_encoder: VaeEncoderFn,
    image_embedder: ImageEmbedderFn,
) -> WanAnimate2ClipConditioning:
    """Build conditioning for a single clip."""
    device = request_state.device
    latent_h, latent_w = request_state.latent_h, request_state.latent_w
    reference_image_height, reference_image_width = (
        request_state.reference_image_height,
        request_state.reference_image_width,
    )

    # gen/noise latent geometry
    num_frames = clip_denoising_metadata.num_frames
    grid_sizes = generation_token_grid(num_frames, latent_h, latent_w)
    latent_t = grid_sizes[0]
    # The clip's own latent frames, without the prepended reference-image frame.
    clip_latent_t = latent_t - 1
    init_noise_shape = _generation_latent_shape(latent_h, latent_w, num_frames)

    with (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16),
        torch.no_grad(),
    ):
        # previous-clip condition over the clip's own latent frames (no reference-image frame)
        # [4, clip_latent_t, latent_h, latent_w] fp32
        prev_clip_condition_mask = make_conditioning_mask(
            clip_latent_t,
            latent_h,
            latent_w,
            clip_denoising_metadata.num_frames_conditioning_for_clip,
            device=device,
        )
        # The clip's pixel video: the given frames first, zeros where the DiT generates.
        # Built on ``device`` so the frames never leave the GPU.
        prev_clip_condition_pixel_space = torch.zeros(
            3,
            num_frames,
            reference_image_height,
            reference_image_width,
            device=device,
        )
        if clip_denoising_metadata.num_frames_conditioning_for_clip > 0:
            # prev_clip_conditioning_frames: the previous clip's last
            # ``num_frames_conditioning`` decoded frames,
            # [C, num_frames_conditioning, reference_image_height, reference_image_width] bf16 in [-1, 1].
            if prev_clip_conditioning_frames is None:
                raise ValueError(
                    "For all clips except clip-0 `prev_clip_conditioning_frames` must be not None."
                )

            prev_clip_condition_pixel_space[
                :, : clip_denoising_metadata.num_frames_conditioning_for_clip
            ] = torch.nn.functional.interpolate(
                prev_clip_conditioning_frames[
                    :, : clip_denoising_metadata.num_frames_conditioning_for_clip
                ],
                size=(reference_image_height, reference_image_width),
                mode="bicubic",
            )

        # [16, clip_latent_t, latent_h, latent_w] fp32
        prev_clip_condition = vae_encoder([prev_clip_condition_pixel_space])[0]

        # [20, clip_latent_t, latent_h, latent_w] bf16
        prev_clip_condition = torch.concat(
            [prev_clip_condition_mask, prev_clip_condition]
        ).to(torch.bfloat16)

        # [20, latent_t, latent_h, latent_w] bf16
        # recall: latent_t = clip_latent_t + 1 (+1 is for the reference image condition in latent space)
        generation_condition = torch.concat(
            [request_state.reference_image_condition, prev_clip_condition], dim=1
        )

        # reference-video slice -> reference_video_latents
        # [t, h, w, c] uint8
        reference_video_clip_np = np.stack(
            request_state.reference_video_frames[
                clip_denoising_metadata.frame_start_index : clip_denoising_metadata.frame_end_index
            ]
        )
        reference_video_pixels = rearrange(
            torch.from_numpy(reference_video_clip_np).to(device), "t h w c -> c t h w"
        )
        # [0, 255] uint8 -> [-1.0, 1.0] bf16; [3, t, h, w]
        reference_video_pixels = (reference_video_pixels.float() / 127.5 - 1).to(
            torch.bfloat16
        )

        # [16, reference_video_latent_t, reference_video_latent_h, reference_video_latent_w] fp32
        # Stays fp32 (vae.encode returns .float()); forward_ref cats it with the
        # bf16 reference_video_condition under autocast type promotion.
        reference_video_latents = vae_encoder([reference_video_pixels])[0]
        (
            _,
            reference_video_latent_t,
            reference_video_latent_h,
            reference_video_latent_w,
        ) = reference_video_latents.shape
        reference_video_grid_sizes = (
            reference_video_latent_t,
            reference_video_latent_h // 2,
            reference_video_latent_w // 2,
        )

        # [3, reference_video_height, reference_video_width] bf16
        reference_video_frame_0 = reference_video_pixels[:, 0]
        reference_video_frame_0_image_embeddings = image_embedder(
            [reference_video_frame_0[:, None, :, :]]
        ).to(torch.bfloat16)  # [1, 257, 1280] bf16

        # reference_video_condition (forward_ref): mask, then VAE latent of the reference-video slice.
        # Every frame of the slice is given, so the latent is reference_video_latents itself;
        # the official code encodes the same pixels a second time here.
        reference_video_condition_mask = make_conditioning_mask(
            reference_video_latent_t,
            reference_video_latent_h,
            reference_video_latent_w,
            num_frames,
            device=device,
        )
        # [20, reference_video_latent_t, reference_video_latent_h, reference_video_latent_w] bf16
        reference_video_condition = torch.concat(
            [reference_video_condition_mask, reference_video_latents]
        ).to(torch.bfloat16)

    # Drawn from the request's seeded generator; clips are built in schedule order, so
    # clip k always gets the k-th draw and (seed, clip_index) identifies the noise. Clip 0's
    # draw already happened in the before-denoising stage (it is batch.latents).
    if (
        clip_denoising_metadata.clip_index == 0
        and request_state.clip_0_init_noise is not None
    ):
        init_noise = request_state.clip_0_init_noise
    else:
        init_noise = torch.randn(
            *init_noise_shape,
            dtype=torch.float32,
            device=device,
            generator=request_state.generator,
        )

    return WanAnimate2ClipConditioning(
        reference_video_latents=reference_video_latents,
        generation_video_grid_sizes=grid_sizes,
        reference_video_frame_0_image_embeddings=reference_video_frame_0_image_embeddings,
        reference_video_condition=reference_video_condition,
        prompt_ref_embeddings=request_state.prompt_ref_embeddings,
        generation_condition=generation_condition,
        reference_video_grid_sizes=reference_video_grid_sizes,
        full_clip_grid_sizes=request_state.full_clip_grid_sizes,
        num_frames=clip_denoising_metadata.num_frames,
        init_noise=init_noise,
    )


class WanAnimate2BeforeDenoisingStage(TextEncodingStage):
    """Native text encoding plus the model-specific reference and clip conditioning."""

    # Text-only stage dedup must not skip fresh image/video conditions or the request RNG.
    deduplicated_output_fields = ()

    def __init__(
        self,
        vae: WanAnimate2VaeAdapter,
        image_encoder: WanAnimate2ImageEncoderAdapter,
        text_encoder: torch.nn.Module,
        tokenizer,
        pipeline_config: Wan_Animate_2_14B_Config,
        scheduler: DPMSolverMultistepScheduler,
    ) -> None:
        super().__init__(text_encoders=[text_encoder], tokenizers=[tokenizer])
        self.vae = vae
        self.image_encoder = image_encoder
        self.pipeline_config = pipeline_config
        self._silent_reference_logged = False
        self._audio_failure_logged = False
        # Shared with the denoising stage, which resets it per clip; used here only to derive
        # batch.timesteps on the same sigma grid.
        self.scheduler = scheduler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        name = self._component_stage_name(stage_name)
        return super().component_uses(server_args, stage_name) + [
            ComponentUse(stage_name=name, component_name="vae"),
            ComponentUse(stage_name=name, component_name="image_encoder"),
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        sp_size = max(1, int(get_sp_world_size()))

        inputs = _WanAnimate2Inputs.from_request(batch)

        # One seeded generator per request; clips draw their noise from it in order.
        generator = torch.Generator(device=device).manual_seed(inputs.seed)

        audio, audio_sample_rate = self._extract_audio(inputs)
        request_state = self._build_request_state(
            inputs,
            device,
            sp_size,
            generator,
            server_args,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
        )
        if len(request_state.schedule) > 1:
            self.log_info(
                "reference video splits into %d clips; each is conditioned in the denoising "
                "loop via build_clip_conditioning.",
                len(request_state.schedule),
            )
        batch.extra[WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY] = request_state
        _mirror_request_state_into_batch(
            batch,
            request_state,
            scheduler=self.scheduler,
            flow_shift=self.pipeline_config.flow_shift,
        )
        return batch

    def _build_request_state(
        self,
        inputs: _WanAnimate2Inputs,
        device: torch.device,
        sp_size: int,
        generator: torch.Generator,
        server_args: ServerArgs,
        *,
        audio: torch.Tensor | None,
        audio_sample_rate: int | None,
    ) -> WanAnimate2RequestState:
        width, height = inputs.width, inputs.height

        reference_image = cv2.imread(inputs.reference_image_path)
        if reference_image is None:
            raise ValueError(
                f"Wan-Animate-2: failed to read reference image at {inputs.reference_image_path!r} "
                "(cv2.imread returned None: missing file or unsupported/corrupt image)."
            )
        reference_image = reference_image[..., ::-1]  # BGR -> RGB
        reference_image, letterbox_info = resize_by_area(
            reference_image, width * height, divisor=16
        )

        # reference video: decode, fps resample, zigzag pad.
        frames = read_reference_video_frames(inputs.reference_video_path, inputs.fps)
        reference_video_frames = [
            resize_by_area(f, width * height, divisor=16)[0] for f in frames
        ]
        num_reference_video_frames = len(frames)
        target_len = get_padding_len(num_reference_video_frames, inputs.clip_len)
        reference_video_frames = zigzag_padding(reference_video_frames, target_len)

        # Keep the official padding and geometry of every needed clip, but do not
        # denoise clips whose entire output would be discarded by the final trim.
        schedule = [
            clip
            for clip in build_schedule(
                reference_video_frames, inputs.clip_len, inputs.num_frames_conditioning
            )
            if clip.frame_start_index + clip.num_frames_conditioning_for_clip
            < num_reference_video_frames
        ]

        # reference-image encodings (constant across clips)
        reference_image_height, reference_image_width = (
            reference_image.shape[0],
            reference_image.shape[1],
        )
        latent_h, latent_w = reference_image_height // 8, reference_image_width // 8

        reference_image_pixel_values = rearrange(
            torch.from_numpy(reference_image).to(device), "h w c -> c 1 h w"
        )
        # [0, 255] uint8 -> [-1.0, 1.0] bf16; [3, 1, H, W]
        reference_image_pixel_values = (
            reference_image_pixel_values.float() / 127.5 - 1
        ).to(torch.bfloat16)

        with (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            # [16, 1, latent_h, latent_w] fp32
            reference_image_latents = self.vae_encoder([reference_image_pixel_values])[
                0
            ]

            # [4, 1, latent_h, latent_w] fp32
            reference_image_mask = make_conditioning_mask(
                1, latent_h, latent_w, 1, device=device
            )

            # [20, 1, latent_h, latent_w] bf16
            reference_image_condition = torch.concat(
                [reference_image_mask, reference_image_latents]
            ).to(torch.bfloat16)

            reference_image_embeddings = self.image_embedder(
                [reference_image_pixel_values]
            ).to(torch.bfloat16)  # [1, 257, 1280] bf16

            # Keep batch-one encoder arithmetic while sharing native caching and residency.
            prompt_embeddings, prompt_ref_embeddings, negative_prompt_embeddings = [
                self.encode_text(prompt, server_args, device=device)[0][0]
                for prompt in (inputs.prompt, inputs.prompt_ref, inputs.negative_prompt)
            ]

        return WanAnimate2RequestState(
            device=device,
            sp_size=sp_size,
            generator=generator,
            inputs=inputs,
            letterbox_info=letterbox_info,
            reference_image_height=reference_image_height,
            reference_image_width=reference_image_width,
            latent_h=latent_h,
            latent_w=latent_w,
            reference_image_condition=reference_image_condition,
            reference_image_embeddings=reference_image_embeddings,
            prompt_embeddings=prompt_embeddings,
            negative_prompt_embeddings=negative_prompt_embeddings,
            prompt_ref_embeddings=prompt_ref_embeddings,
            reference_video_frames=reference_video_frames,
            schedule=schedule,
            num_reference_video_frames=num_reference_video_frames,
            full_clip_grid_sizes=generation_token_grid(
                inputs.clip_len, latent_h, latent_w
            ),
            audio=audio,
            audio_sample_rate=audio_sample_rate,
        )

    def _extract_audio(
        self, inputs: _WanAnimate2Inputs
    ) -> tuple[torch.Tensor | None, int | None]:
        """Per-request opt-out; any failure yields (None, None), a silent video."""
        if not inputs.enable_audio:
            return None, None
        return extract_reference_video_audio(
            inputs.reference_video_path,
            log_info=self.log_info,
            log_no_audio_track=self._log_silent_reference,
            log_warning=self._log_audio_failure,
        )

    def _log_silent_reference(self, msg: str, *args: Any) -> None:
        # A reference video without an audio track is normal input, not a failure:
        # one info line per process, then debug.
        if self._silent_reference_logged:
            self.log_debug(msg, *args)
            return
        self._silent_reference_logged = True
        self.log_info(
            msg + " Set enable_audio=false on the request to skip audio extraction.",
            *args,
        )

    def _log_audio_failure(self, msg: str, *args: Any) -> None:
        # A track the audio reader cannot decode is the same for every request: warn once
        # per process with the remedy, then keep the detail at debug.
        if self._audio_failure_logged:
            self.log_debug(msg, *args)
            return
        self._audio_failure_logged = True
        self.log_warning(
            msg
            + " Set enable_audio=false on the request to skip audio extraction; further "
            "audio failures are logged at debug level.",
            *args,
        )

    def vae_encoder(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        with self.use_declared_component(component_name="vae", module=self.vae.vae):
            return self.vae.encode(videos)

    def image_embedder(self, videos: list[torch.Tensor]) -> torch.Tensor:
        with self.use_declared_component(
            component_name="image_encoder", module=self.image_encoder.model
        ):
            return self.image_encoder.visual(videos)
