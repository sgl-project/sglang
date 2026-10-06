# SPDX-License-Identifier: Apache-2.0
"""SR pipeline stages, grouped by component while retaining separate offload lifetimes."""

from __future__ import annotations

import os
from functools import partial

import torch

from sglang.multimodal_gen.configs.sample.kandinsky6_sr_resolution import (
    resolve_target_hw,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_group_coordinator,
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    can_install_spatial_shard_parallel_decode,
    has_decode_parallel_world,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_CHUNKS_KEY,
    SR_DENOISED_KEY,
    SR_DIT_SPEC_KEY,
    SR_LR_LATENT_KEY,
    SR_PLAN_KEY,
    SR_REQUESTED_HW_KEY,
    SR_SAMPLING_SPEC_KEY,
    SR_TILES_KEY,
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
    build_dit_spec,
    build_sampling_spec,
    uses_latent_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    denoise_chunks,
    module_dtype,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    decode_chunks,
    encode_pixel_tile,
    encode_video_to_lr_latent,
    plan_tiles,
    prepare_tile_latents,
    stitch_tiles,
    upscale_lr_latent_tile,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    VAE_SPATIAL_FACTOR,
    crop_to_hw,
    pad_to_spatial_factor,
    pre_upscale_video,
    resolve_scale_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_io import (
    SOURCE_AUDIO_SAMPLE_RATE,
    DecodedClip,
    decode_clip,
    extract_source_audio,
    synthetic_clip,
    to_output_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_utils import (
    resize_to_target,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    precision_to_dtype,
    resolve_precision,
)

WARMUP_CLIP_FRAMES = 9
WARMUP_CLIP_HW = (256, 384)


class Kandinsky6SRInputStage(PipelineStage):
    """Decode [T, 3, H, W] uint8 video, align fps/frames/size and retain source audio.

    Record requested dimensions before VAE padding so output can remove it."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        tiling_scale, pre_upscale = resolve_scale_request(batch.sr_resolution_scale)
        if batch.is_warmup:
            height, width = WARMUP_CLIP_HW
            clip = DecodedClip(
                frames=synthetic_clip(
                    height=height, width=width, frames=WARMUP_CLIP_FRAMES
                ),
                fps=24,
                source_fps=24.0,
            )
        else:
            path = batch.sampling_params.source_video_path()
            if path is None or not os.path.isfile(path):
                raise ValueError(f"Kandinsky6 SR input video not found: {path!r}")
            self.log_info("Decoding %s", path)
            clip = decode_clip(path)
        video = clip.frames
        if pre_upscale != 1.0:
            video = pre_upscale_video(video, pre_upscale, VAE_SPATIAL_FACTOR)
        frames = video.shape[0]
        # pad every scale for the VAE; output crops padding without resizing source pixels
        video, (requested_h, requested_w) = pad_to_spatial_factor(
            video, VAE_SPATIAL_FACTOR
        )

        batch.extra[SR_VIDEO_KEY] = video
        batch.extra[SR_TILING_SCALE_KEY] = tiling_scale
        batch.extra[SR_REQUESTED_HW_KEY] = (requested_h, requested_w)
        batch.fps = clip.fps
        batch.num_frames = frames
        batch.height = requested_h * tiling_scale
        batch.width = requested_w * tiling_scale
        if not batch.is_warmup:
            audio = extract_source_audio(
                path,
                sample_rate=SOURCE_AUDIO_SAMPLE_RATE,
                max_seconds=frames / clip.fps,
            )
            batch.audio = torch.from_numpy(audio) if audio is not None else None
            batch.audio_sample_rate = (
                SOURCE_AUDIO_SAMPLE_RATE if audio is not None else None
            )
        return batch


class Kandinsky6SREncodeStage(PipelineStage):
    """Encode the whole clip only for the LU path; pixel mode encodes tiles later."""

    def __init__(self, vae, latent_upscaler=None) -> None:
        super().__init__()
        self.vae = vae
        # Only consulted to pick the path (does the bank have this scale?).
        self.latent_upscaler = latent_upscaler

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase="encode_video",
                target_dtype=vae_dtype,
                start_at_stage_entry=False,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        tiling_scale = batch.extra[SR_TILING_SCALE_KEY]
        if not uses_latent_path(self.latent_upscaler, tiling_scale):
            return batch
        video = batch.extra.pop(SR_VIDEO_KEY)
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase="encode_video"
        ) as vae:
            assert vae is not None
            self.vae = vae
            lr_latent = encode_video_to_lr_latent(
                video, vae, device=get_local_torch_device()
            )
        batch.extra[SR_LR_LATENT_KEY] = lr_latent.cpu()
        return batch


class Kandinsky6SRLatentPrepStage(PipelineStage):
    """LU-or-pixel tile encode + upscale -> initial noisy latent, one chunk at a time."""

    def __init__(self, vae, transformer, latent_upscaler=None, scheduler=None) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer  # arch-config / dtype only: never called here
        self.latent_upscaler = latent_upscaler
        self.scheduler = (
            scheduler  # resolved sampler validation only: never stepped here
        )

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        uses: list[ComponentUse] = []
        if self.latent_upscaler is not None:
            uses.append(
                ComponentUse(
                    stage_name,
                    "latent_upscaler",
                    phase="upscale_tiles",
                    target_dtype=torch.bfloat16,
                )
            )
        uses.append(
            ComponentUse(
                stage_name, "vae", phase="encode_tiles", target_dtype=vae_dtype
            )
        )
        return uses

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        arch = server_args.pipeline_config.dit_config.arch_config
        tiling_scale = batch.extra[SR_TILING_SCALE_KEY]
        seed = batch.seed if isinstance(batch.seed, int) else batch.seed[0]
        spec = build_sampling_spec(
            arch=arch,
            tiling_scale=tiling_scale,
            seed=seed,
            num_steps=batch.num_inference_steps,
            tiles_batch_size=batch.sr_tiles_batch_size,
            tile_min_overlap=batch.sr_tile_min_overlap,
            scheduler=self.scheduler,
        )
        dit_spec = build_dit_spec(self.transformer)
        device = get_local_torch_device()

        latent_path = uses_latent_path(self.latent_upscaler, tiling_scale)
        source = batch.extra.pop(SR_LR_LATENT_KEY if latent_path else SR_VIDEO_KEY)
        factor = VAE_SPATIAL_FACTOR if latent_path else 1
        plan = plan_tiles(
            frame_hw=tuple(size * factor for size in source.shape[-2:]),
            visual_size=spec.visual_size,
            tiling_scale=tiling_scale,
            tile_min_overlap=spec.tile_min_overlap,
        )
        with self.use_declared_component(
            component_name="latent_upscaler" if latent_path else "vae",
            module=self.latent_upscaler if latent_path else self.vae,
            phase="upscale_tiles" if latent_path else "encode_tiles",
        ) as component:
            assert component is not None
            if latent_path:
                self.latent_upscaler = component
                tile_encoder = partial(
                    upscale_lr_latent_tile,
                    upscale_fn=partial(component.upscale, scale=tiling_scale),
                    lu_dtype=module_dtype(component),
                    scaling_factor=self.vae.scaling_factor,
                    device=device,
                )
            else:
                self.vae = component
                tile_encoder = partial(
                    encode_pixel_tile,
                    vae=component,
                    scaling_factor=component.scaling_factor,
                    device=device,
                )
            chunks = prepare_tile_latents(
                source,
                plan,
                tile_encoder=tile_encoder,
                latent_path=latent_path,
                dit_spec=dit_spec,
                spec=spec,
                device=device,
            )
        batch.extra[SR_PLAN_KEY] = plan
        batch.extra[SR_CHUNKS_KEY] = chunks
        batch.extra[SR_SAMPLING_SPEC_KEY] = spec
        batch.extra[SR_DIT_SPEC_KEY] = dit_spec
        return batch


class Kandinsky6SRDenoisingStage(DenoisingStage):
    """Runs the bundle's scheduler over every tile chunk prepared by the latent-prep stage."""

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # forward wraps the tiled loop in the shared compile warmup lifecycle
        return True

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Validate tile-chunk state instead of the shared text/CFG loop inputs."""
        result = VerificationResult()
        result.add_check(
            "extra[SR_SAMPLING_SPEC_KEY]",
            batch.extra.get(SR_SAMPLING_SPEC_KEY),
            V.not_none,
        )
        result.add_check(
            "extra[SR_DIT_SPEC_KEY]", batch.extra.get(SR_DIT_SPEC_KEY), V.not_none
        )
        result.add_check(
            "extra[SR_CHUNKS_KEY]", batch.extra.get(SR_CHUNKS_KEY), V.list_not_empty
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "extra[SR_DENOISED_KEY]", batch.extra.get(SR_DENOISED_KEY), V.list_not_empty
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        dit_dtype = precision_to_dtype(
            server_args.pipeline_config.dit_precision, "dit_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="denoise_tiles",
                target_dtype=dit_dtype,
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self._offload_for_torch_compile_warmup(batch):
            return self._denoise_tiles(batch, server_args)

    def _denoise_tiles(self, batch: Req, server_args: ServerArgs) -> Req:
        spec = batch.extra.pop(SR_SAMPLING_SPEC_KEY)
        dit_spec = batch.extra.pop(SR_DIT_SPEC_KEY)
        chunks = batch.extra.pop(SR_CHUNKS_KEY)
        device = get_local_torch_device()

        self._maybe_enable_cache_dit_and_torch_compile(spec.num_steps, batch)

        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise_tiles"
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer
            total = spec.num_steps * len(chunks)
            with self.progress_bar(
                total=total, batch=batch, desc="Kandinsky6 SR denoising"
            ) as progress:
                denoised = denoise_chunks(
                    chunks,
                    transformer,
                    self.scheduler,
                    dit_spec=dit_spec,
                    spec=spec,
                    device=device,
                    step_context=lambda step: set_forward_context(
                        current_timestep=step, attn_metadata=None, forward_batch=batch
                    ),
                    on_step=progress.update,
                )
        self._finish_active_component_use()
        batch.extra[SR_DENOISED_KEY] = denoised
        return batch


class Kandinsky6SRDecodeStage(PipelineStage):
    """VAE-decodes every denoised chunk into row-major uint8 tiles (``[3, T, Hb, Wb]``)."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "vae",
                phase="decode_tiles",
                target_dtype=vae_dtype,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        denoised = batch.extra.pop(SR_DENOISED_KEY)
        device = get_local_torch_device()
        group = None
        if (
            server_args.pipeline_config.vae_config.use_parallel_tiling
            and has_decode_parallel_world()
            and not can_install_spatial_shard_parallel_decode(
                server_args.pipeline_config.vae_config
            )
        ):
            group = get_decode_parallel_group_coordinator()
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase="decode_tiles"
        ) as vae:
            assert vae is not None
            self.vae = vae
            tiles = decode_chunks(
                denoised,
                vae,
                scaling_factor=vae.scaling_factor,
                device=device,
                group=group,
            )
        batch.extra[SR_TILES_KEY] = tiles
        return batch


class Kandinsky6SROutputStage(PipelineStage):
    """Hann-blends the decoded tiles, restores the requested output size, applies the
    optional delivery resize and returns ``output = [1, 3, T, H, W]`` (fp16 in [0, 1])
    plus the mono source audio.

    CPU only: every model phase already ran.  ``batch.height`` / ``batch.width`` are set
    to the final size, ``batch.fps`` was set by the input stage (and now also carried on
    the returned ``OutputBatch``, so a worker-resampled fps reaches the client-side save
    path unchanged).
    """

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        tiles = batch.extra.pop(SR_TILES_KEY)
        plan = batch.extra.pop(SR_PLAN_KEY)
        video = stitch_tiles(tiles, plan)
        del tiles

        requested_hw = batch.extra.pop(SR_REQUESTED_HW_KEY, None)
        if requested_hw is not None:
            requested_h, requested_w = requested_hw
            video = crop_to_hw(
                video,
                (requested_h * plan.tiling_scale, requested_w * plan.tiling_scale),
            )

        target_hw = resolve_target_hw(
            batch.sr_target_resolution,
            tuple(video.shape[-2:]),
            batch.sr_target_resize_mode,
        )
        if target_hw is not None:
            video = resize_to_target(video, target_hw)
        batch.height, batch.width = video.shape[-2:]
        return OutputBatch(
            output=to_output_video(video),
            audio=batch.audio,
            audio_sample_rate=batch.audio_sample_rate,
            fps=batch.fps,
            metrics=batch.metrics,
            usage=batch.usage,
        )
