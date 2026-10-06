# SPDX-License-Identifier: Apache-2.0
"""Latent-preparation stage of Kandinsky 6 video SR: tiles -> initial noisy latent chunks.

LU path: the latent-upscaler bank upscales every (already whole-video-KVAE-encoded) tile;
pixel path (a bank with no entry for the requested scale): every tile is bilinearly enlarged
to the trained base resolution and VAE-encoded directly. Both end in the same
``build_initial_latent`` call per chunk of ``sr_tiles_batch_size`` tiles.

Also resolves and validates this request's :class:`~.sampling.SamplingSpec` -- the
transformer/scheduler head-width check (``run_spec.build_sampling_spec`` /
``_check_sampler_fits_head``) and the step-count checks (``run_spec.check_denoising_request``,
porting FastVideo commit d8d0e79f) run here, before any tile is upscaled or encoded, so a
mismatched bundle fails before the expensive work rather than inside the first DiT call.
"""

from functools import partial

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_CHUNKS_KEY,
    SR_DIT_SPEC_KEY,
    SR_LR_LATENT_KEY,
    SR_PLAN_KEY,
    SR_SAMPLING_SPEC_KEY,
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
    build_dit_spec,
    build_sampling_spec,
    check_denoising_request,
    uses_latent_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
    module_dtype,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    TilePlan,
    plan_tiles,
    prepare_lu_tile_latents,
    prepare_pixel_tile_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    VAE_SPATIAL_FACTOR,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import (
    resolve_precision,
)


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
        check_denoising_request(spec, batch.num_inference_steps)
        dit_spec = build_dit_spec(self.transformer)
        device = get_local_torch_device()

        plan, chunks = self._prepare_chunks(batch, spec, dit_spec, device)
        batch.extra[SR_PLAN_KEY] = plan
        batch.extra[SR_CHUNKS_KEY] = chunks
        batch.extra[SR_SAMPLING_SPEC_KEY] = spec
        batch.extra[SR_DIT_SPEC_KEY] = dit_spec
        return batch

    def _prepare_chunks(
        self, batch: Req, spec: SamplingSpec, dit_spec: DitSpec, device: torch.device
    ) -> tuple[TilePlan, list[torch.Tensor]]:
        common = dict(
            visual_size=spec.visual_size,
            tiling_scale=spec.tiling_scale,
            tile_min_overlap=spec.tile_min_overlap,
        )
        if uses_latent_path(self.latent_upscaler, spec.tiling_scale):
            lr_latent = batch.extra.pop(SR_LR_LATENT_KEY)
            plan = plan_tiles(
                frame_hw=(
                    lr_latent.shape[-2] * VAE_SPATIAL_FACTOR,
                    lr_latent.shape[-1] * VAE_SPATIAL_FACTOR,
                ),
                **common,
            )
            with self.use_declared_component(
                component_name="latent_upscaler",
                module=self.latent_upscaler,
                phase="upscale_tiles",
            ) as bank:
                assert bank is not None
                self.latent_upscaler = bank
                chunks = prepare_lu_tile_latents(
                    lr_latent,
                    plan,
                    upscale_fn=partial(bank.upscale, scale=spec.tiling_scale),
                    lu_dtype=module_dtype(bank),
                    scaling_factor=self.vae.scaling_factor,
                    dit_spec=dit_spec,
                    spec=spec,
                    device=device,
                )
            return plan, chunks

        video = batch.extra.pop(SR_VIDEO_KEY)
        plan = plan_tiles(frame_hw=tuple(video.shape[-2:]), **common)
        with self.use_declared_component(
            component_name="vae", module=self.vae, phase="encode_tiles"
        ) as vae:
            assert vae is not None
            self.vae = vae
            chunks = prepare_pixel_tile_latents(
                video,
                plan,
                vae=vae,
                scaling_factor=vae.scaling_factor,
                dit_spec=dit_spec,
                spec=spec,
                device=device,
            )
        return plan, chunks
