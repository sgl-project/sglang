# SPDX-License-Identifier: Apache-2.0
"""Tiny random Kandinsky 6 SR components and requests for the stage-level tests."""

import copy
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_REQUESTED_HW_KEY,
    SR_TILING_SCALE_KEY,
    SR_VIDEO_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
    denoise_chunks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    LatentUpscaleFn,
    decode_chunks,
    encode_pixel_tile,
    encode_video_to_lr_latent,
    plan_tiles,
    prepare_tile_latents,
    stitch_tiles,
    upscale_lr_latent_tile,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    RESOLUTIONS,
    VAE_SPATIAL_FACTOR,
)

# official transformer layout: total output width, scheduler configured separately
TINY_DIT = dict(
    in_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=16,
    out_visual_dim=12,
    patch_size=[1, 2, 2],
    model_dim=32,
    ff_dim=64,
    num_text_blocks=0,
    num_visual_blocks=2,
    axes_dims=[8, 4, 4],
    visual_cond=False,
    instruct_type="noise",
    use_text=False,
    attribute_overrides=None,
    sr_params=dict(visual_size=[512], scale_factor={"512": [1.0, 1.0, 1.0]}),
)
TINY_PIFLOW = dict(
    n_grid=3,
    nfe=2,
    num_policy_substeps=8,
    final_step_size_scale=0.5,
    shift=5.0,
    eps=1e-6,
)
TINY_KVAE = dict(
    ch=8,
    ch_mult=(1, 1, 2, 2, 2),
    num_res_blocks=2,
    z_channels=4,
    temporal_compress_times=4,
    norm_type="rms_norm",
)
TINY_LU_MODEL = {
    "architecture": "multi_scale",
    "in_channels": 4,
    "hidden_channels": 8,
    "num_pre_blocks": 1,
    "num_mid_blocks": 2,
    "num_post_blocks": 2,
    "expand_ratio": 2,
    "dims": 3,
    "bare_stem": True,
    "input_skip": False,
    "modulated_norm": True,
    "modulated_output_proj": True,
    "temporal_padding": "replicate",
    "upsample_mode": "pxs_v2",
    "upsample_padding_mode": "zeros",
    "enable_x2_entry": True,
    "x2_adapter_blocks": 1,
    "x2_tail_mode": "private_full",
    "x2_finisher": "pxs_residual",
}


def _randomize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)


def build_components(bank_scales, dit_arch=TINY_DIT):
    """``(vae, dit, latent-upscaler bank or None, server args)`` with random weights."""
    vae_config = Kandinsky6SRVAEConfig()
    vae_config.update_model_arch(
        dict(
            vae_type="video-kvae",
            encoder_config=dict(TINY_KVAE, in_channels=3),
            decoder_config=dict(TINY_KVAE, out_ch=3),
            scaling_factor=0.5,
        )
    )
    vae = Kandinsky6SRVAE(vae_config).eval()
    _randomize(vae, 1)
    dit_config = Kandinsky6SRDitConfig()
    dit_config.update_model_arch(dict(dit_arch))
    dit = Kandinsky6SRTransformer3DModel(dit_config, dict(dit_arch)).eval()
    _randomize(dit, 2)
    bank = None
    if bank_scales:
        models = [
            {"target_scale": scale, "model": copy.deepcopy(TINY_LU_MODEL)}
            for scale in bank_scales
        ]
        bank = Kandinsky6SRLatentUpscalerBank(
            models,
            0.5,
            scales=tuple(int(scale.removesuffix("x")) for scale in bank_scales),
        ).eval()
        _randomize(bank, 3)
    # keep initialization reproducible on CPU, then exercise the native device kernels
    device = get_local_torch_device()
    vae.to(device)
    dit.to(device)
    if bank is not None:
        bank.to(device)
    pipeline_config = Kandinsky6SRPipelineConfig()
    pipeline_config.dit_config = dit_config
    server_args = SimpleNamespace(
        pipeline_config=pipeline_config,
        component_precisions={},
        # Kandinsky6SRDenoisingStage reuses the shared DenoisingStage's cache-dit /
        # torch.compile wiring (_maybe_enable_cache_dit_and_torch_compile), which reads
        # these; both stay off for this fixture.
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
    )
    return vae, dit, bank, server_args


def make_stage(stage_cls, server_args, *args, **kwargs):
    stage = stage_cls(*args, **kwargs)
    stage.server_args = server_args  # component_uses() reads the bf16 precisions
    return stage


def make_request(video, *, tiles_batch_size=1, num_steps=5):
    params = Kandinsky6SRSamplingParams(
        video_path="unused.mp4",
        sr_resolution_scale=2,
        sr_tiles_batch_size=tiles_batch_size,
        num_inference_steps=num_steps,
        seed=42,
    )
    batch = Req(sampling_params=params)
    batch.extra[SR_VIDEO_KEY] = video.clone()
    batch.extra[SR_TILING_SCALE_KEY] = 2
    batch.extra[SR_REQUESTED_HW_KEY] = (video.shape[-2], video.shape[-1])
    return batch


def random_video(frames, height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (frames, 3, height, width)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)


@torch.no_grad()
def super_resolve(
    video: torch.Tensor,
    *,
    vae: torch.nn.Module,
    scaling_factor: float,
    dit: Callable[..., torch.Tensor],
    dit_spec: DitSpec,
    spec: SamplingSpec,
    scheduler: Any,
    device: torch.device | str,
    upscale_fn: LatentUpscaleFn | None = None,
    lu_dtype: torch.dtype | None = None,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
    resolutions: Mapping[int, Sequence[tuple[int, int]]] = RESOLUTIONS,
) -> torch.Tensor:
    """Tiled SR of a ``[T, C, H, W]`` uint8 video -> uint8 ``[3, T, H*s, W*s]``.

    ``upscale_fn`` selects the LU path; without it the pixel path runs.
    """
    device = torch.device(device)
    common = dict(
        visual_size=spec.visual_size,
        tiling_scale=spec.tiling_scale,
        tile_min_overlap=spec.tile_min_overlap,
        spatial_factor=spatial_factor,
        resolutions=resolutions,
    )
    if upscale_fn is not None:
        lr_latent = encode_video_to_lr_latent(video, vae, device=device).cpu()
        frame_hw = (
            lr_latent.shape[-2] * spatial_factor,
            lr_latent.shape[-1] * spatial_factor,
        )
        plan = plan_tiles(frame_hw=frame_hw, **common)
        source = lr_latent
        tile_encoder = partial(
            upscale_lr_latent_tile,
            upscale_fn=upscale_fn,
            lu_dtype=lu_dtype,
            scaling_factor=scaling_factor,
            device=device,
        )
    else:
        plan = plan_tiles(frame_hw=tuple(video.shape[-2:]), **common)
        source = video
        tile_encoder = partial(
            encode_pixel_tile,
            vae=vae,
            scaling_factor=scaling_factor,
            device=device,
        )
    chunks = prepare_tile_latents(
        source,
        plan,
        tile_encoder=tile_encoder,
        latent_path=upscale_fn is not None,
        dit_spec=dit_spec,
        spec=spec,
        device=device,
        spatial_factor=spatial_factor,
    )
    denoised = denoise_chunks(
        chunks,
        dit,
        scheduler,
        dit_spec=dit_spec,
        spec=spec,
        device=device,
        step_context=lambda step: set_forward_context(
            current_timestep=step, attn_metadata=None
        ),
    )
    tiles = decode_chunks(denoised, vae, scaling_factor=scaling_factor, device=device)
    return stitch_tiles(tiles, plan)
