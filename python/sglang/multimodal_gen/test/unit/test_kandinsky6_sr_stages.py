# SPDX-License-Identifier: Apache-2.0
"""Stage-level behaviour of Kandinsky 6 video SR on tiny random components.

The tiled algorithm itself is pinned against the reference in
``test_kandinsky6_sr_reference_parity.py``; here the four model-phase stages (encode,
latent-prep, denoising, decode) plus the output stage must reproduce the pure orchestration
function (``tiled.super_resolve``) through ``Req`` / ``batch.extra`` hand-offs, phase by phase.
"""

import copy
import os
from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
from kandinsky6_sr_tiny_components import (
    TINY_LU_MODEL,
    build_components,
    make_request,
    make_stage,
    random_video,
)

from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.decode_stage import (
    Kandinsky6SRDecodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.denoising_stage import (
    Kandinsky6SRDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.encode_stage import (
    Kandinsky6SREncodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.input_stage import (
    WARMUP_CLIP_FRAMES,
    WARMUP_CLIP_HW,
    Kandinsky6SRInputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latent_prep_stage import (
    Kandinsky6SRLatentPrepStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.output_stage import (
    Kandinsky6SROutputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_LR_LATENT_KEY,
    SR_PLAN_KEY,
    SR_REQUESTED_HW_KEY,
    SR_TILES_KEY,
    SR_VIDEO_KEY,
    build_dit_spec,
    build_sampling_spec,
    effective_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    module_dtype,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    TilePlan,
    super_resolve,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    RESOLUTIONS,
)


@pytest.fixture(scope="module", autouse=True)
def single_process_model_parallel():
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29509",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@pytest.fixture(autouse=True)
def tiny_resolutions(monkeypatch):
    monkeypatch.setitem(RESOLUTIONS, 512, [(64, 64), (64, 96), (96, 64)])


@pytest.mark.parametrize(
    "bank_scales",
    [("2x", "4x"), (), ("4x",)],
    ids=["lu_path", "pixel_path", "pixel_fallback"],
)
def test_stage_chain_reproduces_the_pure_orchestration(bank_scales):
    """encode -> latent-prep -> denoise -> decode -> output must give what
    ``super_resolve`` gives on the same clip, for the LU path, the pixel path, and a bank
    that has no entry for the requested scale (which must fall back to the pixel path, not
    fail). The 64x128 clip holds 12 overlapping 32x48 tiles, denoised in chunks of 5, 5 and
    2."""
    vae, dit, bank, server_args = build_components(bank_scales)
    video = random_video(9, 64, 128)
    batch = make_request(video, tiles_batch_size=5)

    encode = make_stage(Kandinsky6SREncodeStage, server_args, vae, bank)
    latent_prep = make_stage(
        Kandinsky6SRLatentPrepStage, server_args, vae, dit, bank, None
    )
    denoise = make_stage(Kandinsky6SRDenoisingStage, server_args, dit, None)
    decode = make_stage(Kandinsky6SRDecodeStage, server_args, vae)
    output = make_stage(Kandinsky6SROutputStage, server_args)

    batch = encode.forward(batch, server_args)
    assert (SR_LR_LATENT_KEY in batch.extra) == ("2x" in bank_scales)
    batch = latent_prep.forward(batch, server_args)
    assert SR_LR_LATENT_KEY not in batch.extra and SR_VIDEO_KEY not in batch.extra
    assert len(batch.extra["kandinsky6_sr_chunks"]) == 3  # 12 tiles / 5 per chunk
    assert isinstance(batch.extra[SR_PLAN_KEY], TilePlan)
    batch = denoise.forward(batch, server_args)
    batch = decode.forward(batch, server_args)
    assert len(batch.extra[SR_TILES_KEY]) == 12
    result = output.forward(batch, server_args)

    arch = dit.config
    use_lu = "2x" in bank_scales
    spec = build_sampling_spec(
        arch=arch,
        tiling_scale=2,
        seed=42,
        num_steps=5,
        tiles_batch_size=5,
        tile_min_overlap=0.2,
    )
    expected = super_resolve(
        video,
        vae=vae,
        scaling_factor=vae.scaling_factor,
        dit=dit,
        dit_spec=build_dit_spec(dit),
        spec=spec,
        scheduler=effective_scheduler(spec, None),
        device=next(dit.parameters()).device,
        upscale_fn=partial(bank.upscale, scale=2) if use_lu else None,
        lu_dtype=module_dtype(bank) if use_lu else None,
    )
    assert isinstance(result, OutputBatch)
    assert result.output.dtype == torch.float16
    assert result.output.shape == (1, 3, 9, 128, 256)
    restored = (result.output[0].float() * 255).to(torch.uint8)
    assert torch.equal(restored, expected)
    assert (batch.height, batch.width) == (128, 256)
    assert SR_TILES_KEY not in batch.extra and SR_PLAN_KEY not in batch.extra


def test_encode_stage_leaves_the_vae_alone_on_the_pixel_path():
    """No LU entry for the scale -> the whole-video encode is skipped, and the VAE (which
    a CPU-offloaded run would otherwise move to the GPU for nothing) is never touched.
    """
    vae = MagicMock(name="vae")
    bank = Kandinsky6SRLatentUpscalerBank(
        [{"target_scale": "4x", "model": copy.deepcopy(TINY_LU_MODEL)}],
        0.5,
        scales=(4,),
    )
    _, _, _, server_args = build_components(())
    video = random_video(9, 64, 64)
    batch = make_request(video)
    stage = make_stage(Kandinsky6SREncodeStage, server_args, vae, bank)
    assert stage.forward(batch, server_args) is batch
    assert torch.equal(batch.extra[SR_VIDEO_KEY], video)
    assert SR_LR_LATENT_KEY not in batch.extra
    assert vae.method_calls == [] and not vae.called


def test_input_stage_runs_on_a_synthetic_clip_for_warmup_requests():
    """Warmup requests carry no video (synthetic warmup) or a real one that must not be
    decoded twice (request warmup): both run the real path on a small synthetic clip,
    without audio, and without needing a prompt."""
    params = Kandinsky6SRSamplingParams(video_path="missing_on_purpose.mp4")
    batch = Req(sampling_params=params)
    batch.is_warmup = True
    Kandinsky6SRInputStage().forward(batch, MagicMock())

    video = batch.extra[SR_VIDEO_KEY]
    height, width = WARMUP_CLIP_HW
    # default scale 2.25 = x1.125 pre-upscale (aligned to 16) then tiling scale 2
    assert video.shape[0] == WARMUP_CLIP_FRAMES and video.dtype == torch.uint8
    assert video.shape[2:] == (288, 432)
    # 288x432 is already a multiple of 16, so alignment padding is a no-op here, and the
    # requested size the output stage will crop back to equals the padded (encoded) size.
    assert batch.extra[SR_REQUESTED_HW_KEY] == (288, 432)
    assert batch.extra["kandinsky6_sr_tiling_scale"] == 2
    assert (batch.fps, batch.num_frames) == (24, WARMUP_CLIP_FRAMES)
    assert (batch.height, batch.width) == (288 * 2, 432 * 2)
    assert batch.audio is None and height == 256


def test_input_stage_pads_a_misaligned_source_to_the_spatial_factor():
    """A source whose pre-upscaled size is not already a multiple of 16 (every integer
    scale, which runs no pixel resize at all) must still be padded before whole-video KVAE
    encoding, and the *requested* (pre-pad) size recorded for the output stage to crop back
    to -- the fix for "whole-video KVAE encoding fails at the fourth PixelUnshuffle" for
    scales other than 2.25."""
    params = Kandinsky6SRSamplingParams(
        video_path="missing_on_purpose.mp4", sr_resolution_scale=2
    )
    batch = Req(sampling_params=params)
    batch.is_warmup = True
    Kandinsky6SRInputStage().forward(batch, MagicMock())

    # WARMUP_CLIP_HW is already 16-aligned, so exercise a genuinely misaligned size by
    # checking the padding helper directly against the same video the stage produced, cut
    # down to a misaligned size.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
        pad_to_spatial_factor,
    )

    misaligned = batch.extra[SR_VIDEO_KEY][:, :, :100, :70]
    padded, requested_hw = pad_to_spatial_factor(misaligned, 16)
    assert requested_hw == (100, 70)
    assert padded.shape[-2] % 16 == 0 and padded.shape[-1] % 16 == 0
    assert torch.equal(padded[:, :, :100, :70], misaligned)


def test_input_stage_rejects_a_missing_video_file():
    batch = Req(sampling_params=Kandinsky6SRSamplingParams(video_path="nope.mp4"))
    with pytest.raises(ValueError, match="not found"):
        Kandinsky6SRInputStage().forward(batch, MagicMock())


def test_output_stage_stitches_crops_resizes_and_carries_the_source_audio():
    """The output stage blends tiles, restores the originally requested (pre-alignment-pad)
    size, applies the delivery resize (fit keeps the aspect ratio and never upscales),
    publishes the *final* size and fps on the request / result, and hands the source audio
    through unchanged."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
        compute_tile_grid_even,
    )

    grid = compute_tile_grid_even(32, 48, (16, 24), 0.2, 16)
    plan = TilePlan(
        tiling_scale=2,
        frame_hw=(32, 48),
        base_hw=(32, 48),
        tile_hw=(16, 24),
        pixel_grid=grid,
    )
    tiles = [
        torch.full((3, 9, 32, 48), 100 + i, dtype=torch.uint8)
        for i in range(grid.total_tiles)
    ]
    audio = torch.linspace(-1, 1, 1000)

    params = Kandinsky6SRSamplingParams(
        video_path="unused.mp4",
        sr_target_resolution="48x32",
        sr_target_resize_mode="fit",
    )
    batch = Req(sampling_params=params)
    batch.extra[SR_TILES_KEY], batch.extra[SR_PLAN_KEY] = tiles, plan
    batch.extra[SR_REQUESTED_HW_KEY] = (32, 48)  # no alignment padding happened
    batch.audio, batch.audio_sample_rate = audio, 44100
    batch.fps = 12

    result = Kandinsky6SROutputStage().forward(batch, MagicMock())

    assert result.output.shape == (1, 3, 9, 32, 48)  # 64x96 fitted into 48x32
    assert (batch.height, batch.width) == (32, 48)
    assert result.audio is audio and result.audio_sample_rate == 44100
    assert result.fps == 12  # the worker's effective fps reaches OutputBatch
    plain = Kandinsky6SRSamplingParams(video_path="unused.mp4")
    batch = Req(sampling_params=plain)
    batch.extra[SR_TILES_KEY], batch.extra[SR_PLAN_KEY] = tiles, plan
    assert Kandinsky6SROutputStage().forward(batch, MagicMock()).output.shape == (
        1,
        3,
        9,
        64,
        96,
    )


def test_output_stage_crops_the_alignment_padding_back_off():
    """A plan whose stitched frame is larger than what was actually requested (the input
    stage padded for alignment) must be cropped down to the requested size before any
    delivery resize -- not left at the padded size."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
        compute_tile_grid_even,
    )

    # Padded frame is 32x48 (aligned), but only 25x40 pixels were actually requested.
    grid = compute_tile_grid_even(32, 48, (16, 24), 0.2, 16)
    plan = TilePlan(
        tiling_scale=2,
        frame_hw=(32, 48),
        base_hw=(32, 48),
        tile_hw=(16, 24),
        pixel_grid=grid,
    )
    tiles = [
        torch.full((3, 9, 32, 48), 100 + i, dtype=torch.uint8)
        for i in range(grid.total_tiles)
    ]
    batch = Req(sampling_params=Kandinsky6SRSamplingParams(video_path="unused.mp4"))
    batch.extra[SR_TILES_KEY], batch.extra[SR_PLAN_KEY] = tiles, plan
    batch.extra[SR_REQUESTED_HW_KEY] = (25, 40)

    result = Kandinsky6SROutputStage().forward(batch, MagicMock())

    assert result.output.shape == (1, 3, 9, 50, 80)  # (25, 40) * tiling_scale(2)
    assert (batch.height, batch.width) == (50, 80)
