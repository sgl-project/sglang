# SPDX-License-Identifier: Apache-2.0
"""LTX's frame callback must preserve the existing decode and final crop."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
    LTXVideoVAEArchConfig,
    LTXVideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sample.joy_echo import JoyEchoSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    _sample_to_uint8_frames,
    _save_video_ffmpeg,
    _try_save_video_with_audio,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
    can_encode_while_decoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joy_echo.setup import (
    JoyEchoMultishotSetupStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
    LTX2AVDecodingStage,
)


def tiny_vae(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    arch = LTXVideoVAEArchConfig(
        latent_channels=4,
        spatial_compression_ratio=2,
        temporal_compression_ratio=2,
        block_out_channels=[8],
        down_block_types=["LTX2VideoDownBlock3D"],
        spatio_temporal_scaling=[True],
        layers_per_block=[1, 1],
        downsample_type=["spatiotemporal"],
        patch_size=1,
        decoder_block_out_channels=[8],
        decoder_spatio_temporal_scaling=[True],
        decoder_layers_per_block=[1, 1],
        decoder_inject_noise=[False, False],
        upsample_residual=[True],
        upsample_factor=[2],
    )
    vae = AutoencoderKLLTX2Video(LTXVideoVAEConfig(arch_config=arch))
    vae.tile_sample_min_num_frames = 8
    vae.tile_sample_stride_num_frames = 4
    return vae.to(device=device, dtype=dtype).eval()


def legacy_temporal_decode(vae, z):
    # Preserve the original two-loop algorithm, independently of the new callback.
    ratio = vae.temporal_compression_ratio
    stride = vae.tile_sample_stride_num_frames
    rows = []
    for start in range(0, z.shape[2], stride // ratio):
        tile = vae.decoder(z[:, :, start : start + 8 // ratio + 1], None)
        rows.append(tile if start == 0 else tile[:, :, :-1])
    kept = []
    for index, tile in enumerate(rows):
        if index:
            tile = vae.blend_t(rows[index - 1], tile, 8 - stride)
        kept.append(tile[:, :, : stride + (index == 0)])
    return torch.cat(kept, dim=2)[:, :, : (z.shape[2] - 1) * ratio + 1]


@pytest.mark.parametrize("batch_size,slicing", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("temporal", [False, True])
@torch.no_grad()
def test_frames_are_final_and_match_legacy_decode(batch_size, slicing, temporal):
    vae = tiny_vae()
    vae.use_framewise_decoding = temporal
    vae.use_slicing = slicing
    z = torch.randn(batch_size, 4, 7, 4, 4)
    if temporal:
        expected = (
            torch.cat([legacy_temporal_decode(vae, sample) for sample in z.split(1)])
            if slicing
            else legacy_temporal_decode(vae, z)
        )
    else:
        expected = vae.decode(z).sample
    chunks = []
    snapshots = []

    def consume(frames):
        chunks.append(frames)
        snapshots.append(frames.clone())

    actual = vae.decode(z, return_dict=False, on_frames=consume)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), actual, rtol=0, atol=0)
    for frames, snapshot in zip(chunks, snapshots):
        torch.testing.assert_close(frames, snapshot, rtol=0, atol=0)
    assert len(chunks) > 1 if temporal and batch_size == 1 else len(chunks) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA encoder")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("audio", [False, True])
@torch.no_grad()
def test_stream_matches_cpu_postprocessing(tmp_path, dtype, audio):
    vae = tiny_vae("cuda", dtype)
    vae.use_framewise_decoding = True
    z = torch.randn(1, 4, 7, 8, 8, device="cuda", dtype=dtype)
    cpu_path, stream_path = tmp_path / "cpu.mp4", tmp_path / "stream.mp4"
    sound = torch.zeros(1, 13000) if audio else None
    request = SimpleNamespace(
        fps=13, num_frames=13, output_compression=None, x264_preset=None
    )
    stream = EncodeWhileDecoding(
        str(stream_path),
        request,
        finish_frames=lambda frames: (frames / 2 + 0.5).clamp(0, 1).float(),
        audio=sound,
        audio_sample_rate=13000 if audio else None,
    )
    decoded = vae.decode(z, on_frames=stream).sample
    assert stream.finish() == [str(stream_path)]
    frames = _sample_to_uint8_frames(
        VideoProcessor().postprocess_video(decoded, output_type="np")[0]
    )
    if audio:
        assert _try_save_video_with_audio(
            save_file_path=str(cpu_path),
            frames=frames,
            fps=13,
            audio=sound,
            audio_sample_rate=13000,
            quality=5,
        )
    else:
        _save_video_ffmpeg(str(cpu_path), frames, fps=13, quality=5)
    assert cpu_path.read_bytes() == stream_path.read_bytes()


@pytest.mark.parametrize(
    "eligible,path,saved",
    [
        (False, None, None),
        (True, None, None),
        (True, "/tmp/clip.mp4", None),
        (True, "/tmp/clip.mp4", ["/tmp/clip.mp4"]),
    ],
)
@pytest.mark.parametrize("initialized,rank", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("enable_memory_bank", [False, True])
def test_audio_order_and_encode_fallback(
    eligible, path, saved, initialized, rank, enable_memory_bank
):
    stage = object.__new__(LTX2AVDecodingStage)
    stage.vae = tiny_vae()
    stage.video_processor = VideoProcessor()
    calls = []
    video = torch.zeros(1, 3, 1, 8, 8)
    request = Req(
        sampling_params=(
            JoyEchoSamplingParams() if enable_memory_bank else SamplingParams()
        ),
        latents=video,
    )
    request.use_diffusion_decoder = False
    request.return_trajectory_decoded = False
    if enable_memory_bank:
        setup = JoyEchoMultishotSetupStage(
            SimpleNamespace(
                multishot_index=0,
                _multishot_session_id=None,
                reset_memory_bank=mock.Mock(),
            )
        )
        setup.forward(request, None)
    module = LTX2AVDecodingStage.__module__
    encoder = mock.Mock()
    encoder.finish.return_value = saved
    with (
        mock.patch.object(stage, "load_model"),
        mock.patch.object(
            stage, "_decode_audio", side_effect=lambda *args: calls.append("audio")
        ),
        mock.patch.object(
            stage,
            "_decode_video",
            side_effect=lambda *args, **kwargs: (calls.append("video"), video)[1],
        ),
        mock.patch(f"{module}.can_encode_while_decoding", return_value=eligible),
        mock.patch(f"{module}.streamed_video_output_path", return_value=path),
        mock.patch(f"{module}.EncodeWhileDecoding", return_value=encoder),
        mock.patch(f"{module}.model_parallel_is_initialized", return_value=initialized),
        mock.patch(
            f"{module}.get_replica_group",
            return_value=SimpleNamespace(rank_in_group=rank),
        ),
        mock.patch.object(
            stage.video_processor,
            "postprocess_video",
            wraps=stage.video_processor.postprocess_video,
        ) as postprocess,
    ):
        result = stage.forward(request, None)
    assert calls == (["audio", "video"] if eligible else ["video", "audio"])
    if saved is not None:
        assert result.output is None
        assert result.output_file_paths == saved
    elif initialized and rank != 0 and not enable_memory_bank:
        assert result.output is None
    else:
        assert result.output.shape == (1, 1, 8, 8, 3)
    assert postprocess.call_count == int(
        saved is None and (enable_memory_bank or not initialized or rank == 0)
    )


@pytest.mark.parametrize("enable_memory_bank", [False, True])
def test_memory_bank_retains_decoded_frames(enable_memory_bank):
    request = Req(
        sampling_params=(
            JoyEchoSamplingParams() if enable_memory_bank else SamplingParams()
        )
    )
    request.save_output = True
    request.return_file_paths_only = True
    request.return_raw_frames = False
    request.data_type = DataType.VIDEO
    request.enable_frame_interpolation = False
    request.enable_upscaling = False
    if enable_memory_bank:
        setup = JoyEchoMultishotSetupStage(
            SimpleNamespace(
                multishot_index=0,
                _multishot_session_id=None,
                reset_memory_bank=mock.Mock(),
            )
        )
        setup.forward(request, None)
    else:
        with pytest.raises(AttributeError):
            request.enable_memory_bank
    server_args = SimpleNamespace(
        enable_torch_compile=False, pipeline_config=PipelineConfig()
    )
    with mock.patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding.current_platform.is_cuda",
        return_value=True,
    ):
        assert can_encode_while_decoding(request, server_args, 1) == (
            not enable_memory_bank
        )
