# SPDX-License-Identifier: Apache-2.0
"""Encoding Wan video while its VAE decodes must not change the MP4.

The frame hand-off itself is covered in
python/sglang/multimodal_gen/test/unit/test_wan_vae_on_frames.py.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import (
    HunyuanVAEArchConfig,
    HunyuanVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import HunyuanConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import _try_save_cuda_video_direct
from sglang.multimodal_gen.runtime.models.vaes.hunyuanvae import (
    AutoencoderKLHunyuanVideo,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

FPS = 16
Z_DIM = 4


def _vae_config(*, wan22: bool) -> WanVAEConfig:
    """A tiny Wan 2.1 VAE, or with ``wan22`` the 2.2 residual blocks and 2x2 patches."""
    channels = 12 if wan22 else 3
    return WanVAEConfig(
        arch_config=WanVAEArchConfig(
            base_dim=8,
            z_dim=Z_DIM,
            dim_mult=(1, 2, 2, 2),
            num_res_blocks=1,
            latents_mean=(0.0,) * Z_DIM,
            latents_std=(0.5,) * Z_DIM,
            is_residual=wan22,
            in_channels=channels,
            out_channels=channels,
            patch_size=2 if wan22 else None,
        ),
        load_encoder=False,
    )


def _vae(config: WanVAEConfig, device: str = "cpu") -> AutoencoderKLWan:
    torch.manual_seed(0)
    return AutoencoderKLWan(config).to(device).eval()


def _server_args(pipeline_config=None) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=pipeline_config or WanT2V480PConfig(),
        disable_autocast=False,
        enable_torch_compile=False,
    )


def _stage(vae, server_args, stage_cls=DecodingStage) -> DecodingStage:
    with mock.patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=server_args,
    ):
        return stage_cls(vae)


def _video_request(**overrides) -> SimpleNamespace:
    fields = dict(
        save_output=True,
        return_file_paths_only=True,
        return_raw_frames=False,
        return_trajectory_decoded=False,
        data_type=DataType.VIDEO,
        enable_frame_interpolation=False,
        enable_upscaling=False,
        extra={},
        latents=torch.zeros(1, Z_DIM, 2, 2, 2),
        fps=FPS,
        num_frames=5,
        output_compression=None,
        x264_preset=None,
        output_file_path=lambda num_outputs, output_idx: "/tmp/clip.mp4",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@unittest.skipUnless(condition=torch.cuda.is_available(), reason="needs CUDA")
class TestStreamedEncode(CustomTestCase):
    @torch.no_grad()
    def test_hunyuan_without_callback_matches_one_shot_mp4(self):
        config = HunyuanVAEConfig(
            arch_config=HunyuanVAEArchConfig(
                latent_channels=4,
                block_out_channels=(8, 8, 8),
                down_block_types=("HunyuanVideoDownBlock3D",) * 3,
                up_block_types=("HunyuanVideoUpBlock3D",) * 3,
                layers_per_block=1,
                norm_num_groups=4,
                temporal_compression_ratio=4,
                mid_block_add_attention=False,
            ),
            load_encoder=False,
        )
        server_args = _server_args(HunyuanConfig(vae_config=config))
        stage = _stage(AutoencoderKLHunyuanVideo(config).cuda().eval(), server_args)
        with tempfile.TemporaryDirectory() as tmp:
            streamed, one_shot = Path(tmp, "stream.mp4"), Path(tmp, "normal.mp4")
            stream = EncodeWhileDecoding(str(streamed), _video_request())
            frames = stage.decode(
                torch.randn(1, 4, 2, 10, 12, device="cuda"),
                server_args,
                vae_dtype=torch.float32,
                on_frames=stream,
            )
            self.assertEqual(stream.finish(), [str(streamed)])
            self.assertTrue(
                _try_save_cuda_video_direct(
                    save_file_path=str(one_shot),
                    sample=frames[0],
                    fps=FPS,
                    audio_sample_rate=None,
                    output_compression=None,
                )
            )
            self.assertEqual(one_shot.read_bytes(), streamed.read_bytes())

    @torch.no_grad()
    def test_streamed_mp4_matches_the_one_shot_save(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        # Wan 2.1 decodes under bf16 autocast, Wan 2.2 in fp32; 40x48 is not a
        # macroblock multiple, so this also covers the scale filter
        for wan22, vae_dtype, latent_size in (
            (False, torch.bfloat16, (5, 6)),
            (True, torch.float32, (3, 4)),
        ):
            config = _vae_config(wan22=wan22)
            server_args = _server_args(WanT2V480PConfig(vae_config=config))
            stage = _stage(_vae(config, "cuda"), server_args)
            latents = torch.randn(
                1, Z_DIM, 6, *latent_size, device="cuda", generator=generator
            )
            with (
                self.subTest(wan22=wan22),
                tempfile.TemporaryDirectory() as tmp,
            ):
                one_shot, streamed = (
                    Path(tmp, "one_shot.mp4"),
                    Path(tmp, "streamed.mp4"),
                )
                stream = EncodeWhileDecoding(
                    str(streamed), _video_request(num_frames=21)
                )
                frames = stage.decode(
                    latents, server_args, vae_dtype=vae_dtype, on_frames=stream
                )
                self.assertEqual(stream.finish(), [str(streamed)])
                self.assertEqual(tuple(frames.shape[1:3]), (3, 21))
                self.assertTrue(
                    _try_save_cuda_video_direct(
                        save_file_path=str(one_shot),
                        sample=frames[0],
                        fps=FPS,
                        audio_sample_rate=None,
                        output_compression=None,
                    )
                )
                self.assertEqual(one_shot.read_bytes(), streamed.read_bytes())

    @unittest.skipUnless(
        current_platform.is_cuda(), "streaming is gated on the CUDA platform, not ROCm"
    )
    def test_only_file_path_video_requests_stream(self):
        server_args = _server_args()
        stage = _stage(_vae(_vae_config(wan22=False)), server_args)
        self.assertIsInstance(
            stage._encode_while_decoding(_video_request(), server_args),
            EncodeWhileDecoding,
        )
        for field, value in (
            ("return_file_paths_only", False),
            ("return_raw_frames", True),
            ("return_trajectory_decoded", True),
            ("data_type", DataType.IMAGE),
            ("enable_frame_interpolation", True),
            ("enable_upscaling", True),
            ("extra", {"dynamic_batch_output_paths": ["a.mp4"]}),
            ("latents", torch.zeros(2, Z_DIM, 2, 2, 2)),
        ):
            with self.subTest(field=field):
                request = _video_request(**{field: value})
                self.assertIsNone(stage._encode_while_decoding(request, server_args))

        class OwnPostDecoding(WanT2V480PConfig):
            def post_decoding(self, frames, server_args):
                return frames

        class OwnDecode(DecodingStage):
            def decode(self, latents, server_args, *, vae_dtype):
                return latents

        for name, candidate, args in (
            ("post_decoding", stage, _server_args(OwnPostDecoding())),
            (
                "torch_compile",
                stage,
                SimpleNamespace(**{**vars(server_args), "enable_torch_compile": True}),
            ),
            ("stage_decode", _stage(stage.vae, server_args, OwnDecode), server_args),
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    candidate._encode_while_decoding(_video_request(), args)
                )

    @torch.no_grad()
    def test_decoder_without_callback_hands_over_the_complete_clip(self):
        config = _vae_config(wan22=False)
        server_args = _server_args(WanT2V480PConfig(vae_config=config))
        stage = _stage(_vae(config, "cuda"), server_args)
        # The shared stage must not inject a keyword into an ordinary decoder.
        stage.vae.supports_decode_on_frames = False
        self.assertIsNone(stage._encode_while_decoding(_video_request(), server_args))
        expected = torch.randn(1, 3, 5, 40, 48, device="cuda")
        decode = mock.Mock(side_effect=lambda latents: expected)
        for output in (expected, expected[:, :, 0]):
            with self.subTest(shape=output.shape):
                decode.side_effect = lambda latents: output
                received = []
                with mock.patch.object(
                    stage, "_get_vae_decode_fn", return_value=decode
                ):
                    actual = stage.decode(
                        torch.zeros(1, Z_DIM, 2, 2, 2, device="cuda"),
                        server_args,
                        vae_dtype=torch.float32,
                        on_frames=received.append,
                    )
                self.assertEqual(len(received), int(output.ndim == 5))
                if received:
                    torch.testing.assert_close(received[0], actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
