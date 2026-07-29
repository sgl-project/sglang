import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_2_TI2V_5B_Config,
    Wan2_2_I2V_A14B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import vae_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _backfill_ltx2_audio_vae_latent_stats,
    _should_use_channels_last_3d,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae


class _FakeServerArgs:
    def __init__(self, pipeline_config, num_gpus=1):
        self.pipeline_config = pipeline_config
        self.num_gpus = num_gpus


class TestVAELoader(unittest.TestCase):
    def test_backfill_ltx2_audio_vae_latent_stats_maps_official_keys(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([3.0, 4.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_does_not_override_existing(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0, 2.0]),
            "per_channel_statistics.std-of-means": torch.tensor([3.0, 4.0]),
            "latents_mean": torch.tensor([5.0, 6.0]),
            "latents_std": torch.tensor([7.0, 8.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "audio_vae")

        self.assertTrue(torch.equal(loaded["latents_mean"], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(loaded["latents_std"], torch.tensor([7.0, 8.0])))

    def test_backfill_ltx2_audio_vae_latent_stats_skips_non_audio_vae(self):
        loaded = {
            "per_channel_statistics.mean-of-means": torch.tensor([1.0]),
            "per_channel_statistics.std-of-means": torch.tensor([2.0]),
        }

        _backfill_ltx2_audio_vae_latent_stats(loaded, "vae")

        self.assertNotIn("latents_mean", loaded)
        self.assertNotIn("latents_std", loaded)

    def test_channels_last_3d_defaults_true_for_qwen_image_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertTrue(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_fast_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(FastWan2_2_TI2V_5B_Config(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_wan_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(Wan2_2_I2V_A14B_Config(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_true_for_single_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=1)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_defaults_false_for_multi_gpu_ltx_on_cuda(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertFalse(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_can_be_disabled_by_env(self):
        with (
            patch.dict(
                "os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "false"}
            ),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    def test_channels_last_3d_can_be_enabled_by_env(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "true"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)
            self.assertTrue(_should_use_channels_last_3d(server_args, "video_vae"))

    def test_channels_last_3d_auto_uses_model_policy(self):
        with (
            patch.dict("os.environ", {"SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D": "auto"}),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            wan_args = _FakeServerArgs(WanT2V480PConfig(), num_gpus=1)
            ltx_args = _FakeServerArgs(LTX2PipelineConfig(), num_gpus=2)

            self.assertTrue(_should_use_channels_last_3d(wan_args, "video_vae"))
            self.assertFalse(_should_use_channels_last_3d(ltx_args, "video_vae"))

    def test_channels_last_3d_skips_non_video_vae_components(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=True),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "audio_vae"))

    def test_channels_last_3d_skips_unsupported_platforms(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(vae_loader.current_platform, "is_cuda", return_value=False),
            patch.object(vae_loader.current_platform, "is_rocm", return_value=False),
        ):
            server_args = _FakeServerArgs(QwenImagePipelineConfig())
            self.assertFalse(_should_use_channels_last_3d(server_args, "vae"))

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_skips_non_cuda_platforms(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=False),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertIs(out, x)

    @unittest.skipUnless(
        hasattr(torch, "channels_last_3d"), "channels_last_3d is unavailable"
    )
    def test_match_conv3d_input_format_uses_channels_last_3d_on_cuda(self):
        x = torch.randn(1, 3, 2, 4, 4)
        weight = torch.randn(3, 3, 1, 1, 1).contiguous(
            memory_format=torch.channels_last_3d
        )

        with (
            patch.object(wanvae.current_platform, "is_cuda", return_value=True),
            patch.object(wanvae.current_platform, "is_rocm", return_value=False),
        ):
            out = wanvae.match_conv3d_input_format(x, weight)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))


if __name__ == "__main__":
    unittest.main()
