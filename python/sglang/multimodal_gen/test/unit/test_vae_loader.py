import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn

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
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _backfill_ltx2_audio_vae_latent_stats,
    _match_checkpoint_dtypes,
    _require_native_loader_for_quantized_vae,
    _should_use_channels_last_3d,
)
from sglang.multimodal_gen.runtime.loader.utils import keep_checkpoint_mapped
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    host_memory_budget,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae


class _FakeServerArgs:
    def __init__(self, pipeline_config, num_gpus=1):
        self.pipeline_config = pipeline_config
        self.num_gpus = num_gpus
        self.model_paths = {}
        self.revision = "test-revision"
        self.trust_remote_code = True
        self.layerwise_components = set()

    def resolve_component_attention_backend(self, _component_name):
        return None, None

    def should_start_component_on_cpu(self, _component_name):
        return False

    def should_configure_layerwise_offload_for_lazy_component(self, component_name):
        return component_name in self.layerwise_components


class TestKeepCheckpointMapped(unittest.TestCase):
    """The mapping is for hosts that cannot afford the whole deployment."""

    def test_a_small_deployment_on_a_roomy_host_copies(self):
        with unittest.mock.patch.object(
            host_memory_budget, "host_memory_available_bytes", lambda: 64 * 1024**3
        ):
            self.assertFalse(
                keep_checkpoint_mapped(weight_bytes=3 * 1024**3, component="vae (VAE)"),
                "copies are the faster choice when the host has room: their "
                "pages are resident where a mapping's first use pays a fault",
            )

    def test_a_deployment_larger_than_the_host_stays_mapped(self):
        with unittest.mock.patch.object(
            host_memory_budget, "host_memory_available_bytes", lambda: 19 * 1024**3
        ):
            self.assertTrue(
                keep_checkpoint_mapped(
                    weight_bytes=117 * 1024**3, component="vae (VAE)"
                )
            )


class TestMatchCheckpointDtypes(unittest.TestCase):
    """Assignment replaces a parameter, so only matching dtypes may stay mapped."""

    def test_a_matching_tensor_is_left_alone(self):
        loaded = {"w": torch.zeros(4, dtype=torch.float32)}
        before = loaded["w"]
        _match_checkpoint_dtypes(loaded, {"w": torch.zeros(4, dtype=torch.float32)})
        self.assertIs(loaded["w"], before)

    def test_a_mismatched_tensor_is_converted(self):
        loaded = {"w": torch.zeros(4, dtype=torch.float32)}
        _match_checkpoint_dtypes(loaded, {"w": torch.zeros(4, dtype=torch.bfloat16)})
        self.assertEqual(loaded["w"].dtype, torch.bfloat16)

    def test_a_tensor_the_module_does_not_want_is_left_alone(self):
        loaded = {"extra": torch.zeros(4, dtype=torch.float32)}
        before = loaded["extra"]
        _match_checkpoint_dtypes(loaded, {})
        self.assertIs(loaded["extra"], before)


class TestVAELoader(unittest.TestCase):
    def test_mps_layerwise_load_uses_residency_api(self):
        loader = vae_loader.VAELoader()
        server_args = _FakeServerArgs(QwenImagePipelineConfig())
        server_args.layerwise_components.add("vae")

        with patch.object(vae_loader.current_platform, "is_mps", return_value=True):
            self.assertEqual(
                loader.customized_load_kwargs_for_component(server_args, "vae"),
                {"cpu_offload_flag": True},
            )
            self.assertEqual(
                loader.customized_load_kwargs_for_component(server_args, "audio_vae"),
                {},
            )

    def test_quantized_vae_admission_leaves_plain_configs_unchanged(self):
        _require_native_loader_for_quantized_vae(
            {"_class_name": "AutoencoderKL"}, "vae"
        )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "compression_config"
        ):
            _require_native_loader_for_quantized_vae(
                {
                    "_class_name": "AutoencoderKL",
                    "compression_config": {"quant_method": "compressed-tensors"},
                },
                "vae",
            )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            r"text_config\.quantization_config",
        ):
            _require_native_loader_for_quantized_vae(
                {
                    "_class_name": "AutoencoderKL",
                    "text_config": {
                        "quantization_config": {
                            "quant_method": "bitsandbytes",
                            "load_in_4bit": True,
                        }
                    },
                },
                "vae",
            )

    def test_quantized_vae_routes_to_diffusers_native_loader(self):
        loader = vae_loader.VAELoader()
        server_args = _FakeServerArgs(QwenImagePipelineConfig())
        native_vae = nn.Linear(1, 1)

        with (
            TemporaryDirectory() as component_path,
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value={
                    "_class_name": "AutoencoderKL",
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_4bit": True,
                    },
                },
            ),
            patch(
                "diffusers.AutoModel.from_pretrained",
                return_value=native_vae,
            ) as native_load,
            patch.object(loader, "target_device", return_value=torch.device("cpu")),
            patch.object(native_vae, "to", wraps=native_vae.to) as module_to,
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                side_effect=[10.0, 9.0],
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.get_memory_usage_of_component",
                return_value=1.0,
            ),
        ):
            loaded, consumed = loader.load(
                component_path, server_args, "vae", "diffusers"
            )

        self.assertIs(loaded, native_vae)
        self.assertFalse(loaded.training)
        self.assertEqual(consumed, 1.0)
        self.assertEqual(server_args.model_paths["vae"], component_path)
        native_load.assert_called_once_with(
            component_path,
            revision="test-revision",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        module_to.assert_called_once_with(torch.device("cpu"))

    def test_native_only_quantized_vae_fails_closed(self):
        pipeline_config = QwenImagePipelineConfig()
        pipeline_config.native_only_components = ("vae",)
        server_args = _FakeServerArgs(pipeline_config)
        loader = vae_loader.VAELoader()

        with (
            patch.object(
                vae_loader,
                "get_diffusers_component_config",
                return_value={
                    "_class_name": "AutoencoderKL",
                    "quantization_config": {
                        "quant_method": "bitsandbytes",
                        "load_in_4bit": True,
                    },
                },
            ),
            patch("diffusers.AutoModel.from_pretrained") as native_load,
            patch.object(
                vae_loader.current_platform,
                "get_available_gpu_memory",
                return_value=10.0,
            ),
        ):
            with self.assertRaisesRegex(
                ComponentCheckpointUnsupportedError, "native-only SGLang"
            ):
                loader.load("/quantized/vae", server_args, "vae", "diffusers")

        native_load.assert_not_called()

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
