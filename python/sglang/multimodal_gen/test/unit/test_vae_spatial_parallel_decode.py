import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.vaes.base import (
    VAEConfig,
    is_spatial_shard_parallel_decode_mode,
    should_use_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.configs.models.vaes.ernie_image import ErnieImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig, FluxVAEConfig
from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_audio import LTXAudioVAEConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
from sglang.multimodal_gen.configs.utils import update_config_from_args
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelCausalConv3d,
    SpatialParallelConv2d,
    SpatialParallelConv3d,
    chunk_height_by_sizes,
    split_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage import (
    QwenImageAttentionBlock,
    QwenImageDecoder3d,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vaes.hunyuanvae import (
    HunyuanVideoDecoder3D,
    HunyuanVideoMidBlock3D,
    HunyuanVideoResnetBlockCausal3D,
    _enable_hunyuan_decoder_spatial_parallel,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    LTX2VideoCausalConv3d,
    LTX2VideoDecoder3d,
    _enable_ltx_decoder_spatial_parallel,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanDecoder3d,
    WanDistAttentionBlock,
)
from sglang.multimodal_gen.runtime.utils.distributed import RankGenerator
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class _DispatchProbeVAE(ParallelTiledVAE):
    def __init__(self, config):
        super().__init__(config)
        self.used_decode = False
        self.used_parallel_tiled_decode = False

    def _encode(self, x):
        return x

    def _decode(self, z):
        self.used_decode = True
        return z

    def parallel_tiled_decode(self, z):
        self.used_parallel_tiled_decode = True
        raise AssertionError(
            "spatial_shard decode should not use parallel_tiled_decode"
        )


class TestVAESpatialParallelDecode(unittest.TestCase):
    def test_base_vae_config_defaults_to_auto_parallel_decode(self):
        config = VAEConfig()

        self.assertTrue(config.use_parallel_decode)
        self.assertEqual(config.parallel_decode_mode, "auto")

    def test_image_video_vae_configs_default_to_auto_parallel_decode(self):
        configs = (
            ErnieImageVAEConfig(),
            FluxVAEConfig(),
            Flux2VAEConfig(),
            GlmImageVAEConfig(),
            HunyuanVAEConfig(),
            LTXVideoVAEConfig(),
            QwenImageVAEConfig(),
            SanaVAEConfig(),
            StableDiffusion3VAEConfig(),
            WanVAEConfig(),
        )

        for config in configs:
            with self.subTest(config=type(config).__name__):
                self.assertTrue(config.use_parallel_decode)
                self.assertEqual(config.parallel_decode_mode, "auto")

    def test_auto_parallel_decode_policy_is_conservative(self):
        self.assertFalse(is_spatial_shard_parallel_decode_mode("auto"))
        self.assertFalse(should_use_spatial_shard_parallel_decode(VAEConfig()))
        self.assertFalse(should_use_spatial_shard_parallel_decode(QwenImageVAEConfig()))
        self.assertTrue(should_use_spatial_shard_parallel_decode(LTXVideoVAEConfig()))
        self.assertTrue(should_use_spatial_shard_parallel_decode(WanVAEConfig()))

        ltx23_config = LTXVideoVAEConfig()
        ltx23_config.arch_config.video_decoder_variant = "ltx_2_3"
        self.assertTrue(should_use_spatial_shard_parallel_decode(ltx23_config))
        ltx23_config.parallel_decode_mode = "spatial_shard"
        self.assertTrue(should_use_spatial_shard_parallel_decode(ltx23_config))

        config = QwenImageVAEConfig()
        self.assertFalse(
            should_use_spatial_shard_parallel_decode(
                config, torch.empty(1, 16, 1, 128, 128), 2
            )
        )
        config.parallel_decode_mode = "spatial_shard"
        self.assertTrue(should_use_spatial_shard_parallel_decode(config))

        hunyuan_config = HunyuanVAEConfig()
        self.assertTrue(should_use_spatial_shard_parallel_decode(hunyuan_config))
        self.assertFalse(
            should_use_spatial_shard_parallel_decode(
                hunyuan_config, torch.empty(1, 16, 9, 16, 16), 2
            )
        )
        self.assertTrue(
            should_use_spatial_shard_parallel_decode(
                hunyuan_config, torch.empty(1, 16, 9, 96, 96), 2
            )
        )

    def test_unsupported_vae_configs_opt_out_of_spatial_parallel_decode(self):
        configs = (Hunyuan3DVAEConfig(), LTXAudioVAEConfig())

        for config in configs:
            with self.subTest(config=type(config).__name__):
                self.assertFalse(config.use_parallel_decode)
                self.assertEqual(config.parallel_decode_mode, "tiled")

    def test_vae_nested_cli_defaults_do_not_override_model_defaults(self):
        parser = FlexibleArgumentParser()
        VAEConfig.add_cli_args(parser)
        parsed = vars(parser.parse_args([]))
        config = FluxVAEConfig()

        update_config_from_args(config, parsed, "vae_config")

        self.assertTrue(config.use_parallel_decode)
        self.assertEqual(config.parallel_decode_mode, "auto")

    def test_vae_nested_cli_explicit_args_override_model_defaults(self):
        parser = FlexibleArgumentParser()
        VAEConfig.add_cli_args(parser)
        parsed = vars(
            parser.parse_args(
                [
                    "--vae-config.use-parallel-decode",
                    "false",
                    "--vae-config.parallel-decode-mode",
                    "patch",
                ]
            )
        )
        config = FluxVAEConfig()

        update_config_from_args(config, parsed, "vae_config")

        self.assertFalse(config.use_parallel_decode)
        self.assertEqual(config.parallel_decode_mode, "patch")

    def test_base_decode_prefers_spatial_parallel_dispatch(self):
        config = VAEConfig()
        config.arch_config.temporal_compression_ratio = 1
        config.arch_config.spatial_compression_ratio = 1
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        vae = _DispatchProbeVAE(config)
        z = torch.randn(1, 1, 1, 2, 2)

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_group_coordinator",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_world_size",
                return_value=2,
            ),
        ):
            out = vae.decode(z)

        self.assertTrue(vae.used_decode)
        self.assertFalse(vae.used_parallel_tiled_decode)
        torch.testing.assert_close(out, z)

    def test_auto_decode_does_not_fall_back_to_parallel_tiling(self):
        config = VAEConfig()
        config.arch_config.temporal_compression_ratio = 1
        config.arch_config.spatial_compression_ratio = 1
        config.use_parallel_decode = True
        config.parallel_decode_mode = "auto"
        vae = _DispatchProbeVAE(config)
        z = torch.randn(1, 1, 1, 2, 2)

        with patch(
            "sglang.multimodal_gen.runtime.models.vaes.common.get_sp_world_size",
            return_value=2,
        ):
            out = vae.decode(z)

        self.assertTrue(vae.used_decode)
        self.assertFalse(vae.used_parallel_tiled_decode)
        torch.testing.assert_close(out, z)

    def test_spatial_alias_still_uses_spatial_shard_dispatch(self):
        self.assertTrue(is_spatial_shard_parallel_decode_mode("spatial"))

    def test_decode_parallel_group_uses_dedicated_group(self):
        old_decode = parallel_state._VAE_DECODE
        decode_group = SimpleNamespace(world_size=4, rank_in_group=2)
        try:
            parallel_state._VAE_DECODE = decode_group
            self.assertIs(
                parallel_state.get_decode_parallel_group_coordinator(), decode_group
            )
            self.assertEqual(parallel_state.get_decode_parallel_world_size(), 4)
            self.assertEqual(parallel_state.get_decode_parallel_rank(), 2)
        finally:
            parallel_state._VAE_DECODE = old_decode

    def test_decode_rank_groups_cover_non_dp_parallel_axes(self):
        rank_generator = RankGenerator(
            tp=2,
            sp=2,
            pp=1,
            cfg=2,
            dp=2,
            order="tp-sp-pp-cfg-dp",
        )

        self.assertEqual(
            parallel_state._get_vae_decode_group_ranks(rank_generator),
            [list(range(0, 8)), list(range(8, 16))],
        )

    def test_spatial_split_keeps_uneven_height_lossless(self):
        x = torch.arange(5).view(1, 1, 1, 5, 1)

        rank0, expected_height = split_for_parallel_decode(
            x, upsample_count=1, world_size=2, rank=0
        )
        rank1, _ = split_for_parallel_decode(x, upsample_count=1, world_size=2, rank=1)

        self.assertEqual(expected_height, 10)
        self.assertEqual(rank0.shape[-2], 3)
        self.assertEqual(rank1.shape[-2], 2)
        torch.testing.assert_close(torch.cat([rank0, rank1], dim=-2), x)

    def test_chunk_height_by_sizes_keeps_original_partition(self):
        x = torch.arange(10).view(1, 1, 10, 1)

        with patch(
            "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
            return_value=0,
        ):
            rank0 = chunk_height_by_sizes(x, [6, 4])
        with patch(
            "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
            return_value=1,
        ):
            rank1 = chunk_height_by_sizes(x, [6, 4])

        self.assertEqual(rank0.shape[-2], 6)
        self.assertEqual(rank1.shape[-2], 4)
        torch.testing.assert_close(torch.cat([rank0, rank1], dim=-2), x)

    def test_qwen_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage.get_decode_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = QwenImageDecoder3d(
                dim=4,
                z_dim=2,
                dim_mult=(1, 1),
                num_res_blocks=1,
                attn_scales=(),
                temperal_upsample=(False,),
                input_channels=3,
                use_parallel_decode=True,
            )

        self.assertTrue(
            any(isinstance(m, SpatialParallelCausalConv3d) for m in decoder.modules())
        )
        self.assertTrue(
            any(
                isinstance(m, QwenImageAttentionBlock) and m.spatial_parallel
                for m in decoder.modules()
            )
        )

    def test_wan_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.wanvae.get_decode_parallel_rank",
                return_value=0,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = WanDecoder3d(
                dim=4,
                z_dim=2,
                dim_mult=(1, 1),
                num_res_blocks=1,
                attn_scales=(),
                temperal_upsample=(False,),
                out_channels=3,
                use_parallel_decode=True,
            )

        self.assertTrue(
            any(isinstance(m, SpatialParallelCausalConv3d) for m in decoder.modules())
        )
        self.assertTrue(
            any(isinstance(m, WanDistAttentionBlock) for m in decoder.modules())
        )

    def test_diffusers_2d_decoder_uses_spatial_parallel_components(self):
        config = StableDiffusion3VAEConfig()
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        config.arch_config.latent_channels = 2
        config.arch_config.block_out_channels = (4, 4)
        config.arch_config.down_block_types = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        )
        config.arch_config.up_block_types = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        )
        config.arch_config.layers_per_block = 1
        config.arch_config.norm_num_groups = 1
        config.arch_config.sample_size = 8

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.model_parallel_is_initialized",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_group_coordinator",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.multimodal_gen.runtime.models.vaes.common.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            vae = AutoencoderKL(config)

        self.assertTrue(vae._spatial_parallel_decode_enabled)
        self.assertTrue(
            any(isinstance(m, SpatialParallelConv2d) for m in vae.decoder.modules())
        )

    def test_ltx_decoder_uses_spatial_parallel_conv3d(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = LTX2VideoDecoder3d(
                in_channels=4,
                out_channels=3,
                block_out_channels=(8,),
                spatio_temporal_scaling=(False,),
                layers_per_block=(1, 1),
                patch_size=1,
                upsample_residual=(False,),
                upsample_factor=(1,),
                spatial_padding_mode="reflect",
            )
            _enable_ltx_decoder_spatial_parallel(decoder)

        causal_convs = [
            m for m in decoder.modules() if isinstance(m, LTX2VideoCausalConv3d)
        ]
        self.assertGreater(len(causal_convs), 0)
        self.assertTrue(
            all(isinstance(m.conv, SpatialParallelConv3d) for m in causal_convs)
        )

    def test_hunyuan_decoder_uses_spatial_parallel_components(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.parallel_conv.get_decode_parallel_rank",
                return_value=0,
            ),
        ):
            decoder = HunyuanVideoDecoder3D(
                in_channels=4,
                out_channels=3,
                up_block_types=("HunyuanVideoUpBlock3D", "HunyuanVideoUpBlock3D"),
                block_out_channels=(4, 8),
                layers_per_block=1,
                norm_num_groups=1,
                mid_block_add_attention=True,
                time_compression_ratio=4,
                spatial_compression_ratio=2,
            )
            _enable_hunyuan_decoder_spatial_parallel(decoder)

        self.assertTrue(decoder.spatial_parallel)
        self.assertTrue(
            any(
                isinstance(m, HunyuanVideoMidBlock3D) and m.spatial_parallel
                for m in decoder.modules()
            )
        )
        self.assertTrue(
            any(isinstance(m, SpatialParallelConv3d) for m in decoder.modules())
        )
        shortcut_convs = [
            m.conv_shortcut.conv
            for m in decoder.modules()
            if isinstance(m, HunyuanVideoResnetBlockCausal3D)
            and m.conv_shortcut is not None
        ]
        self.assertTrue(shortcut_convs)
        self.assertTrue(
            all(not isinstance(conv, SpatialParallelConv3d) for conv in shortcut_convs)
        )


if __name__ == "__main__":
    unittest.main()
