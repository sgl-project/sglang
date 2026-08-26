import unittest
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_1_T2V_480P_Config,
    Wan2_1_Fun_1_3B_InP_Config,
    Wan2_1_T2V_1_3B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanSelfAttention,
    _wan_default_attention_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_WAN = "sglang.multimodal_gen.runtime.models.dits.wanvideo"


class TestWanAttentionBackendRole(unittest.TestCase):
    def test_cross_attention_role_is_forwarded_to_usp(self):
        with (
            patch(f"{_WAN}.ColumnParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.RowParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.get_tp_world_size", return_value=1),
            patch(f"{_WAN}.USPAttention") as usp_attention,
        ):
            WanSelfAttention(
                dim=128,
                num_heads=1,
                qk_norm=False,
                is_cross_attention=True,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
            )

        self.assertTrue(usp_attention.call_args.kwargs["is_cross_attention"])
        self.assertTrue(usp_attention.call_args.kwargs["skip_sequence_parallel"])

    def test_default_backend_is_forwarded_to_usp(self):
        with (
            patch(f"{_WAN}.ColumnParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.RowParallelLinear", return_value=nn.Identity()),
            patch(f"{_WAN}.get_tp_world_size", return_value=1),
            patch(f"{_WAN}.USPAttention") as usp_attention,
        ):
            WanSelfAttention(
                dim=128,
                num_heads=1,
                qk_norm=False,
                default_attention_backend=AttentionBackendEnum.TORCH_CUDNN_SDPA,
            )

        self.assertEqual(
            usp_attention.call_args.kwargs["default_attention_backend"],
            AttentionBackendEnum.TORCH_CUDNN_SDPA,
        )

    @patch(f"{_WAN}.get_ring_parallel_world_size", return_value=1)
    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_fastwan_prefers_cudnn_sdpa_on_sm120(self, _mock_is_sm120, _mock_ring):
        config = FastWan2_1_T2V_480P_Config()

        self.assertTrue(config.dit_config.prefer_cudnn_sdpa_on_sm120)
        self.assertEqual(
            _wan_default_attention_backend(config.dit_config),
            AttentionBackendEnum.TORCH_CUDNN_SDPA,
        )

    @patch(f"{_WAN}.get_ring_parallel_world_size", return_value=1)
    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_wan21_1_3b_prefers_cudnn_sdpa_on_sm120(self, _mock_is_sm120, _mock_ring):
        config = Wan2_1_T2V_1_3B_Config()

        self.assertTrue(config.dit_config.prefer_cudnn_sdpa_on_sm120)
        self.assertEqual(
            _wan_default_attention_backend(config.dit_config),
            AttentionBackendEnum.TORCH_CUDNN_SDPA,
        )

    @patch(f"{_WAN}.get_ring_parallel_world_size", return_value=1)
    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_wan21_fun_1_3b_prefers_cudnn_sdpa_on_sm120(
        self, _mock_is_sm120, _mock_ring
    ):
        config = Wan2_1_Fun_1_3B_InP_Config()

        self.assertTrue(config.dit_config.prefer_cudnn_sdpa_on_sm120)
        self.assertEqual(
            _wan_default_attention_backend(config.dit_config),
            AttentionBackendEnum.TORCH_CUDNN_SDPA,
        )

    @patch(f"{_WAN}.current_platform.is_sm120", return_value=False)
    def test_fastwan_keeps_platform_default_outside_sm120(self, _mock_is_sm120):
        config = FastWan2_1_T2V_480P_Config()

        self.assertIsNone(_wan_default_attention_backend(config.dit_config))

    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_fastwan_keeps_compile_backend_on_sm120(self, _mock_is_sm120):
        config = FastWan2_1_T2V_480P_Config()

        self.assertIsNone(
            _wan_default_attention_backend(
                config.dit_config,
                enable_torch_compile=True,
            )
        )

    @patch(f"{_WAN}.get_ring_parallel_world_size", return_value=2)
    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_fastwan_keeps_ring_backend_on_sm120(self, _mock_is_sm120, _mock_ring):
        config = FastWan2_1_T2V_480P_Config()

        self.assertIsNone(_wan_default_attention_backend(config.dit_config))

    @patch(f"{_WAN}.current_platform.is_sm120", return_value=True)
    def test_regular_wan_keeps_platform_default_on_sm120(self, _mock_is_sm120):
        config = WanT2V480PConfig()

        self.assertFalse(config.dit_config.prefer_cudnn_sdpa_on_sm120)
        self.assertIsNone(_wan_default_attention_backend(config.dit_config))


if __name__ == "__main__":
    unittest.main()
