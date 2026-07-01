import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestSM120TritonAttentionBackend(unittest.TestCase):
    def test_cli_name_is_normalized(self):
        self.assertEqual(
            ServerArgs._normalize_attention_backend_name("sm120_triton_attn"),
            "sm120_triton_attn",
        )

    def test_dit_configs_allow_backend(self):
        self.assertIn(
            AttentionBackendEnum.SM120_TRITON_ATTN,
            DiTConfig().arch_config._supported_attention_backends,
        )

    def test_cuda_selector_accepts_backend_on_sm120(self):
        with patch.object(CudaPlatformBase, "is_sm120", return_value=True):
            self.assertEqual(
                CudaPlatformBase.get_attn_backend_cls_str(
                    AttentionBackendEnum.SM120_TRITON_ATTN,
                    head_size=128,
                    dtype=torch.bfloat16,
                ),
                "sglang.multimodal_gen.runtime.layers.attention.backends.sm120_triton_attn.SM120TritonAttentionBackend",
            )

    def test_cuda_selector_rejects_backend_off_sm120(self):
        with patch.object(CudaPlatformBase, "is_sm120", return_value=False):
            with self.assertRaisesRegex(ValueError, "SM12.x"):
                CudaPlatformBase.get_attn_backend_cls_str(
                    AttentionBackendEnum.SM120_TRITON_ATTN,
                    head_size=128,
                    dtype=torch.bfloat16,
                )


if __name__ == "__main__":
    unittest.main()
