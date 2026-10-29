import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.xpu import XpuPlatform
from sglang.test.test_utils import CustomTestCase

SAGE_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn."
    "SageAttentionBackend"
)


@unittest.skipIf(importlib.util.find_spec("triton") is None, "triton not installed")
class TestXpuSageAttentionResolution(CustomTestCase):
    def test_sage_is_not_replaced_by_sdpa(self):
        """An explicit sage_attn request on XPU must not run torch SDPA instead."""
        cls_str = XpuPlatform.get_attn_backend_cls_str(
            AttentionBackendEnum.SAGE_ATTN, 128, torch.bfloat16
        )
        self.assertEqual(cls_str, SAGE_BACKEND_CLS_STR)

    def test_unsupported_inputs_raise_for_the_selector(self):
        # the component-aware selector owns the fallback policy for these
        with self.assertRaisesRegex(ValueError, "head size 256"):
            XpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.SAGE_ATTN, 256, torch.bfloat16
            )
        with self.assertRaisesRegex(ValueError, "fp16/bf16"):
            XpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.SAGE_ATTN, 128, torch.float32
            )

    def test_missing_kernel_falls_back_to_flash_attention(self):
        with patch.dict(sys.modules, {"sglang.kernels.ops.diffusion": None}):
            self.assertIsNone(
                XpuPlatform._get_sage_attn_backend_cls_str(128, torch.bfloat16)
            )


if __name__ == "__main__":
    unittest.main()
