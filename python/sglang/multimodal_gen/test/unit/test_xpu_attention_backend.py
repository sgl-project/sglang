import importlib.util
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import sage_attn
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


@unittest.skipIf(importlib.util.find_spec("triton") is None, "triton not installed")
class TestXpuSageAttentionArkDispatch(CustomTestCase):
    def test_lse_calls_never_reach_ark(self):
        """ARK returns one LSE per 16 query rows, so an LSE call routed to it
        would silently corrupt the ring-attention merge."""
        ark_sagev1 = MagicMock()
        q, k, v = (torch.randn(1, 16, 2, 64) for _ in range(3))
        with patch(
            "sglang.kernels.ops.diffusion.sage_attn_qk_int8_pv_fp16"
        ) as triton_sage:
            sage_attn._xpu_sage_attn(
                q,
                k,
                v,
                ark_sagev1=ark_sagev1,
                causal=False,
                sm_scale=0.125,
                return_lse=True,
            )
        ark_sagev1.assert_not_called()
        triton_sage.assert_called_once()

    def test_unloadable_ark_library_falls_back_to_triton(self):
        """When ARK's XPU library fails to load (libiomp5.so missing from
        LD_LIBRARY_PATH), ARK only prints and every sagev1 call raises."""
        fake_ark = types.ModuleType("auto_round_kernel")
        fake_ark.xpu_lib = None
        fake_ark.sagev1 = MagicMock()
        fake_version = types.ModuleType("auto_round_kernel.version")
        fake_version.__version__ = "0.16.0"
        fake_modules = {
            "auto_round_kernel": fake_ark,
            "auto_round_kernel.version": fake_version,
        }
        with patch.dict(sys.modules, fake_modules):
            self.assertIsNone(sage_attn._load_ark_sagev1.__wrapped__())

    def test_ark_inputs_get_dense_strides_on_size_one_dims(self):
        # a sequence slice of a batch-1 tensor passes is_contiguous() but keeps
        # the parent's batch stride, which ARK's exact stride check rejects
        x = torch.randn(1, 10, 2, 4)[:, 2:7]
        dense = sage_attn._as_dense_nhd(x)
        self.assertEqual(dense.stride(), (40, 8, 4, 1))
        self.assertTrue(torch.equal(dense, x))


if __name__ == "__main__":
    unittest.main()
