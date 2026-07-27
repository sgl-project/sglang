import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
)

SDPA_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)
FA_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn."
    "FlashAttentionBackend"
)


class FakeCudaPlatform(CudaPlatformBase):
    is_sm120_device = False
    is_blackwell_device = False
    supports_flash_attention = True
    device_capability = DeviceCapability(9, 0)

    @classmethod
    def is_sm120(cls):
        return cls.is_sm120_device

    @classmethod
    def is_blackwell(cls):
        return cls.is_blackwell_device

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls.supports_flash_attention

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        del device_id
        return cls.device_capability


class TestCudaAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        FakeCudaPlatform.is_sm120_device = False
        FakeCudaPlatform.is_blackwell_device = False
        FakeCudaPlatform.supports_flash_attention = True
        FakeCudaPlatform.device_capability = DeviceCapability(9, 0)

    def resolve(
        self,
        selected_backend: AttentionBackendEnum | None,
        dtype: torch.dtype = torch.float16,
        head_size: int = 128,
    ) -> str:
        return FakeCudaPlatform.get_attn_backend_cls_str(
            selected_backend=selected_backend,
            head_size=head_size,
            dtype=dtype,
        )

    def test_direct_torch_sdpa_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.TORCH_SDPA), SDPA_BACKEND_CLS_STR
        )

    def test_direct_aiter_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.AITER),
            "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend",
        )

    def test_default_backend_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_requested_flash_attention_uses_torch_sdpa_on_sm120(self):
        FakeCudaPlatform.is_sm120_device = True

        self.assertEqual(self.resolve(AttentionBackendEnum.FA), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_for_non_flash_attention_dtype(self):
        self.assertEqual(self.resolve(None, torch.float32), SDPA_BACKEND_CLS_STR)

    def test_default_backend_falls_back_without_flash_attention_capability(self):
        FakeCudaPlatform.supports_flash_attention = False

        self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

    def test_blackwell_falls_back_when_flash_attention_prepare_fails(self):
        FakeCudaPlatform.is_blackwell_device = True
        FakeCudaPlatform.device_capability = DeviceCapability(10, 0)

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=False,
        ) as prepare_flash_attention:
            self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

        prepare_flash_attention.assert_called_once_with()

    def test_sm100_falls_back_when_fa4_cute_is_unavailable(self):
        FakeCudaPlatform.is_blackwell_device = True
        FakeCudaPlatform.device_capability = DeviceCapability(10, 0)

        with patch(
            "sglang.multimodal_gen.runtime.platforms.cuda.import_module",
            side_effect=ModuleNotFoundError("flash_attn.cute"),
        ):
            self.assertEqual(
                self.resolve(AttentionBackendEnum.FA, head_size=72),
                SDPA_BACKEND_CLS_STR,
            )

    def test_sm100_propagates_fa4_import_initialization_errors(self):
        FakeCudaPlatform.is_blackwell_device = True
        FakeCudaPlatform.device_capability = DeviceCapability(10, 0)

        with patch(
            "sglang.multimodal_gen.runtime.platforms.cuda.import_module",
            side_effect=RuntimeError("FA4 initialization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "FA4 initialization failed"):
                self.resolve(AttentionBackendEnum.FA, head_size=72)

    def test_sm100_preparation_enables_fa4(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            flash_attn as fa_backend,
        )

        FakeCudaPlatform.is_blackwell_device = True
        original_fa_version = fa_backend.fa_ver
        try:
            fa_backend.fa_ver = 3
            with patch(
                "sglang.multimodal_gen.runtime.platforms.cuda.import_module",
                return_value=SimpleNamespace(flash_attn_varlen_func=lambda: None),
            ):
                self.assertTrue(
                    FakeCudaPlatform._prepare_flash_attention_for_blackwell()
                )
            self.assertEqual(fa_backend.fa_ver, 4)
        finally:
            fa_backend.fa_ver = original_fa_version

    def test_sm100_fa4_accepts_bagel_head_size_72(self):
        FakeCudaPlatform.is_blackwell_device = True
        FakeCudaPlatform.device_capability = DeviceCapability(10, 0)

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=True,
        ):
            self.assertEqual(
                self.resolve(AttentionBackendEnum.FA, head_size=72),
                FA_BACKEND_CLS_STR,
            )

    def test_hopper_rejects_bagel_head_size_72_for_fa3(self):
        FakeCudaPlatform.device_capability = DeviceCapability(9, 0)

        self.assertEqual(
            self.resolve(AttentionBackendEnum.FA, head_size=72),
            SDPA_BACKEND_CLS_STR,
        )

    def test_sm103_rejects_bagel_head_size_72(self):
        FakeCudaPlatform.is_blackwell_device = True
        FakeCudaPlatform.device_capability = DeviceCapability(10, 3)

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=True,
        ):
            self.assertEqual(
                self.resolve(AttentionBackendEnum.FA, head_size=72),
                SDPA_BACKEND_CLS_STR,
            )

    def test_invalid_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid attention backend"):
            self.resolve(AttentionBackendEnum.AITER_SAGE)


if __name__ == "__main__":
    unittest.main()
