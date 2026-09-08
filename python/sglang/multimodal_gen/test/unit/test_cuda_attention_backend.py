import sys
import types
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms.cuda import (
    CudaPlatformBase,
    _SageAttentionBackendResolver,
    _SpargeAttentionBackendResolver,
)
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
)

SDPA_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)


class FakeCudaPlatform(CudaPlatformBase):
    is_sm120_device = False
    is_blackwell_device = False
    is_hopper_device = False
    supports_flash_attention = True
    device_capability = DeviceCapability(8, 0)

    @classmethod
    def is_sm120(cls):
        return cls.is_sm120_device

    @classmethod
    def is_blackwell(cls):
        return cls.is_blackwell_device

    @classmethod
    def is_hopper(cls):
        return cls.is_hopper_device

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls.supports_flash_attention

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.device_capability


class TestCudaAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        FakeCudaPlatform.is_sm120_device = False
        FakeCudaPlatform.is_blackwell_device = False
        FakeCudaPlatform.is_hopper_device = False
        FakeCudaPlatform.supports_flash_attention = True
        FakeCudaPlatform.device_capability = DeviceCapability(8, 0)
        _cached_get_attn_backend.cache_clear()

    def resolve(
        self,
        selected_backend: AttentionBackendEnum | None,
        dtype: torch.dtype = torch.float16,
    ) -> str:
        return FakeCudaPlatform.get_attn_backend_cls_str(
            selected_backend=selected_backend,
            head_size=128,
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

    def test_direct_cube_sparse_selection(self):
        self.assertEqual(
            self.resolve(AttentionBackendEnum.CUBE_SPARSE_ATTN),
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "cube_sparse_attn.CubeSparseAttentionBackend",
        )

    def test_blackwell_cube_selection_initializes_fa4_for_token_refiner(self):
        FakeCudaPlatform.is_blackwell_device = True
        module_name = (
            "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn"
        )
        fake_flash_attn = ModuleType(module_name)
        fake_flash_attn.set_fa_ver = Mock()

        with patch.dict("sys.modules", {module_name: fake_flash_attn}):
            self.resolve(AttentionBackendEnum.CUBE_SPARSE_ATTN)

        fake_flash_attn.set_fa_ver.assert_called_once_with(4)

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

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=False,
        ) as prepare_flash_attention:
            self.assertEqual(self.resolve(None), SDPA_BACKEND_CLS_STR)

        prepare_flash_attention.assert_called_once_with()

    def test_default_backend_prefers_dynamic_cudnn_sdpa_on_blackwell(self):
        FakeCudaPlatform.is_blackwell_device = True

        with patch.object(
            FakeCudaPlatform,
            "_prepare_flash_attention_for_blackwell",
            return_value=True,
        ):
            self.assertEqual(
                self.resolve(None),
                "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.DynamicCudnnSDPABackend",
            )

    def test_invalid_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid attention backend"):
            self.resolve(AttentionBackendEnum.AITER_SAGE)

    def test_hopper_sage_attention_without_sm90_fix_falls_back(self):
        FakeCudaPlatform.is_hopper_device = True
        sageattention = types.ModuleType("sageattention")
        sageattention.__path__ = []
        sageattention.sageattn = object()
        sm90_compile = types.ModuleType("sageattention.sm90_compile")

        with patch.dict(
            sys.modules,
            {
                "sageattention": sageattention,
                "sageattention.sm90_compile": sm90_compile,
            },
        ):
            self.assertEqual(
                _SageAttentionBackendResolver.resolve(FakeCudaPlatform),
                AttentionBackendEnum.FA,
            )

    def test_sparge_attention_resolver(self):
        module = types.ModuleType("spas_sage_attn")
        module.spas_sage2_attn_meansim_topk_cuda = object()
        backend_module = (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sparge_attn"
        )
        try:
            with patch.dict(sys.modules, {"spas_sage_attn": module}):
                self.assertEqual(
                    self.resolve(AttentionBackendEnum.SPARGE_ATTN),
                    f"{backend_module}.SpargeAttentionBackend",
                )
        finally:
            sys.modules.pop(backend_module, None)

    def test_sparge_attention_rejects_pre_ampere_cuda(self):
        FakeCudaPlatform.device_capability = DeviceCapability(7, 5)
        with self.assertRaisesRegex(ValueError, "found 7.5"):
            _SpargeAttentionBackendResolver.resolve(FakeCudaPlatform)

    def test_sparge_attention_rejects_unsupported_blackwell_cuda(self):
        FakeCudaPlatform.device_capability = DeviceCapability(10, 0)
        with self.assertRaisesRegex(ValueError, "found 10.0"):
            _SpargeAttentionBackendResolver.resolve(FakeCudaPlatform)

    def test_sparge_attention_missing_dependency_fails_closed(self):
        with patch.dict(sys.modules, {"spas_sage_attn": None}):
            with self.assertRaisesRegex(
                ImportError, "SpargeAttention is not installed"
            ):
                _SpargeAttentionBackendResolver.resolve(FakeCudaPlatform)

    def test_explicit_backend_is_not_rejected_by_model_preferences(self):
        class FakeAITERBackend:
            @classmethod
            def get_enum(cls):
                return AttentionBackendEnum.AITER

        with (
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform",
                FakeCudaPlatform,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.resolve_obj_by_qualname",
                return_value=FakeAITERBackend,
            ),
        ):
            backend = _cached_get_attn_backend(
                128,
                torch.float16,
                (AttentionBackendEnum.FA,),
                AttentionBackendEnum.AITER,
            )

        self.assertEqual(backend.get_enum(), AttentionBackendEnum.AITER)


if __name__ == "__main__":
    unittest.main()
