import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"


class _ServerArgs:
    def __init__(self, backend: str, *, explicit: bool) -> None:
        self.attention_backend = backend
        self._explicit = explicit

    def is_arg_explicitly_set(self, arg_name: str) -> bool:
        return arg_name == "attention_backend" and self._explicit


class _FakeSDPABackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def unsupported_requirements(cls, _requirements) -> tuple[str, ...]:
        return ()


class _FakeAITERBackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER

    @classmethod
    def unsupported_requirements(
        cls, requirements: AttentionRequirements
    ) -> tuple[str, ...]:
        return ("packed varlen attention",) if requirements.packed_varlen else ()


class _FakePlatform:
    device_name = "test"
    selected_backend = None

    @classmethod
    def get_attn_backend_cls_str(cls, selected_backend, _head_size, _dtype):
        cls.selected_backend = selected_backend
        if selected_backend == AttentionBackendEnum.AITER:
            return "fake.AITERBackend"
        return "fake.SDPABackend"


class TestAttentionBackendFallback(unittest.TestCase):
    def setUp(self) -> None:
        _cached_get_attn_backend.cache_clear()
        _FakePlatform.selected_backend = None

    def _resolve(
        self,
        backend: AttentionBackendEnum,
        *,
        explicit: bool,
        is_cross_attention: bool,
        supported: set[AttentionBackendEnum],
        attention_requirements: AttentionRequirements | None = None,
    ):
        server_args = _ServerArgs(backend.name.lower(), explicit=explicit)
        with (
            patch(f"{_SELECTOR}.get_global_forced_attn_backend", return_value=None),
            patch(f"{_SELECTOR}.get_component_forced_attn_backend", return_value=None),
            patch(f"{_SELECTOR}.get_global_server_args", return_value=server_args),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform",
                _FakePlatform,
            ),
            patch(
                f"{_SELECTOR}.resolve_obj_by_qualname",
                side_effect=lambda name: (
                    _FakeAITERBackend
                    if name == "fake.AITERBackend"
                    else _FakeSDPABackend
                ),
            ),
        ):
            return get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends=supported,
                attention_requirements=attention_requirements,
                is_cross_attention=is_cross_attention,
            )

    def test_implicit_platform_preference_falls_back(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            is_cross_attention=False,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeSDPABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_implicit_preference_falls_back_for_missing_capability(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            is_cross_attention=False,
            supported={AttentionBackendEnum.AITER, AttentionBackendEnum.TORCH_SDPA},
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )

        self.assertIs(backend, _FakeSDPABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_explicit_dense_mismatch_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            self._resolve(
                AttentionBackendEnum.AITER,
                explicit=True,
                is_cross_attention=False,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )

    def test_sparse_backend_falls_back_for_cross_attention(self):
        backend = self._resolve(
            AttentionBackendEnum.LASER_ATTN,
            explicit=True,
            is_cross_attention=True,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeSDPABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_sparse_backend_mismatch_fails_for_self_attention(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            self._resolve(
                AttentionBackendEnum.LASER_ATTN,
                explicit=True,
                is_cross_attention=False,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )


if __name__ == "__main__":
    unittest.main()
