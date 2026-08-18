import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.roles import AttentionRole
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    _cached_get_attn_backend,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_SELECTOR = "sglang.multimodal_gen.runtime.layers.attention.selector"


class _ServerArgs(ServerArgs):
    def __init__(self, backend: str, *, explicit: bool) -> None:
        self.attention_backend = backend
        self._explicit_arg_names = {"attention_backend"} if explicit else set()


class _FakeSDPABackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def unsupported_requirements(cls, _requirements) -> tuple[str, ...]:
        return ()


class _FakeFABackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

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
        if selected_backend in (None, AttentionBackendEnum.FA):
            return "fake.FABackend"
        return "fake.SDPABackend"


_FAKE_BACKENDS = {
    "fake.AITERBackend": _FakeAITERBackend,
    "fake.FABackend": _FakeFABackend,
    "fake.SDPABackend": _FakeSDPABackend,
}


class TestAttentionBackendFallback(unittest.TestCase):
    def setUp(self) -> None:
        _cached_get_attn_backend.cache_clear()
        _FakePlatform.selected_backend = None

    def _resolve(
        self,
        backend: AttentionBackendEnum,
        *,
        explicit: bool,
        attention_role: AttentionRole = AttentionRole.SELF,
        supported: set[AttentionBackendEnum],
        attention_requirements: AttentionRequirements | None = None,
        server_args: object | None = None,
        backend_by_role: dict[AttentionRole, AttentionBackendEnum] | None = None,
        component_backend: AttentionBackendEnum | None = None,
    ):
        if server_args is None:
            server_args = _ServerArgs(backend.name.lower(), explicit=explicit)

        def _role_forced(role: AttentionRole) -> AttentionBackendEnum | None:
            return (backend_by_role or {}).get(role)

        with (
            patch(f"{_SELECTOR}.get_global_forced_attn_backend", return_value=None),
            patch(
                f"{_SELECTOR}.get_component_forced_attn_backend_for_role",
                side_effect=_role_forced,
            ),
            patch(
                f"{_SELECTOR}.get_component_forced_attn_backend",
                return_value=component_backend,
            ),
            patch(f"{_SELECTOR}.get_global_server_args", return_value=server_args),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform",
                _FakePlatform,
            ),
            patch(
                f"{_SELECTOR}.resolve_obj_by_qualname",
                side_effect=_FAKE_BACKENDS.__getitem__,
            ),
        ):
            return get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends=supported,
                attention_requirements=attention_requirements,
                attention_role=attention_role,
            )

    def test_implicit_platform_preference_falls_back(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            attention_role=AttentionRole.SELF,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_implicit_preference_falls_back_for_missing_capability(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            attention_role=AttentionRole.SELF,
            supported={AttentionBackendEnum.AITER, AttentionBackendEnum.TORCH_SDPA},
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )

        self.assertIs(backend, _FakeSDPABackend)
        self.assertEqual(
            _FakePlatform.selected_backend, AttentionBackendEnum.TORCH_SDPA
        )

    def test_lightweight_server_args_are_treated_as_implicit(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            attention_role=AttentionRole.SELF,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            server_args=SimpleNamespace(attention_backend="aiter"),
        )

        self.assertIs(backend, _FakeFABackend)

    def test_explicit_dense_mismatch_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            self._resolve(
                AttentionBackendEnum.AITER,
                explicit=True,
                attention_role=AttentionRole.SELF,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )

    def test_sparse_backend_falls_back_for_cross_attention(self):
        backend = self._resolve(
            AttentionBackendEnum.LASER_ATTN,
            explicit=True,
            attention_role=AttentionRole.CROSS,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_sparse_backend_mismatch_fails_for_self_attention(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            self._resolve(
                AttentionBackendEnum.LASER_ATTN,
                explicit=True,
                attention_role=AttentionRole.SELF,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )

    def test_role_backend_selects_for_cross(self):
        # Component-wide default is SDPA (implicit), but the cross role forces FA.
        backend = self._resolve(
            AttentionBackendEnum.TORCH_SDPA,
            explicit=False,
            attention_role=AttentionRole.CROSS,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            backend_by_role={AttentionRole.CROSS: AttentionBackendEnum.FA},
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertEqual(_FakePlatform.selected_backend, AttentionBackendEnum.FA)

    def test_role_backend_not_applied_to_self(self):
        # The cross role override must not leak into self-attention, which keeps
        # the explicit SDPA selection.
        backend = self._resolve(
            AttentionBackendEnum.TORCH_SDPA,
            explicit=True,
            attention_role=AttentionRole.SELF,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            backend_by_role={AttentionRole.CROSS: AttentionBackendEnum.FA},
        )

        self.assertIs(backend, _FakeSDPABackend)

    def test_role_backend_overrides_component_wide(self):
        # Role-qualified selection takes precedence over the component-wide backend.
        backend = self._resolve(
            AttentionBackendEnum.TORCH_SDPA,
            explicit=False,
            attention_role=AttentionRole.CROSS,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            backend_by_role={AttentionRole.CROSS: AttentionBackendEnum.FA},
            component_backend=AttentionBackendEnum.TORCH_SDPA,
        )

        self.assertIs(backend, _FakeFABackend)

    def test_role_sparse_backend_downgrades_for_cross(self):
        # A sparse backend chosen via the cross role still downgrades to dense.
        backend = self._resolve(
            AttentionBackendEnum.LASER_ATTN,
            explicit=False,
            attention_role=AttentionRole.CROSS,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            backend_by_role={AttentionRole.CROSS: AttentionBackendEnum.LASER_ATTN},
        )

        self.assertIs(backend, _FakeFABackend)


if __name__ == "__main__":
    unittest.main()
