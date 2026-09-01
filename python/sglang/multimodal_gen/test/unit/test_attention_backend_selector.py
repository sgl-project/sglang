import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    ComponentAttentionBackendNotAppliedError,
    _cached_get_attn_backend,
    _record_component_attn_backend,
    claim_deferred_component_attn_backend,
    component_attn_backend_context_manager,
    get_attn_backend,
    get_component_attn_backend_context,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
    GenericComponentLoader,
    NativeComponentLoaderRequired,
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersPipeline,
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


class _FakeSparseBackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.LASER_ATTN

    @classmethod
    def unsupported_requirements(cls, _requirements) -> tuple[str, ...]:
        return ()


class _FakePlatform:
    device_name = "test"
    selected_backend = None

    @classmethod
    def get_attn_backend_cls_str(cls, selected_backend, _head_size, _dtype):
        cls.selected_backend = selected_backend
        if selected_backend == AttentionBackendEnum.AITER:
            return "fake.AITERBackend"
        if selected_backend == AttentionBackendEnum.LASER_ATTN:
            return "fake.SparseBackend"
        if selected_backend in (None, AttentionBackendEnum.FA):
            return "fake.FABackend"
        return "fake.SDPABackend"


_FAKE_BACKENDS = {
    "fake.AITERBackend": _FakeAITERBackend,
    "fake.FABackend": _FakeFABackend,
    "fake.SparseBackend": _FakeSparseBackend,
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
        is_cross_attention: bool,
        supported: set[AttentionBackendEnum],
        attention_requirements: AttentionRequirements | None = None,
        default_attention_backend: AttentionBackendEnum | None = None,
        component_backend: AttentionBackendEnum | None = None,
        allow_global_backend_fallback: bool = False,
        server_args: object | None = None,
    ):
        if server_args is None:
            server_args = _ServerArgs(backend.name.lower(), explicit=explicit)
        with (
            patch(f"{_SELECTOR}.get_global_forced_attn_backend", return_value=None),
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
            component_attn_backend_context_manager(
                component_backend,
                component_name="text_encoder",
                allow_global_backend_fallback=allow_global_backend_fallback,
                require_backend_selection=component_backend is not None,
            ),
        ):
            return get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends=supported,
                attention_requirements=attention_requirements,
                default_attention_backend=default_attention_backend,
                is_cross_attention=is_cross_attention,
            )

    def test_implicit_platform_preference_falls_back(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            is_cross_attention=False,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_component_override_requires_an_sglang_attention_layer(self):
        with self.assertRaisesRegex(
            ValueError, "did not construct any SGLang-selectable attention layers"
        ):
            with component_attn_backend_context_manager(
                AttentionBackendEnum.FA, component_name="vae"
            ):
                pass

        self.assertIsNone(get_component_attn_backend_context())

    def test_component_override_preserves_load_error_and_resets_context(self):
        with self.assertRaisesRegex(RuntimeError, "component failed"):
            with component_attn_backend_context_manager(
                AttentionBackendEnum.FA, component_name="vae"
            ):
                raise RuntimeError("component failed")

        self.assertIsNone(get_component_attn_backend_context())

    def test_automatic_component_backend_may_skip_sglang_attention_layer(self):
        with component_attn_backend_context_manager(
            AttentionBackendEnum.TORCH_SDPA,
            component_name="text_encoder",
            require_component_backend_selection=False,
        ):
            pass

        self.assertIsNone(get_component_attn_backend_context())

    def test_implicit_preference_falls_back_for_missing_capability(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=False,
            is_cross_attention=False,
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
            is_cross_attention=False,
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
                is_cross_attention=False,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )

    def test_explicit_global_backend_uses_component_default(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=True,
            is_cross_attention=False,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            default_attention_backend=AttentionBackendEnum.TORCH_SDPA,
        )

        self.assertIs(backend, _FakeSDPABackend)
        self.assertEqual(
            _FakePlatform.selected_backend, AttentionBackendEnum.TORCH_SDPA
        )

    def test_explicit_global_backend_falls_back_for_auxiliary_component(self):
        backend = self._resolve(
            AttentionBackendEnum.AITER,
            explicit=True,
            is_cross_attention=False,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            allow_global_backend_fallback=True,
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_explicit_component_backend_remains_strict(self):
        with self.assertRaisesRegex(
            ValueError, "not supported by this attention layer"
        ):
            self._resolve(
                AttentionBackendEnum.FA,
                explicit=True,
                is_cross_attention=False,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
                component_backend=AttentionBackendEnum.AITER,
                allow_global_backend_fallback=True,
            )

    def test_explicit_component_backend_is_consumed(self):
        backend = self._resolve(
            AttentionBackendEnum.TORCH_SDPA,
            explicit=True,
            is_cross_attention=False,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            component_backend=AttentionBackendEnum.FA,
        )

        self.assertIs(backend, _FakeFABackend)

    def test_sparse_backend_falls_back_for_cross_attention(self):
        backend = self._resolve(
            AttentionBackendEnum.LASER_ATTN,
            explicit=True,
            is_cross_attention=True,
            supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

        self.assertIs(backend, _FakeFABackend)
        self.assertIsNone(_FakePlatform.selected_backend)

    def test_sparse_backend_falls_back_for_unconstrained_cross_attention(self):
        backend = self._resolve(
            AttentionBackendEnum.LASER_ATTN,
            explicit=True,
            is_cross_attention=True,
            supported=set(),
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
                is_cross_attention=False,
                supported={AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
            )


class TestComponentAttentionBackendScope(unittest.TestCase):
    def _load_with_policy(self, allow_global_backend_fallback: bool):
        captured_context = None

        class _Loader(ComponentLoader):
            def load_customized(self, *_args):
                nonlocal captured_context
                captured_context = get_component_attn_backend_context()
                return object()

        class _Args:
            component_precisions = {}
            component_quantizations = {}

            @staticmethod
            def requested_component_attention_backend(_component_name):
                return None

            @staticmethod
            def should_direct_gpu_weight_load_component(_component_name):
                return False

            @staticmethod
            def should_use_fsdp_for_component(_component_name):
                return False

        _Loader.allow_global_attention_backend_fallback = allow_global_backend_fallback
        with (
            patch.object(ComponentLoader, "for_component_type", return_value=_Loader()),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.current_platform.get_available_gpu_memory",
                return_value=1.0,
            ),
        ):
            PipelineComponentLoader.load_component(
                component_name="text_encoder",
                component_model_path="unused",
                transformers_or_diffusers="transformers",
                server_args=_Args(),
                component_attn_name="text_encoder",
            )
        return captured_context

    def test_auxiliary_loader_enables_global_fallback(self):
        context = self._load_with_policy(True)

        self.assertIsNotNone(context)
        self.assertTrue(context.allow_global_backend_fallback)

    def test_dit_loader_keeps_global_backend_strict(self):
        context = self._load_with_policy(False)

        self.assertIsNotNone(context)
        self.assertFalse(context.allow_global_backend_fallback)

    def test_builtin_loader_scopes(self):
        self.assertFalse(TransformerLoader.allow_global_attention_backend_fallback)
        self.assertFalse(GenericComponentLoader.allow_global_attention_backend_fallback)
        self.assertTrue(TextEncoderLoader.allow_global_attention_backend_fallback)
        self.assertTrue(VAELoader.allow_global_attention_backend_fallback)

    def test_explicit_backend_must_be_consumed(self):
        with self.assertRaisesRegex(
            ComponentAttentionBackendNotAppliedError,
            "did not construct any SGLang-selectable attention layers",
        ):
            with component_attn_backend_context_manager(
                AttentionBackendEnum.FA,
                component_name="image_encoder",
                require_backend_selection=True,
            ):
                pass

    def test_deferred_selection_satisfies_construction_contract(self):
        with component_attn_backend_context_manager(
            AttentionBackendEnum.FA,
            component_name="transformer",
            require_backend_selection=True,
        ):
            self.assertIs(
                claim_deferred_component_attn_backend(),
                AttentionBackendEnum.FA,
            )

    def test_fixed_component_load_rejects_explicit_backend(self):
        class _Loader(ComponentLoader):
            def load_customized(self, *_args):
                return object()

        class _Args:
            component_precisions = {}
            component_quantizations = {}

            @staticmethod
            def requested_component_attention_backend(_component_name):
                return "fa"

            @staticmethod
            def should_direct_gpu_weight_load_component(_component_name):
                return False

            @staticmethod
            def should_use_fsdp_for_component(_component_name):
                return False

        with (
            patch.object(ComponentLoader, "for_component_type", return_value=_Loader()),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.current_platform.get_available_gpu_memory",
                return_value=1.0,
            ),
            self.assertRaisesRegex(
                ComponentAttentionBackendNotAppliedError,
                "did not construct any SGLang-selectable attention layers",
            ),
        ):
            PipelineComponentLoader.load_component(
                component_name="image_encoder",
                component_model_path="unused",
                transformers_or_diffusers="transformers",
                server_args=_Args(),
                component_attn_backend=AttentionBackendEnum.FA,
                component_attn_name="image_encoder",
            )

    def test_unexplained_mixed_backend_is_rejected(self):
        with self.assertRaisesRegex(
            ComponentAttentionBackendNotAppliedError,
            "also selected torch_sdpa without an allowed fallback",
        ):
            with component_attn_backend_context_manager(
                AttentionBackendEnum.FA,
                component_name="transformer",
                require_backend_selection=True,
            ):
                _record_component_attn_backend("fa", None)
                _record_component_attn_backend("torch_sdpa", None)
                _record_component_attn_backend(
                    "torch_sdpa", "dense cross-attention fallback"
                )

    def test_explicit_backend_preserves_customized_load_failure(self):
        native_load_called = False

        class _Loader(ComponentLoader):
            def load_customized(self, *_args):
                claim_deferred_component_attn_backend()
                raise RuntimeError("customized load failed")

            def load_native(self, *_args):
                nonlocal native_load_called
                native_load_called = True
                return object()

        class _Args:
            component_precisions = {}
            component_quantizations = {}
            pipeline_config = SimpleNamespace(native_only_components=())

            @staticmethod
            def requested_component_attention_backend(_component_name):
                return "fa"

            @staticmethod
            def should_direct_gpu_weight_load_component(_component_name):
                return False

            @staticmethod
            def should_use_fsdp_for_component(_component_name):
                return False

        with (
            patch.object(ComponentLoader, "for_component_type", return_value=_Loader()),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.current_platform.get_available_gpu_memory",
                return_value=1.0,
            ),
            self.assertRaisesRegex(RuntimeError, "customized load failed"),
        ):
            PipelineComponentLoader.load_component(
                component_name="text_encoder",
                component_model_path="unused",
                transformers_or_diffusers="transformers",
                server_args=_Args(),
                component_attn_backend=AttentionBackendEnum.FA,
                component_attn_name="text_encoder",
            )
        self.assertFalse(native_load_called)

    def test_legacy_fallback_uses_a_fresh_selection_context(self):
        customized_context = None
        native_context = None

        class _Loader(ComponentLoader):
            def load_customized(self, *_args):
                nonlocal customized_context
                customized_context = get_component_attn_backend_context()
                _record_component_attn_backend("fa", None)
                raise NativeComponentLoaderRequired("use native loader")

            def load_native(self, *_args):
                nonlocal native_context
                native_context = get_component_attn_backend_context()
                return object()

        class _Args:
            component_precisions = {}
            component_quantizations = {}
            pipeline_config = SimpleNamespace(native_only_components=())

            @staticmethod
            def requested_component_attention_backend(_component_name):
                return None

            @staticmethod
            def should_direct_gpu_weight_load_component(_component_name):
                return False

            @staticmethod
            def should_use_fsdp_for_component(_component_name):
                return False

        with (
            patch.object(ComponentLoader, "for_component_type", return_value=_Loader()),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.current_platform.get_available_gpu_memory",
                return_value=1.0,
            ),
        ):
            PipelineComponentLoader.load_component(
                component_name="text_encoder",
                component_model_path="unused",
                transformers_or_diffusers="transformers",
                server_args=_Args(),
                component_attn_name="text_encoder",
            )

        self.assertIsNotNone(customized_context)
        self.assertIsNotNone(native_context)
        self.assertIsNot(customized_context, native_context)
        self.assertEqual(customized_context.selected_backends, {"fa": None})
        self.assertEqual(native_context.selected_backends, {})

    def test_diffusers_backend_rejects_component_override(self):
        with self.assertRaisesRegex(
            ValueError, "supported only by native SGLang diffusion pipelines"
        ):
            DiffusersPipeline(
                "/unused",
                SimpleNamespace(
                    has_requested_component_attention_backends=lambda: True
                ),
            )


if __name__ == "__main__":
    unittest.main()
