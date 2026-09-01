import re
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.loader.component_loaders.bridge_loader import (
    BridgeLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
    NativeComponentLoaderRequired,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestTransformerLoaderFallbackAdmission(unittest.TestCase):
    @staticmethod
    def _server_args(*, fsdp_requested=False, **overrides):
        values = {
            "component_precisions": {},
            "component_quantizations": {},
            "component_weights_paths": {},
            "component_quantization_ignored_layers": {},
            "transformer_weights_path": None,
            "nunchaku_config": None,
            "quantization": None,
            "pipeline_config": SimpleNamespace(native_only_components=()),
            "tp_size": 1,
            "sp_degree": 1,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "kv_gather_degree": 1,
            "enable_cfg_parallel": False,
            "dp_size": 1,
            "use_fsdp_inference": False,
            "resolve_component_attention_backend": mock.Mock(return_value=(None, None)),
            "requested_component_attention_backend": mock.Mock(return_value=None),
            "should_direct_gpu_weight_load_component": mock.Mock(return_value=False),
            "should_use_fsdp_for_component": mock.Mock(return_value=fsdp_requested),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _mocked_load(loader):
        customized_load = mock.patch.object(
            loader,
            "_load_customized_with_context",
            side_effect=NativeComponentLoaderRequired("native loader required"),
        )
        native_load = mock.patch.object(
            loader, "_load_native_with_context", return_value=object()
        )
        available_memory = mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.current_platform.get_available_gpu_memory",
            return_value=0.0,
        )
        return customized_load, native_load, available_memory

    def test_distributed_execution_rejects_before_native_load(self):
        cases = (
            ("tp_size", 2, "tp_size=2"),
            ("sp_degree", 2, "sp_degree=2"),
            ("ulysses_degree", 2, "ulysses_degree=2"),
            ("ring_degree", 2, "ring_degree=2"),
            ("kv_gather_degree", 2, "kv_gather_degree=2"),
            ("fsdp_requested", True, "FSDP"),
        )

        for field, value, expected_error in cases:
            with self.subTest(field=field):
                loader = TransformerLoader()
                server_args = self._server_args(**{field: value})
                customized_load, native_load, available_memory = self._mocked_load(
                    loader
                )

                with customized_load, native_load as native, available_memory:
                    with self.assertRaisesRegex(
                        RuntimeError, re.escape(expected_error)
                    ):
                        loader.load(
                            "/model/transformer_2",
                            server_args,
                            "transformer_2",
                            "diffusers",
                        )

                native.assert_not_called()
                server_args.should_use_fsdp_for_component.assert_called_once_with(
                    "transformer_2"
                )

    def test_replicated_cfg_and_dp_keep_native_fallback(self):
        loader = TransformerLoader()
        server_args = self._server_args(
            enable_cfg_parallel=True,
            dp_size=2,
            use_fsdp_inference=True,
        )
        customized_load, native_load, available_memory = self._mocked_load(loader)

        with customized_load, native_load as native, available_memory:
            component, consumed = loader.load(
                "/model/transformer_2",
                server_args,
                "transformer_2",
                "diffusers",
            )

        self.assertIsNotNone(component)
        self.assertEqual(consumed, 0.0)
        native.assert_called_once()
        server_args.should_use_fsdp_for_component.assert_called_once_with(
            "transformer_2"
        )

    def test_parallel_execution_rejects_native_fallback(self):
        cases = (
            ({"tp_size": 2}, "tp_size=2"),
            ({"sp_degree": 2}, "sp_degree=2"),
            ({"ulysses_degree": 2}, "ulysses_degree=2"),
            ({"ring_degree": 2}, "ring_degree=2"),
            ({"kv_gather_degree": 2}, "kv_gather_degree=2"),
            ({"fsdp_requested": True}, "FSDP"),
        )

        for overrides, expected_error in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    TransformerLoader().validate_native_fallback(
                        self._server_args(**overrides), "transformer_2"
                    )

    def test_replicated_execution_keeps_native_fallback_available(self):
        self.assertIsNone(
            TransformerLoader().validate_native_fallback(
                self._server_args(), "transformer_2"
            )
        )

    def test_only_fsdp_materializers_keep_the_component_request(self):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.use_fsdp_inference = True
        server_args._fsdp_disabled_components = set()
        server_args.residency_mode = lambda _component: RESIDENT

        ComponentLoader().disable_unsupported_component_fsdp(
            server_args, "text_encoder"
        )
        self.assertFalse(server_args.should_use_fsdp_for_component("text_encoder"))

        TransformerLoader().disable_unsupported_component_fsdp(
            server_args, "transformer"
        )
        BridgeLoader().disable_unsupported_component_fsdp(
            server_args, "dual_tower_bridge"
        )
        self.assertTrue(server_args.should_use_fsdp_for_component("transformer"))
        self.assertTrue(server_args.should_use_fsdp_for_component("dual_tower_bridge"))


if __name__ == "__main__":
    unittest.main()
