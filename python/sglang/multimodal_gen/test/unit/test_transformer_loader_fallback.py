import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.loader.component_loaders.bridge_loader import (
    BridgeLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
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
            "tp_size": 1,
            "sp_degree": 1,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "should_use_fsdp_for_component": lambda _component: fsdp_requested,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_parallel_execution_rejects_native_fallback(self):
        cases = (
            ({"tp_size": 2}, "tp_size=2"),
            ({"sp_degree": 2}, "sp_degree=2"),
            ({"ulysses_degree": 2}, "ulysses_degree=2"),
            ({"ring_degree": 2}, "ring_degree=2"),
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
