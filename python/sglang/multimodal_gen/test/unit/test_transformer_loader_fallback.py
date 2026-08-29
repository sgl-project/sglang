import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)


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


if __name__ == "__main__":
    unittest.main()
