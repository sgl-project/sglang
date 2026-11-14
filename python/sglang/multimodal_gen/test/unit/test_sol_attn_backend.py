import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn import (
    SolAttnImpl,
    _get_sol_attn_runtime_config,
    _parse_layer_ranges,
)


class TestSolAttnBackend(unittest.TestCase):
    def test_parse_layer_ranges(self):
        self.assertEqual(_parse_layer_ranges("0,1,3-5"), frozenset({0, 1, 3, 4, 5}))

    def test_dense_backend_aliases(self):
        for raw, expected in (
            ("fa", "fa"),
            ("sage", "sage_attn"),
            ("sage_attn", "sage_attn"),
        ):
            server_args = MagicMock()
            server_args.attention_backend_config = {"dense_backend": raw}
            with patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ):
                self.assertEqual(
                    _get_sol_attn_runtime_config()["dense_backend"], expected
                )
        server_args = MagicMock()
        server_args.attention_backend_config = {"dense_backend": "torch_sdpa"}
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            self.assertRaises(ValueError),
        ):
            _get_sol_attn_runtime_config()

    def test_dense_guard_uses_early_steps(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 3
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertTrue(impl._should_use_dense())

    def test_sparse_layer_after_dense_guard(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 20
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertFalse(impl._should_use_dense())


if __name__ == "__main__":
    unittest.main()
