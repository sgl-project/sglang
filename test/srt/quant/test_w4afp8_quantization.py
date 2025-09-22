import unittest
import torch
from unittest.mock import MagicMock, patch

from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod


class TestW4AFp8Config(unittest.TestCase):
    def test_config_init(self):
        config = W4AFp8Config(
            is_checkpoint_fp8_serialized=True,
            is_checkpoint_w4afp8_serialized=True,
            moe_activation_scheme="static",
            group_size=128,
        )
        self.assertTrue(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.moe_activation_scheme, "static")
        self.assertEqual(config.group_size, 128)

    def test_invalid_activation_scheme(self):
        with self.assertRaises(ValueError):
            W4AFp8Config(moe_activation_scheme="invalid")

    def test_from_config(self):
        config = W4AFp8Config.from_config({"quant_method": "w4afp8"})
        self.assertTrue(config.is_checkpoint_w4afp8_serialized)


class TestW4AFp8MoEMethod(unittest.TestCase):
    def setUp(self):
        self.config = W4AFp8Config()
        self.method = W4AFp8MoEMethod(self.config)

    def test_supported_backends(self):
        from sglang.srt.layers.moe import MoeRunnerBackend
        backends = self.method.get_supported_backends()
        self.assertIn(MoeRunnerBackend.CUTLASS_W4A8, backends)

    @patch('sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe')
    def test_apply(self, mock_cutlass):
        from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

        # Mock layer and dispatch output
        layer = MagicMock()
        layer.num_experts = 8

        dispatch_output = MagicMock(spec=StandardDispatchOutput)
        dispatch_output.hidden_states = torch.randn(16, 512)
        dispatch_output.topk_output = (
            torch.randn(16, 2),  # weights
            torch.randint(0, 8, (16, 2)),  # ids
            None
        )

        # Test apply
        mock_cutlass.return_value = torch.randn(16, 512)
        result = self.method.apply(layer, dispatch_output)

        self.assertIsNotNone(result)
        mock_cutlass.assert_called_once()


if __name__ == "__main__":
    unittest.main()