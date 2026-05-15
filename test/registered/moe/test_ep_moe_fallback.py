import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add python path
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        ),
        "python",
    )
)

# Try to import DeepEPMoE
try:
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    print("DeepEPMoE imported successfully")
except ImportError as e:
    print(f"Failed to import DeepEPMoE: {e}")
    DeepEPMoE = None


class TestDeepEPMoEFallback(unittest.TestCase):
    def setUp(self):
        if DeepEPMoE is None:
            self.skipTest("DeepEPMoE could not be imported")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    @patch("sglang.srt.layers.moe.ep_moe.layer.deep_gemm_wrapper")
    @patch("sglang.srt.layers.moe.ep_moe.layer.get_deepep_mode")
    @patch("sglang.srt.layers.moe.ep_moe.layer.get_moe_runner_backend")
    @patch("sglang.srt.layers.moe.ep_moe.layer.outplace_fused_experts")
    @patch("sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.__init__", return_value=None)
    def test_forward_fused_moe_fallback(
        self,
        mock_init,
        mock_outplace_fused_experts,
        mock_get_moe_runner_backend,
        mock_get_deepep_mode,
        mock_deep_gemm_wrapper,
    ):
        """Test that forward_fused_moe_fallback calls outplace_fused_experts with correct arguments."""

        # Instantiate layer (mocked __init__)
        layer = DeepEPMoE(
            num_experts=8, top_k=2, hidden_size=64, intermediate_size=128, layer_id=0
        )

        # Manually set attributes
        layer.num_local_experts = 8
        layer.w13_weight = MagicMock()
        layer.w2_weight = MagicMock()
        layer.moe_runner_config = MagicMock()
        layer.moe_runner_config.activation = "silu"
        layer.moe_runner_config.is_gated = True
        layer.moe_runner_config.apply_router_weight_on_input = False
        layer.moe_runner_config.routed_scaling_factor = 1.0

        # Create dummy dispatch_output
        batch_size = 2
        top_k = 2
        hidden_size = 64

        hidden_states = torch.randn(batch_size * top_k, hidden_size, device="cuda")
        # num_tokens_per_expert for 8 experts.
        # Total tokens = 4. Experts 0, 1, 2, 3 get 1 token each.
        num_tokens_per_expert = torch.tensor(
            [1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.randn(batch_size * top_k, device="cuda")

        dispatch_output = MagicMock()
        dispatch_output.hidden_states = hidden_states
        dispatch_output.num_tokens_per_expert = num_tokens_per_expert
        dispatch_output.topk_weights = topk_weights

        # Call the fallback method
        layer.forward_fused_moe_fallback(dispatch_output)

        # Verify outplace_fused_experts is called
        mock_outplace_fused_experts.assert_called_once()

        call_kwargs = mock_outplace_fused_experts.call_args.kwargs
        self.assertTrue(torch.equal(call_kwargs["hidden_states"], hidden_states))

        # Verify topk_ids construction
        # experts_range = [0..7]
        # repeat_interleave([0..7], [1,1,1,1,0,0,0,0]) -> [0, 1, 2, 3]
        expected_topk_ids = torch.tensor(
            [[0], [1], [2], [3]], dtype=torch.int32, device="cuda"
        )

        self.assertTrue(torch.equal(call_kwargs["topk_ids"], expected_topk_ids))
        self.assertTrue(call_kwargs["no_combine"])

    @patch("sglang.srt.layers.moe.ep_moe.layer.deep_gemm_wrapper")
    @patch("sglang.srt.layers.moe.ep_moe.layer.get_deepep_mode")
    @patch("sglang.srt.layers.moe.ep_moe.layer.get_moe_runner_backend")
    @patch(
        "sglang.srt.layers.moe.ep_moe.layer.FusedMoE.__init__"
    )  # Mock super init to avoid heavy lifting
    def test_init_warning_fallback(
        self,
        mock_super_init,
        mock_get_moe_runner_backend,
        mock_get_deepep_mode,
        mock_deep_gemm_wrapper,
    ):
        """Test that __init__ warns instead of asserts when DeepGEMM is missing."""

        # Setup mocks to trigger the check
        mock_deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = False

        mock_mode = MagicMock()
        mock_mode.enable_low_latency.return_value = True
        mock_get_deepep_mode.return_value = mock_mode

        mock_backend = MagicMock()
        mock_backend.is_flashinfer_cutedsl.return_value = False
        mock_get_moe_runner_backend.return_value = mock_backend

        mock_quant_config = MagicMock()
        mock_quant_config.get_name.return_value = "unquantized"

        # Capture logs
        with self.assertLogs(
            "sglang.srt.layers.moe.ep_moe.layer", level="WARNING"
        ) as cm:
            layer = DeepEPMoE(
                num_experts=8,
                top_k=2,
                hidden_size=64,
                intermediate_size=128,
                layer_id=0,
                quant_config=mock_quant_config,
            )

        self.assertTrue(any("DeepGEMM is not available" in o for o in cm.output))


if __name__ == "__main__":
    unittest.main()
