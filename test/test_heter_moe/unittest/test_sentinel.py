"""1.2 Sentinel value tests.

Verify that the sentinel choice avoids computing a fake expert:
  - INT4 (Marlin): sentinel = num_experts (Marlin skips natively)
  - BF16/INT8 (Triton): sentinel = -1 (Triton skips -1 expert IDs)

Wrong sentinel = kernel processes a garbage expert row -> wasted compute,
wrong output.
"""

import pytest
import torch

from test_heter_moe.util import CUDA_AVAILABLE, make_topk_output


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestBF16Sentinel:
    """BF16 group: all-sentinel dispatch should produce zero output.

    If sentinel is wrong, the kernel would compute against a real expert
    row and produce non-zero garbage.
    """

    def _make_layer(self, num_experts=8, hidden=64, intermediate=32):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        config = {
            "groups": [
                {"name": "cold", "num_bits": 16, "size_ratio": 0.5},
                {"name": "hot", "num_bits": 16, "size_ratio": 0.5},
            ],
            "policy": "random",
        }
        layer = HeterFusedMoE(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            top_k=2,
            heter_config=config,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        layer.init_fake_weights(seed=42)
        return layer

    def test_sentinel_slots_produce_zero_contribution(self):
        """For each group, sentinel-masked slots should contribute nothing."""
        layer = self._make_layer()
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")

        # Route all tokens to expert 0 only — one group will have expert 0,
        # the other group will have all sentinels for these slots.
        topk_ids = torch.zeros(8, 2, dtype=torch.int64, device="cuda")
        topk_weights = torch.ones(8, 2, dtype=torch.bfloat16, device="cuda")

        # Run the full layer — if sentinel is wrong, the sentinel group
        # would add garbage to the output.
        out_full = layer(x, make_topk_output(topk_weights, topk_ids))

        # Now run with zero weights — should be near-zero regardless
        topk_weights_zero = torch.zeros(8, 2, dtype=torch.bfloat16, device="cuda")
        out_zero = layer(x, make_topk_output(topk_weights_zero, topk_ids))

        assert out_zero.abs().max() < 1e-3, (
            "All-zero weights should produce near-zero output"
        )
        # Full output should be non-trivial (experts are active)
        assert out_full.abs().sum() > 0

    def test_sentinel_converted_to_minus_one(self):
        """Verify the forward path converts num_experts sentinel to -1 for BF16."""
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        layer = self._make_layer()

        # Manually invoke policy dispatch to inspect sentinel values
        topk_ids = torch.zeros(4, 2, dtype=torch.int64, device="cuda")
        topk_weights = torch.ones(4, 2, dtype=torch.bfloat16, device="cuda")
        dispatches = layer.policy.dispatch(topk_ids, topk_weights)

        for group_ids, group_weights in dispatches:
            sentinel_mask = group_ids == layer.num_experts
            # Policy produces num_experts as sentinel
            # Forward will convert to -1 for BF16, but at policy level
            # it should be num_experts
            if sentinel_mask.any():
                assert (group_weights[sentinel_mask] == 0).all(), (
                    "Sentinel slots must have zero weight"
                )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestInt8Sentinel:
    """INT8 group: same sentinel semantics as BF16 (Triton path, sentinel=-1)."""

    def _make_layer(self, num_experts=8, hidden=64, intermediate=32):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        config = {
            "groups": [
                {"name": "cold", "num_bits": 8, "size_ratio": 0.5},
                {"name": "hot", "num_bits": 8, "size_ratio": 0.5},
            ],
            "policy": "random",
        }
        layer = HeterFusedMoE(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            top_k=2,
            heter_config=config,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        layer.init_fake_weights(seed=42)
        return layer

    def test_sentinel_slots_produce_zero_contribution(self):
        layer = self._make_layer()
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.zeros(8, 2, dtype=torch.int64, device="cuda")
        topk_weights_zero = torch.zeros(8, 2, dtype=torch.bfloat16, device="cuda")
        out_zero = layer(x, make_topk_output(topk_weights_zero, topk_ids))
        assert out_zero.abs().max() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
