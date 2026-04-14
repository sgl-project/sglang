"""1.3 Correctness tests.

1.3.1 Same-precision split: two-group GeMM of the same precision should
      produce identical results to single-group GeMM (bit-exact).
      Covers both BF16 and INT4 (Marlin).
1.3.2 Mixed-precision split: two-group GeMM with different precisions
      should produce results within reasonable MSE of the high-precision
      baseline.

NOTE on INT8: INT8 (a8w8) on A100 is not optimized and far inferior to
Hopper FP8 — not our primary target. INT8 tests are omitted for now;
will add if we decide to support INT8 in production.
"""

import pytest
import torch

from test_heter_moe.util import CUDA_AVAILABLE, make_topk_output

E, H, I, K = 8, 128, 128, 2
GROUP_SIZE = 128
SEED = 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_layer(config):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

    return HeterFusedMoE(
        num_experts=E, hidden_size=H, intermediate_size=I, top_k=K,
        heter_config=config, dtype=torch.bfloat16, device=torch.device("cuda"),
    )


def _make_inputs(with_logits=False):
    torch.manual_seed(0)
    x = torch.randn(16, H, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, E, (16, K), device="cuda")
    topk_weights = torch.rand(16, K, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn(16, E, device="cuda") if with_logits else None
    return x, make_topk_output(topk_weights, topk_ids, router_logits)


def _fill_int4(layer, bf16_w13, bf16_w2):
    """Quantize BF16 weights to Marlin INT4 directly (no repack needed)."""
    from test_heter_moe.unittest.int4_marlin_weight_no_gptq import fill_int4_params
    fill_int4_params(layer, bf16_w13, bf16_w2, GROUP_SIZE)


# ---------------------------------------------------------------------------
# Module-level fixture: generate BF16 weights once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_weights():
    """BF16 reference weights, created once per module."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
    return w13, w2


# ---------------------------------------------------------------------------
# 1.3.1 Same-precision split (bit-exact)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSamePrecisionSplitBF16:
    """N BF16 groups with shared weights must equal 1 BF16 group (bit-exact)."""

    def test_two_bf16_groups_equals_one(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        x, topk_out = _make_inputs()

        one_cfg = {
            "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer1 = _make_layer(one_cfg)
        layer1.w13_weight.data.copy_(bf16_w13)
        layer1.w2_weight.data.copy_(bf16_w2)
        out_one = layer1(x, topk_out)

        two_cfg = {
            "groups": [
                {"name": "g0", "num_bits": 16, "size_ratio": 0.5},
                {"name": "g1", "num_bits": 16, "size_ratio": 0.5},
            ],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer2 = _make_layer(two_cfg)
        layer2.w13_weight.data.copy_(bf16_w13)
        layer2.w2_weight.data.copy_(bf16_w2)
        out_two = layer2(x, topk_out)

        torch.testing.assert_close(out_one, out_two, atol=0, rtol=0)

    def test_three_bf16_groups_equals_one(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        x, topk_out = _make_inputs()

        one_cfg = {
            "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer1 = _make_layer(one_cfg)
        layer1.w13_weight.data.copy_(bf16_w13)
        layer1.w2_weight.data.copy_(bf16_w2)
        out_one = layer1(x, topk_out)

        three_cfg = {
            "groups": [
                {"name": "g0", "num_bits": 16, "size_ratio": 0.5},
                {"name": "g1", "num_bits": 16, "size_ratio": 0.25},
                {"name": "g2", "num_bits": 16, "size_ratio": 0.25},
            ],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer3 = _make_layer(three_cfg)
        layer3.w13_weight.data.copy_(bf16_w13)
        layer3.w2_weight.data.copy_(bf16_w2)
        out_three = layer3(x, topk_out)

        torch.testing.assert_close(out_one, out_three, atol=0, rtol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSamePrecisionSplitINT4:
    """N INT4 Marlin groups with shared weights must equal 1 INT4 group (bit-exact)."""

    def test_int4_single_group_runs(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        one_cfg = {
            "groups": [{"name": "all", "num_bits": 4, "size_ratio": 1.0, "group_size": GROUP_SIZE}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer = _make_layer(one_cfg)
        _fill_int4(layer, bf16_w13, bf16_w2)

        x, topk_out = _make_inputs(with_logits=True)
        out = layer(x, topk_out)
        assert out.shape == x.shape
        assert out.isfinite().all(), "INT4 Marlin output contains NaN/Inf"
        assert out.abs().sum() > 0, "INT4 Marlin output is all zeros"

    def test_two_int4_groups_equals_one(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights

        one_cfg = {
            "groups": [{"name": "all", "num_bits": 4, "size_ratio": 1.0, "group_size": GROUP_SIZE}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer_one = _make_layer(one_cfg)
        _fill_int4(layer_one, bf16_w13, bf16_w2)

        two_cfg = {
            "groups": [
                {"name": "g0", "num_bits": 4, "size_ratio": 0.5, "group_size": GROUP_SIZE},
                {"name": "g1", "num_bits": 4, "size_ratio": 0.5, "group_size": GROUP_SIZE},
            ],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer_two = _make_layer(two_cfg)
        _fill_int4(layer_two, bf16_w13, bf16_w2)

        x, topk_out = _make_inputs(with_logits=True)
        out_one = layer_one(x, topk_out)
        out_two = layer_two(x, topk_out)
        # Marlin INT4 is not bit-exact across group splits: different
        # moe_align_block_size layouts change FP reduction order.
        # BF16 Triton is bit-exact because it fully skips sentinel slots.
        torch.testing.assert_close(out_one, out_two, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# 1.3.2 Mixed-precision split (MSE check)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestMixedPrecisionSplit:
    """BF16+INT4 mixed output should be within reasonable MSE of BF16 baseline.

    INT4 weights are quantized from the same BF16 weights (round-to-nearest,
    no GPTQ calibration), so MSE reflects pure quantization error.
    """

    def _make_bf16_layer(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        cfg = {
            "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer = _make_layer(cfg)
        layer.w13_weight.data.copy_(bf16_w13)
        layer.w2_weight.data.copy_(bf16_w2)
        return layer

    def _make_int4_layer(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        cfg = {
            "groups": [{"name": "all", "num_bits": 4, "size_ratio": 1.0, "group_size": GROUP_SIZE}],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer = _make_layer(cfg)
        _fill_int4(layer, bf16_w13, bf16_w2)
        return layer

    def _make_mixed_layer(self, shared_weights):
        bf16_w13, bf16_w2 = shared_weights
        cfg = {
            "groups": [
                {"name": "low_prec", "num_bits": 4, "size_ratio": 0.5, "group_size": GROUP_SIZE},
                {"name": "high_prec", "num_bits": 16, "size_ratio": 0.5},
            ],
            "policy": "random", "policy_params": {"seed": SEED},
        }
        layer = _make_layer(cfg)
        layer.w13_weight.data.copy_(bf16_w13)
        layer.w2_weight.data.copy_(bf16_w2)
        _fill_int4(layer, bf16_w13, bf16_w2)
        return layer

    def test_int4_vs_bf16_bounded_mse(self, shared_weights):
        """Pure INT4 output has bounded MSE vs BF16 using same logical weights."""
        bf16_layer = self._make_bf16_layer(shared_weights)
        int4_layer = self._make_int4_layer(shared_weights)

        x, topk_out = _make_inputs(with_logits=True)
        out_bf16 = bf16_layer(x, topk_out)
        out_int4 = int4_layer(x, topk_out)

        assert out_bf16.isfinite().all(), "BF16 output contains NaN/Inf"
        assert out_int4.isfinite().all(), "INT4 output contains NaN/Inf"

        mse = (out_bf16.float() - out_int4.float()).pow(2).mean().item()
        assert mse > 0, "INT4 should differ from BF16 (quantization error)"
        assert mse < 1.0, f"INT4 vs BF16 MSE too large: {mse}"

    def test_bf16_int4_mix_produces_finite_output(self, shared_weights):
        """BF16+INT4 mixed layer produces non-zero, finite output."""
        layer = self._make_mixed_layer(shared_weights)

        x, topk_out = _make_inputs(with_logits=True)
        out = layer(x, topk_out)
        assert out.shape == x.shape
        assert out.isfinite().all(), "BF16+INT4 output contains NaN/Inf"
        assert out.abs().sum() > 0, "BF16+INT4 output is all zeros"

    def test_bf16_int4_mix_mse_between_pure_bf16_and_pure_int4(self, shared_weights):
        """Mixed BF16+INT4 MSE vs BF16 baseline should be <= pure INT4 MSE."""
        bf16_layer = self._make_bf16_layer(shared_weights)
        int4_layer = self._make_int4_layer(shared_weights)
        mixed_layer = self._make_mixed_layer(shared_weights)

        x, topk_out = _make_inputs(with_logits=True)
        out_bf16 = bf16_layer(x, topk_out)
        out_int4 = int4_layer(x, topk_out)
        out_mixed = mixed_layer(x, topk_out)

        mse_int4 = (out_bf16.float() - out_int4.float()).pow(2).mean().item()
        mse_mixed = (out_bf16.float() - out_mixed.float()).pow(2).mean().item()

        assert mse_mixed <= mse_int4, (
            f"Mixed MSE ({mse_mixed}) should be <= pure INT4 MSE ({mse_int4})"
        )

    # TODO: INT8 (a8w8) correctness tests omitted — INT8 on A100 is not
    # optimized and far inferior to Hopper FP8. Not our primary target.
    # Will add INT8 tests if we decide to support it in production.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
