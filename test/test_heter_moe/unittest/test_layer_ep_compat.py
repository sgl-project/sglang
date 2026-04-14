"""Test HeterFusedMoE compatibility with Expert Parallelism (EP).

EP shards experts across ranks: each rank holds num_experts/ep_size experts.
The token dispatcher remaps global expert IDs to local IDs before the MoE
layer, and sets non-local experts to sentinel=-1 (Triton) or sentinel=E (Marlin).

These tests validate that HeterFusedMoE produces correct results when:
  1. Initialized with a local subset of experts (simulating one EP rank)
  2. The EP remapping in forward() correctly maps global → local IDs
  3. Sum of all EP ranks' outputs equals the global (non-EP) output

No actual multi-GPU communication — this tests the kernel-level + remapping
behavior that would be needed for EP integration.
"""

import pytest
import torch

from test_heter_moe.util import CUDA_AVAILABLE, make_topk_output

E_GLOBAL = 8       # Total experts across all EP ranks
EP_SIZE = 2         # Number of EP ranks
E_LOCAL = E_GLOBAL // EP_SIZE  # Experts per rank
H, I, K = 128, 128, 2
GROUP_SIZE = 128
SEED = 42
BATCH = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(num_experts, config, ep_size=1, ep_rank=0, num_global_experts=None):
    from sglang.srt.layers.moe.heter_moe import HeterFusedMoE
    return HeterFusedMoE(
        num_experts=num_experts, hidden_size=H, intermediate_size=I,
        top_k=K, heter_config=config, dtype=torch.bfloat16,
        device=torch.device("cuda"),
        ep_size=ep_size, ep_rank=ep_rank,
        num_global_experts=num_global_experts,
    )


def _fill_int4(layer, bf16_w13, bf16_w2):
    from test_heter_moe.unittest.int4_marlin_weight_no_gptq import fill_int4_params
    fill_int4_params(layer, bf16_w13, bf16_w2, GROUP_SIZE)


def _make_bf16_config():
    return {
        "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
        "policy": "random", "policy_params": {"seed": SEED},
    }


def _make_mixed_config():
    return {
        "groups": [
            {"name": "cold_int4", "num_bits": 4, "size_ratio": 0.5, "group_size": GROUP_SIZE},
            {"name": "hot_bf16", "num_bits": 16, "size_ratio": 0.5},
        ],
        "policy": "random", "policy_params": {"seed": SEED},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def global_weights():
    """BF16 weights for all E_GLOBAL experts."""
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    w13 = torch.randn(E_GLOBAL, 2 * I, H, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
    w2 = torch.randn(E_GLOBAL, H, I, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
    return w13, w2


# ---------------------------------------------------------------------------
# 1. BF16: EP remapping produces correct split across ranks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestEPCompatBF16:
    """Validate BF16 HeterFusedMoE with EP expert remapping."""

    def test_local_subset_matches_global(self, global_weights):
        """When all tokens route to rank 0's experts, EP layer with remapping
        should produce the same output as a global (non-EP) layer."""
        w13_global, w2_global = global_weights

        # Global layer (no EP): all E_GLOBAL experts
        global_layer = _make_layer(E_GLOBAL, _make_bf16_config())
        global_layer.w13_weight.data.copy_(w13_global)
        global_layer.w2_weight.data.copy_(w2_global)

        # Inputs: all tokens route to experts [0, E_LOCAL) — rank 0's experts
        torch.manual_seed(0)
        x = torch.randn(BATCH, H, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, E_LOCAL, (BATCH, K), device="cuda")
        topk_weights = torch.rand(BATCH, K, dtype=torch.bfloat16, device="cuda")

        out_global = global_layer(x, make_topk_output(topk_weights, topk_ids))

        # EP layer (rank 0): E_LOCAL experts, with EP remapping enabled
        ep_layer = _make_layer(
            E_LOCAL, _make_bf16_config(),
            ep_size=EP_SIZE, ep_rank=0, num_global_experts=E_GLOBAL,
        )
        ep_layer.w13_weight.data.copy_(w13_global[:E_LOCAL])
        ep_layer.w2_weight.data.copy_(w2_global[:E_LOCAL])

        # Same global topk_ids — the layer's forward() should remap them
        out_ep = ep_layer(x, make_topk_output(topk_weights, topk_ids))

        torch.testing.assert_close(out_global, out_ep, atol=0, rtol=0)

    def test_two_rank_sum_equals_global(self, global_weights):
        """Sum of rank 0 + rank 1 outputs should equal global output (BF16)."""
        w13_global, w2_global = global_weights

        # Global layer
        global_layer = _make_layer(E_GLOBAL, _make_bf16_config())
        global_layer.w13_weight.data.copy_(w13_global)
        global_layer.w2_weight.data.copy_(w2_global)

        torch.manual_seed(1)
        x = torch.randn(BATCH, H, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, E_GLOBAL, (BATCH, K), device="cuda")
        topk_weights = torch.rand(BATCH, K, dtype=torch.bfloat16, device="cuda")

        out_global = global_layer(x, make_topk_output(topk_weights, topk_ids))

        # Rank 0: experts [0, E_LOCAL), with EP remapping
        rank0 = _make_layer(
            E_LOCAL, _make_bf16_config(),
            ep_size=EP_SIZE, ep_rank=0, num_global_experts=E_GLOBAL,
        )
        rank0.w13_weight.data.copy_(w13_global[:E_LOCAL])
        rank0.w2_weight.data.copy_(w2_global[:E_LOCAL])
        out_r0 = rank0(x, make_topk_output(topk_weights, topk_ids))

        # Rank 1: experts [E_LOCAL, E_GLOBAL), with EP remapping
        rank1 = _make_layer(
            E_LOCAL, _make_bf16_config(),
            ep_size=EP_SIZE, ep_rank=1, num_global_experts=E_GLOBAL,
        )
        rank1.w13_weight.data.copy_(w13_global[E_LOCAL:])
        rank1.w2_weight.data.copy_(w2_global[E_LOCAL:])
        out_r1 = rank1(x, make_topk_output(topk_weights, topk_ids))

        # In real EP, the all-reduce sums these. Simulate here.
        out_combined = out_r0 + out_r1

        torch.testing.assert_close(out_combined, out_global, atol=0, rtol=0)

    def test_non_local_experts_produce_zero(self, global_weights):
        """Tokens routed entirely to non-local experts produce zero output."""
        w13_global, w2_global = global_weights

        # Rank 0 layer: owns experts [0, E_LOCAL)
        rank0 = _make_layer(
            E_LOCAL, _make_bf16_config(),
            ep_size=EP_SIZE, ep_rank=0, num_global_experts=E_GLOBAL,
        )
        rank0.w13_weight.data.copy_(w13_global[:E_LOCAL])
        rank0.w2_weight.data.copy_(w2_global[:E_LOCAL])

        torch.manual_seed(2)
        x = torch.randn(BATCH, H, dtype=torch.bfloat16, device="cuda")
        # All tokens route to rank 1's experts [E_LOCAL, E_GLOBAL)
        topk_ids = torch.randint(E_LOCAL, E_GLOBAL, (BATCH, K), device="cuda")
        topk_weights = torch.rand(BATCH, K, dtype=torch.bfloat16, device="cuda")

        out = rank0(x, make_topk_output(topk_weights, topk_ids))
        assert out.abs().max() == 0, "Non-local experts should produce zero output"


# ---------------------------------------------------------------------------
# 2. Mixed BF16+INT4: EP remapping with mixed precision
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestEPCompatMixed:
    """Validate mixed BF16+INT4 HeterFusedMoE with EP remapping."""

    def test_local_mixed_produces_finite_output(self, global_weights):
        """Mixed-precision EP layer runs without errors."""
        w13_global, w2_global = global_weights
        w13_local = w13_global[:E_LOCAL].clone()
        w2_local = w2_global[:E_LOCAL].clone()

        layer = _make_layer(
            E_LOCAL, _make_mixed_config(),
            ep_size=EP_SIZE, ep_rank=0, num_global_experts=E_GLOBAL,
        )
        layer.w13_weight.data.copy_(w13_local)
        layer.w2_weight.data.copy_(w2_local)
        _fill_int4(layer, w13_local, w2_local)

        torch.manual_seed(3)
        x = torch.randn(BATCH, H, dtype=torch.bfloat16, device="cuda")
        # Global IDs spanning all experts — EP remapping should handle this
        topk_ids = torch.randint(0, E_GLOBAL, (BATCH, K), device="cuda")
        topk_weights = torch.rand(BATCH, K, dtype=torch.bfloat16, device="cuda")
        router_logits = torch.randn(BATCH, E_LOCAL, device="cuda")

        out = layer(x, make_topk_output(topk_weights, topk_ids, router_logits))

        assert out.shape == x.shape
        assert out.isfinite().all(), "EP mixed output contains NaN/Inf"

    def test_two_rank_mixed_sum_finite_and_reasonable(self, global_weights):
        """Sum of two EP ranks' mixed outputs is finite and has reasonable magnitude."""
        w13_global, w2_global = global_weights

        torch.manual_seed(4)
        x = torch.randn(BATCH, H, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, E_GLOBAL, (BATCH, K), device="cuda")
        topk_weights = torch.rand(BATCH, K, dtype=torch.bfloat16, device="cuda")

        # Global mixed layer (no EP)
        global_layer = _make_layer(E_GLOBAL, _make_mixed_config())
        global_layer.w13_weight.data.copy_(w13_global)
        global_layer.w2_weight.data.copy_(w2_global)
        _fill_int4(global_layer, w13_global, w2_global)
        router_logits_global = torch.randn(BATCH, E_GLOBAL, device="cuda")
        out_global = global_layer(
            x, make_topk_output(topk_weights, topk_ids, router_logits_global)
        )

        # Rank 0
        r0 = _make_layer(
            E_LOCAL, _make_mixed_config(),
            ep_size=EP_SIZE, ep_rank=0, num_global_experts=E_GLOBAL,
        )
        r0.w13_weight.data.copy_(w13_global[:E_LOCAL])
        r0.w2_weight.data.copy_(w2_global[:E_LOCAL])
        _fill_int4(r0, w13_global[:E_LOCAL], w2_global[:E_LOCAL])
        r0_logits = torch.randn(BATCH, E_LOCAL, device="cuda")
        out_r0 = r0(x, make_topk_output(topk_weights, topk_ids, r0_logits))

        # Rank 1
        r1 = _make_layer(
            E_LOCAL, _make_mixed_config(),
            ep_size=EP_SIZE, ep_rank=1, num_global_experts=E_GLOBAL,
        )
        r1.w13_weight.data.copy_(w13_global[E_LOCAL:])
        r1.w2_weight.data.copy_(w2_global[E_LOCAL:])
        _fill_int4(r1, w13_global[E_LOCAL:], w2_global[E_LOCAL:])
        r1_logits = torch.randn(BATCH, E_LOCAL, device="cuda")
        out_r1 = r1(x, make_topk_output(topk_weights, topk_ids, r1_logits))

        out_combined = out_r0 + out_r1

        assert out_combined.isfinite().all(), "Combined EP output has NaN/Inf"
        assert out_global.isfinite().all(), "Global output has NaN/Inf"

        # Mixed precision dispatch differs between global (8 experts) and
        # per-rank (4 experts each) due to different policy groupings.
        # We verify both are finite and in the same ballpark.
        ratio = out_combined.abs().mean() / (out_global.abs().mean() + 1e-8)
        assert 0.1 < ratio < 10.0, (
            f"EP combined vs global magnitude ratio {ratio:.3f} is unreasonable"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
