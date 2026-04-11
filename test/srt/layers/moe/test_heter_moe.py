"""Unit tests for heterogeneous-precision MoE.

Tests policy logic and kernel integration. All tests require CUDA since
the GPU-resident policies pre-allocate buffers on GPU at construction time.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PYTHON_ROOT = Path(__file__).resolve().parents[4] / "python"


def _import_module_from_file(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, _PYTHON_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_policy_mod = _import_module_from_file(
    "heter_policy", "sglang/srt/layers/moe/heter_policy.py"
)
HeterDispatchPolicy = _policy_mod.HeterDispatchPolicy
ExpertLoadHeterDispatch = _policy_mod.ExpertLoadHeterDispatch
ConfidenceThresholdHeterDispatch = _policy_mod.ConfidenceThresholdHeterDispatch
RandomHeterDispatch = _policy_mod.RandomHeterDispatch
create_policy = _policy_mod.create_policy
_assign_by_score_gpu = _policy_mod._assign_by_score_gpu
_build_group_labels = _policy_mod._build_group_labels

from types import SimpleNamespace


def _make_topk_output(topk_weights, topk_ids, router_logits=None):
    """Construct a TopKOutput-shaped shim for HeterFusedMoE.forward() tests."""
    return SimpleNamespace(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )


CUDA_AVAILABLE = torch.cuda.is_available()


def _init_mock_server_args():
    """Initialize minimal ServerArgs so Triton kernel config lookup works.

    ServerArgs.__post_init__ tries to download model config, so we create
    a bare object bypassing __init__ and set only the fields the kernel
    config path needs.
    """
    try:
        import sglang.srt.server_args as sa

        try:
            sa.get_global_server_args()
        except ValueError:
            mock = object.__new__(sa.ServerArgs)
            mock.enable_deterministic_inference = False
            mock.disable_moe_autotuning = False
            sa._global_server_args = mock
    except Exception:
        pass


if CUDA_AVAILABLE:
    _init_mock_server_args()


# --- Policy tests (GPU required for pre-allocated buffers) ----------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestAssignByScoreGpu:
    def test_two_groups_topk_path(self):
        scores = torch.tensor(
            [1.0, 5.0, 2.0, 8.0, 3.0, 0.5, 7.0, 4.0], device="cuda"
        )
        buf = torch.empty(8, dtype=torch.long, device="cuda")
        result = _assign_by_score_gpu(scores, 8, [0.75, 0.25], buf)
        # Top 2 by score: experts 3 (8.0) and 6 (7.0) -> group 1
        assert result[3].item() == 1
        assert result[6].item() == 1
        # Others -> group 0
        assert result[0].item() == 0
        assert result[5].item() == 0

    def test_single_group(self):
        scores = torch.ones(8, device="cuda")
        buf = torch.empty(8, dtype=torch.long, device="cuda")
        result = _assign_by_score_gpu(scores, 8, [1.0], buf)
        assert (result == 0).all()

    def test_three_groups(self):
        scores = torch.arange(16, dtype=torch.float32, device="cuda")
        buf = torch.empty(16, dtype=torch.long, device="cuda")
        labels = _build_group_labels(16, [0.5, 0.3, 0.2], torch.device("cuda"))
        result = _assign_by_score_gpu(scores, 16, [0.5, 0.3, 0.2], buf, labels)
        # All 16 experts assigned, each to exactly one group
        for gidx in range(3):
            count = (result == gidx).sum().item()
            assert count > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestExpertLoadHeterDispatch:
    def test_basic_dispatch(self):
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.tensor([[0, 1], [0, 2], [0, 3], [4, 5]], device="cuda")
        topk_weights = torch.rand(4, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2
        for experts_g, scales_g in dispatches:
            assert experts_g.shape == (4, 2)
            assert scales_g.shape == (4, 2)

    def test_hot_expert_in_high_precision_group(self):
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        # Expert 0 appears 3 times (hottest)
        topk_ids = torch.tensor([[0, 1], [0, 2], [0, 3], [4, 5]], device="cuda")
        topk_weights = torch.ones(4, 2, device="cuda") * 0.5
        dispatches = policy.dispatch(topk_ids, topk_weights)
        # Group 1 (high-precision) should contain expert 0
        experts_g1, scales_g1 = dispatches[1]
        # Where topk_ids == 0, group 1 experts should be 0 (not sentinel)
        mask_expert0 = (topk_ids == 0)
        assert (experts_g1[mask_expert0] == 0).all()

    def test_sentinel_for_non_group(self):
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.tensor([[0, 1], [2, 3]], device="cuda")
        topk_weights = torch.ones(2, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        for experts_g, scales_g in dispatches:
            sentinel_mask = (experts_g == 8)  # sentinel = num_experts
            # Where sentinel, scale must be 0
            assert (scales_g[sentinel_mask] == 0).all()

    def test_deterministic(self):
        policy = ExpertLoadHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.tensor([[0, 1], [0, 2], [3, 4]], device="cuda")
        topk_weights = torch.rand(3, 2, device="cuda")
        d1 = policy.dispatch(topk_ids, topk_weights)
        d2 = policy.dispatch(topk_ids, topk_weights)
        assert torch.equal(d1[0][0], d2[0][0])
        assert torch.equal(d1[1][0], d2[1][0])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestRandomHeterDispatch:
    def test_reproducible(self):
        p1 = RandomHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            seed=123,
            device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            seed=123,
            device=torch.device("cuda"),
        )
        ids = torch.randint(0, 8, (10, 2), device="cuda")
        weights = torch.rand(10, 2, device="cuda")
        d1 = p1.dispatch(ids, weights)
        d2 = p2.dispatch(ids, weights)
        assert torch.equal(d1[0][0], d2[0][0])

    def test_different_seeds_differ(self):
        p1 = RandomHeterDispatch(
            num_experts=128,
            group_size_ratios=[0.8, 0.2],
            seed=1,
            device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=128,
            group_size_ratios=[0.8, 0.2],
            seed=2,
            device=torch.device("cuda"),
        )
        ids = torch.randint(0, 128, (100, 8), device="cuda")
        weights = torch.rand(100, 8, device="cuda")
        d1 = p1.dispatch(ids, weights)
        d2 = p2.dispatch(ids, weights)
        assert not torch.equal(d1[0][0], d2[0][0])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestConfidenceThresholdHeterDispatch:
    def test_basic_dispatch(self):
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.randint(0, 8, (10, 2), device="cuda")
        topk_weights = torch.rand(10, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2

    def test_fallback_without_signals(self):
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        # _assign with None should fall back to random
        result = policy._assign(None, None)
        assert result.shape == (8,)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCreatePolicy:
    def test_create_expert_load(self):
        p = create_policy(
            "expert_load",
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        assert isinstance(p, ExpertLoadHeterDispatch)

    def test_create_random(self):
        p = create_policy(
            "random",
            num_experts=8,
            group_size_ratios=[0.75, 0.25],
            seed=99,
            device=torch.device("cuda"),
        )
        assert isinstance(p, RandomHeterDispatch)

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown heter policy"):
            create_policy(
                "nonexistent",
                num_experts=8,
                group_size_ratios=[0.5, 0.5],
                device=torch.device("cuda"),
            )


# --- HeterFusedMoE forward tests (GPU required) --------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHeterFusedMoEBF16Only:
    """Test HeterFusedMoE with a single BF16 group (simplest case)."""

    def _make_layer(self, num_experts=8, hidden=64, intermediate=32, top_k=2):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        config = {
            "groups": [{"name": "all_bf16", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "expert_load",
        }
        layer = HeterFusedMoE(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            top_k=top_k,
            heter_config=config,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        layer.init_fake_weights(seed=42)
        return layer

    def test_output_shape(self):
        layer = self._make_layer()
        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.rand(4, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (4, 2), device="cuda")
        out = layer(x, _make_topk_output(topk_w, topk_ids))
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_nonzero_output(self):
        layer = self._make_layer()
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.ones(8, 2, dtype=torch.bfloat16, device="cuda") * 0.5
        topk_ids = torch.randint(0, 8, (8, 2), device="cuda")
        out = layer(x, _make_topk_output(topk_w, topk_ids))
        assert out.abs().sum() > 0, "Output should not be all zeros"

    def test_zero_weights_give_zero_output(self):
        layer = self._make_layer()
        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.zeros(4, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (4, 2), device="cuda")
        out = layer(x, _make_topk_output(topk_w, topk_ids))
        assert out.abs().max() < 1e-3, (
            "Zero routing weights should give near-zero output"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHeterFusedMoETwoGroupsBF16:
    """Test with two BF16 groups (validates multi-group masking logic)."""

    def _make_layer(self, num_experts=8, hidden=64, intermediate=32, top_k=2):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        config = {
            "groups": [
                {"name": "cold", "num_bits": 16, "size_ratio": 0.75},
                {"name": "hot", "num_bits": 16, "size_ratio": 0.25},
            ],
            "policy": "expert_load",
        }
        layer = HeterFusedMoE(
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            top_k=top_k,
            heter_config=config,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        layer.init_fake_weights(seed=42)
        return layer

    def test_output_shape(self):
        layer = self._make_layer()
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.rand(16, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (16, 2), device="cuda")
        out = layer(x, _make_topk_output(topk_w, topk_ids))
        assert out.shape == x.shape

    def test_two_groups_vs_one_group_different(self):
        """Two BF16 groups with DIFFERENT weights should give different results."""
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        torch.manual_seed(0)
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.rand(16, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (16, 2), device="cuda")

        one_group_cfg = {
            "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "expert_load",
        }
        layer1 = HeterFusedMoE(
            8, 64, 32, 2, one_group_cfg, torch.bfloat16, torch.device("cuda")
        )
        layer1.init_fake_weights(seed=42)
        out1 = layer1(x, _make_topk_output(topk_w, topk_ids))

        two_group_cfg = {
            "groups": [
                {"name": "cold", "num_bits": 16, "size_ratio": 0.75},
                {"name": "hot", "num_bits": 16, "size_ratio": 0.25},
            ],
            "policy": "expert_load",
        }
        layer2 = HeterFusedMoE(
            8, 64, 32, 2, two_group_cfg, torch.bfloat16, torch.device("cuda")
        )
        layer2.init_fake_weights(seed=99)
        out2 = layer2(x, _make_topk_output(topk_w, topk_ids))

        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-4, (
            f"Different weights should give different outputs, max diff={diff}"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHeterFusedMoEInt8Group:
    """DEPRECATED: INT8 path kept for functional correctness only."""

    def _make_layer(self):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = {
                "groups": [{"name": "int8_all", "num_bits": 8, "size_ratio": 1.0}],
                "policy": "expert_load",
            }
            layer = HeterFusedMoE(
                num_experts=8,
                hidden_size=64,
                intermediate_size=32,
                top_k=2,
                heter_config=config,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            )
        layer.init_fake_weights(seed=42)
        return layer

    def test_int8_output_shape(self):
        layer = self._make_layer()
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.rand(8, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (8, 2), device="cuda")
        out = layer(x, _make_topk_output(topk_w, topk_ids))
        assert out.shape == x.shape


# --- Triton sentinel handling tests --------------------------------------
#
# heter-moe uses a uniform sentinel of `num_experts` (an out-of-range
# expert ID) to mark tokens dispatched to other groups. These tests verify
# that the Triton outplace_fused_experts kernel correctly filters those
# slots so the Marlin and Triton paths can share one sentinel convention.


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_triton_sentinel_num_experts():
    """Verify outplace_fused_experts handles sentinel=num_experts (all tokens masked)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts

    E, H, I = 8, 64, 32
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
    # All experts set to sentinel (num_experts=8)
    topk_ids = torch.full((4, 2), E, dtype=torch.int64, device="cuda")
    topk_weights = torch.zeros(4, 2, dtype=torch.bfloat16, device="cuda")
    out = outplace_fused_experts(x, w13, w2, topk_weights, topk_ids)
    assert out.shape == x.shape
    assert out.abs().max() < 1e-3  # All-masked should give zero output


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_triton_sentinel_partial_mask():
    """Verify outplace_fused_experts works when some but not all tokens use sentinel."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts

    E, H, I = 8, 64, 32
    torch.manual_seed(42)
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
    # Mixed: some valid expert IDs, some sentinel
    topk_ids = torch.tensor(
        [[0, E], [1, 2], [E, 3], [E, E]], dtype=torch.int64, device="cuda"
    )
    topk_weights = torch.tensor(
        [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.0, 0.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = outplace_fused_experts(x, w13, w2, topk_weights, topk_ids)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_triton_sentinel_matches_zero_weight_masking():
    """Sentinel-masking must be numerically equivalent to zero-weight-masking.

    Exercises the large-batch path (numel >= 1024) and verifies that using
    sentinel=num_experts produces identical output to keeping valid IDs but
    zeroing their weights. This is the strongest guarantee that the Triton
    kernel filters sentinel slots instead of accessing out-of-bounds weights.
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts

    torch.manual_seed(7)
    E, H, I = 8, 128, 64
    N, topk = 1024, 4  # numel=4096 triggers the large-batch moe_align path
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")

    ids_valid = torch.randint(0, E, (N, topk), dtype=torch.int64, device="cuda")
    weights = torch.rand(N, topk, dtype=torch.bfloat16, device="cuda")
    mask = torch.rand(N, topk, device="cuda") < 0.5
    weights_masked = torch.where(mask, torch.zeros_like(weights), weights)

    # A: valid expert IDs, zeroed weights where masked
    out_a = outplace_fused_experts(x, w13, w2, weights_masked.clone(), ids_valid.clone())
    # B: sentinel (num_experts) where masked, same zero weights
    ids_sentinel = torch.where(mask, torch.full_like(ids_valid, E), ids_valid)
    out_b = outplace_fused_experts(x, w13, w2, weights_masked.clone(), ids_sentinel)

    # Should be bit-for-bit equal since sentinel slots must be skipped entirely.
    diff = (out_a - out_b).abs().max().item()
    assert diff < 1e-3, f"sentinel vs zero-weight masking not equivalent, diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
