"""Unit tests for heterogeneous-precision MoE.

Tests policy logic (CPU-only) and kernel integration (GPU required).
GPU tests are skipped if CUDA is not available.
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
HeterDispatchPlan = _policy_mod.HeterDispatchPlan
BaseHeterPolicy = _policy_mod.BaseHeterPolicy
TokenCountPolicy = _policy_mod.TokenCountPolicy
FixedPolicy = _policy_mod.FixedPolicy
RandomPolicy = _policy_mod.RandomPolicy
create_policy = _policy_mod.create_policy

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


# ─── Policy tests (CPU only) ─────────────────────────────────────────────


class TestHeterDispatchPlan:
    def test_validate_correct(self):
        plan = HeterDispatchPlan(group_assignments=[[0, 1, 2], [3, 4, 5, 6, 7]])
        plan.validate(num_experts=8)

    def test_validate_missing_expert(self):
        plan = HeterDispatchPlan(group_assignments=[[0, 1], [3, 4]])
        with pytest.raises(AssertionError):
            plan.validate(num_experts=5)

    def test_validate_duplicate_expert(self):
        plan = HeterDispatchPlan(group_assignments=[[0, 1, 2], [2, 3, 4]])
        with pytest.raises(AssertionError):
            plan.validate(num_experts=5)

    def test_get_expert_to_group(self):
        plan = HeterDispatchPlan(group_assignments=[[0, 3, 5], [1, 2, 4, 6, 7]])
        mapping = plan.get_expert_to_group(8)
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert mapping[3] == 0
        assert mapping[4] == 1


class TestTokenCountPolicy:
    def test_basic_assignment(self):
        policy = TokenCountPolicy()
        topk_ids = torch.tensor([[0, 1], [0, 2], [0, 3], [4, 5]])
        plan = policy.assign(topk_ids, num_experts=8, group_ratios=[0.75, 0.25])
        plan.validate(8)
        assert len(plan.group_assignments) == 2
        cold_group = plan.group_assignments[0]
        hot_group = plan.group_assignments[1]
        assert len(cold_group) == 6
        assert len(hot_group) == 2
        # Expert 0 appears 3 times (hottest) → should be in hot group
        assert 0 in hot_group

    def test_all_experts_one_group(self):
        policy = TokenCountPolicy()
        topk_ids = torch.tensor([[0, 1], [2, 3]])
        plan = policy.assign(topk_ids, num_experts=4, group_ratios=[1.0])
        plan.validate(4)
        assert len(plan.group_assignments[0]) == 4

    def test_even_split(self):
        policy = TokenCountPolicy()
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        plan = policy.assign(topk_ids, num_experts=8, group_ratios=[0.5, 0.5])
        plan.validate(8)
        assert len(plan.group_assignments[0]) == 4
        assert len(plan.group_assignments[1]) == 4

    def test_three_groups(self):
        policy = TokenCountPolicy()
        topk_ids = torch.randint(0, 16, (32, 4))
        plan = policy.assign(topk_ids, num_experts=16, group_ratios=[0.5, 0.3, 0.2])
        plan.validate(16)
        assert len(plan.group_assignments) == 3
        total = sum(len(g) for g in plan.group_assignments)
        assert total == 16

    def test_deterministic(self):
        policy = TokenCountPolicy()
        topk_ids = torch.tensor([[0, 1], [0, 2], [3, 4]])
        plan1 = policy.assign(topk_ids, num_experts=8, group_ratios=[0.75, 0.25])
        plan2 = policy.assign(topk_ids, num_experts=8, group_ratios=[0.75, 0.25])
        assert plan1.group_assignments == plan2.group_assignments


class TestFixedPolicy:
    def test_fixed_assignment(self):
        fixed = [[0, 1, 2, 3], [4, 5, 6, 7]]
        policy = FixedPolicy(fixed_assignments=fixed)
        topk_ids = torch.randint(0, 8, (10, 2))
        plan = policy.assign(topk_ids, num_experts=8, group_ratios=[0.5, 0.5])
        assert plan.group_assignments == fixed


class TestRandomPolicy:
    def test_reproducible(self):
        p1 = RandomPolicy(seed=123)
        p2 = RandomPolicy(seed=123)
        ids = torch.randint(0, 8, (10, 2))
        plan1 = p1.assign(ids, num_experts=8, group_ratios=[0.75, 0.25])
        plan2 = p2.assign(ids, num_experts=8, group_ratios=[0.75, 0.25])
        assert plan1.group_assignments == plan2.group_assignments

    def test_different_seeds_differ(self):
        p1 = RandomPolicy(seed=1)
        p2 = RandomPolicy(seed=2)
        ids = torch.randint(0, 128, (100, 8))
        plan1 = p1.assign(ids, num_experts=128, group_ratios=[0.8, 0.2])
        plan2 = p2.assign(ids, num_experts=128, group_ratios=[0.8, 0.2])
        assert plan1.group_assignments != plan2.group_assignments


class TestCreatePolicy:
    def test_create_token_count(self):
        p = create_policy("token_count")
        assert isinstance(p, TokenCountPolicy)

    def test_create_random(self):
        p = create_policy("random", seed=99)
        assert isinstance(p, RandomPolicy)

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown heter policy"):
            create_policy("nonexistent")


# ─── HeterFusedMoE forward tests (GPU required) ──────────────────────────


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHeterFusedMoEBF16Only:
    """Test HeterFusedMoE with a single BF16 group (simplest case)."""

    def _make_layer(self, num_experts=8, hidden=64, intermediate=32, top_k=2):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        config = {
            "groups": [{"name": "all_bf16", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "token_count",
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
        out = layer(x, topk_w, topk_ids)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_nonzero_output(self):
        layer = self._make_layer()
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.ones(8, 2, dtype=torch.bfloat16, device="cuda") * 0.5
        topk_ids = torch.randint(0, 8, (8, 2), device="cuda")
        out = layer(x, topk_w, topk_ids)
        assert out.abs().sum() > 0, "Output should not be all zeros"

    def test_zero_weights_give_zero_output(self):
        layer = self._make_layer()
        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.zeros(4, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (4, 2), device="cuda")
        out = layer(x, topk_w, topk_ids)
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
            "policy": "token_count",
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
        out = layer(x, topk_w, topk_ids)
        assert out.shape == x.shape

    def test_two_groups_vs_one_group_different(self):
        """Two BF16 groups with DIFFERENT weights should give different results than one group."""
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE

        torch.manual_seed(0)
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        topk_w = torch.rand(16, 2, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.randint(0, 8, (16, 2), device="cuda")

        one_group_cfg = {
            "groups": [{"name": "all", "num_bits": 16, "size_ratio": 1.0}],
            "policy": "token_count",
        }
        layer1 = HeterFusedMoE(
            8, 64, 32, 2, one_group_cfg, torch.bfloat16, torch.device("cuda")
        )
        layer1.init_fake_weights(seed=42)
        out1 = layer1(x, topk_w, topk_ids)

        two_group_cfg = {
            "groups": [
                {"name": "cold", "num_bits": 16, "size_ratio": 0.75},
                {"name": "hot", "num_bits": 16, "size_ratio": 0.25},
            ],
            "policy": "token_count",
        }
        layer2 = HeterFusedMoE(
            8, 64, 32, 2, two_group_cfg, torch.bfloat16, torch.device("cuda")
        )
        layer2.init_fake_weights(seed=99)
        out2 = layer2(x, topk_w, topk_ids)

        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-4, (
            f"Different weights should give different outputs, max diff={diff}"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHeterFusedMoEInt8Group:
    """DEPRECATED: INT8 path kept for functional correctness only.

    Triton INT8 on A100 achieves ~6% of peak tensor core throughput.
    See: https://github.com/triton-lang/triton/issues/2818
    Use Marlin INT4 (num_bits=4) for production workloads.
    """

    def _make_layer(self):
        from sglang.srt.layers.moe.heter_moe import HeterFusedMoE
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = {
                "groups": [{"name": "int8_all", "num_bits": 8, "size_ratio": 1.0}],
                "policy": "token_count",
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
        out = layer(x, topk_w, topk_ids)
        assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
