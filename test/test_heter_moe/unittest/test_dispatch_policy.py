"""1.1 Dispatch policy tests.

Tests heter dispatch logic: group assignment, sentinel masking, determinism.
All tests require CUDA (GPU-resident policy buffers).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"


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
TotalWeightHeterDispatch = _policy_mod.TotalWeightHeterDispatch
RandomHeterDispatch = _policy_mod.RandomHeterDispatch
create_policy = _policy_mod.create_policy
_assign_by_score_gpu = _policy_mod._assign_by_score_gpu
_build_group_labels = _policy_mod._build_group_labels

CUDA_AVAILABLE = torch.cuda.is_available()


# --- _assign_by_score_gpu ---------------------------------------------------


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
        for gidx in range(3):
            count = (result == gidx).sum().item()
            assert count > 0


# --- ExpertLoadHeterDispatch ------------------------------------------------


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
        topk_ids = torch.tensor([[0, 1], [0, 2], [0, 3], [4, 5]], device="cuda")
        topk_weights = torch.ones(4, 2, device="cuda") * 0.5
        dispatches = policy.dispatch(topk_ids, topk_weights)
        experts_g1, _ = dispatches[1]
        mask_expert0 = topk_ids == 0
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
            sentinel_mask = experts_g == 8  # sentinel = num_experts
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


# --- RandomHeterDispatch ----------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestRandomHeterDispatch:
    def test_reproducible(self):
        p1 = RandomHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            seed=123, device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            seed=123, device=torch.device("cuda"),
        )
        ids = torch.randint(0, 8, (10, 2), device="cuda")
        weights = torch.rand(10, 2, device="cuda")
        d1 = p1.dispatch(ids, weights)
        d2 = p2.dispatch(ids, weights)
        assert torch.equal(d1[0][0], d2[0][0])

    def test_different_seeds_differ(self):
        p1 = RandomHeterDispatch(
            num_experts=128, group_size_ratios=[0.8, 0.2],
            seed=1, device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=128, group_size_ratios=[0.8, 0.2],
            seed=2, device=torch.device("cuda"),
        )
        ids = torch.randint(0, 128, (100, 8), device="cuda")
        weights = torch.rand(100, 8, device="cuda")
        d1 = p1.dispatch(ids, weights)
        d2 = p2.dispatch(ids, weights)
        assert not torch.equal(d1[0][0], d2[0][0])


# --- ConfidenceThresholdHeterDispatch ---------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestConfidenceThresholdHeterDispatch:
    def test_basic_dispatch(self):
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.randint(0, 8, (10, 2), device="cuda")
        topk_weights = torch.rand(10, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2

    def test_fallback_without_signals(self):
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        result = policy._assign(None, None)
        assert result.shape == (8,)


# --- TotalWeightHeterDispatch ----------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestTotalWeightHeterDispatch:
    def test_basic_dispatch(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        topk_ids = torch.randint(0, 8, (10, 2), device="cuda")
        topk_weights = torch.rand(10, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2

    def test_fallback_without_signals(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        result = policy._assign(None, None)
        assert result.shape == (8,)

    def test_high_total_weight_expert_in_high_group(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            device=torch.device("cuda"),
        )
        # Expert 0 appears many times with high weight -> highest total
        topk_ids = torch.tensor(
            [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6]], device="cuda"
        )
        topk_weights = torch.ones(5, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        experts_g1, _ = dispatches[1]
        mask_expert0 = topk_ids == 0
        assert (experts_g1[mask_expert0] == 0).all()


# --- create_policy registry ------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCreatePolicy:
    def test_create_expert_load(self):
        p = create_policy(
            "expert_load", num_experts=8,
            group_size_ratios=[0.75, 0.25], device=torch.device("cuda"),
        )
        assert isinstance(p, ExpertLoadHeterDispatch)

    def test_create_random(self):
        p = create_policy(
            "random", num_experts=8,
            group_size_ratios=[0.75, 0.25], seed=99, device=torch.device("cuda"),
        )
        assert isinstance(p, RandomHeterDispatch)

    def test_create_total_weight(self):
        p = create_policy(
            "total_weight", num_experts=8,
            group_size_ratios=[0.75, 0.25], device=torch.device("cuda"),
        )
        assert isinstance(p, TotalWeightHeterDispatch)

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown heter policy"):
            create_policy(
                "nonexistent", num_experts=8,
                group_size_ratios=[0.5, 0.5], device=torch.device("cuda"),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
