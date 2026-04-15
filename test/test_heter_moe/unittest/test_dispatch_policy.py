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
BatchSizeGatedHeterDispatch = _policy_mod.BatchSizeGatedHeterDispatch
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


# --- BatchSizeGatedHeterDispatch -------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestBatchSizeGatedHeterDispatch:
    """Conventions for these tests:

    - INT4 group at index 0, BF16 group at index 1 (matches the binary
      assumption baked into BatchSizeGatedHeterDispatch defaults).
    - Sentinel = -1 (Triton convention) so unselected slots are negative.
    """

    def _make_policy(self, num_experts=8, threshold=128, int4_only_mask=None):
        return BatchSizeGatedHeterDispatch(
            num_experts=num_experts,
            group_size_ratios=[0.5, 0.5],  # plumbing only; gate ignores it
            threshold=threshold,
            device=torch.device("cuda"),
            int4_only_mask=int4_only_mask,
            int4_group_idx=0,
        )

    def test_below_threshold_assigns_all_to_int4(self):
        policy = self._make_policy()
        # 64 tokens, top_k=2 => batch tokens = 64 < 128
        topk_ids = torch.randint(0, 8, (64, 2), device="cuda")
        topk_weights = torch.rand(64, 2, device="cuda")
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, int4_w = d[0]
        bf16_ids, bf16_w = d[1]
        # All real slots (>=0 in topk_ids) flow into INT4 group
        live = topk_ids >= 0
        assert (int4_ids[live] == topk_ids[live]).all()
        # BF16 group is fully sentineled
        assert (bf16_ids == -1).all()
        assert (bf16_w == 0).all()

    def test_at_threshold_assigns_all_to_bf16(self):
        policy = self._make_policy()
        topk_ids = torch.randint(0, 8, (128, 2), device="cuda")
        topk_weights = torch.rand(128, 2, device="cuda")
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        live = topk_ids >= 0
        assert (bf16_ids[live] == topk_ids[live]).all()
        assert (int4_ids == -1).all()

    def test_above_threshold_assigns_all_to_bf16(self):
        policy = self._make_policy()
        topk_ids = torch.randint(0, 8, (256, 2), device="cuda")
        topk_weights = torch.rand(256, 2, device="cuda")
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        live = topk_ids >= 0
        assert (bf16_ids[live] == topk_ids[live]).all()
        assert (int4_ids == -1).all()

    def test_int4_only_mask_respected_above_threshold(self):
        # Experts 0 and 3 are INT4-only; even at large batch they stay INT4.
        mask = torch.zeros(8, dtype=torch.bool, device="cuda")
        mask[[0, 3]] = True
        policy = self._make_policy(int4_only_mask=mask)
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]] * 64, device="cuda"
        )  # 256 tokens
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        # INT4-only experts (0, 3) routed to INT4 group
        for eid in (0, 3):
            mask_e = topk_ids == eid
            assert (int4_ids[mask_e] == eid).all()
            assert (bf16_ids[mask_e] == -1).all()
        # Other experts go to BF16
        for eid in (1, 2, 4, 5, 6, 7):
            mask_e = topk_ids == eid
            assert (bf16_ids[mask_e] == eid).all()
            assert (int4_ids[mask_e] == -1).all()

    def test_custom_threshold(self):
        policy = self._make_policy(threshold=64)
        # 32 tokens < 64 -> INT4
        ids = torch.randint(0, 8, (32, 2), device="cuda")
        w = torch.rand(32, 2, device="cuda")
        d = policy.dispatch(ids, w, sentinel=-1)
        assert (d[1][0] == -1).all()
        # 64 tokens -> BF16
        ids = torch.randint(0, 8, (64, 2), device="cuda")
        w = torch.rand(64, 2, device="cuda")
        d = policy.dispatch(ids, w, sentinel=-1)
        assert (d[0][0] == -1).all()

    def test_should_skip_group_no_int4_only(self):
        policy = self._make_policy()
        # Below threshold: skip BF16 (group 1), keep INT4 (group 0)
        assert policy.should_skip_group(0, 64) is False
        assert policy.should_skip_group(1, 64) is True
        # At/above threshold: skip INT4 (group 0), keep BF16 (group 1)
        assert policy.should_skip_group(0, 128) is True
        assert policy.should_skip_group(1, 128) is False
        assert policy.should_skip_group(0, 1024) is True

    def test_should_skip_group_with_int4_only(self):
        mask = torch.zeros(8, dtype=torch.bool, device="cuda")
        mask[[2, 5]] = True
        policy = self._make_policy(int4_only_mask=mask)
        # Below threshold: BF16 still skipped
        assert policy.should_skip_group(1, 64) is True
        # At/above threshold: INT4 NOT skipped because INT4-only experts exist
        assert policy.should_skip_group(0, 256) is False
        assert policy.should_skip_group(1, 256) is False

    def test_assign_with_no_routing(self):
        policy = self._make_policy()
        result = policy._assign(None, None)
        # No routing info -> default to INT4 (cold)
        assert (result == 0).all()
        assert result.shape == (8,)

    def test_create_via_registry(self):
        p = create_policy(
            "batch_size",
            num_experts=8,
            group_size_ratios=[0.5, 0.5],
            device=torch.device("cuda"),
            threshold=64,
        )
        assert isinstance(p, BatchSizeGatedHeterDispatch)
        assert p.threshold == 64


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
