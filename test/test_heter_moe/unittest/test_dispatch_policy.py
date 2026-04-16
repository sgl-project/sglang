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
ExpertBatchGatedHeterDispatch = _policy_mod.ExpertBatchGatedHeterDispatch
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


# --- ExpertBatchGatedHeterDispatch -------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestExpertBatchGatedHeterDispatch:
    """Per-expert hot/cold gating via routed-token count.

    - INT4 group at index 0, BF16 group at index 1.
    - Sentinel = -1 (Triton convention) so unselected slots are negative.
    - Expert is hot iff its routed-token count >= threshold.
    """

    def _make_policy(self, num_experts=8, threshold=128, int4_only_mask=None):
        return ExpertBatchGatedHeterDispatch(
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

    def test_uniform_routing_above_threshold_stays_cold(self):
        # 256 tokens * top_k=2 = 512 slots spread over 8 experts ~ 64/expert.
        # None reach threshold=128, so all experts cold -> INT4.
        policy = self._make_policy()
        topk_ids = torch.randint(0, 8, (256, 2), device="cuda")
        topk_weights = torch.rand(256, 2, device="cuda")
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        live = topk_ids >= 0
        assert (int4_ids[live] == topk_ids[live]).all()
        assert (bf16_ids == -1).all()

    def test_concentrated_routing_makes_expert_hot(self):
        # All tokens route to expert 0; count[0] >= threshold -> BF16.
        policy = self._make_policy(threshold=128)
        n = 150
        topk_ids = torch.zeros(n, 1, dtype=torch.long, device="cuda")
        topk_weights = torch.ones(n, 1, device="cuda")
        d = policy.dispatch(topk_ids, topk_weights, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        assert (bf16_ids == 0).all()
        assert (int4_ids == -1).all()

    def test_mixed_hot_cold_per_expert(self):
        # Expert 0: 200 slots (hot). Expert 1: 20 slots (cold).
        policy = self._make_policy(num_experts=4, threshold=128)
        ids = torch.cat([
            torch.zeros(200, dtype=torch.long, device="cuda"),
            torch.ones(20, dtype=torch.long, device="cuda"),
        ]).unsqueeze(1)
        w = torch.ones_like(ids, dtype=torch.float32)
        d = policy.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        is_expert_0 = ids == 0
        is_expert_1 = ids == 1
        # Hot expert 0 -> BF16 only
        assert (bf16_ids[is_expert_0] == 0).all()
        assert (int4_ids[is_expert_0] == -1).all()
        # Cold expert 1 -> INT4 only
        assert (int4_ids[is_expert_1] == 1).all()
        assert (bf16_ids[is_expert_1] == -1).all()

    def test_hot_int4_only_expert_forced_to_int4(self):
        # Expert 0 is INT4-only. Even when hot, it stays INT4.
        mask = torch.zeros(4, dtype=torch.bool, device="cuda")
        mask[0] = True
        policy = self._make_policy(num_experts=4, threshold=128,
                                    int4_only_mask=mask)
        ids = torch.zeros(200, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids, dtype=torch.float32)
        d = policy.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        assert (int4_ids == 0).all()
        assert (bf16_ids == -1).all()

    def test_hot_and_cold_with_int4_only_mask(self):
        # Experts: 0 hot (normal), 1 hot (int4-only), 2 cold, 3 cold.
        mask = torch.zeros(4, dtype=torch.bool, device="cuda")
        mask[1] = True
        policy = self._make_policy(num_experts=4, threshold=128,
                                    int4_only_mask=mask)
        ids = torch.cat([
            torch.zeros(150, dtype=torch.long, device="cuda"),  # hot
            torch.ones(150, dtype=torch.long, device="cuda"),   # hot, int4-only
            torch.full((10,), 2, dtype=torch.long, device="cuda"),  # cold
            torch.full((10,), 3, dtype=torch.long, device="cuda"),  # cold
        ]).unsqueeze(1)
        w = torch.ones_like(ids, dtype=torch.float32)
        d = policy.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        # Expert 0 hot, not int4-only -> BF16
        m0 = ids == 0
        assert (bf16_ids[m0] == 0).all()
        assert (int4_ids[m0] == -1).all()
        # Expert 1 hot, int4-only -> INT4
        m1 = ids == 1
        assert (int4_ids[m1] == 1).all()
        assert (bf16_ids[m1] == -1).all()
        # Experts 2, 3 cold -> INT4
        for eid in (2, 3):
            me = ids == eid
            assert (int4_ids[me] == eid).all()
            assert (bf16_ids[me] == -1).all()

    def test_custom_threshold(self):
        policy = self._make_policy(threshold=64)
        # Expert 0 gets 100 slots > 64 -> hot; rest scarce -> cold.
        ids = torch.cat([
            torch.zeros(100, dtype=torch.long, device="cuda"),
            torch.randint(1, 8, (5,), dtype=torch.long, device="cuda"),
        ]).unsqueeze(1)
        w = torch.ones_like(ids, dtype=torch.float32)
        d = policy.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        m0 = ids == 0
        assert (bf16_ids[m0] == 0).all()
        # Expert 0 does not appear in INT4 group.
        assert (int4_ids[m0] == -1).all()

    def test_should_skip_group_below_threshold(self):
        # Below threshold: BF16 provably empty, INT4 may be populated.
        policy = self._make_policy()
        assert policy.should_skip_group(0, 64) is False
        assert policy.should_skip_group(1, 64) is True

    def test_should_skip_group_at_or_above_threshold(self):
        # At/above threshold: neither group can be skipped host-side,
        # because per-expert hot/cold depends on tensor values.
        policy = self._make_policy()
        assert policy.should_skip_group(0, 128) is False
        assert policy.should_skip_group(1, 128) is False
        assert policy.should_skip_group(0, 1024) is False
        assert policy.should_skip_group(1, 1024) is False

    def test_should_skip_group_ignores_int4_only_mask(self):
        # Short-circuit is host-side only; int4_only_mask must not affect it.
        mask = torch.zeros(8, dtype=torch.bool, device="cuda")
        mask[[2, 5]] = True
        p_with = self._make_policy(int4_only_mask=mask)
        p_without = self._make_policy(int4_only_mask=None)
        for n in (1, 64, 128, 1024):
            for g in (0, 1):
                assert (p_with.should_skip_group(g, n)
                        == p_without.should_skip_group(g, n))

    def test_short_circuit_host_side_only(self):
        # Passing a Python int must work without any tensor inputs.
        policy = self._make_policy()
        assert isinstance(policy.should_skip_group(1, 1), bool)
        assert isinstance(policy.should_skip_group(0, 999999), bool)

    def test_global_batch_below_threshold_helper(self):
        policy = self._make_policy(threshold=100)
        assert policy._global_batch_below_threshold(0) is True
        assert policy._global_batch_below_threshold(99) is True
        assert policy._global_batch_below_threshold(100) is False
        assert policy._global_batch_below_threshold(1000) is False

    def test_assign_reuses_count_buffer_across_calls(self):
        # Per-call zero_() must wipe stale counts; back-to-back calls
        # with different concentrations must not leak state.
        policy = self._make_policy(num_experts=4, threshold=128)
        ids_hot = torch.zeros(200, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids_hot, dtype=torch.float32)
        policy.dispatch(ids_hot, w, sentinel=-1)
        # Now a cold batch -- expert 0 should become cold again.
        ids_cold = torch.ones(10, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids_cold, dtype=torch.float32)
        d = policy.dispatch(ids_cold, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        assert (int4_ids == 1).all()
        assert (bf16_ids == -1).all()

    def test_assign_with_no_routing(self):
        policy = self._make_policy()
        result = policy._assign(None, None)
        # No routing info -> default to INT4 (cold)
        assert (result == 0).all()
        assert result.shape == (8,)

    def test_create_via_registry(self):
        p = create_policy(
            "expert_batch",
            num_experts=8,
            group_size_ratios=[0.5, 0.5],
            device=torch.device("cuda"),
            threshold=64,
        )
        assert isinstance(p, ExpertBatchGatedHeterDispatch)
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
