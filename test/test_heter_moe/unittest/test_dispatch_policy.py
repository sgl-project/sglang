"""1.1 Dispatch policy tests.

Tests heter dispatch logic: group assignment, sentinel masking, determinism,
and the universal BF16 promotion threshold (now owned by the ABC).

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
HessianWeightedRoutingWeightsDispatch = _policy_mod.HessianWeightedRoutingWeightsDispatch
RandomHeterDispatch = _policy_mod.RandomHeterDispatch
ExpertBatchGatedHeterDispatch = _policy_mod.ExpertBatchGatedHeterDispatch
create_policy = _policy_mod.create_policy
_assign_by_score_gpu = _policy_mod._assign_by_score_gpu
_build_group_labels = _policy_mod._build_group_labels

CUDA_AVAILABLE = torch.cuda.is_available()


# Sentinel "threshold so high it never fires" for tests that want to assert
# pure policy-scoring behavior without the ABC threshold rule promoting anyone.
HUGE_THR = 10**9


def _counts_for(policy, ids):
    """Helper: produce the per-expert token-count tensor the ABC would compute."""
    return policy._compute_token_counts(ids)


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
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
            seed=123, device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
            seed=1, device=torch.device("cuda"),
        )
        p2 = RandomHeterDispatch(
            num_experts=128, group_size_ratios=[0.8, 0.2],
            bf16_promotion_threshold=HUGE_THR,
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
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        topk_ids = torch.randint(0, 8, (10, 2), device="cuda")
        topk_weights = torch.rand(10, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2

    def test_fallback_without_signals(self):
        policy = ConfidenceThresholdHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        counts = _counts_for(policy, None)
        result = policy._assign(counts, None, None)
        assert result.shape == (8,)


# --- TotalWeightHeterDispatch ----------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestTotalWeightHeterDispatch:
    def test_basic_dispatch(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        topk_ids = torch.randint(0, 8, (10, 2), device="cuda")
        topk_weights = torch.rand(10, 2, device="cuda")
        dispatches = policy.dispatch(topk_ids, topk_weights)
        assert len(dispatches) == 2

    def test_fallback_without_signals(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        counts = _counts_for(policy, None)
        result = policy._assign(counts, None, None)
        assert result.shape == (8,)

    def test_high_total_weight_expert_in_high_group(self):
        policy = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
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


# --- HessianWeightedRoutingWeightsDispatch ---------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHessianWeightedRoutingWeightsDispatch:
    """score(E) = importance(E) × sum(scales for E).

    The ABC's BF16 promotion rule is layered on top: any expert whose
    routed-token count crosses ``bf16_promotion_threshold`` goes BF16
    regardless of its hessian score, with ``int4_only_mask`` as the final
    override.
    """

    def _make(self, num_experts=8, importance=None, ratios=(0.75, 0.25),
              bf16_promotion_threshold=HUGE_THR, int4_only_mask=None):
        if importance is None:
            importance = torch.ones(num_experts, device="cuda")
        return HessianWeightedRoutingWeightsDispatch(
            num_experts=num_experts,
            group_size_ratios=list(ratios),
            importance=importance,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=torch.device("cuda"),
            int4_only_mask=int4_only_mask,
        )

    def test_identity_importance_matches_total_weight(self):
        # Uniform importance=1 reduces to TotalWeightHeterDispatch.
        topk_ids = torch.tensor(
            [[0, 1], [0, 2], [0, 3], [0, 4], [5, 6]], device="cuda"
        )
        topk_weights = torch.rand(5, 2, device="cuda")
        p_ref = TotalWeightHeterDispatch(
            num_experts=8, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        p = self._make()
        c_ref = _counts_for(p_ref, topk_ids)
        c_new = _counts_for(p, topk_ids)
        g_ref = p_ref._assign(c_ref, topk_ids, topk_weights).clone()
        g_new = p._assign(c_new, topk_ids, topk_weights).clone()
        assert torch.equal(g_ref, g_new)

    def test_zero_importance_expert_never_in_bf16(self):
        # Expert 0 dominates routing, but importance=0 -> must stay out of BF16
        # at the policy layer. Threshold is HUGE so the ABC rule never fires.
        imp = torch.ones(8, device="cuda")
        imp[0] = 0.0
        p = self._make(importance=imp)
        topk_ids = torch.tensor(
            [[0, 0], [0, 0], [0, 0], [0, 1], [2, 3]], device="cuda"
        )
        topk_weights = torch.ones(5, 2, device="cuda")
        d = p.dispatch(topk_ids, topk_weights)
        experts_g1, _ = d[1]  # BF16 group
        mask_e0 = topk_ids == 0
        # Expert 0 slots in BF16 group must all be sentineled (default -1).
        assert (experts_g1[mask_e0] == -1).all()

    def test_threshold_rule_promotes_low_importance_expert(self):
        # Expert 0 has zero importance (policy puts it in INT4) but receives
        # >=threshold tokens. The ABC's universal threshold rule must promote
        # it to BF16.
        imp = torch.ones(8, device="cuda")
        imp[0] = 0.0
        p = self._make(importance=imp, bf16_promotion_threshold=5)
        # Expert 0 receives 6 tokens (>=5). Threshold rule fires.
        ids = torch.zeros(6, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids, dtype=torch.float32)
        d = p.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        # Expert 0 lands in BF16 group despite zero importance.
        assert (bf16_ids == 0).all()
        assert (int4_ids == -1).all()

    def test_int4_only_mask_overrides_threshold(self):
        # Expert 0 is INT4-only AND receives >=threshold tokens. The mask is
        # the final word: it must stay INT4 (no BF16 weights loaded).
        mask = torch.zeros(8, dtype=torch.bool, device="cuda")
        mask[0] = True
        imp = torch.ones(8, device="cuda")
        p = self._make(importance=imp, bf16_promotion_threshold=5,
                       int4_only_mask=mask)
        ids = torch.zeros(10, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids, dtype=torch.float32)
        d = p.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        assert (int4_ids == 0).all()
        assert (bf16_ids == -1).all()

    def test_importance_breaks_tie_on_equal_weight(self):
        # Two experts with equal routing weight: higher importance wins BF16.
        imp = torch.zeros(4, device="cuda")
        imp[1] = 10.0  # winner
        imp[2] = 1.0   # loser
        p = self._make(num_experts=4, importance=imp, ratios=(0.75, 0.25))
        # Experts 1 and 2 both get one slot of weight 1.0; 0 and 3 none.
        topk_ids = torch.tensor([[1, 2]], device="cuda")
        topk_weights = torch.ones(1, 2, device="cuda")
        counts = _counts_for(p, topk_ids)
        g = p._assign(counts, topk_ids, topk_weights)
        # k_high = round(4*0.25) = 1, so top-1 by score goes BF16.
        assert g[1].item() == 1
        assert g[2].item() == 0

    def test_fallback_without_signals(self):
        p = self._make()
        counts = _counts_for(p, None)
        result = p._assign(counts, None, None)
        assert result.shape == (8,)

    def test_shape_assertion(self):
        with pytest.raises(AssertionError):
            HessianWeightedRoutingWeightsDispatch(
                num_experts=8,
                group_size_ratios=[0.75, 0.25],
                importance=torch.ones(4, device="cuda"),  # wrong length
                bf16_promotion_threshold=HUGE_THR,
                device=torch.device("cuda"),
            )


# --- ExpertBatchGatedHeterDispatch ------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestExpertBatchGatedHeterDispatch:
    """No-scoring policy: every expert defaults to INT4.

    The ABC's universal BF16 promotion rule then promotes any expert whose
    routed-token count crosses ``bf16_promotion_threshold`` to BF16.
    ``int4_only_mask`` is the final word: those experts stay INT4 regardless.
    """

    def _make_policy(self, num_experts=8, threshold=128, int4_only_mask=None):
        return ExpertBatchGatedHeterDispatch(
            num_experts=num_experts,
            group_size_ratios=[0.5, 0.5],  # plumbing only; gate ignores it
            bf16_promotion_threshold=threshold,
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
        zero_counts = torch.zeros(8, device="cuda", dtype=torch.float32)
        result = policy._assign(zero_counts, None, None)
        # No routing info -> default to INT4 (cold)
        assert (result == 0).all()
        assert result.shape == (8,)

    def test_create_via_registry(self):
        p = create_policy(
            "expert_batch",
            num_experts=8,
            group_size_ratios=[0.5, 0.5],
            bf16_promotion_threshold=64,
            device=torch.device("cuda"),
        )
        assert isinstance(p, ExpertBatchGatedHeterDispatch)
        assert p.bf16_promotion_threshold == 64


# --- Universal threshold rule (ABC) -----------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestUniversalThresholdRule:
    """Tests the ABC's BF16 promotion rule across non-trivial policies."""

    def test_threshold_promotes_in_total_weight(self):
        # Expert 0 receives many tokens but with TINY weight; total-weight
        # ranking would NOT pick it. The threshold rule must still promote.
        p = TotalWeightHeterDispatch(
            num_experts=4, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=10,
            device=torch.device("cuda"),
        )
        # Expert 0: 20 slots × weight 1e-6 (tiny total weight)
        # Expert 1: 1 slot × weight 100 (high total weight, would win)
        ids = torch.cat([
            torch.zeros(20, dtype=torch.long, device="cuda"),
            torch.ones(1, dtype=torch.long, device="cuda"),
        ]).unsqueeze(1)
        w = torch.cat([
            torch.full((20,), 1e-6, device="cuda"),
            torch.full((1,), 100.0, device="cuda"),
        ]).unsqueeze(1)
        d = p.dispatch(ids, w, sentinel=-1)
        bf16_ids, _ = d[1]
        # Expert 0 has 20 routed tokens >= 10 -> threshold promotes to BF16.
        m0 = ids == 0
        assert (bf16_ids[m0] == 0).all()

    def test_int4_only_overrides_universal_threshold(self):
        # Expert 0 is INT4-only and would be promoted by threshold; mask wins.
        mask = torch.zeros(4, dtype=torch.bool, device="cuda")
        mask[0] = True
        p = ExpertLoadHeterDispatch(
            num_experts=4, group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=5,
            device=torch.device("cuda"),
            int4_only_mask=mask,
        )
        ids = torch.zeros(10, 1, dtype=torch.long, device="cuda")
        w = torch.ones_like(ids, dtype=torch.float32)
        d = p.dispatch(ids, w, sentinel=-1)
        int4_ids, _ = d[0]
        bf16_ids, _ = d[1]
        assert (int4_ids == 0).all()
        assert (bf16_ids == -1).all()

    def test_threshold_required(self):
        # ABC ctor must require bf16_promotion_threshold (no default).
        with pytest.raises(TypeError):
            ExpertLoadHeterDispatch(
                num_experts=4, group_size_ratios=[0.75, 0.25],
                device=torch.device("cuda"),
            )

    def test_each_slot_in_exactly_one_group(self):
        # Invariant: across all groups returned by dispatch, every token slot
        # appears as a non-sentinel in exactly one group. Guarantees that no
        # expert is ever "assigned to two precisions" -- the dispatch builds
        # mutually-exclusive (experts, scales) tuples per group.
        p = ExpertLoadHeterDispatch(
            num_experts=8, group_size_ratios=[0.5, 0.5],
            bf16_promotion_threshold=5,
            device=torch.device("cuda"),
        )
        ids = torch.randint(0, 8, (32, 2), device="cuda")
        w = torch.ones_like(ids, dtype=torch.float32)
        dispatches = p.dispatch(ids, w, sentinel=-1)
        in_groups = [(g_ids != -1) for g_ids, _ in dispatches]
        # Each slot lives in exactly one group's experts tensor.
        appearances = sum(m.long() for m in in_groups)
        assert (appearances == 1).all(), (
            "every slot must appear in exactly one group; got "
            f"distribution {appearances.unique(return_counts=True)}"
        )


# --- create_policy registry -------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCreatePolicy:
    def test_create_expert_load(self):
        p = create_policy(
            "expert_load", num_experts=8,
            group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        assert isinstance(p, ExpertLoadHeterDispatch)

    def test_create_random(self):
        p = create_policy(
            "random", num_experts=8,
            group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            seed=99, device=torch.device("cuda"),
        )
        assert isinstance(p, RandomHeterDispatch)

    def test_create_total_weight(self):
        p = create_policy(
            "total_weight", num_experts=8,
            group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
        )
        assert isinstance(p, TotalWeightHeterDispatch)

    def test_create_hessian_weighted_routing_weights(self):
        p = create_policy(
            "hessian_weighted_routing_weights", num_experts=8,
            group_size_ratios=[0.75, 0.25],
            bf16_promotion_threshold=HUGE_THR,
            device=torch.device("cuda"),
            importance=torch.ones(8, device="cuda"),
        )
        assert isinstance(p, HessianWeightedRoutingWeightsDispatch)

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown heter policy"):
            create_policy(
                "nonexistent", num_experts=8,
                group_size_ratios=[0.5, 0.5],
                bf16_promotion_threshold=HUGE_THR,
                device=torch.device("cuda"),
            )


# --- CUDA graph & torch.compile compat across all policies -----------------


_ALL_POLICY_NAMES = [
    "random",
    "expert_load",
    "confidence",
    "total_weight",
    "hessian_weighted_routing_weights",
    "expert_batch",
]


def _build_policy(name, num_experts=8, ratios=(0.5, 0.5),
                  bf16_promotion_threshold=4):
    """Build any of the 6 registered policies with sensible defaults.

    Threshold defaults to 4 (low) so the universal promotion rule fires
    on the dummy input and the ABC's threshold path actually runs under
    the compat checks; ``int4_only_mask``/``bf16_only_mask`` left None.
    """
    kwargs = dict(
        num_experts=num_experts,
        group_size_ratios=list(ratios),
        bf16_promotion_threshold=bf16_promotion_threshold,
        device=torch.device("cuda"),
    )
    if name == "hessian_weighted_routing_weights":
        kwargs["importance"] = torch.ones(num_experts, device="cuda")
    return create_policy(name, **kwargs)


def _dummy_dispatch_inputs(num_experts=8, batch=32, top_k=2):
    """Deterministic (ids, weights) covering all experts uniformly.

    All ``num_experts`` get >=1 routed slot to keep downstream kernels
    happy; weights are non-zero so the threshold rule sees real counts.
    """
    n_slots = batch * top_k
    base = torch.arange(num_experts, device="cuda").repeat(
        (n_slots + num_experts - 1) // num_experts
    )[:n_slots]
    perm_gen = torch.Generator(device="cuda").manual_seed(0)
    perm = torch.randperm(n_slots, device="cuda", generator=perm_gen)
    ids = base[perm].view(batch, top_k).contiguous()
    weights = torch.full(
        (batch, top_k), 0.5, dtype=torch.float32, device="cuda"
    )
    return ids, weights


def _dispatch_to_tensor(dispatch_result):
    """Stack per-group (experts, scales) tuples into one [G, 2, N, K] tensor.

    Lets ``torch.testing.assert_close`` work on the entire dispatch output
    as a single tensor (and gives us a single value to capture in graphs).
    """
    return torch.stack(
        [torch.stack([e, s.float()], dim=0) for e, s in dispatch_result],
        dim=0,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPolicyCudaGraphCompat:
    """All 6 policies must be capturable in a CUDA graph and replay
    deterministically.  Captures only the policy's ``dispatch`` (no kernel
    calls), so any host-side branch in the new ABC dispatch path would
    surface here.
    """

    @pytest.mark.parametrize("policy_name", _ALL_POLICY_NAMES)
    def test_capture_and_replay(self, policy_name):
        policy = _build_policy(policy_name)
        ids, weights = _dummy_dispatch_inputs()

        # Warm up to materialize lazy buffers (e.g. ``_ones_buf`` grow path)
        # before capture; otherwise the first-time alloc lands inside the
        # captured region and replays would re-trigger it.
        for _ in range(3):
            policy.dispatch(ids, weights, sentinel=-1)
        torch.cuda.synchronize()

        eager = _dispatch_to_tensor(
            policy.dispatch(ids, weights, sentinel=-1)
        ).clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = policy.dispatch(ids, weights, sentinel=-1)
            captured_tensor = _dispatch_to_tensor(captured)
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()

        assert captured_tensor.shape == eager.shape
        torch.testing.assert_close(captured_tensor, eager, atol=0, rtol=0)

    @pytest.mark.parametrize("policy_name", _ALL_POLICY_NAMES)
    def test_replay_is_deterministic(self, policy_name):
        # Two replays of the same captured graph must produce identical
        # values; would catch any uninitialized scratch in the new ABC path.
        policy = _build_policy(policy_name)
        ids, weights = _dummy_dispatch_inputs()
        for _ in range(3):
            policy.dispatch(ids, weights, sentinel=-1)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = policy.dispatch(ids, weights, sentinel=-1)
            captured_tensor = _dispatch_to_tensor(captured)

        graph.replay()
        torch.cuda.synchronize()
        first = captured_tensor.clone()

        graph.replay()
        torch.cuda.synchronize()
        second = captured_tensor.clone()

        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPolicyTorchCompileCompat:
    """All 6 policies must be ``torch.compile``-clean.

    The compiled dispatch must produce the same outputs as eager (after
    the compile warmup).
    """

    @pytest.mark.parametrize("policy_name", _ALL_POLICY_NAMES)
    def test_compile_matches_eager(self, policy_name):
        policy = _build_policy(policy_name)
        ids, weights = _dummy_dispatch_inputs()

        eager_tensor = _dispatch_to_tensor(
            policy.dispatch(ids, weights, sentinel=-1)
        ).clone()

        def _run(i, w):
            return _dispatch_to_tensor(policy.dispatch(i, w, sentinel=-1))

        compiled = torch.compile(_run)
        # Compile warmup — first call triggers tracing.
        for _ in range(3):
            compiled(ids, weights)
        compiled_tensor = compiled(ids, weights)

        assert compiled_tensor.shape == eager_tensor.shape
        torch.testing.assert_close(
            compiled_tensor, eager_tensor, atol=0, rtol=0
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
