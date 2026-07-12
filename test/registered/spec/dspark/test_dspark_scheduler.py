import functools
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    VerifyBudgetDecision,
    compute_verify_token_budget,
    graph_tier_fill_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
    schedule_verify_lens_topk_from_survival,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _flat_table(
    steps_per_sec: float = 1.0, max_batch_tokens: int = 4096
) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[steps_per_sec],
        max_batch_tokens=max_batch_tokens,
    )


def _cliff_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_steps_per_sec=[1.0, 1.0, 1.0, 0.5, 0.45, 0.44, 0.43, 0.42],
        max_batch_tokens=64,
    )


def _additive_table() -> SpsAdditiveCostTable:
    return SpsAdditiveCostTable(
        bias_seconds=0.01,
        bs_probes=[1, 100],
        alpha_seconds=[0.0, 0.05],
        m_probes=[1, 200],
        theta_seconds=[0.0, 0.02],
    )


def _survival_from_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence, dim=1)


def _bruteforce_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: SpsCostTable,
    cfg: DSparkScheduleConfig,
) -> int:
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()
    candidates = history_survival_probs[:, :max_len].flatten()
    candidates = [float(x) for x in candidates.tolist() if float(x) >= cfg.survival_eps]
    candidates.sort(reverse=True)

    best_extra, best_theta = 0, float("-inf")
    for extra in range(len(candidates) + 1):
        tau_star = num_requests + sum(candidates[:extra])
        theta = tau_star * sps_table.lookup(num_requests + extra)
        if theta > best_theta:
            best_theta, best_extra = theta, extra
    return best_extra


def schedule_verify_lens_topk_vanilla(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    cfg.validate()
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    valid_rows = (survival_probs >= cfg.survival_eps).tolist()
    survival_rows = survival_probs.to(torch.float64).tolist()

    candidates: list[tuple[float, int, int]] = []
    for request in range(num_requests):
        for position in range(min(max_len, len(valid_rows[request]))):
            if valid_rows[request][position]:
                candidates.append((survival_rows[request][position], position, request))

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

    selected_extra = [0] * num_requests
    for _survival, _position, request in candidates[: max(int(budget), 0)]:
        selected_extra[request] += 1

    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = [
        min(max(cfg.min_verify_len + extra, lower_bound), max_len)
        for extra in selected_extra
    ]
    return torch.tensor(verify_lens, dtype=torch.int32, device=device)


_TOPK_IMPLS = (
    schedule_verify_lens_topk_from_survival,
    schedule_verify_lens_topk_vanilla,
)


def _for_each_impl(test_method):
    @functools.wraps(test_method)
    def wrapper(self):
        for impl in _TOPK_IMPLS:
            with self.subTest(impl=impl.__name__):
                test_method(self, impl)

    return wrapper


class TestComputeVerifyTokenBudget(CustomTestCase):
    def test_budget_argmax_matches_bruteforce_scan_across_sps_cliffs(self):
        torch.manual_seed(1)
        cfg = DSparkScheduleConfig(gamma=7)
        table = _cliff_table()
        for _ in range(20):
            confidence = torch.rand(3, 7, dtype=torch.float32) * 0.5 + 0.45
            survival = _survival_from_confidence(confidence)
            expected = _bruteforce_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            )
            actual = compute_verify_token_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            ).budget
            self.assertEqual(actual, expected)

    def test_budget_pool_is_independent_of_min_verify_len(self):
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg_no_min = DSparkScheduleConfig(gamma=4, min_verify_len=0)
        cfg_min2 = DSparkScheduleConfig(gamma=4, min_verify_len=2)
        table = _flat_table()
        budget_no_min = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_no_min
        ).budget
        budget_min2 = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg_min2
        ).budget
        # The budget candidate pool spans all positions; the per-request floor
        # is enforced later at verify-lens scheduling, not at budget time.
        self.assertEqual(budget_no_min, 4)
        self.assertEqual(budget_min2, 4)

    def test_budget_drops_candidates_below_survival_eps(self):
        survival = torch.tensor([[0.9, 1e-9, 1e-12]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3, survival_eps=1e-6)
        table = _flat_table()
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        ).budget
        self.assertLessEqual(budget, 1)

    def test_decision_predicted_step_matches_additive_table_at_budget(self):
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        table = _additive_table()
        decision = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        # Reference via the independent scalar interpolation path (step_time),
        # not the tensor helper the implementation itself uses.
        self.assertAlmostEqual(
            decision.predicted_step_seconds,
            table.step_time(num_reqs=1, budget=int(decision.budget)),
            places=5,
        )
        self.assertGreater(decision.predicted_theta, 0.0)

    def test_decision_predicted_step_is_inverse_sps_for_diagonal_table(self):
        survival = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=4)
        table = _cliff_table()
        decision = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        expected_sps = table.lookup(1 + decision.budget)
        self.assertIsNotNone(decision.predicted_step_seconds)
        self.assertAlmostEqual(
            decision.predicted_step_seconds, 1.0 / expected_sps, places=9
        )
        self.assertGreater(decision.predicted_theta, 0.0)


def _make_budget_planner() -> HostConfidenceBudgetPlanner:
    return HostConfidenceBudgetPlanner(
        sps_table=_flat_table(),
        cfg=DSparkScheduleConfig(gamma=4),
        model_runner=None,
    )


class TestBudgetDecisionLifecycle(CustomTestCase):
    def test_take_last_decision_is_consume_once(self):
        planner = _make_budget_planner()
        planner.last_decision = VerifyBudgetDecision(
            budget=3, predicted_step_seconds=0.01, predicted_theta=100.0
        )
        first = planner.take_last_decision()
        self.assertEqual(first.budget, 3)
        self.assertIsNone(planner.take_last_decision())

    def test_note_non_decode_step_clears_decision(self):
        planner = _make_budget_planner()
        planner.last_decision = VerifyBudgetDecision(budget=1)
        planner.note_non_decode_step()
        self.assertIsNone(planner.take_last_decision())


class TestScheduleVerifyLensTopk(CustomTestCase):
    @_for_each_impl
    def test_topk_does_not_exceed_budget(self, impl):
        torch.manual_seed(2)
        survival = _survival_from_confidence(torch.rand(5, 7) * 0.4 + 0.55)
        cfg = DSparkScheduleConfig(gamma=7)
        floor = max(cfg.min_verify_len, 1)
        for budget in (0, 1, 5, 12, 100):
            verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
            total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
            self.assertLessEqual(total_extra, budget)
            self.assertGreaterEqual(int(verify_lens.min().item()), 1)

    @_for_each_impl
    def test_total_equals_anchors_plus_lens(self, impl):
        survival = torch.tensor(
            [[0.90, 0.80, 0.30, 0.20], [0.85, 0.70, 0.25, 0.15]],
            dtype=torch.float32,
        )
        num_requests, max_len, budget = 2, 4, 2
        cfg = DSparkScheduleConfig(gamma=max_len, min_verify_len=1)
        verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
        actual_total = num_requests + int(verify_lens.to(torch.int64).sum().item())
        admitted = budget
        expected_total = num_requests + num_requests * cfg.min_verify_len + admitted
        self.assertEqual(actual_total, expected_total)

    @_for_each_impl
    def test_admission_is_contiguous_prefix(self, impl):
        survival = torch.tensor(
            [[0.99, 0.98, 0.50, 0.10], [0.97, 0.20, 0.05, 0.01]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        budget = 3
        verify_lens = impl(survival_probs=survival, budget=budget, cfg=cfg)
        num_requests, max_len = survival.shape[0], cfg.resolved_max_verify_len()
        flat = [
            (float(survival[r, p]), p, r)
            for r in range(num_requests)
            for p in range(min(max_len, survival.shape[1]))
            if float(survival[r, p]) >= cfg.survival_eps
        ]
        flat.sort(key=lambda e: (-e[0], e[1], e[2]))
        admitted_positions: dict[int, list[int]] = {r: [] for r in range(num_requests)}
        for _prob, position, request in flat[:budget]:
            admitted_positions[request].append(position)
        for request in range(num_requests):
            positions = sorted(admitted_positions[request])
            count = int(verify_lens[request].item()) - cfg.min_verify_len
            self.assertEqual(
                len(positions),
                count,
                f"request {request}: verify_len count mismatch",
            )
            expected_prefix = list(range(count))
            self.assertEqual(
                positions,
                expected_prefix,
                f"request {request} admitted {positions}, not the prefix "
                f"{expected_prefix}",
            )

    @_for_each_impl
    def test_higher_confidence_admitted_first(self, impl):
        survival = torch.tensor(
            [[0.99, 0.98, 0.97, 0.96], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        verify_lens = impl(survival_probs=survival, budget=2, cfg=cfg)
        extra = verify_lens.to(torch.int64) - cfg.min_verify_len
        self.assertEqual(int(extra[0].item()), 2)
        self.assertEqual(int(extra[1].item()), 0)

    @_for_each_impl
    def test_min_and_max_enter_the_budget(self, impl):
        survival = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=5, min_verify_len=1, max_verify_len=3)
        verify_lens = impl(survival_probs=survival, budget=100, cfg=cfg)
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertLessEqual(int(verify_lens.max().item()), 3)

    @_for_each_impl
    def test_budget_zero_returns_min_verify_len(self, impl):
        survival = torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=2, min_verify_len=1)
        verify_lens = impl(survival_probs=survival, budget=0, cfg=cfg)
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    @_for_each_impl
    def test_large_budget_selects_all_candidates(self, impl):
        survival = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        verify_lens = impl(survival_probs=survival, budget=1000, cfg=cfg)
        # anchor (min_verify_len=1) + all 3 admitted drafts
        self.assertEqual(int(verify_lens[0].item()), 4)

    @_for_each_impl
    def test_tie_break_is_value_independent(self, impl):
        survival = torch.tensor([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)
        floor = max(cfg.min_verify_len, 1)
        verify_lens = impl(survival_probs=survival, budget=3, cfg=cfg)
        total_extra = int((verify_lens.to(torch.int64) - floor).sum().item())
        self.assertEqual(total_extra, 3)


class TestVerifyLenAnchorContract(CustomTestCase):

    @_for_each_impl
    def test_explicit_zero_min_still_clamped_to_anchor(self, impl):
        survival = _survival_from_confidence(
            torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32)
        )
        cfg = DSparkScheduleConfig(gamma=3, min_verify_len=0)
        verify_lens = impl(survival_probs=survival, budget=0, cfg=cfg)
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([1, 1], dtype=torch.int32))
        )

    def test_non_flat_table_small_budget_feeds_ragged_layout(self):
        table = SpsCostTable(
            sample_batch_tokens=[2, 3],
            sample_steps_per_sec=[1.0, 0.1],
            max_batch_tokens=64,
        )
        cfg = DSparkScheduleConfig(gamma=3)
        survival = _survival_from_confidence(
            torch.tensor([[0.90, 0.80, 0.70], [0.85, 0.60, 0.40]], dtype=torch.float32)
        )
        budget = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        ).budget
        self.assertEqual(budget, 0)
        verify_lens = schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=budget, cfg=cfg
        )
        self.assertGreaterEqual(int(verify_lens.min().item()), 1)
        verify_lens_cpu = verify_lens.to(torch.int64).tolist()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=torch.device("cpu"),
            grid=[sum(verify_lens_cpu)],
        )
        self.assertEqual(layout.verify_lens_cpu, verify_lens_cpu)


class TestNonAnticipating(CustomTestCase):
    @_for_each_impl
    def test_lens_topk_non_anticipating_under_future_perturbation(self, impl):
        base = torch.tensor(
            [
                [0.95, 0.90, 0.80, 0.40],
                [0.92, 0.70, 0.30, 0.10],
                [0.99, 0.98, 0.50, 0.05],
            ],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4)
        budget = 5
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)

        request, cut = 1, 2
        for delta in (-0.05, -0.2, 0.05, 0.0):
            perturbed = base.clone()
            future = perturbed[request, cut:]
            perturbed[request, cut:] = torch.clamp(
                torch.minimum(future + delta, base[request, cut - 1]), min=0.0
            )
            verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
            admitted_prefix_unchanged = min(int(baseline[request].item()), cut) == min(
                int(verify_lens[request].item()), cut
            )
            self.assertTrue(
                admitted_prefix_unchanged,
                msg=f"prefix admission changed under future perturbation delta={delta}",
            )

    @_for_each_impl
    def test_other_requests_unaffected_by_one_request_future(self, impl):
        base = torch.tensor(
            [[0.95, 0.90, 0.20], [0.93, 0.88, 0.15]], dtype=torch.float32
        )
        cfg = DSparkScheduleConfig(gamma=3)
        budget = 2
        baseline = impl(survival_probs=base, budget=budget, cfg=cfg)
        perturbed = base.clone()
        perturbed[0, 2] = 0.01
        verify_lens = impl(survival_probs=perturbed, budget=budget, cfg=cfg)
        self.assertEqual(int(baseline[1].item()), int(verify_lens[1].item()))


class TestVanillaMatchesReference(CustomTestCase):
    def test_random_inputs_match_reference(self):
        torch.manual_seed(20260630)
        num_trials = 4000
        for trial in range(num_trials):
            num_requests = int(torch.randint(1, 6, ()).item())
            gamma = int(torch.randint(1, 9, ()).item())

            dtype = torch.float32 if trial % 2 == 0 else torch.float64
            confidence = torch.rand(num_requests, gamma, dtype=dtype)
            if trial % 3 == 0:
                confidence = (confidence * 4).round() / 4
            if trial % 7 == 0:
                confidence = torch.ones(num_requests, gamma, dtype=dtype)
            survival = torch.cumprod(confidence, dim=1)

            min_verify_len = int(torch.randint(0, gamma + 1, ()).item())
            if torch.rand(()).item() < 0.5:
                max_verify_len = 0
            else:
                max_verify_len = int(
                    torch.randint(min_verify_len, gamma + 1, ()).item()
                )
            survival_eps = float(
                [1e-6, 1e-3, 0.1, 0.5][int(torch.randint(0, 4, ()).item())]
            )
            budget = int(torch.randint(0, num_requests * gamma + 3, ()).item())

            cfg = DSparkScheduleConfig(
                gamma=gamma,
                min_verify_len=min_verify_len,
                max_verify_len=max_verify_len,
                survival_eps=survival_eps,
            )
            reference = schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            vanilla = schedule_verify_lens_topk_vanilla(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            self.assertTrue(
                torch.equal(reference, vanilla),
                msg=(
                    f"mismatch on trial {trial}: budget={budget} "
                    f"min={min_verify_len} max={max_verify_len} eps={survival_eps} "
                    f"survival={survival.tolist()} "
                    f"reference={reference.tolist()} vanilla={vanilla.tolist()}"
                ),
            )


class TestDSparkScheduleConfig(CustomTestCase):
    def test_validate_rejects_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            DSparkScheduleConfig(gamma=4, min_verify_len=3, max_verify_len=2).validate()

    def test_validate_rejects_max_greater_than_gamma_plus_one(self):
        with self.assertRaises(ValueError):
            DSparkScheduleConfig(gamma=4, max_verify_len=6).validate()

    def test_zero_max_resolves_to_gamma_plus_one(self):
        cfg = DSparkScheduleConfig(gamma=7)
        self.assertEqual(cfg.resolved_max_verify_len(), 8)


class TestGraphTierFillBudget(CustomTestCase):
    def test_below_cap_is_tier_minus_floor(self):
        """A tier under the per-request cap yields budget = tier - floor_tokens."""
        self.assertEqual(
            graph_tier_fill_budget(
                graph_num_tokens=48, bs=8, verify_num_draft_tokens=6, min_verify_len=1
            ),
            48 - 8,
        )

    def test_clamped_at_draft_capacity(self):
        """A tier above bs*K is capped at bs*K before subtracting the floor."""
        self.assertEqual(
            graph_tier_fill_budget(
                graph_num_tokens=100, bs=8, verify_num_draft_tokens=6, min_verify_len=1
            ),
            8 * 6 - 8,
        )

    def test_floor_scales_with_min_verify_len(self):
        """The subtracted floor is bs * max(min_verify_len, 1)."""
        self.assertEqual(
            graph_tier_fill_budget(
                graph_num_tokens=60, bs=10, verify_num_draft_tokens=6, min_verify_len=2
            ),
            60 - 20,
        )

    def test_never_negative(self):
        """A tier below the floor clamps the budget to zero, never negative."""
        self.assertEqual(
            graph_tier_fill_budget(
                graph_num_tokens=4, bs=8, verify_num_draft_tokens=6, min_verify_len=1
            ),
            0,
        )

    def test_feeding_budget_fills_topk_total_to_tier(self):
        """Feeding the fill budget to the top-k lifts the total to min(tier, bs*K)."""
        bs = 6
        cfg = DSparkScheduleConfig(gamma=6, min_verify_len=1)
        cap = cfg.resolved_max_verify_len()
        survival = torch.full((bs, cap), 0.99, dtype=torch.float32)
        for graph_num_tokens in (bs, bs * cap // 2, bs * cap, bs * cap + 12):
            budget = graph_tier_fill_budget(
                graph_num_tokens=graph_num_tokens,
                bs=bs,
                verify_num_draft_tokens=cap,
                min_verify_len=cfg.min_verify_len,
            )
            verify_lens = schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=budget, cfg=cfg
            )
            total = int(verify_lens.to(torch.int64).sum().item())
            self.assertEqual(total, min(graph_num_tokens, bs * cap))


if __name__ == "__main__":
    unittest.main()
