import tempfile
import unittest
from pathlib import Path

from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    build_capture_derived_sps_table,
    build_uninitialized_sps_table,
    is_uninitialized_sps_table,
    load_sps_table_from_path,
    profile_sps_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=24, suite="base-a-test-cpu")


def _make_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[8, 16, 32, 64],
        sample_steps_per_sec=[1000.0, 950.0, 500.0, 480.0],
        max_batch_tokens=128,
    )


class TestSpsCostTableInvariants(CustomTestCase):
    def test_rejects_non_increasing_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 8, 16],
                sample_steps_per_sec=[1.0, 2.0, 3.0],
                max_batch_tokens=16,
            )

    def test_rejects_unsorted_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[16, 8],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=16,
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0],
                max_batch_tokens=16,
            )

    def test_rejects_empty_table(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[],
                sample_steps_per_sec=[],
                max_batch_tokens=0,
            )

    def test_rejects_max_below_largest_probe(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=15,
            )


class TestSpsCostTableLookup(CustomTestCase):
    def test_lookup_exact_probe_returns_that_sps(self):
        table = _make_table()
        self.assertEqual(table.lookup(8), 1000.0)
        self.assertEqual(table.lookup(16), 950.0)
        self.assertEqual(table.lookup(32), 500.0)
        self.assertEqual(table.lookup(64), 480.0)

    def test_lookup_floors_to_lower_captured_probe(self):
        table = _make_table()
        self.assertEqual(table.lookup(31), 950.0)
        self.assertEqual(table.lookup(63), 500.0)

    def test_lookup_below_first_probe_clamps_to_first(self):
        table = _make_table()
        self.assertEqual(table.lookup(1), 1000.0)
        self.assertEqual(table.lookup(7), 1000.0)

    def test_lookup_above_last_probe_clamps_to_last(self):
        table = _make_table()
        self.assertEqual(table.lookup(65), 480.0)
        self.assertEqual(table.lookup(10_000), 480.0)


class TestLoadSpsTableFromPath(CustomTestCase):
    def test_load_from_path_round_trips_table_and_lookup(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = load_sps_table_from_path(str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)
        for batch_tokens in (1, 8, 31, 64, 200):
            self.assertEqual(loaded.lookup(batch_tokens), table.lookup(batch_tokens))


class TestFlatTableLookupIsConstant(CustomTestCase):
    def test_flat_table_lookup_is_one_for_any_batch(self):
        flat = SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=4096,
        )
        for batch_tokens in (0, 1, 2, 17, 256, 100_000):
            self.assertEqual(flat.lookup(batch_tokens), 1.0)


class TestProfileSpsTable(CustomTestCase):
    def test_profile_sorts_out_of_order_probes(self):
        table = profile_sps_table(
            probes=[(32, 500.0), (8, 1000.0), (16, 950.0)],
        )
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32])
        self.assertEqual(table.sample_steps_per_sec, [1000.0, 950.0, 500.0])

    def test_profile_rejects_duplicate_batch_tokens(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[(8, 1000.0), (8, 900.0)])

    def test_profile_rejects_empty_probes(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[])

    def test_profile_max_batch_tokens_defaults_to_largest_probe(self):
        table = profile_sps_table(probes=[(8, 1000.0), (64, 480.0), (16, 950.0)])
        self.assertEqual(table.max_batch_tokens, 64)

    def test_profile_honors_explicit_max_batch_tokens(self):
        table = profile_sps_table(
            probes=[(8, 1000.0), (16, 950.0)], max_batch_tokens=256
        )
        self.assertEqual(table.max_batch_tokens, 256)


def _build_sps_cost_table_for(testcase, *, sps_table_path):
    from sglang.srt.runtime_context import get_context, get_server_args
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        build_sps_cost_table,
    )

    # The table bound reads `max_running_requests` from the published bags, so
    # the case publishes it; the table path stays on the handed record, which is
    # what `build_sps_cost_table` takes.
    override = get_context().override_server_args(
        speculative_dspark_sps_table_path=sps_table_path,
        max_running_requests=4,
    )
    override.install()
    testcase.addCleanup(override.restore)
    return build_sps_cost_table(
        server_args=get_server_args(), verify_num_draft_tokens=5
    )


class TestBuildSpsCostTableContract(CustomTestCase):
    def test_unset_table_path_returns_flat_table(self):
        for sps_table_path in (None, ""):
            table = _build_sps_cost_table_for(self, sps_table_path=sps_table_path)
            self.assertEqual(table.sample_batch_tokens, [1])
            self.assertEqual(table.sample_steps_per_sec, [1.0])
            self.assertEqual(table.max_batch_tokens, 20)

    def test_real_path_loads_table(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = _build_sps_cost_table_for(self, sps_table_path=str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)


class TestIsUninitializedSpsTable(CustomTestCase):
    def test_additive_table_is_never_uninitialized(self):
        table = SpsAdditiveCostTable(
            bias_seconds=0.1,
            bs_probes=[128, 192, 256],
            alpha_seconds=[0.0, 0.008, 0.016],
            m_probes=[384, 512, 1024],
            theta_seconds=[0.0, 0.02, 0.1],
        )
        self.assertFalse(is_uninitialized_sps_table(table))

    def test_placeholder_diagonal_table_is_uninitialized(self):
        self.assertTrue(
            is_uninitialized_sps_table(
                build_uninitialized_sps_table(max_batch_tokens=128)
            )
        )

    def test_real_diagonal_table_is_initialized(self):
        self.assertFalse(is_uninitialized_sps_table(_make_table()))


class TestBuildCaptureDerivedSpsTable(CustomTestCase):
    """The capture-time cost model, built from two ladders of (shape, seconds, spread) triples.

    Verify graphs are keyed by total verify tokens and draft graphs by request count -- the two axes
    SpsAdditiveCostTable separates. Returning None is the contract for "unusable measurement": the
    caller then keeps the uninitialized table, i.e. the engine behaves exactly as it did before the
    derivation existed.
    """

    def test_rejects_too_few_probes(self):
        self.assertIsNone(
            build_capture_derived_sps_table(verify_probes=[], draft_probes=[])
        )
        self.assertIsNone(
            build_capture_derived_sps_table(
                verify_probes=[(8, 0.01, 0.0)], draft_probes=[]
            )
        )

    def test_rejects_non_positive_timings(self):
        self.assertIsNone(
            build_capture_derived_sps_table(
                verify_probes=[(8, 0.0, 0.0), (64, -1.0, 0.0)], draft_probes=[]
            )
        )

    def test_rejects_curve_with_too_little_dynamic_range(self):
        # A near-constant cost curve carries no scheduling signal, which is the degenerate case the
        # derivation exists to replace.
        flat = [(8, 0.010, 0.0), (64, 0.0102, 0.0), (128, 0.0104, 0.0)]
        self.assertIsNone(
            build_capture_derived_sps_table(verify_probes=flat, draft_probes=[])
        )

    def test_emits_the_additive_form(self):
        """The additive table is the point, not an implementation detail: it is the only one of the
        two shapes whose per-request term the planner holds fixed while it sweeps the budget.
        """
        table = build_capture_derived_sps_table(
            verify_probes=[
                (8, 0.00873, 3e-5),
                (64, 0.00964, 5e-5),
                (128, 0.01016, 5e-5),
                (384, 0.01980, 1e-4),
            ],
            draft_probes=[(1, 0.00120, 1e-5), (48, 0.00460, 1e-5)],
        )
        self.assertIsInstance(table, SpsAdditiveCostTable)
        self.assertEqual(table.m_probes, [8, 64, 128, 384])
        self.assertEqual(table.bs_probes, [1, 48])
        self.assertFalse(is_uninitialized_sps_table(table))

    def test_draft_cost_stays_on_the_request_axis(self):
        """The draft forward has already run, at full width, for every request by the time a budget
        is chosen. Folding its cost into the token axis instead would price trimming as
        though it also removed requests, so the verify term must come through untouched.
        """
        table = build_capture_derived_sps_table(
            verify_probes=[(48, 0.010, 0.0), (96, 0.015, 0.0), (288, 0.030, 0.0)],
            draft_probes=[(8, 0.004, 0.0), (48, 0.009, 0.0)],
        )
        for measured, stored in zip([0.010, 0.015, 0.030], table.theta_seconds):
            self.assertAlmostEqual(stored, measured, places=9)
        self.assertEqual(table.alpha_seconds, [0.004, 0.009])

    def test_trimming_the_budget_does_not_discount_the_draft(self):
        """The behavioural consequence, at a fixed request count: only the verify term may move.
        A draft term that shrinks with the budget is not a harmless constant, because the objective
        is a ratio -- it moves the argmax toward trimming."""
        table = build_capture_derived_sps_table(
            verify_probes=[(48, 0.010, 0.0), (96, 0.015, 0.0), (288, 0.030, 0.0)],
            draft_probes=[(8, 0.004, 0.0), (48, 0.009, 0.0)],
        )
        verify_all = table.step_time(num_reqs=48, budget=240)
        trimmed = table.step_time(num_reqs=48, budget=48)
        self.assertAlmostEqual(verify_all - trimmed, 0.030 - 0.015, places=9)

    def test_builds_without_a_draft_ladder(self):
        """A draft model with no captured graphs must not sink the derivation: a zero per-request
        term is the verify-only model, worse than the additive one but still a real curve.
        """
        table = build_capture_derived_sps_table(
            verify_probes=[(8, 0.010, 0.0), (64, 0.020, 0.0)], draft_probes=[]
        )
        self.assertIsNotNone(table)
        self.assertEqual(table.alpha_seconds, [0.0])

    def test_sorts_and_deduplicates_probes(self):
        table = build_capture_derived_sps_table(
            verify_probes=[
                (128, 0.02, 0.0),
                (8, 0.01, 0.0),
                (8, 0.011, 0.0),
                (64, 0.015, 0.0),
            ],
            draft_probes=[],
        )
        self.assertEqual(table.m_probes, [8, 64, 128])

    def test_repairs_a_non_monotone_curve(self):
        # A curve that says a wider step is cheaper can make the planner prefer the wider step, so
        # cost must never fall as tokens are added.
        table = build_capture_derived_sps_table(
            verify_probes=[(8, 0.010, 0.0), (64, 0.008, 0.0), (128, 0.020, 0.0)],
            draft_probes=[],
        )
        theta = table.theta_seconds
        self.assertTrue(all(theta[i] <= theta[i + 1] for i in range(len(theta) - 1)))

    def test_repairs_a_non_monotone_draft_ladder(self):
        # The request axis carries the same invariant: adding requests cannot make the draft cheaper.
        table = build_capture_derived_sps_table(
            verify_probes=[(8, 0.010, 0.0), (128, 0.020, 0.0)],
            draft_probes=[(1, 0.004, 0.0), (8, 0.002, 0.0), (48, 0.009, 0.0)],
        )
        alpha = table.alpha_seconds
        self.assertTrue(all(alpha[i] <= alpha[i + 1] for i in range(len(alpha) - 1)))

    def test_pools_differences_the_measurement_cannot_resolve(self):
        # Neighbours closer together than the measurement spread must come out equal-cost, so the
        # planner cannot trim on noise in a region where the true cost is flat.
        table = build_capture_derived_sps_table(
            verify_probes=[
                (8, 0.01000, 0.0005),
                (16, 0.01002, 0.0005),
                (24, 0.01004, 0.0005),
                (384, 0.02000, 0.0005),
            ],
            draft_probes=[],
        )
        unresolvable = table.theta_seconds[:3]
        self.assertAlmostEqual(min(unresolvable), max(unresolvable), places=9)

    def test_keeps_differences_the_measurement_does_resolve(self):
        table = build_capture_derived_sps_table(
            verify_probes=[
                (8, 0.01000, 1e-5),
                (16, 0.01200, 1e-5),
                (24, 0.01400, 1e-5),
                (384, 0.02000, 1e-5),
            ],
            draft_probes=[],
        )
        theta = table.theta_seconds
        self.assertLess(theta[0], theta[1])
        self.assertLess(theta[1], theta[2])


class TestDerivationSkippedWhenGraphsAreNeverReplayed(CustomTestCase):
    """Capture records a verify graph per token tier on any backend, but the runner's admission test
    begins at `attn_backend.supports_ragged_verify_graph`. A backend that leaves the base-class
    default of False -- a hybrid linear-attention target, whose GDN backend does not override it --
    fails that test on every step, so the captured graphs are recorded and never replayed.

    Timing them there prices a path the engine does not take. Worse than useless: on a Qwen3.6-27B
    hybrid-GDN target the derivation installed a curve, the planner began trimming, and the first
    decode step died with an illegal memory access inside the eager compact verify path. This guard
    is therefore a crash guard, not only a performance one, and it must key on the capability flag
    rather than on whether graphs were captured.
    """

    def _skips(self, *, supports_ragged, ragged_mode=True, runner_present=True):
        from unittest.mock import MagicMock

        from sglang.srt.speculative.dspark_components.dspark_planner import (
            ragged_verify_graphs_are_replayable,
        )

        model_runner = MagicMock()
        if not runner_present:
            model_runner.decode_cuda_graph_runner = None
        else:
            runner = model_runner.decode_cuda_graph_runner
            runner.ragged_verify_mode = ragged_mode
            runner.attn_backend.supports_ragged_verify_graph = supports_ragged
        return not ragged_verify_graphs_are_replayable(model_runner=model_runner)

    def test_backend_without_ragged_verify_support_is_skipped(self):
        self.assertTrue(self._skips(supports_ragged=False))

    def test_backend_with_ragged_verify_support_proceeds(self):
        self.assertFalse(self._skips(supports_ragged=True))

    def test_non_ragged_mode_is_skipped(self):
        # Static mode captures fixed-width verify graphs on a different axis; the derived table is a
        # compact-mode object and must not be built from them.
        self.assertTrue(self._skips(supports_ragged=True, ragged_mode=False))

    def test_absent_runner_is_skipped(self):
        self.assertTrue(self._skips(supports_ragged=True, runner_present=False))

    def test_install_survives_an_absent_runner(self):
        """The rank-0 skip log must not dereference the runner the guard just found missing.

        A server launched with decode CUDA graphs disabled and the derivation enabled has
        `decode_cuda_graph_runner is None`; the guard correctly declines, but an earlier revision's
        skip message then read `.attn_backend` off that None and turned a clean decline into an
        AttributeError at startup.
        """
        from unittest.mock import MagicMock

        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_context, get_server_args
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkVerifyPlanner,
        )

        override = get_context().override_server_args(
            speculative_dspark_align_verify_tokens_to_graph_tier=False,
            speculative_dspark_confidence_sts_path=None,
            speculative_dspark_sps_table_path=None,
            max_running_requests=4,
        )
        override.install()
        self.addCleanup(override.restore)
        # Compact-mode construction requires a confidence head; the MagicMock default (non-None)
        # satisfies it without loading weights.
        draft_model = MagicMock()
        model_runner = MagicMock()
        model_runner.decode_cuda_graph_runner = None
        with envs.SGLANG_RAGGED_VERIFY_MODE.override(
            "compact"
        ), envs.SGLANG_DSPARK_ENABLE_CAPTURE_DERIVED_SPS.override(True):
            planner = DSparkVerifyPlanner(
                draft_model=draft_model,
                gamma=8,
                model_runner=model_runner,
                device="cpu",
                tp_rank=0,
                server_args=get_server_args(),
                verify_num_draft_tokens=8,
            )
            planner.install_capture_derived_sps_table(draft_model_runner=None)
        self.assertTrue(planner.is_verify_all)


class TestFullWindowBudgetKeepsTheFastPath(CustomTestCase):
    """Installing a real cost table clears `_is_verify_all` for the engine's lifetime, because the
    planner *can* now trim. But on a deployment with no headroom it chooses a full-width budget on
    every step, and without this the engine pays the per-step top-k schedule and its host<->device
    round-trips to reproduce the layout the uniform cache already holds.
    """

    def _covers(self, *, bs, budget, gamma=7, min_verify_len=1, max_verify_len=0):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
            DSparkVerifyPlanner,
        )

        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._schedule_cfg = DSparkScheduleConfig(
            gamma=gamma, min_verify_len=min_verify_len, max_verify_len=max_verify_len
        )
        return planner._budget_covers_full_window(bs=bs, budget=budget)

    def test_the_fast_path_is_inert_until_a_derived_table_is_installed(self):
        """Drives the real `schedule_layout`, not a copy of its condition.

        A deployment running a profiled table from disk, or one that simply did not opt in, must keep
        the scheduling path it has today. An earlier revision of this guard re-stated the gate inline,
        which would have stayed green if someone dropped `_derived_sps_installed` from the production
        expression -- exactly the regression it exists to catch.
        """
        from unittest.mock import patch

        import torch

        from sglang.srt.speculative.dspark_components import dspark_planner as mod
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
            DSparkVerifyPlanner,
        )
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        bs, gamma = 4, 7
        first_step = {}

        def _record(kind):
            # Only the FIRST branch entered per run is the gate's decision. A miss then falls back
            # to a uniform layout further down, which is correct but not what is under test.
            def _fn(**kwargs):
                first_step.setdefault(installed, kind)
                return "uniform-layout" if kind == "fast-path" else None

            return _fn

        for installed in (False, True):
            planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
            planner._schedule_cfg = DSparkScheduleConfig(gamma=gamma)
            planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
            planner._is_verify_all = False
            planner._derived_sps_installed = installed
            planner._uniform_layout_cache = {}
            planner.verify_num_draft_tokens = gamma + 1
            planner.model_runner = None
            planner._schedule_verify_lens = _record("scheduler")
            planner._budget_aligned_to_graph_tier = lambda **kwargs: kwargs["budget"]

            with patch.object(mod, "uniform_ragged_layout", _record("fast-path")):
                planner.schedule_layout(
                    req_pool_indices=torch.zeros(bs, dtype=torch.int64),
                    prefix_lens=torch.zeros(bs, dtype=torch.int64),
                    device="cpu",
                    confidence=torch.zeros(bs, gamma),
                    budget=bs * gamma,
                )

        # Not installed -> the scheduler runs, as it does upstream today.
        # Installed -> the cached uniform layout is served instead.
        self.assertEqual(first_step[False], "scheduler")
        self.assertEqual(first_step[True], "fast-path")

    def test_budget_that_fills_every_window_takes_the_fast_path(self):
        # gamma=7 -> window 8, minus the always-verified anchor -> 7 selectable slots per request.
        self.assertTrue(self._covers(bs=48, budget=48 * 7))
        self.assertTrue(self._covers(bs=48, budget=48 * 7 + 1))

    def test_one_token_short_falls_through_to_the_scheduler(self):
        # The boundary is the whole point: a budget that cannot fill every window must be scheduled,
        # or requests that should have been trimmed would silently verify their full window.
        self.assertFalse(self._covers(bs=48, budget=48 * 7 - 1))

    def test_absent_budget_is_not_treated_as_full(self):
        # `budget=None` reaches here from paths that never consulted the planner; treating it as
        # full-window would turn "no decision yet" into "verify everything".
        self.assertFalse(self._covers(bs=48, budget=None))

    def test_honours_a_capped_max_verify_len(self):
        # With max_verify_len capping the window, fewer slots per request are selectable, so the
        # fast path must engage at a proportionally smaller budget.
        self.assertTrue(self._covers(bs=10, budget=10 * 3, gamma=7, max_verify_len=4))
        self.assertFalse(
            self._covers(bs=10, budget=10 * 3 - 1, gamma=7, max_verify_len=4)
        )


class TestCaptureDerivedInstallIsInertInStaticMode(CustomTestCase):
    """Static is the DEFAULT ragged-verify mode, and the install hook runs on every DSpark start.

    The compact-only branch of DSparkVerifyPlanner.__init__ is where the derivation decision is
    made, so a regression that leaves the decision attribute unset there does not show up in any
    compact-mode test -- it shows up as a startup crash for every default user.
    """

    def _planner_in_static_mode(self):
        from unittest.mock import MagicMock

        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_context, get_server_args
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkVerifyPlanner,
        )

        # The planner reads the spec/schedule bags as well as the handed record,
        # so the case publishes; the record it is handed is the published one.
        override = get_context().override_server_args(
            speculative_dspark_align_verify_tokens_to_graph_tier=False,
            speculative_dspark_confidence_sts_path=None,
            speculative_dspark_sps_table_path=None,
            max_running_requests=4,
        )
        override.install()
        self.addCleanup(override.restore)
        draft_model = MagicMock()
        draft_model.confidence_head = None
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            return DSparkVerifyPlanner(
                draft_model=draft_model,
                gamma=8,
                model_runner=MagicMock(),
                device="cpu",
                tp_rank=0,
                server_args=get_server_args(),
                verify_num_draft_tokens=8,
            )

    def test_planner_constructs_and_install_is_a_no_op(self):
        planner = self._planner_in_static_mode()
        # Must not raise: the hook is called unconditionally from DSparkWorkerV2.init_cuda_graphs.
        planner.install_capture_derived_sps_table(draft_model_runner=None)
        self.assertTrue(planner.is_verify_all)


class TestMinPredictedGainGuard(CustomTestCase):
    """Trimming has to earn its keep: fewer verified tokens is a real loss, so a predicted win
    smaller than the threshold must leave the schedule at verify-all."""

    _STEEP = [1000.0, 700.0, 400.0, 100.0]
    _NEARLY_FLAT = [1000.0, 999.5, 999.0, 998.5]
    _VERIFY_ALL = 8 * 7  # 8 requests x 7 survival columns, all clearing survival_eps

    @staticmethod
    def _survival():
        import torch

        return torch.tensor([[0.5**k for k in range(1, 8)]] * 8, dtype=torch.float32)

    @staticmethod
    def _table(steps_per_sec):
        return SpsCostTable(
            sample_batch_tokens=[8, 64, 128, 384],
            sample_steps_per_sec=steps_per_sec,
            max_batch_tokens=384,
        )

    def _budget(self, *, min_predicted_gain, steps_per_sec):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
            compute_verify_token_budget,
        )

        cfg = DSparkScheduleConfig(gamma=8, min_predicted_gain=min_predicted_gain)
        return compute_verify_token_budget(
            history_survival_probs=self._survival(),
            sps_table=self._table(steps_per_sec),
            cfg=cfg,
        ).budget

    def _unguarded_argmax(self, steps_per_sec):
        """The objective's argmax, rebuilt here independently of the code under test."""
        import torch

        from sglang.srt.speculative.dspark_components.dspark_planner import (
            _lookup_sps_tensor,
        )

        survival = self._survival()
        num_requests = survival.shape[0]
        candidates = torch.sort(
            survival.flatten().to(torch.float64), descending=True
        ).values
        tau_star = num_requests + torch.cat(
            [torch.zeros(1, dtype=torch.float64), torch.cumsum(candidates, dim=0)]
        )
        batch_tokens = num_requests + torch.arange(tau_star.numel(), dtype=torch.int64)
        sps = _lookup_sps_tensor(
            sps_table=self._table(steps_per_sec), batch_tokens=batch_tokens
        )
        return int(torch.argmax(tau_star * sps))

    def test_zero_threshold_is_exactly_the_unguarded_argmax(self):
        for steps_per_sec in (self._STEEP, self._NEARLY_FLAT):
            self.assertEqual(
                self._budget(min_predicted_gain=0.0, steps_per_sec=steps_per_sec),
                self._unguarded_argmax(steps_per_sec),
            )

    def test_a_steep_curve_still_trims_under_the_threshold(self):
        self.assertLess(
            self._budget(min_predicted_gain=0.05, steps_per_sec=self._STEEP),
            self._VERIFY_ALL,
        )

    def test_a_nearly_flat_curve_declines_to_trim(self):
        # Precondition: without the guard this curve does trim, so the guard is what changes the
        # answer rather than the curve being uninteresting.
        self.assertLess(
            self._budget(min_predicted_gain=0.0, steps_per_sec=self._NEARLY_FLAT),
            self._VERIFY_ALL,
        )
        self.assertEqual(
            self._budget(min_predicted_gain=0.05, steps_per_sec=self._NEARLY_FLAT),
            self._VERIFY_ALL,
        )

    def test_raising_the_threshold_never_trims_more(self):
        budgets = [
            self._budget(min_predicted_gain=g, steps_per_sec=self._STEEP)
            for g in (0.0, 0.05, 0.2, 0.9)
        ]
        self.assertEqual(budgets, sorted(budgets))
        self.assertEqual(budgets[-1], self._VERIFY_ALL)


class TestSimulateAccLenGate(CustomTestCase):
    """SGLANG_SIMULATE_ACC_LEN > 1 requires a verify-all schedule: a constant simulated
    correct_len can exceed a trimmed request's budget and break the cutoff accounting.
    DSparkWorkerV2 refuses that combination at construction, and deriving a cost table afterwards
    would flip the planner off verify-all behind that check. Both the admission check and the
    derivation read this predicate, so these cases pin the boundary it draws -- in particular that
    exactly 1.0 is safe (it yields correct_len 0) while anything else above 0 is not."""

    def _needs_verify_all(self, value):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            simulate_acc_len_needs_verify_all,
        )

        if value is None:
            with envs.SGLANG_SIMULATE_ACC_LEN.override(0.0):
                return simulate_acc_len_needs_verify_all()
        with envs.SGLANG_SIMULATE_ACC_LEN.override(value):
            return simulate_acc_len_needs_verify_all()

    def test_matches_the_worker_admission_rule(self):
        # DSparkWorkerV2 raises for `acc_len > 0 and acc_len != 1.0` unless the schedule is
        # verify-all, so those are exactly the values that must block derivation.
        for value, blocked in (
            (None, False),
            (0.0, False),
            (1.0, False),
            (1.5, True),
            (2.0, True),
            (3.5, True),
        ):
            with self.subTest(simulate_acc_len=value):
                self.assertEqual(self._needs_verify_all(value), blocked)


class TestStepCostProbeBroadcastIsDeadlockFree(CustomTestCase):
    """The cost measurement ends in a collective, and a rank that skips it strands its peers.

    The shape broadcast must therefore depend only on the captured tier list, which every rank
    derives identically -- never on how many probes this rank managed to measure -- and the
    broadcast must still happen when the local measurement fails outright. Both properties are
    invisible on one GPU and fatal on two, which is why they are pinned here.
    """

    def _planner_with(self, measured, *, raises=False):
        from unittest.mock import MagicMock, patch

        from sglang.srt.speculative.dspark_components import dspark_planner as mod

        planner = mod.DSparkVerifyPlanner.__new__(mod.DSparkVerifyPlanner)
        planner.model_runner = MagicMock()
        planner.device = "cpu"
        planner.verify_num_draft_tokens = 8
        runner = planner.model_runner.decode_cuda_graph_runner
        if raises:
            runner.measure_captured_replay_seconds.side_effect = RuntimeError("boom")
        else:
            runner.measure_captured_replay_seconds.return_value = measured
        draft_runner = MagicMock()
        draft_runner.decode_cuda_graph_runner = None

        group = MagicMock()
        tiers = [8, 16, 24, 32]
        with patch.object(
            mod, "ragged_capture_num_tokens", return_value=tiers
        ), patch.object(
            mod, "verify_lens_broadcast_group", return_value=(group, 2)
        ), patch.object(
            mod, "get_parallel", return_value=MagicMock(tp_size=2)
        ):
            probes = planner._measure_step_cost_probes(draft_model_runner=draft_runner)
        return tiers, group, probes

    def test_broadcast_shape_is_fixed_however_little_was_measured(self):
        from sglang.srt.speculative.dspark_components import dspark_planner as mod

        for measured in ([], [(8, 0.01, 0.0)], [(8, 0.01, 0.0), (32, 0.02, 0.0)]):
            with self.subTest(num_measured=len(measured)):
                tiers, group, _ = self._planner_with(measured)
                group.broadcast.assert_called_once()
                sent = group.broadcast.call_args.args[0]
                self.assertEqual(
                    tuple(sent.shape),
                    (len(tiers) + mod.CAPTURE_DERIVED_SPS_MAX_PROBES, 3),
                )

    def test_broadcast_still_happens_when_the_local_measurement_raises(self):
        _, group, (verify_probes, draft_probes) = self._planner_with([], raises=True)
        group.broadcast.assert_called_once()
        self.assertEqual(verify_probes, [])
        self.assertEqual(draft_probes, [])

    def test_only_measured_tiers_are_returned(self):
        _, _, (verify_probes, _) = self._planner_with(
            [(8, 0.01, 0.001), (32, 0.02, 0.002)]
        )
        self.assertEqual([tier for tier, _, _ in verify_probes], [8, 32])

    def test_a_draft_runner_without_captured_graphs_yields_no_request_axis(self):
        """`decode_cuda_graph_runner is None` is the real shape of a draft model whose graphs were
        never captured. The measurement must return an empty draft ladder rather than fail, because
        the builder degrades to the verify-only model on it."""
        _, _, (verify_probes, draft_probes) = self._planner_with([(8, 0.01, 0.0)])
        self.assertEqual(draft_probes, [])
        self.assertTrue(verify_probes)


if __name__ == "__main__":
    unittest.main()
