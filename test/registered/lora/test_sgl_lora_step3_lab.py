"""Step-3 lab foundation: schema axes, exec specs, poison probe, ledger.

The plan §63.1 P1 additions (as amended by the first S3 review) are all
evidence machinery — a wrong axis here corrupts every Step-3 comparison
silently, so each one gets a pin before any schedule arm consumes it:

* the K1 per-bridge precision axes (all FIVE materialized bridges) must
  enter the content digest, reject unknown values, and default to the
  Step-1/2 semantics so old archive records still decode;
* the A-candidate identity is the typed LoraAExecutionSpec, NOT a case
  field: it must express the review's required combinations
  (indexed+CuTeDSL, grouped+split-K) and reject impossible ones
  (token-dedup down site);
* executors fail CLOSED — the serial control raises on a non-default
  bridge axis, and the production executor raises at ENTRY on a case it
  cannot honor (the skip_reason alone was fail-open: a driver that forgot
  to consult it silently executed grouped/BF16 under any declaration);
* the inactive-rank-tail poison probe is per-A-site selective: poisoning
  B tails too would fail a COMPLIANT A kernel (stock B multiplies its own
  NaN tail columns by zeros), so the probe poisons ONE A factor and is
  judged at that site's immediate A output;
* ledger rows are evidence-bound: provenance comes from the suite and a
  row cannot cite a record the suite did not measure.

Triton only: no SM100 requirement, so this also runs on the H200 pod.
"""

from __future__ import annotations

import json
import tempfile
import unittest

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    MoeLoraBenchCase,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.crossover_ledger import (
    MIN_MARGIN,
    decide_cell,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.production_runner import (
    prepare_production_forward,
    production_runner_skip_reason,
)
from benchmark.kernels.lora_moe.serial_control import (
    run_serial_materialized_control,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    TimingRecord,
    new_suite,
    write_suite,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

BRIDGE_FIELDS = (
    "bridge_gate_a_out",
    "bridge_gate_up_delta",
    "bridge_activation_lora_input",
    "bridge_down_a_out",
    "bridge_down_delta",
)


def _case(**overrides) -> MoeLoraBenchCase:
    params = dict(
        device="cuda:0",
        model_preset="tiny_smoke",
        adapter_cell=AdapterCell(
            active_adapters=2, include_base_rows=True, slot_capacity=4
        ),
        route_generator="iid",
        num_tokens=16,
        active_rank=8,
        physical_rank=16,
        source_revision="test",
    )
    params.update(overrides)
    return build_case(**params)


class Step3CaseAxes(CustomTestCase):
    def test_defaults_are_step2_semantics(self):
        case = _case()
        for field in BRIDGE_FIELDS:
            self.assertEqual(getattr(case, field), "bf16")

    def test_unknown_axis_values_reject(self):
        with self.assertRaisesRegex(ValueError, "bridge_down_a_out"):
            _case(bridge_down_a_out="fp64")

    def test_axes_enter_the_content_digest(self):
        base = _case()
        for field in BRIDGE_FIELDS:
            self.assertNotEqual(base.case_id, _case(**{field: "fp32"}).case_id)

    def test_step2_archive_records_decode_with_defaults(self):
        # A record written before the Step-3 axes existed carries no such
        # keys; decoding into the new struct must yield the declared
        # defaults rather than an error, so old archives stay readable.
        encoded = msgspec.json.encode(_case())
        stripped = {
            key: value
            for key, value in msgspec.json.decode(encoded).items()
            if not key.startswith("bridge_")
        }
        decoded = msgspec.json.decode(
            msgspec.json.encode(stripped), type=MoeLoraBenchCase
        )
        self.assertEqual(decoded.bridge_gate_up_delta, "bf16")
        self.assertEqual(decoded.bridge_activation_lora_input, "bf16")


class ExecutionSpecIdentity(CustomTestCase):
    def test_required_review_combinations_are_expressible(self):
        indexed_cutedsl = LoraAExecutionSpec(
            site="gate_up", ownership="indexed", implementation="cutedsl"
        )
        grouped_splitk = LoraAExecutionSpec(
            site="down", ownership="grouped", reduction="deterministic_split_k"
        )
        self.assertEqual(indexed_cutedsl.key(), "gate_up_indexed_cutedsl")
        self.assertEqual(grouped_splitk.key(), "down_grouped_splitk")

    def test_impossible_and_unknown_specs_reject(self):
        with self.assertRaisesRegex(ValueError, "token-dedup down-A"):
            LoraAExecutionSpec(
                site="down", ownership="grouped", shared_handling="token_dedup"
            )
        with self.assertRaisesRegex(ValueError, "ownership"):
            LoraAExecutionSpec(site="gate_up", ownership="turbo")

    def test_identity_defaults_stay_out_of_the_key(self):
        self.assertEqual(
            LoraAExecutionSpec(site="gate_up", ownership="grouped").key(),
            "gate_up_grouped",
        )


class Step3ExecutorDeclarations(CustomTestCase):
    def test_serial_control_raises_on_non_default_bridge(self):
        device = torch.device("cuda:0")
        case = _case(bridge_activation_lora_input="fp32")
        tensors = materialize_case_tensors(case)
        with self.assertRaisesRegex(ValueError, "bridge_activation_lora_input"):
            run_serial_materialized_control(case, tensors, device=device)

    def test_production_executor_fails_closed_at_entry(self):
        case = _case(bridge_gate_a_out="fp32")
        self.assertIn("BF16", production_runner_skip_reason(case))
        tensors = materialize_case_tensors(case)
        # The entry raise must fire even when a driver forgets skip_reason.
        with self.assertRaisesRegex(ValueError, "does not execute"):
            prepare_production_forward(case, tensors, device=torch.device("cuda:0"))


class InactiveRankTailPoison(CustomTestCase):
    def test_selective_probe_discriminates_the_poisoned_site(self):
        device = torch.device("cuda:0")
        case = _case()  # active_rank=8 < physical_rank=16: tails exist
        poisoned = materialize_case_tensors(case, poison_inactive_rank_tails="down_a")

        # Only the down-A factor carries NaN tails; every other factor keeps
        # the contractual zero tails a compliant pipeline may multiply by.
        self.assertTrue(
            bool(poisoned.lora_a_down[:, :, case.active_rank :, :].isnan().all())
        )
        self.assertTrue(bool(poisoned.lora_a_gate_up.isfinite().all()))
        self.assertTrue(bool(poisoned.lora_b_gate_up.isfinite().all()))
        self.assertTrue(bool(poisoned.lora_b_down.isfinite().all()))

        result = run_serial_materialized_control(case, poisoned, device=device)
        # Judge at the immediate A output: the full-physical-rank grouped
        # kernel writes NaN into exactly the tail columns of the poisoned
        # site, while the UNPOISONED site's A output stays finite — the
        # discrimination the review required.
        import benchmark.kernels.lora_moe.reference as reference

        stages = reference.reference_pair_stages(case, poisoned)
        lora_rows = (stages.pair_adapter >= 0).to(device)
        down_a = result.down_lora_a[lora_rows]
        self.assertFalse(bool(down_a[:, case.active_rank :].isfinite().all()))
        self.assertTrue(bool(down_a[:, : case.active_rank].isfinite().all()))
        gate_a = result.gate_up_lora_a[lora_rows]
        self.assertTrue(bool(gate_a.isfinite().all()))

        # The contractual zero-tail materialization stays finite end to end.
        clean = materialize_case_tensors(case)
        result_clean = run_serial_materialized_control(case, clean, device=device)
        self.assertTrue(bool(result_clean.output.isfinite().all()))

    def test_unknown_site_rejects(self):
        with self.assertRaisesRegex(ValueError, "poison_inactive_rank_tails"):
            materialize_case_tensors(_case(), poison_inactive_rank_tails="both")


class CrossoverCellDecision(CustomTestCase):
    def test_unanimous_with_margin_decides(self):
        decision = decide_cell(
            arm_a="jit",
            samples_a=[1.10, 1.20, 1.15],
            arm_b="fused",
            samples_b=[1.00, 1.00, 1.00],
        )
        self.assertEqual(decision.winner, "fused")
        self.assertTrue(decision.unanimous)
        self.assertGreaterEqual(decision.margin(), MIN_MARGIN)

    def test_non_unanimous_ties_even_with_large_geo_margin(self):
        decision = decide_cell(
            arm_a="jit",
            samples_a=[2.0, 2.0, 0.99],
            arm_b="fused",
            samples_b=[1.0, 1.0, 1.0],
        )
        self.assertIsNone(decision.winner)
        self.assertFalse(decision.unanimous)

    def test_unanimous_below_margin_ties(self):
        decision = decide_cell(
            arm_a="jit",
            samples_a=[1.01, 1.02, 1.01],
            arm_b="fused",
            samples_b=[1.00, 1.00, 1.00],
        )
        self.assertIsNone(decision.winner)
        self.assertTrue(decision.unanimous)
        self.assertEqual(decision.margin(), 1.0)


class EvidenceBoundLedger(CustomTestCase):
    @staticmethod
    def _record(
        record_id: str,
        case_id: str,
        suite,
        candidate: str = "gate_up_grouped",
        num_tokens: int = 64,
        extra: dict | None = None,
    ) -> TimingRecord:
        return TimingRecord(
            record_id=record_id,
            candidate=candidate,
            boundary=BOUNDARY_ISOLATED,
            cache_state="l2_hot_graph",
            params={"case_id": case_id, "T": num_tokens, **(extra or {})},
            median_s=1e-5,
            mean_s=1e-5,
            p25_s=1e-5,
            p75_s=1e-5,
            replicate_s=(1e-5,),
            memory_footprint_bytes=None,
            bandwidth_gib_s=None,
            graph_replay=True,
            device_name=suite.device_name,
            source_revision=suite.source_revision,
        )

    def test_entries_bind_to_measured_records(self):
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        suite.add(self._record("rec-a", "c" * 16, suite))
        suite.add(self._record("rec-b", "d" * 16, suite, candidate="gate_up_indexed"))
        suite.add(self._record("rec-c", "e" * 16, suite, num_tokens=96))
        suite.add(
            self._record(
                "rec-d", "f" * 16, suite, candidate="gate_up_indexed", num_tokens=96
            )
        )

        entry = suite.site_crossover(
            site="gate_up_a",
            boundary=BOUNDARY_ISOLATED,
            candidates=("gate_up_grouped", "gate_up_indexed"),
            axis="num_tokens",
            crossover_location="T in (64, 96]",
            bracketing_low_record_ids=("rec-a", "rec-b"),
            bracketing_high_record_ids=("rec-c", "rec-d"),
            cache_state="l2_hot_graph",
            axis_param="T",
            workload_params=(),
            notes="unit fixture",
        )
        self.assertEqual(entry.bracketing_high_record_ids, ("rec-c", "rec-d"))
        # Provenance is DERIVED, and the bracketing case ids come from the
        # records themselves — a hand-typed wrong revision cannot exist.
        self.assertEqual(entry.source_revision, "test-rev")
        self.assertEqual(entry.device, suite.device_name)
        self.assertEqual(
            entry.bracketing_case_ids,
            ("c" * 16, "d" * 16, "e" * 16, "f" * 16),
        )

        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            write_suite(suite, handle.name)
            payload = json.load(open(handle.name))
        self.assertEqual(len(payload["ledger"]), 1)
        self.assertEqual(payload["ledger"][0]["axis"], "num_tokens")

    def test_uncited_or_mismatched_records_reject(self):
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        suite.add(self._record("rec-a", "c" * 16, suite))
        with self.assertRaisesRegex(ValueError, "not in this suite"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-missing"),
                bracketing_high_record_ids=("rec-a",),
                cache_state="l2_hot_graph",
            )
        suite.add(self._record("rec-b", "d" * 16, suite))
        with self.assertRaisesRegex(ValueError, "measured at"):
            suite.site_crossover(
                site="gate_up_a",
                boundary="route_inclusive",
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a",),
                bracketing_high_record_ids=("rec-b",),
                cache_state="l2_hot_graph",
            )

    def test_candidate_membership_is_enforced(self):
        # The review's exact exploit: a grouped-vs-indexed claim citing two
        # GROUPED records must be refused, in both directions — a cited
        # record outside the declared pair, and a declared arm with no
        # evidence at all.
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        suite.add(self._record("rec-a", "c" * 16, suite))
        suite.add(self._record("rec-b", "d" * 16, suite))
        suite.add(self._record("rec-e", "g" * 16, suite))
        with self.assertRaisesRegex(ValueError, "BOTH arms measured in BOTH"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-b"),
                bracketing_high_record_ids=("rec-e",),
                cache_state="l2_hot_graph",
            )
        # Distinct record IDs whose params share the axis value are NOT two
        # axis cells (fifth S3 review's exact exploit: both sides T=64
        # under a "T in (64, 96]" claim).
        suite.add(self._record("rec-f", "h" * 16, suite, candidate="gate_up_indexed"))
        with self.assertRaisesRegex(ValueError, "same 'T' value"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-b"),
                bracketing_high_record_ids=("rec-e", "rec-f"),
                cache_state="l2_hot_graph",
                axis_param="T",
                workload_params=(),
            )
        # Identical low/high citation sets cannot bracket a flip (fourth
        # S3 review).
        with self.assertRaisesRegex(ValueError, "DISTINCT cells"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-b"),
                bracketing_high_record_ids=("rec-a", "rec-b"),
                cache_state="l2_hot_graph",
            )
        # One-armed HIGH cell (third S3 review): both candidates measured
        # somewhere is not enough — each bracketing cell needs both arms.
        suite.add(self._record("rec-c", "e" * 16, suite, candidate="gate_up_indexed"))
        with self.assertRaisesRegex(ValueError, "high bracketing cell"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("gate_up_grouped", "gate_up_indexed"),
                axis="num_tokens",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-c"),
                bracketing_high_record_ids=("rec-b",),
                cache_state="l2_hot_graph",
            )
        with self.assertRaisesRegex(ValueError, "not one of the declared"):
            suite.site_crossover(
                site="gate_up_a",
                boundary=BOUNDARY_ISOLATED,
                candidates=("down_grouped", "down_indexed"),
                axis="num_tokens (declared-candidates mismatch)",
                crossover_location="T in (64, 96]",
                bracketing_low_record_ids=("rec-a", "rec-b"),
                bracketing_high_record_ids=("rec-e",),
                cache_state="l2_hot_graph",
            )

    def test_workload_signature_is_explicit_and_fail_closed(self):
        # Sixth S3 review: the previous validator compared EVERY parameter,
        # which broke the real producer (candidates record their own tuning
        # configs), while never comparing the two cells' workloads at all.
        def add(suite, rid, candidate, T, rank, config):
            suite.add(
                self._record(
                    rid,
                    "c" * 16,
                    suite,
                    candidate=candidate,
                    num_tokens=T,
                    extra={"rank": rank, "config": config},
                )
            )

        common = dict(
            site="gate_up_a",
            boundary=BOUNDARY_ISOLATED,
            candidates=("gate_up_grouped", "gate_up_indexed"),
            axis="num_tokens",
            crossover_location="T in (64, 96]",
            cache_state="l2_hot_graph",
            axis_param="T",
        )

        # (1) Candidate-specific configs within a cell are LEGAL — this is
        # the producer-breaking regression the review found.
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        add(suite, "lo-g", "gate_up_grouped", 64, 16, {"BLOCK": 32})
        add(suite, "lo-i", "gate_up_indexed", 64, 16, {"BLOCK": 128})
        add(suite, "hi-g", "gate_up_grouped", 96, 16, {"BLOCK": 64})
        add(suite, "hi-i", "gate_up_indexed", 96, 16, {"BLOCK": 16})
        entry = suite.site_crossover(
            bracketing_low_record_ids=("lo-g", "lo-i"),
            bracketing_high_record_ids=("hi-g", "hi-i"),
            workload_params=("rank",),
            **common,
        )
        self.assertEqual(entry.bracketing_low_record_ids, ("lo-g", "lo-i"))

        # (2) The two cells must be one workload: a rank mismatch between
        # cells is a different comparison, not two points on one axis.
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        add(suite, "lo-g", "gate_up_grouped", 64, 16, {})
        add(suite, "lo-i", "gate_up_indexed", 64, 16, {})
        add(suite, "hi-g", "gate_up_grouped", 96, 64, {})
        add(suite, "hi-i", "gate_up_indexed", 96, 64, {})
        with self.assertRaisesRegex(ValueError, "workload parameter 'rank'"):
            suite.site_crossover(
                bracketing_low_record_ids=("lo-g", "lo-i"),
                bracketing_high_record_ids=("hi-g", "hi-i"),
                workload_params=("rank",),
                **common,
            )
        # (3) ... and the signature must be declared — axis validation
        # without workload_params is refused rather than silently partial.
        with self.assertRaisesRegex(ValueError, "workload_params"):
            suite.site_crossover(
                bracketing_low_record_ids=("lo-g", "lo-i"),
                bracketing_high_record_ids=("hi-g", "hi-i"),
                **common,
            )

        # (4) One candidate's records WITHIN a cell must agree on all their
        # parameters — intra-arm config drift inside a cell is a bug.
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        add(suite, "lo-g1", "gate_up_grouped", 64, 16, {"BLOCK": 32})
        add(suite, "lo-g2", "gate_up_grouped", 64, 16, {"BLOCK": 64})
        add(suite, "lo-i", "gate_up_indexed", 64, 16, {})
        add(suite, "hi-g", "gate_up_grouped", 96, 16, {"BLOCK": 32})
        add(suite, "hi-i", "gate_up_indexed", 96, 16, {})
        with self.assertRaisesRegex(ValueError, "disagree on non-axis"):
            suite.site_crossover(
                bracketing_low_record_ids=("lo-g1", "lo-g2", "lo-i"),
                bracketing_high_record_ids=("hi-g", "hi-i"),
                workload_params=("rank",),
                **common,
            )

        # (5) Two cells at the SAME axis value cannot bracket a flip even
        # with disjoint record sets.
        suite = new_suite("step3_lab_test", source_revision="test-rev")
        add(suite, "lo-g", "gate_up_grouped", 64, 16, {})
        add(suite, "lo-i", "gate_up_indexed", 64, 16, {})
        add(suite, "hi-g", "gate_up_grouped", 64, 16, {})
        add(suite, "hi-i", "gate_up_indexed", 64, 16, {})
        with self.assertRaisesRegex(ValueError, "same 'T' value"):
            suite.site_crossover(
                bracketing_low_record_ids=("lo-g", "lo-i"),
                bracketing_high_record_ids=("hi-g", "hi-i"),
                workload_params=("rank",),
                **common,
            )


if __name__ == "__main__":
    unittest.main()
