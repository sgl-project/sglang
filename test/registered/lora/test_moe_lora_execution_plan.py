"""CPU tests for the typed, whole-pipeline MoE-LoRA execution contract."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

# Load the pure contract directly so this CPU test does not execute
# ``sglang.__init__`` (which legitimately imports the CUDA/Triton runtime).
_SOURCE = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/lora/moe/execution_plan.py"
)
_SPEC = importlib.util.spec_from_file_location("_moe_lora_execution_plan", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SERIAL_MATERIALIZED_REFERENCE = _MODULE.SERIAL_MATERIALIZED_REFERENCE
ActivationFamily = _MODULE.ActivationFamily
EarlyOverlap = _MODULE.EarlyOverlap
FactorContract = _MODULE.FactorContract
FactorLayout = _MODULE.FactorLayout
FactorSite = _MODULE.FactorSite
FinalizeFamily = _MODULE.FinalizeFamily
FinalizeSpec = _MODULE.FinalizeSpec
LateOverlap = _MODULE.LateOverlap
LoraAFamily = _MODULE.LoraAFamily
LoraASpec = _MODULE.LoraASpec
LoraBFamily = _MODULE.LoraBFamily
LoraBSpec = _MODULE.LoraBSpec
MiddleFamily = _MODULE.MiddleFamily
MiddleSpec = _MODULE.MiddleSpec
MoeLoraExecutionPlan = _MODULE.MoeLoraExecutionPlan
RouteBuilderFamily = _MODULE.RouteBuilderFamily
RouteRequirement = _MODULE.RouteRequirement


def _a(
    site: FactorSite,
    family: LoraAFamily = LoraAFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: FactorLayout = FactorLayout.PAIR_MAJOR,
) -> LoraASpec:
    return LoraASpec(site, family, is_shared_outer, layout)


def _b(
    site: FactorSite,
    family: LoraBFamily = LoraBFamily.ONE_LAUNCH_SLICED,
    is_shared_outer: bool = False,
    layout: FactorLayout = FactorLayout.PAIR_MAJOR,
) -> LoraBSpec:
    return LoraBSpec(site, family, is_shared_outer, layout)


def _factor(
    site: FactorSite,
    is_shared_outer: bool = False,
    layout: FactorLayout = FactorLayout.PAIR_MAJOR,
) -> FactorContract:
    return FactorContract(site, is_shared_outer, layout)


def _plan(**changes) -> MoeLoraExecutionPlan:
    values = {
        "gate_a": _a(FactorSite.GATE_UP),
        "gate_b": _b(FactorSite.GATE_UP),
        "middle": MiddleSpec(MiddleFamily.MATERIALIZED, ActivationFamily.SWIGLU),
        "down_a": _a(FactorSite.DOWN),
        "down_b": _b(FactorSite.DOWN),
        "finalize": FinalizeSpec(FinalizeFamily.MATERIALIZED),
    }
    values.update(changes)
    return MoeLoraExecutionPlan(**values)


class TestFactorAndKernelSpecs(unittest.TestCase):
    def test_specs_require_a_bool_ownership_flag(self):
        with self.assertRaises(TypeError):
            _a(FactorSite.GATE_UP, is_shared_outer=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            _b(FactorSite.DOWN, is_shared_outer=1)  # type: ignore[arg-type]

    def test_a_rejects_unqualified_family_site_ownership_and_layout(self):
        invalid = (
            lambda: _a(
                FactorSite.DOWN,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                FactorLayout.TOKEN_MAJOR,
            ),
            lambda: _a(
                FactorSite.GATE_UP,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                FactorLayout.PAIR_MAJOR,
            ),
            lambda: _a(
                FactorSite.GATE_UP,
                LoraAFamily.INDEXED,
            ),
            lambda: _a(
                FactorSite.GATE_UP,
                LoraAFamily.INDEXED,
                True,
            ),
            lambda: _a(
                FactorSite.GATE_UP,
                LoraAFamily.GROUPED,
                layout=FactorLayout.TOKEN_MAJOR,
            ),
        )
        for construct in invalid:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()

    def test_b_rejects_down_token_layout(self):
        with self.assertRaisesRegex(ValueError, "down bridge"):
            _b(
                FactorSite.DOWN,
                LoraBFamily.ONE_LAUNCH_SLICED,
                layout=FactorLayout.TOKEN_MAJOR,
            )

    def test_specs_require_typed_enums(self):
        with self.assertRaises(TypeError):
            LoraASpec("gate_up", LoraAFamily.GROUPED)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LoraBSpec(FactorSite.GATE_UP, "one_launch_sliced")  # type: ignore[arg-type]

    def test_only_gate_a_and_down_b_may_be_shared_outer(self):
        self.assertEqual(
            _a(
                FactorSite.GATE_UP,
                is_shared_outer=True,
            ).is_shared_outer,
            True,
        )
        self.assertEqual(
            _b(
                FactorSite.DOWN,
                is_shared_outer=True,
            ).is_shared_outer,
            True,
        )
        with self.assertRaisesRegex(ValueError, "down A is always per-expert"):
            _a(
                FactorSite.DOWN,
                is_shared_outer=True,
            )
        with self.assertRaisesRegex(ValueError, "gate/up B is always per-expert"):
            _b(
                FactorSite.GATE_UP,
                is_shared_outer=True,
            )


class TestFusionOwnership(unittest.TestCase):
    def test_middle_rejects_missing_duplicate_and_wrong_site_consumers(self):
        with self.assertRaisesRegex(ValueError, "requires.*gate B"):
            MiddleSpec(MiddleFamily.B_ACTIVATION, ActivationFamily.SWIGLU)
        with self.assertRaisesRegex(ValueError, "does not consume.*gate B"):
            MiddleSpec(
                MiddleFamily.MATERIALIZED,
                ActivationFamily.SWIGLU,
                consumed_gate_b=_factor(FactorSite.GATE_UP),
            )
        with self.assertRaisesRegex(ValueError, "gate/up site"):
            MiddleSpec(
                MiddleFamily.B_ACTIVATION,
                ActivationFamily.SWIGLU,
                consumed_gate_b=_factor(FactorSite.DOWN),
            )
        with self.assertRaisesRegex(ValueError, "gate/up B must be per-expert"):
            MiddleSpec(
                MiddleFamily.B_ACTIVATION,
                ActivationFamily.SWIGLU,
                consumed_gate_b=_factor(FactorSite.GATE_UP, True),
            )

    def test_finalize_rejects_missing_or_wrong_ownership_consumer(self):
        with self.assertRaisesRegex(ValueError, "requires.*down B"):
            FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE)
        with self.assertRaisesRegex(ValueError, "does not consume.*down B"):
            FinalizeSpec(FinalizeFamily.MATERIALIZED, _factor(FactorSite.DOWN))
        with self.assertRaisesRegex(ValueError, "shared-outer"):
            FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, _factor(FactorSite.DOWN))


class TestWholePipelineValidation(unittest.TestCase):
    def test_reference_is_immutable_complete_and_aligned(self):
        self.assertIs(
            SERIAL_MATERIALIZED_REFERENCE.validate(),
            SERIAL_MATERIALIZED_REFERENCE,
        )
        self.assertEqual(
            SERIAL_MATERIALIZED_REFERENCE.route_requirements(),
            frozenset((RouteRequirement.ALIGNED_PER_EXPERT,)),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SERIAL_MATERIALIZED_REFERENCE.gate_b = None  # type: ignore[misc]

    def test_plan_validates_against_the_resident_ownership_flag(self):
        self.assertIs(
            SERIAL_MATERIALIZED_REFERENCE.validate_ownership(False),
            SERIAL_MATERIALIZED_REFERENCE,
        )
        with self.assertRaisesRegex(ValueError, "gate-A ownership"):
            SERIAL_MATERIALIZED_REFERENCE.validate_ownership(True)
        with self.assertRaisesRegex(ValueError, "down-B ownership"):
            _plan(
                gate_a=_a(FactorSite.GATE_UP, is_shared_outer=True)
            ).validate_ownership(True)
        with self.assertRaises(TypeError):
            SERIAL_MATERIALIZED_REFERENCE.validate_ownership(1)  # type: ignore

    def test_every_required_middle_and_finalize_composition_constructs(self):
        gate = _factor(FactorSite.GATE_UP)
        shared_down = _factor(FactorSite.DOWN, is_shared_outer=True)
        plans = (
            _plan(
                gate_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_b=gate,
                ),
            ),
            _plan(
                gate_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_b=gate,
                ),
                down_b=None,
                finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down),
            ),
        )
        for plan in plans:
            self.assertIs(plan.validate(), plan)

    def test_exactly_one_stage_owner_is_required(self):
        gate = _factor(FactorSite.GATE_UP)
        shared_down = _factor(FactorSite.DOWN, is_shared_outer=True)
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_b=gate,
                )
            )
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(gate_b=None)
        with self.assertRaisesRegex(ValueError, "standalone stage"):
            _plan(down_a=None)
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down))

    def test_bridge_layouts_must_match_at_both_sites(self):
        token_gate_a = _a(
            FactorSite.GATE_UP,
            LoraAFamily.TOKEN_DEDUP_GROUPED,
            True,
            FactorLayout.TOKEN_MAJOR,
        )
        with self.assertRaisesRegex(ValueError, "gate A output layout"):
            _plan(gate_a=token_gate_a)

        token_gate_b = _b(
            FactorSite.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            FactorLayout.TOKEN_MAJOR,
        )
        plan = _plan(gate_a=token_gate_a, gate_b=token_gate_b)
        self.assertIs(plan.validate(), plan)

    def test_overlaps_reject_consumed_stages(self):
        gate = _factor(FactorSite.GATE_UP)
        shared_down = _factor(FactorSite.DOWN, is_shared_outer=True)
        with self.assertRaisesRegex(ValueError, "gate-A\\+B"):
            _plan(
                gate_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_b=gate,
                ),
                early_overlap=EarlyOverlap.GATE_A_B,
            )
        with self.assertRaisesRegex(ValueError, "standalone down B"):
            _plan(
                down_b=None,
                finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down),
                late_overlap=LateOverlap.DOWN_B,
            )

    def test_indexed_a_is_the_down_a_only_composition(self):
        down_only = _plan(
            down_a=_a(FactorSite.DOWN, LoraAFamily.INDEXED),
        )
        self.assertEqual(
            down_only.route_requirements(),
            frozenset(
                (
                    RouteRequirement.RAW,
                    RouteRequirement.ALIGNED_PER_EXPERT,
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "down site"):
            _plan(gate_a=_a(FactorSite.GATE_UP, LoraAFamily.INDEXED))

    def test_pdl_edges_are_explicit_and_fail_closed(self):
        both = _plan(gate_a_to_b_pdl=True, down_a_to_b_pdl=True)
        self.assertTrue(both.gate_a_to_b_pdl_eligible())
        self.assertTrue(both.down_a_to_b_pdl_eligible())

        with self.assertRaisesRegex(ValueError, "gate-A -> gate-B"):
            _plan(
                gate_b=_b(FactorSite.GATE_UP, LoraBFamily.INDEXED_PAIRS),
                gate_a_to_b_pdl=True,
            )
        with self.assertRaisesRegex(ValueError, "down-A -> down-B"):
            _plan(
                late_overlap=LateOverlap.DOWN_A,
                down_a_to_b_pdl=True,
            )
        with self.assertRaisesRegex(ValueError, "down-A -> down-B"):
            _plan(
                down_a=_a(FactorSite.DOWN, LoraAFamily.INDEXED),
                down_a_to_b_pdl=True,
            )

    def test_base_pdl_edge_requires_a_direct_supported_consumer(self):
        gateup = _plan(base_gateup_to_middle_pdl=True)
        self.assertTrue(gateup.base_gateup_to_middle_pdl_eligible())

        for overlap in (EarlyOverlap.GATE_A, EarlyOverlap.GATE_A_B):
            with self.assertRaisesRegex(ValueError, "base gate/up -> middle"):
                _plan(
                    early_overlap=overlap,
                    base_gateup_to_middle_pdl=True,
                )


class TestRouteRequirementUnion(unittest.TestCase):
    def test_shared_token_a_and_per_expert_b_require_both_products(self):
        plan = _plan(
            gate_a=_a(
                FactorSite.GATE_UP,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                FactorLayout.TOKEN_MAJOR,
            ),
            gate_b=_b(
                FactorSite.GATE_UP,
                LoraBFamily.ONE_LAUNCH_SLICED,
                False,
                FactorLayout.TOKEN_MAJOR,
            ),
        )
        self.assertEqual(
            plan.route_requirements(),
            frozenset(
                (
                    RouteRequirement.SHARED_TOKEN_PLAN,
                    RouteRequirement.ALIGNED_PER_EXPERT,
                )
            ),
        )

    def test_joint_builder_requires_both_aligned_pair_plans(self):
        with self.assertRaisesRegex(ValueError, "requires both aligned"):
            _plan(
                route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
                route_pdl=False,
            )

        shared_down_b = _b(
            FactorSite.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            True,
        )
        joint = _plan(
            down_b=shared_down_b,
            route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
            route_pdl=False,
        )
        self.assertEqual(
            joint.route_requirements(),
            frozenset(
                (
                    RouteRequirement.ALIGNED_PER_EXPERT,
                    RouteRequirement.ALIGNED_SHARED_OUTER,
                )
            ),
        )

    def test_joint_builder_requires_explicit_pdl_control(self):
        shared_down_b = _b(
            FactorSite.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            True,
        )
        with self.assertRaisesRegex(ValueError, "explicit route_pdl"):
            _plan(
                down_b=shared_down_b,
                route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
            )
        for enabled in (False, True):
            plan = _plan(
                down_b=shared_down_b,
                route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
                route_pdl=enabled,
            )
            self.assertIs(plan.route_pdl, enabled)

    def test_standard_builder_keeps_pdl_default_off_and_accepts_explicit_twins(self):
        self.assertIsNone(_plan().route_pdl)
        for enabled in (False, True):
            self.assertIs(_plan(route_pdl=enabled).route_pdl, enabled)

    def test_shared_finalize_adds_raw_without_hiding_aligned_routes(self):
        shared_down = _factor(FactorSite.DOWN, is_shared_outer=True)
        plan = _plan(
            down_b=None,
            finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down),
        )
        self.assertEqual(
            plan.route_requirements(),
            frozenset(
                (
                    RouteRequirement.ALIGNED_PER_EXPERT,
                    RouteRequirement.RAW,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
