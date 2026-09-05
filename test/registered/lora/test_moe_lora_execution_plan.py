"""CPU tests for the typed, whole-pipeline MoE-LoRA execution contract."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

import pydantic

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")

_SOURCE = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/lora/moe/execution_plan.py"
)
_SPEC = importlib.util.spec_from_file_location("_moe_lora_execution_plan", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ActivationFn = _MODULE.ActivationFn
GateUpOverlap = _MODULE.GateUpOverlap
BridgeLayout = _MODULE.BridgeLayout
Site = _MODULE.Site
FinalizeFamily = _MODULE.FinalizeFamily
FinalizeSpec = _MODULE.FinalizeSpec
DownOverlap = _MODULE.DownOverlap
LoraAFamily = _MODULE.LoraAFamily
LoraASpec = _MODULE.LoraASpec
LoraBFamily = _MODULE.LoraBFamily
LoraBSpec = _MODULE.LoraBSpec
ActFamily = _MODULE.ActFamily
ActSpec = _MODULE.ActSpec
MoeLoraExecutionPlan = _MODULE.MoeLoraExecutionPlan
RouteBuilderFamily = _MODULE.RouteBuilderFamily
RouteRequirement = _MODULE.RouteRequirement


def _a(
    site: Site,
    family: LoraAFamily = LoraAFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> LoraASpec:
    return LoraASpec(site, family, is_shared_outer, layout)


def _b(
    site: Site,
    family: LoraBFamily = LoraBFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> LoraBSpec:
    return LoraBSpec(site, family, is_shared_outer, layout)


def _plan(**changes) -> MoeLoraExecutionPlan:
    values = {
        "gate_up_a": _a(Site.GATE_UP),
        "gate_up_b": _b(Site.GATE_UP),
        "act": ActSpec(ActFamily.MATERIALIZED, ActivationFn.SILU),
        "down_a": _a(Site.DOWN),
        "down_b": _b(Site.DOWN),
        "finalize": FinalizeSpec(FinalizeFamily.MATERIALIZED),
    }
    values.update(changes)
    return MoeLoraExecutionPlan(**values)


SERIAL_MATERIALIZED_REFERENCE = _plan()


class TestFactorAndKernelSpecs(unittest.TestCase):
    def test_specs_require_a_bool_ownership_flag(self):
        with self.assertRaises(pydantic.ValidationError):
            _a(Site.GATE_UP, is_shared_outer=1)  # type: ignore[arg-type]
        with self.assertRaises(pydantic.ValidationError):
            _b(Site.DOWN, is_shared_outer=1)  # type: ignore[arg-type]

    def test_a_rejects_unqualified_family_site_ownership_and_layout(self):
        invalid = (
            lambda: _a(
                Site.DOWN,
                LoraAFamily.TOKEN_GROUPED,
                True,
                BridgeLayout.TOKEN_MAJOR,
            ),
            lambda: _a(
                Site.GATE_UP,
                LoraAFamily.TOKEN_GROUPED,
                True,
                BridgeLayout.PAIR_MAJOR,
            ),
            lambda: _a(
                Site.GATE_UP,
                LoraAFamily.GROUPED,
                layout=BridgeLayout.TOKEN_MAJOR,
            ),
        )
        for construct in invalid:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()

        # Indexed A supports both ownership modes, but requires pair-major output.
        for site, shared in (
            (Site.GATE_UP, False),
            (Site.GATE_UP, True),
            (Site.DOWN, True),
        ):
            with self.subTest(site=site, shared=shared):
                spec = _a(site, LoraAFamily.PER_PAIR, shared)
                self.assertIs(spec.output_layout, BridgeLayout.PAIR_MAJOR)
        with self.assertRaises(ValueError):
            _a(Site.DOWN, LoraAFamily.PER_PAIR, layout=BridgeLayout.TOKEN_MAJOR)

    def test_b_rejects_down_token_layout(self):
        with self.assertRaisesRegex(ValueError, "down bridge"):
            _b(
                Site.DOWN,
                LoraBFamily.GROUPED,
                layout=BridgeLayout.TOKEN_MAJOR,
            )

    def test_specs_require_typed_enums(self):
        with self.assertRaises(pydantic.ValidationError):
            LoraASpec("gate_up", LoraAFamily.GROUPED)  # type: ignore[arg-type]
        with self.assertRaises(pydantic.ValidationError):
            LoraBSpec(Site.GATE_UP, "grouped")  # type: ignore[arg-type]

    def test_specs_reject_unknown_field_names(self):
        with self.assertRaises(pydantic.ValidationError):
            LoraASpec(Site.GATE_UP, LoraAFamily.GROUPED, nonsense=True)  # type: ignore
        with self.assertRaises(pydantic.ValidationError):
            dataclasses.replace(
                SERIAL_MATERIALIZED_REFERENCE,
                down_b_scater=True,  # type: ignore[call-arg]
            )

    def test_only_gate_up_a_and_down_b_may_be_shared_outer(self):
        self.assertEqual(
            _a(
                Site.GATE_UP,
                is_shared_outer=True,
            ).is_shared_outer,
            True,
        )
        self.assertEqual(
            _b(
                Site.DOWN,
                is_shared_outer=True,
            ).is_shared_outer,
            True,
        )
        # The weight loader, not the plan, restricts shared-outer sites.
        self.assertEqual(_a(Site.DOWN, is_shared_outer=True).is_shared_outer, True)
        self.assertEqual(_b(Site.GATE_UP, is_shared_outer=True).is_shared_outer, True)


class TestFusionOwnership(unittest.TestCase):
    def test_ownership_lives_on_the_family_alone(self):
        # A consuming family records nothing of the absorbed stage; the
        # exactly-one-owner rules are plan-level (tested below).
        self.assertIsNotNone(ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU))

    def test_shared_finalize_requires_shared_ownership(self):
        with self.assertRaisesRegex(ValueError, "shared-outer"):
            FinalizeSpec(FinalizeFamily.SHARED_TOKEN_DELTA)
        self.assertTrue(
            FinalizeSpec(
                FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
            ).is_shared_outer
        )


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
            SERIAL_MATERIALIZED_REFERENCE.gate_up_b = None  # type: ignore[misc]

    def test_every_required_act_and_finalize_composition_constructs(self):
        plans = (
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
            ),
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
                down_b=None,
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                ),
            ),
        )
        for plan in plans:
            self.assertIs(plan.validate(), plan)

    def test_exactly_one_stage_owner_is_required(self):
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU))
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(gate_up_b=None)
        with self.assertRaises(pydantic.ValidationError):
            _plan(down_a=None)  # required field: down A always runs standalone
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                )
            )

    def test_bridge_layouts_must_match_at_both_sites(self):
        token_gate_up_a = _a(
            Site.GATE_UP,
            LoraAFamily.TOKEN_GROUPED,
            True,
            BridgeLayout.TOKEN_MAJOR,
        )
        with self.assertRaisesRegex(ValueError, "gate/up A output layout"):
            _plan(gate_up_a=token_gate_up_a)

        token_gate_up_b = _b(
            Site.GATE_UP,
            LoraBFamily.GROUPED,
            False,
            BridgeLayout.TOKEN_MAJOR,
        )
        plan = _plan(gate_up_a=token_gate_up_a, gate_up_b=token_gate_up_b)
        self.assertIs(plan.validate(), plan)

    def test_overlaps_reject_consumed_stages(self):
        with self.assertRaisesRegex(ValueError, "gate/up-A\\+B"):
            _plan(
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
                gate_up_overlap=GateUpOverlap.GATE_UP_A_B,
            )
        with self.assertRaisesRegex(ValueError, "standalone down B"):
            _plan(
                down_b=None,
                finalize=FinalizeSpec(
                    FinalizeFamily.SHARED_TOKEN_DELTA, is_shared_outer=True
                ),
                down_overlap=DownOverlap.DOWN_B,
            )

    def test_indexed_a_is_the_down_a_only_composition(self):
        down_only = _plan(
            down_a=_a(Site.DOWN, LoraAFamily.PER_PAIR),
        )
        self.assertEqual(
            down_only.route_requirements(),
            frozenset(
                (
                    RouteRequirement.RAW_PER_EXPERT,
                    RouteRequirement.ALIGNED_PER_EXPERT,
                )
            ),
        )
        # An indexed gate/up-A is coherent (the kernel visits pairs at either
        # site); only its pair-major bridge is pinned.
        indexed_gate_up = _plan(gate_up_a=_a(Site.GATE_UP, LoraAFamily.PER_PAIR))
        self.assertEqual(
            indexed_gate_up.route_requirements(),
            frozenset(
                (RouteRequirement.RAW_PER_EXPERT, RouteRequirement.ALIGNED_PER_EXPERT)
            ),
        )


class TestRouteRequirementUnion(unittest.TestCase):
    def test_shared_token_a_and_per_expert_b_require_both_products(self):
        plan = _plan(
            gate_up_a=_a(
                Site.GATE_UP,
                LoraAFamily.TOKEN_GROUPED,
                True,
                BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=_b(
                Site.GATE_UP,
                LoraBFamily.GROUPED,
                False,
                BridgeLayout.TOKEN_MAJOR,
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

    def test_parallel_builder_yields_both_aligned_pair_plans(self):
        shared_down_b = _b(
            Site.DOWN,
            LoraBFamily.GROUPED,
            True,
        )
        parallel = _plan(
            down_b=shared_down_b,
            route_builder=RouteBuilderFamily.PARALLEL_SHARED_OUTER,
        )
        self.assertEqual(
            parallel.route_requirements(),
            frozenset(
                (
                    RouteRequirement.ALIGNED_PER_EXPERT,
                    RouteRequirement.ALIGNED_SHARED_OUTER,
                )
            ),
        )

    def test_shared_finalize_adds_raw_without_hiding_aligned_routes(self):
        plan = _plan(
            down_b=None,
            finalize=FinalizeSpec(FinalizeFamily.SHARED_ONE_PASS, is_shared_outer=True),
        )
        self.assertEqual(
            plan.route_requirements(),
            frozenset(
                (
                    RouteRequirement.ALIGNED_PER_EXPERT,
                    RouteRequirement.RAW_SHARED_OUTER,
                )
            ),
        )


class TestTokenDenseBridgeOwnership(unittest.TestCase):
    def test_token_dense_a_requires_a_standalone_per_pair_gate_up_b(self):
        token_dense_a = _a(
            Site.GATE_UP, LoraAFamily.TOKEN_DENSE, True, BridgeLayout.TOKEN_MAJOR
        )
        _plan(
            gate_up_a=token_dense_a,
            gate_up_b=_b(
                Site.GATE_UP, LoraBFamily.PER_PAIR, True, BridgeLayout.TOKEN_MAJOR
            ),
        )
        with self.assertRaises(ValueError):
            _plan(
                gate_up_a=token_dense_a,
                gate_up_b=_b(
                    Site.GATE_UP, LoraBFamily.GROUPED, True, BridgeLayout.TOKEN_MAJOR
                ),
            )
        with self.assertRaises(ValueError):
            _plan(
                gate_up_a=token_dense_a,
                gate_up_b=None,
                act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
            )


if __name__ == "__main__":
    unittest.main()
