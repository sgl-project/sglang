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

ActivationFamily = _MODULE.ActivationFamily
EarlyOverlap = _MODULE.EarlyOverlap
StageContract = _MODULE.StageContract
BridgeLayout = _MODULE.BridgeLayout
Site = _MODULE.Site
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
    site: Site,
    family: LoraAFamily = LoraAFamily.GROUPED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> LoraASpec:
    return LoraASpec(site, family, is_shared_outer, layout)


def _b(
    site: Site,
    family: LoraBFamily = LoraBFamily.ONE_LAUNCH_SLICED,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> LoraBSpec:
    return LoraBSpec(site, family, is_shared_outer, layout)


def _factor(
    site: Site,
    is_shared_outer: bool = False,
    layout: BridgeLayout = BridgeLayout.PAIR_MAJOR,
) -> StageContract:
    return StageContract(site, is_shared_outer, layout)


def _plan(**changes) -> MoeLoraExecutionPlan:
    values = {
        "gate_up_a": _a(Site.GATE_UP),
        "gate_up_b": _b(Site.GATE_UP),
        "middle": MiddleSpec(MiddleFamily.MATERIALIZED, ActivationFamily.SWIGLU),
        "down_a": _a(Site.DOWN),
        "down_b": _b(Site.DOWN),
        "finalize": FinalizeSpec(FinalizeFamily.MATERIALIZED),
    }
    values.update(changes)
    return MoeLoraExecutionPlan(**values)


# The simplest correct pipeline: every stage standalone, nothing fused, no
# overlap window.  Other plans are described as departures from it.
SERIAL_MATERIALIZED_REFERENCE = _plan()


class TestFactorAndKernelSpecs(unittest.TestCase):
    def test_specs_require_a_bool_ownership_flag(self):
        # strict pydantic fields: no int -> bool coercion
        with self.assertRaises(pydantic.ValidationError):
            _a(Site.GATE_UP, is_shared_outer=1)  # type: ignore[arg-type]
        with self.assertRaises(pydantic.ValidationError):
            _b(Site.DOWN, is_shared_outer=1)  # type: ignore[arg-type]

    def test_a_rejects_unqualified_family_site_ownership_and_layout(self):
        invalid = (
            lambda: _a(
                Site.DOWN,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                BridgeLayout.TOKEN_MAJOR,
            ),
            lambda: _a(
                Site.GATE_UP,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
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

        # An indexed A is not pinned to a site or to per-expert ownership:
        # the kernel visits pairs and derives shared-outer keys from the
        # route either way.  Only its pair-major bridge is definitional.
        for site, shared in (
            (Site.GATE_UP, False),
            (Site.GATE_UP, True),
            (Site.DOWN, True),
        ):
            with self.subTest(site=site, shared=shared):
                spec = _a(site, LoraAFamily.INDEXED, shared)
                self.assertIs(spec.output_layout, BridgeLayout.PAIR_MAJOR)
        with self.assertRaises(ValueError):
            _a(Site.DOWN, LoraAFamily.INDEXED, layout=BridgeLayout.TOKEN_MAJOR)

    def test_b_rejects_down_token_layout(self):
        with self.assertRaisesRegex(ValueError, "down bridge"):
            _b(
                Site.DOWN,
                LoraBFamily.ONE_LAUNCH_SLICED,
                layout=BridgeLayout.TOKEN_MAJOR,
            )

    def test_specs_require_typed_enums(self):
        # strict pydantic fields: no str -> enum coercion
        with self.assertRaises(pydantic.ValidationError):
            LoraASpec("gate_up", LoraAFamily.GROUPED)  # type: ignore[arg-type]
        with self.assertRaises(pydantic.ValidationError):
            LoraBSpec(Site.GATE_UP, "one_launch_sliced")  # type: ignore[arg-type]

    def test_specs_reject_unknown_field_names(self):
        # extra="forbid": a mistyped kwarg (including through
        # dataclasses.replace) is a hard error, exactly as it was for the
        # stdlib dataclasses — never a silently unchanged copy.
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
        # Which sites may be shared-outer is the adapter weight format,
        # validated where the weights load into the memory buffer — the plan
        # model only carries the flag.
        self.assertEqual(_a(Site.DOWN, is_shared_outer=True).is_shared_outer, True)
        self.assertEqual(_b(Site.GATE_UP, is_shared_outer=True).is_shared_outer, True)


class TestFusionOwnership(unittest.TestCase):
    def test_middle_rejects_missing_duplicate_and_wrong_site_consumers(self):
        with self.assertRaisesRegex(ValueError, "requires.*gate/up B"):
            MiddleSpec(MiddleFamily.B_ACTIVATION, ActivationFamily.SWIGLU)
        with self.assertRaisesRegex(ValueError, "does not consume.*gate/up B"):
            MiddleSpec(
                MiddleFamily.MATERIALIZED,
                ActivationFamily.SWIGLU,
                consumed_gate_up_b=_factor(Site.GATE_UP),
            )
        with self.assertRaisesRegex(ValueError, "gate/up site"):
            MiddleSpec(
                MiddleFamily.B_ACTIVATION,
                ActivationFamily.SWIGLU,
                consumed_gate_up_b=_factor(Site.DOWN),
            )
        # The consumed contract's ownership is the adapter weight format,
        # validated at weight load; the middle only needs it to name the
        # gate/up site.
        self.assertIsNotNone(
            MiddleSpec(
                MiddleFamily.B_ACTIVATION,
                ActivationFamily.SWIGLU,
                consumed_gate_up_b=_factor(Site.GATE_UP, True),
            ).consumed_gate_up_b
        )

    def test_finalize_rejects_missing_or_wrong_ownership_consumer(self):
        with self.assertRaisesRegex(ValueError, "requires.*down B"):
            FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE)
        with self.assertRaisesRegex(ValueError, "does not consume.*down B"):
            FinalizeSpec(FinalizeFamily.MATERIALIZED, _factor(Site.DOWN))
        with self.assertRaisesRegex(ValueError, "shared-outer"):
            FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, _factor(Site.DOWN))


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

    def test_every_required_middle_and_finalize_composition_constructs(self):
        gate = _factor(Site.GATE_UP)
        shared_down = _factor(Site.DOWN, is_shared_outer=True)
        plans = (
            _plan(
                gate_up_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_up_b=gate,
                ),
            ),
            _plan(
                gate_up_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_up_b=gate,
                ),
                down_b=None,
                finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down),
            ),
        )
        for plan in plans:
            self.assertIs(plan.validate(), plan)

    def test_exactly_one_stage_owner_is_required(self):
        gate = _factor(Site.GATE_UP)
        shared_down = _factor(Site.DOWN, is_shared_outer=True)
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_up_b=gate,
                )
            )
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(gate_up_b=None)
        with self.assertRaises(pydantic.ValidationError):
            _plan(down_a=None)  # required field: down A always runs standalone
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _plan(finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down))

    def test_bridge_layouts_must_match_at_both_sites(self):
        token_gate_up_a = _a(
            Site.GATE_UP,
            LoraAFamily.TOKEN_DEDUP_GROUPED,
            True,
            BridgeLayout.TOKEN_MAJOR,
        )
        with self.assertRaisesRegex(ValueError, "gate/up A output layout"):
            _plan(gate_up_a=token_gate_up_a)

        token_gate_up_b = _b(
            Site.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            BridgeLayout.TOKEN_MAJOR,
        )
        plan = _plan(gate_up_a=token_gate_up_a, gate_up_b=token_gate_up_b)
        self.assertIs(plan.validate(), plan)

    def test_overlaps_reject_consumed_stages(self):
        gate = _factor(Site.GATE_UP)
        shared_down = _factor(Site.DOWN, is_shared_outer=True)
        with self.assertRaisesRegex(ValueError, "gate/up-A\\+B"):
            _plan(
                gate_up_b=None,
                middle=MiddleSpec(
                    MiddleFamily.B_ACTIVATION,
                    ActivationFamily.SWIGLU,
                    consumed_gate_up_b=gate,
                ),
                early_overlap=EarlyOverlap.GATE_UP_A_B,
            )
        with self.assertRaisesRegex(ValueError, "standalone down B"):
            _plan(
                down_b=None,
                finalize=FinalizeSpec(FinalizeFamily.SHARED_RANK_REDUCE, shared_down),
                late_overlap=LateOverlap.DOWN_B,
            )

    def test_indexed_a_is_the_down_a_only_composition(self):
        down_only = _plan(
            down_a=_a(Site.DOWN, LoraAFamily.INDEXED),
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
        # An indexed gate/up-A is coherent (the kernel visits pairs at either
        # site); only its pair-major bridge is pinned.
        indexed_gate_up = _plan(gate_up_a=_a(Site.GATE_UP, LoraAFamily.INDEXED))
        self.assertEqual(
            indexed_gate_up.route_requirements(),
            frozenset((RouteRequirement.RAW, RouteRequirement.ALIGNED_PER_EXPERT)),
        )


class TestRouteRequirementUnion(unittest.TestCase):
    def test_shared_token_a_and_per_expert_b_require_both_products(self):
        plan = _plan(
            gate_up_a=_a(
                Site.GATE_UP,
                LoraAFamily.TOKEN_DEDUP_GROUPED,
                True,
                BridgeLayout.TOKEN_MAJOR,
            ),
            gate_up_b=_b(
                Site.GATE_UP,
                LoraBFamily.ONE_LAUNCH_SLICED,
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

    def test_joint_builder_yields_both_aligned_pair_plans(self):
        shared_down_b = _b(
            Site.DOWN,
            LoraBFamily.ONE_LAUNCH_SLICED,
            True,
        )
        joint = _plan(
            down_b=shared_down_b,
            route_builder=RouteBuilderFamily.JOINT_SHARED_OUTER,
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

    def test_shared_finalize_adds_raw_without_hiding_aligned_routes(self):
        shared_down = _factor(Site.DOWN, is_shared_outer=True)
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
