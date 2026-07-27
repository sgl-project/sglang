# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DAG disaggregation topology and request scheduler."""

import unittest
from pathlib import Path

from sglang.multimodal_gen.runtime.disaggregation.dag import (
    DagSpec,
    ExecutionPlan,
    PlanValidationError,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.predicate import (
    PredicateError,
    evaluate_predicate,
    predicate_names,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.scheduler import (
    CompleteRequest,
    DagRequestScheduler,
    EdgeTransfer,
    FailRequest,
    SourceDispatch,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.state import TransferHandle
from sglang.test.test_utils import CustomTestCase

HD = {"width": 1280, "height": 720}
UHD = {"width": 3840, "height": 2160}


def _linear_spec() -> dict:
    return {
        "source": "encoder",
        "roles": [
            {"name": "encoder", "stages": ["TextEncodingStage"]},
            {"name": "denoiser", "stages": ["DenoisingStage"]},
            {
                "name": "decoder",
                "stages": ["DecodingStage"],
                "terminal": True,
                "emit": ["output"],
            },
        ],
        "pools": [
            {"role": "encoder", "urls": ["tcp://e:1"]},
            {"role": "denoiser", "urls": ["tcp://d:1"]},
            {"role": "decoder", "urls": ["tcp://c:1"]},
        ],
        "routes": [
            {"src": "encoder", "dst": "denoiser"},
            {"src": "denoiser", "dst": "decoder"},
        ],
    }


def _diamond_spec() -> dict:
    """Conditional VSR plus an independently scaled audio branch."""
    big = "width * height > 1920 * 1080"
    return {
        "source": "encoder",
        "roles": [
            {"name": "encoder", "stages": ["TextEncodingStage"]},
            {"name": "denoiser", "stages": ["DenoisingStage"]},
            {"name": "vsr", "stages": ["UpsampleStage"]},
            {
                "name": "vae_video",
                "stages": ["VideoDecodingStage"],
                "terminal": True,
                "emit": ["output"],
            },
            {
                "name": "vae_audio",
                "stages": ["AudioDecodingStage"],
                "terminal": True,
                "emit": ["audio"],
            },
        ],
        "pools": [
            {"role": "encoder", "urls": ["tcp://e:1"]},
            {"role": "denoiser", "urls": ["tcp://d:1"]},
            {"role": "vsr", "urls": ["tcp://s:1"]},
            {"role": "vae_video", "urls": ["tcp://v:1", "tcp://v:2"]},
            {"role": "vae_audio", "urls": ["tcp://a:1"]},
        ],
        "routes": [
            {"src": "encoder", "dst": "denoiser"},
            {"src": "denoiser", "dst": "vsr", "when": big},
            {"src": "denoiser", "dst": "vae_video", "when": f"not ({big})"},
            {"src": "vsr", "dst": "vae_video"},
            {"src": "denoiser", "dst": "vae_audio", "fields": ["audio_latents"]},
        ],
    }


def _handle(**kwargs) -> TransferHandle:
    base = {"manifest": {"latents": [{"offset": 0}]}, "scalar_fields": {}}
    base.update(kwargs)
    return TransferHandle(**base)


class TestPredicate(CustomTestCase):
    def test_arithmetic_comparison(self):
        self.assertTrue(evaluate_predicate("width * height > 1920 * 1080", UHD))
        self.assertFalse(evaluate_predicate("width * height > 1920 * 1080", HD))

    def test_boolean_and_negation(self):
        ctx = {"generate_audio": True, "width": 100}
        self.assertTrue(evaluate_predicate("generate_audio and width > 50", ctx))
        self.assertFalse(evaluate_predicate("not generate_audio", ctx))

    def test_missing_name_is_falsy_not_an_error(self):
        self.assertFalse(evaluate_predicate("nonexistent > 5", {}))
        self.assertTrue(evaluate_predicate("nonexistent == None", {}))

    def test_subscript_into_nested_metadata(self):
        ctx = {"extra": {"mode": "fast"}}
        self.assertTrue(evaluate_predicate("extra['mode'] == 'fast'", ctx))
        self.assertFalse(evaluate_predicate("extra['missing'] == 'fast'", ctx))

    def test_calls_and_attributes_are_rejected(self):
        for expr in (
            "__import__('os').system('true')",
            "width.bit_length()",
            "open('/etc/passwd')",
            "[x for x in range(3)]",
        ):
            with self.assertRaises(PredicateError):
                evaluate_predicate(expr, {"width": 1})

    def test_predicate_names(self):
        self.assertEqual(
            predicate_names("width * height > 1920 and generate_audio"),
            {"width", "height", "generate_audio"},
        )


class TestPlanCompilation(CustomTestCase):
    def test_linear_plan(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_linear_spec()))
        self.assertEqual(plan.node_names, ["encoder", "denoiser", "decoder"])
        self.assertEqual(plan.source, "encoder")
        self.assertEqual(plan.terminals, ["decoder"])
        self.assertTrue(plan.is_sender("encoder"))
        self.assertFalse(plan.is_receiver("encoder"))
        self.assertTrue(plan.is_receiver("decoder"))
        self.assertFalse(plan.is_sender("decoder"))

    def test_scheduler_init_inferred_from_stage_names(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        self.assertTrue(plan.node("denoiser").needs_scheduler_init)
        self.assertFalse(plan.node("encoder").needs_scheduler_init)
        self.assertFalse(plan.node("vae_video").needs_scheduler_init)

    def test_classic_roles_compile_to_linear_plan(self):
        plan = ExecutionPlan.from_classic_roles(
            ["tcp://e:1"], ["tcp://d:1", "tcp://d:2"], ["tcp://c:1"]
        )
        self.assertEqual(plan.node_names, ["encoder", "denoiser", "decoder"])
        self.assertTrue(plan.node("denoiser").needs_scheduler_init)
        self.assertEqual(plan.node("denoiser").num_instances, 2)

    def test_cycle_is_rejected(self):
        spec = _linear_spec()
        spec["routes"].append({"src": "decoder", "dst": "encoder"})
        with self.assertRaises(PlanValidationError) as ctx:
            ExecutionPlan.compile(DagSpec.parse(spec))
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_stage_claimed_twice_is_rejected(self):
        spec = _linear_spec()
        spec["roles"][1]["stages"].append("TextEncodingStage")
        with self.assertRaises(PlanValidationError) as ctx:
            ExecutionPlan.compile(DagSpec.parse(spec))
        self.assertIn("exactly one role", str(ctx.exception))

    def test_dangling_non_terminal_is_rejected(self):
        spec = _linear_spec()
        spec["roles"][2]["terminal"] = False
        with self.assertRaises(PlanValidationError) as ctx:
            ExecutionPlan.compile(DagSpec.parse(spec))
        self.assertIn("terminal", str(ctx.exception))

    def test_terminals_may_not_emit_the_same_field(self):
        spec = _diamond_spec()
        spec["roles"][4]["emit"] = ["output"]
        with self.assertRaises(PlanValidationError) as ctx:
            ExecutionPlan.compile(DagSpec.parse(spec))
        self.assertIn("both emit output field", str(ctx.exception))

    def test_empty_pool_is_rejected(self):
        spec = _linear_spec()
        spec["pools"][1]["urls"] = []
        with self.assertRaises(PlanValidationError) as ctx:
            ExecutionPlan.compile(DagSpec.parse(spec))
        self.assertIn("no pool instances", str(ctx.exception))

    def test_stage_coverage_detects_drift(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_linear_spec()))
        self.assertEqual(
            plan.validate_stage_coverage(
                ["TextEncodingStage", "DenoisingStage", "DecodingStage"]
            ),
            [],
        )
        errors = plan.validate_stage_coverage(
            ["TextEncodingStage", "DenoisingStage", "DecodingStage", "NewStage"]
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("NewStage", errors[0])

    def test_edge_field_filter_restricts_manifest_and_scalars(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        edge = plan.edge("denoiser->vae_audio")
        manifest = {"latents": 1, "audio_latents": 2}
        self.assertEqual(edge.filter_manifest(manifest), {"audio_latents": 2})
        scalars = {"height": 720, "audio_latents": 3, "_trace_state": {}}
        self.assertEqual(
            edge.filter_scalars(scalars), {"audio_latents": 3, "_trace_state": {}}
        )


class TestShippedTopologies(CustomTestCase):
    """The example topologies must stay compilable as stages get renamed."""

    topology_dir = Path(__file__).resolve().parents[2] / "configs" / "disagg_topologies"

    def test_examples_compile(self):
        configs = [
            self.topology_dir / "zimage_linear.yaml",
            self.topology_dir / "zimage_conditional_join.yaml",
        ]
        for path in configs:
            with self.subTest(topology=path.name):
                self.assertTrue(path.is_file(), f"missing topology {path}")
                plan = ExecutionPlan.compile(DagSpec.load(str(path)))
                self.assertTrue(plan.terminals)
                self.assertFalse(plan.is_receiver(plan.source))

    def test_zimage_linear_is_a_three_node_chain(self):
        plan = ExecutionPlan.compile(
            DagSpec.load(str(self.topology_dir / "zimage_linear.yaml"))
        )
        self.assertEqual(list(plan.node_names), ["encoder", "denoiser", "decoder"])
        self.assertEqual(plan.terminals, ["decoder"])
        self.assertEqual(plan.stage_owner("denoising_stage"), "denoiser")
        self.assertEqual(plan.stage_owner("decoding_stage"), "decoder")

    def test_zimage_conditional_join_routes_by_resolution(self):
        plan = ExecutionPlan.compile(
            DagSpec.load(str(self.topology_dir / "zimage_conditional_join.yaml"))
        )
        self.assertEqual(
            [e.edge_id for e in plan.live_out_edges("encoder", HD)],
            ["encoder->relay"],
        )
        self.assertEqual(
            [e.edge_id for e in plan.live_out_edges("encoder", UHD)],
            ["encoder->denoiser"],
        )
        self.assertEqual(plan.node("denoiser").in_degree, 2)
        self.assertEqual(plan.stage_owner("TimestepPreparationStage"), "relay")


class TestDagRequestScheduler(CustomTestCase):
    def _run_linear(self) -> tuple[DagRequestScheduler, list]:
        plan = ExecutionPlan.compile(DagSpec.parse(_linear_spec()))
        sched = DagRequestScheduler(plan)
        actions = sched.submit("r1", b"client", b"payload", dict(HD))
        return sched, actions

    def test_linear_flow_reaches_completion(self):
        sched, actions = self._run_linear()
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], SourceDispatch)
        self.assertEqual(actions[0].node, "encoder")
        self.assertEqual(actions[0].payload, b"payload")

        actions = sched.on_node_staged("r1", "encoder", _handle())
        transfers = [a for a in actions if isinstance(a, EdgeTransfer)]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].edge_id, "encoder->denoiser")
        self.assertEqual(transfers[0].expected_inputs, 1)

        # Encoder holds its slot until the push drains.
        self.assertEqual(sched.stats()["nodes"]["encoder"]["free_slots"], [3])
        sched.on_edge_pushed("r1", "encoder->denoiser")
        self.assertEqual(sched.stats()["nodes"]["encoder"]["free_slots"], [4])

        actions = sched.on_node_staged("r1", "denoiser", _handle())
        self.assertEqual(
            [a.edge_id for a in actions if isinstance(a, EdgeTransfer)],
            ["denoiser->decoder"],
        )
        sched.on_edge_pushed("r1", "denoiser->decoder")

        actions = sched.on_terminal_result("r1", "decoder", {"output": "pixels"})
        done = [a for a in actions if isinstance(a, CompleteRequest)]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].fields, {"output": "pixels"})
        self.assertIsNone(sched.get("r1"))

    def test_conditional_route_skips_vsr_at_low_resolution(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")

        actions = sched.on_node_staged("r1", "denoiser", _handle())
        edges = sorted(a.edge_id for a in actions if isinstance(a, EdgeTransfer))
        self.assertEqual(edges, ["denoiser->vae_audio", "denoiser->vae_video"])

        state = sched.get("r1")
        self.assertEqual(state.node_status["vsr"].value, "skipped")

    def test_conditional_route_engages_vsr_at_high_resolution(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(UHD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")

        actions = sched.on_node_staged("r1", "denoiser", _handle())
        edges = sorted(a.edge_id for a in actions if isinstance(a, EdgeTransfer))
        self.assertEqual(edges, ["denoiser->vae_audio", "denoiser->vsr"])

        # vae_video waits: its denoiser edge was pruned but the vsr edge is live.
        state = sched.get("r1")
        self.assertEqual(state.node_status["vae_video"].value, "pending")

        sched.on_edge_pushed("r1", "denoiser->vsr")
        sched.on_edge_pushed("r1", "denoiser->vae_audio")
        actions = sched.on_node_staged("r1", "vsr", _handle())
        self.assertEqual(
            [a.edge_id for a in actions if isinstance(a, EdgeTransfer)],
            ["vsr->vae_video"],
        )

    def test_predicate_sees_values_produced_mid_flight(self):
        """An upstream stage that changes resolution steers the route."""
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        # The encoder upsamples, so the denoiser's routing sees 4K.
        sched.on_node_staged(
            "r1", "encoder", _handle(scalar_fields={"width": 3840, "height": 2160})
        )
        sched.on_edge_pushed("r1", "encoder->denoiser")
        actions = sched.on_node_staged("r1", "denoiser", _handle())
        edges = sorted(a.edge_id for a in actions if isinstance(a, EdgeTransfer))
        self.assertEqual(edges, ["denoiser->vae_audio", "denoiser->vsr"])

    def test_fanout_holds_producer_slot_until_all_pushes_drain(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")
        sched.on_node_staged("r1", "denoiser", _handle())

        free = lambda: sched.stats()["nodes"]["denoiser"]["free_slots"][0]
        self.assertEqual(free(), 3)
        sched.on_edge_pushed("r1", "denoiser->vae_video")
        self.assertEqual(free(), 3)
        sched.on_edge_pushed("r1", "denoiser->vae_audio")
        self.assertEqual(free(), 4)

    def test_dual_terminal_outputs_are_merged(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")
        sched.on_node_staged("r1", "denoiser", _handle())
        sched.on_edge_pushed("r1", "denoiser->vae_video")
        sched.on_edge_pushed("r1", "denoiser->vae_audio")

        actions = sched.on_terminal_result("r1", "vae_video", {"output": "pixels"})
        self.assertEqual([a for a in actions if isinstance(a, CompleteRequest)], [])

        actions = sched.on_terminal_result("r1", "vae_audio", {"audio": "wav"})
        done = [a for a in actions if isinstance(a, CompleteRequest)]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].fields, {"output": "pixels", "audio": "wav"})

    def test_terminal_only_keeps_the_fields_it_declares(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")
        sched.on_node_staged("r1", "denoiser", _handle())
        sched.on_edge_pushed("r1", "denoiser->vae_video")
        sched.on_edge_pushed("r1", "denoiser->vae_audio")
        sched.on_terminal_result("r1", "vae_video", {"output": "px", "audio": "junk"})
        actions = sched.on_terminal_result("r1", "vae_audio", {"audio": "wav"})
        done = [a for a in actions if isinstance(a, CompleteRequest)][0]
        self.assertEqual(done.fields, {"output": "px", "audio": "wav"})

    def test_failure_releases_every_held_slot(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_diamond_spec()))
        sched = DagRequestScheduler(plan)
        sched.submit("r1", b"c", b"p", dict(HD))
        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")

        actions = sched.on_node_error("r1", "denoiser", "CUDA OOM")
        failures = [a for a in actions if isinstance(a, FailRequest)]
        self.assertEqual(len(failures), 1)
        self.assertIn("CUDA OOM", failures[0].error)
        self.assertIsNone(sched.get("r1"))

        for node, stats in sched.stats()["nodes"].items():
            capacity = plan.node(node).capacity
            self.assertEqual(
                stats["free_slots"], [capacity] * plan.node(node).num_instances, node
            )

    def test_capacity_backpressure_queues_then_releases(self):
        spec = _linear_spec()
        spec["pools"][0]["capacity"] = 1
        plan = ExecutionPlan.compile(DagSpec.parse(spec))
        sched = DagRequestScheduler(plan, max_inflight=8)

        first = sched.submit("r1", b"c", b"p1", dict(HD))
        second = sched.submit("r2", b"c", b"p2", dict(HD))
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

        sched.on_node_staged("r1", "encoder", _handle())
        actions = sched.on_edge_pushed("r1", "encoder->denoiser")
        dispatched = [a for a in actions if isinstance(a, SourceDispatch)]
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].request_id, "r2")

    def test_admission_control_caps_inflight_requests(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_linear_spec()))
        sched = DagRequestScheduler(plan, max_inflight=1)
        sched.submit("r1", b"c", b"p1", dict(HD))
        sched.submit("r2", b"c", b"p2", dict(HD))
        self.assertEqual(sched.stats()["inflight"], 1)
        self.assertEqual(sched.stats()["admission_queue"], 1)

        sched.on_node_staged("r1", "encoder", _handle())
        sched.on_edge_pushed("r1", "encoder->denoiser")
        sched.on_node_staged("r1", "denoiser", _handle())
        sched.on_edge_pushed("r1", "denoiser->decoder")
        actions = sched.on_terminal_result("r1", "decoder", {"output": "px"})

        self.assertEqual(
            [a.request_id for a in actions if isinstance(a, SourceDispatch)], ["r2"]
        )

    def test_timeout_fails_the_request(self):
        plan = ExecutionPlan.compile(DagSpec.parse(_linear_spec()))
        sched = DagRequestScheduler(plan, timeout_s=0.0)
        sched.submit("r1", b"c", b"p", dict(HD))
        actions = sched.check_timeouts()
        failures = [a for a in actions if isinstance(a, FailRequest)]
        self.assertEqual(len(failures), 1)
        self.assertIn("timed out", failures[0].error)


if __name__ == "__main__":
    unittest.main()
