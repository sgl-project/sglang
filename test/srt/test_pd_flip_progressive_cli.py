import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_controller.py"
)
RUN_CONTROLLER_PATH = SCRIPT_PATH.with_name("pd_flip_docker") / "run_controller.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_progressive_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeSnapshot:
    def to_dict(self):
        return {"prefill_slo_attainment": 0.5, "decode_slo_attainment": 1.0}


class FakeMonitor:
    def collect_cluster(self, nodes):
        list(nodes)
        return FakeSnapshot()


class ProgressiveCLITest(unittest.TestCase):
    def test_atomic_rid_match_accepts_reordering_but_rejects_duplicates(self):
        module = load_module()

        self.assertTrue(module._same_atomic_rids(("r0", "r1"), ("r1", "r0")))
        self.assertFalse(module._same_atomic_rids(("r0", "r1"), ("r0", "r0")))
        self.assertFalse(module._same_atomic_rids(("r0", "r1"), ("r0",)))

    def test_docker_wrapper_forwards_progressive_trace_and_fixed_pair(self):
        source = RUN_CONTROLLER_PATH.read_text(encoding="utf-8")

        self.assertIn("monitor-progressive)", source)
        self.assertIn("PD_FLIP_TRACE_SLO_LEDGER", source)
        self.assertIn('MIGRATION_TARGET_NAME', source)
        self.assertIn('--trace-slo-ledger', source)
        self.assertIn('--migration-target-name', source)

    def test_parser_accepts_progressive_trace_and_fixed_pair(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--router-url",
                "http://router",
                "--node",
                "name=node0,worker_url=http://node0,router_worker_id=node0",
                "monitor-progressive",
                "--trace-slo-ledger",
                "/raw/ledger.jsonl",
                "--source-name",
                "node2",
                "--migration-target-name",
                "node3",
                "--iterations",
                "120",
                "--poll-interval",
                "0.25",
            ]
        )

        self.assertEqual(args.command, "monitor-progressive")
        self.assertEqual(args.trace_slo_ledger, "/raw/ledger.jsonl")
        self.assertEqual(args.source_name, "node2")
        self.assertEqual(args.migration_target_name, "node3")
        self.assertEqual(args.iterations, 120)
        self.assertEqual(args.poll_interval, 0.25)

    def test_api_key_can_be_resolved_from_environment_without_argv_secret(self):
        module = load_module()
        args = module.build_arg_parser().parse_args(
            [
                "--router-url",
                "http://router",
                "--node",
                "name=node0,worker_url=http://node0,router_worker_id=node0",
                "--api-key-env",
                "PD_FLIP_SECRET",
                "metrics",
            ]
        )
        previous = os.environ.get("PD_FLIP_SECRET")
        os.environ["PD_FLIP_SECRET"] = "secret-from-env"
        try:
            self.assertEqual(module.resolve_api_key(args), "secret-from-env")
        finally:
            if previous is None:
                os.environ.pop("PD_FLIP_SECRET", None)
            else:
                os.environ["PD_FLIP_SECRET"] = previous

    def test_progressive_monitor_honors_fixed_pair_over_load_sorting(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            config = module.PDClusterConfig(
                router_url="http://router",
                nodes=[module.PDNode("node0", "http://node0", "node0")],
                session_journal_path=str(Path(directory) / "journal.json"),
            )
            controller = module.PDFlipController(config, client=object())
            controller.collect_metrics = lambda: [
                module.NodeMetrics(
                    "node0", "http://node0", "node0", worker_role="prefill"
                ),
                module.NodeMetrics(
                    "node1",
                    "http://node1",
                    "node1",
                    worker_role="decode",
                    running_reqs=99,
                ),
                module.NodeMetrics(
                    "node2",
                    "http://node2",
                    "node2",
                    worker_role="decode",
                    running_reqs=1,
                ),
                module.NodeMetrics(
                    "node3", "http://node3", "node3", worker_role="decode"
                ),
            ]
            controller._evaluate_progressive_snapshot = (
                lambda snapshot, observing: module.ProgressiveDecision.START
            )
            controller._progressive_observability_fields = (
                lambda snapshot, selection: {}
            )
            controller._execute_progressive_d_to_p = (
                lambda **kwargs: (kwargs["source"].name, kwargs["target"].name)
            )

            selected = controller.monitor_progressive(
                FakeMonitor(),
                iterations=1,
                poll_interval_seconds=0,
                source_name="node2",
                migration_target_name="node3",
            )

        self.assertEqual(selected, ("node2", "node3"))


if __name__ == "__main__":
    unittest.main()
