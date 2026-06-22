import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_controller.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("pd_flip_controller", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self):
        self.gets = []
        self.posts = []

    def get_json(self, base_url, path):
        self.gets.append((base_url, path))
        if base_url == "http://router" and path == "/pd_flip/router/workers":
            return {
                "workers": [
                    {
                        "worker_id": "node-a",
                        "url": "http://node-a:30000",
                        "role": "decode",
                        "draining": False,
                        "active_load": 3,
                        "bootstrap_port": None,
                    },
                    {
                        "worker_id": "node-b",
                        "url": "http://node-b:30000",
                        "role": "decode",
                        "draining": False,
                        "active_load": 1,
                        "bootstrap_port": None,
                    },
                    {
                        "worker_id": "node-c",
                        "url": "http://node-c:30000",
                        "role": "prefill",
                        "draining": False,
                        "active_load": 0,
                        "bootstrap_port": 8997,
                    },
                ]
            }
        if path == "/pd_flip/runtime_role/status":
            role = "prefill" if "node-c" in base_url else "decode"
            return [
                {
                    "success": True,
                    "role": role,
                    "status": {
                        "role": role,
                        "is_idle": "node-c" in base_url,
                        "admission_paused": False,
                    },
                }
            ]
        if path == "/v1/loads?include=all":
            running = 3 if "node-a" in base_url else 1 if "node-b" in base_url else 0
            return {
                "loads": [
                    {
                        "dp_rank": 0,
                        "num_running_reqs": running,
                        "num_waiting_reqs": 2,
                        "num_total_tokens": 1000 + running,
                        "token_usage": 0.4 + running / 10,
                    }
                ]
            }
        raise AssertionError(f"unexpected GET {base_url} {path}")

    def post_json(self, base_url, path, payload):
        self.posts.append((base_url, path, payload))
        return {"success": True}


class TestPDFlipController(unittest.TestCase):
    def setUp(self):
        self.script = load_script_module()
        self.client = FakeClient()
        self.config = self.script.PDClusterConfig(
            router_url="http://router",
            nodes=[
                self.script.PDNode(
                    name="node-a",
                    worker_url="http://node-a:30000",
                    router_worker_id="node-a",
                    bootstrap_port=8997,
                ),
                self.script.PDNode(
                    name="node-b",
                    worker_url="http://node-b:30000",
                    router_worker_id="node-b",
                    bootstrap_port=8997,
                ),
                self.script.PDNode(
                    name="node-c",
                    worker_url="http://node-c:30000",
                    router_worker_id="node-c",
                    bootstrap_port=8997,
                ),
            ],
        )

    def test_collect_metrics_merges_router_role_status_and_loads(self):
        controller = self.script.PDFlipController(self.config, self.client)

        metrics = controller.collect_metrics()

        by_name = {metric.name: metric for metric in metrics}
        self.assertEqual(by_name["node-a"].router_role, "decode")
        self.assertEqual(by_name["node-a"].worker_role, "decode")
        self.assertEqual(by_name["node-a"].router_active_load, 3)
        self.assertEqual(by_name["node-a"].running_reqs, 3)
        self.assertEqual(by_name["node-a"].waiting_reqs, 2)
        self.assertEqual(by_name["node-c"].worker_role, "prefill")
        self.assertTrue(by_name["node-c"].is_idle)

    def test_d_to_p_dry_run_builds_safe_action_order_without_posts(self):
        controller = self.script.PDFlipController(self.config, self.client)

        plan = controller.dry_run(direction="d_to_p", source_name="node-a")

        self.assertTrue(plan.dry_run)
        self.assertEqual(plan.source, "node-a")
        self.assertEqual(plan.target_role, "prefill")
        self.assertEqual(plan.migration_target, "node-b")
        self.assertEqual([action.step for action in plan.actions], [
            "router_drain_source",
            "pause_source_admission",
            "start_decode_migration_source",
            "prepare_decode_migration_target",
            "wait_decode_migration",
            "finish_decode_migration_source",
            "set_source_runtime_role",
            "refresh_router_source_role",
            "resume_source_admission",
            "router_undrain_source",
        ])
        self.assertEqual(
            plan.actions[0].payload,
            {"worker_id": "node-a", "draining": True},
        )
        self.assertEqual(
            plan.actions[2].payload,
            {
                "session_id": "pd-flip-node-a-to-node-b",
                "target_url": "http://node-b:30000",
            },
        )
        self.assertEqual(
            plan.actions[7].payload,
            {
                "worker_id": "node-a",
                "role": "prefill",
                "bootstrap_port": 8997,
                "draining": False,
            },
        )
        self.assertEqual(self.client.posts, [])

    def test_d_to_p_dry_run_auto_selects_highest_load_decode_source(self):
        controller = self.script.PDFlipController(self.config, self.client)

        plan = controller.dry_run(direction="d_to_p")

        self.assertEqual(plan.source, "node-a")
        self.assertEqual(plan.migration_target, "node-b")

    def test_p_to_d_dry_run_has_no_active_decode_migration(self):
        controller = self.script.PDFlipController(self.config, self.client)

        plan = controller.dry_run(direction="p_to_d", source_name="node-c")

        self.assertEqual(plan.source, "node-c")
        self.assertEqual(plan.target_role, "decode")
        self.assertIsNone(plan.migration_target)
        self.assertNotIn(
            "start_decode_migration_source",
            [action.step for action in plan.actions],
        )
        self.assertEqual(
            plan.actions[4].payload,
            {
                "worker_id": "node-c",
                "role": "decode",
                "bootstrap_port": None,
                "draining": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
