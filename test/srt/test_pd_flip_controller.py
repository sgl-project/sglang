import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
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


class ExecutingFakeClient(FakeClient):
    def __init__(self, *, fail_target_prepare=False):
        super().__init__()
        self.fail_target_prepare = fail_target_prepare
        self.migration_status_gets = {}
        self.runtime_status_gets = {}

    def get_json(self, base_url, path):
        if path == "/pd_flip/migration/status":
            count = self.migration_status_gets.get(base_url, 0)
            self.migration_status_gets[base_url] = count + 1
            state = "source_transferred" if "node-a" in base_url else "target_transferred"
            return [
                {
                    "success": True,
                    "status": {
                        "state": state,
                        "pending_reqs": 0,
                        "transferred_reqs": 1,
                        "failed_reqs": 0,
                    },
                    "manifests": [{"rid": "rid-1"}],
                }
            ]

        if path == "/pd_flip/runtime_role/status":
            self.gets.append((base_url, path))
            count = self.runtime_status_gets.get(base_url, 0)
            self.runtime_status_gets[base_url] = count + 1
            role = "prefill" if "node-c" in base_url else "decode"
            is_idle = "node-c" in base_url or count > 0
            return [
                {
                    "success": True,
                    "role": role,
                    "status": {
                        "role": role,
                        "is_idle": is_idle,
                        "admission_paused": False,
                    },
                }
            ]

        return super().get_json(base_url, path)

    def post_json(self, base_url, path, payload):
        self.posts.append((base_url, path, payload))
        if path == "/pd_flip/migration/source/start":
            return [
                {
                    "success": True,
                    "status": {
                        "state": "source_started",
                        "pending_reqs": 1,
                        "failed_reqs": 0,
                    },
                    "manifests": [
                        {
                            "rid": "rid-1",
                            "origin_input_ids": [1, 2],
                            "output_ids": [3],
                            "kv_committed_len": 2,
                        }
                    ],
                }
            ]
        if path == "/pd_flip/migration/target/prepare":
            if self.fail_target_prepare:
                raise RuntimeError("target prepare failed")
            return [
                {
                    "success": True,
                    "status": {
                        "state": "target_prepared",
                        "pending_reqs": 1,
                        "failed_reqs": 0,
                    },
                    "manifests": payload["manifests"],
                }
            ]
        if path == "/pd_flip/migration/source/finish":
            return [
                {
                    "success": True,
                    "status": {
                        "state": "source_released",
                        "pending_reqs": 0,
                        "released_reqs": len(payload.get("released_rids") or []),
                        "failed_reqs": 0,
                    },
                    "manifests": [{"rid": "rid-1"}],
                }
            ]
        return {"success": True, "status": {"role": payload.get("role")}}


class MonitorFakeClient(FakeClient):
    def get_text(self, base_url, path):
        if path != "/metrics":
            raise AssertionError(f"unexpected GET text {base_url} {path}")
        return """
sglang:time_to_first_token_seconds_bucket{le="0.2"} 10
sglang:time_to_first_token_seconds_bucket{le="+Inf"} 10
sglang:inter_token_latency_seconds_bucket{le="0.02"} 20
sglang:inter_token_latency_seconds_bucket{le="+Inf"} 20
"""


class SequenceSLOMonitor:
    def __init__(self, script, prefill_attainments):
        self.script = script
        self.prefill_attainments = list(prefill_attainments)
        self.collects = []

    def collect_cluster(self, nodes):
        self.collects.append(list(nodes))
        idx = min(len(self.collects) - 1, len(self.prefill_attainments) - 1)
        return self.script.ClusterSLOSnapshot(
            timestamp=float(len(self.collects)),
            prefill_nodes=1,
            decode_nodes=2,
            prefill_slo_attainment=self.prefill_attainments[idx],
            decode_slo_attainment=1.0,
            nodes=[],
        )


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
            "wait_decode_migration_source",
            "wait_decode_migration_target",
            "finish_decode_migration_source",
            "wait_source_idle",
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
            plan.actions[9].payload,
            {
                "worker_id": "node-a",
                "role": "prefill",
                "bootstrap_port": 8997,
                "draining": False,
            },
        )
        self.assertEqual(plan.actions[3].payload["adopt_on_success"], True)
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

    def test_d_to_p_execute_runs_migration_and_role_switch(self):
        client = ExecutingFakeClient()
        controller = self.script.PDFlipController(self.config, client)

        result = controller.execute(direction="d_to_p", source_name="node-a")

        self.assertTrue(result.success)
        self.assertEqual(result.source, "node-a")
        self.assertEqual(result.migration_target, "node-b")
        self.assertEqual(result.target_role, "prefill")
        self.assertEqual(
            [record.step for record in result.actions],
            [
                "router_drain_source",
                "pause_source_admission",
                "start_decode_migration_source",
                "prepare_decode_migration_target",
                "wait_decode_migration_source",
                "wait_decode_migration_target",
                "finish_decode_migration_source",
                "wait_source_idle",
                "set_source_runtime_role",
                "refresh_router_source_role",
                "resume_source_admission",
                "router_undrain_source",
            ],
        )
        target_prepare = [
            post for post in client.posts if post[1] == "/pd_flip/migration/target/prepare"
        ][0]
        self.assertEqual(target_prepare[0], "http://node-b:30000")
        self.assertEqual(target_prepare[2]["manifests"][0]["rid"], "rid-1")
        finish_source = [
            post for post in client.posts if post[1] == "/pd_flip/migration/source/finish"
        ][0]
        self.assertEqual(finish_source[2]["released_rids"], ["rid-1"])
        self.assertGreaterEqual(result.total_seconds, 0.0)
        self.assertGreaterEqual(result.migration_seconds, 0.0)

    def test_p_to_d_execute_waits_idle_and_switches_role(self):
        client = ExecutingFakeClient()
        controller = self.script.PDFlipController(self.config, client)

        result = controller.execute(direction="p_to_d", source_name="node-c")

        self.assertTrue(result.success)
        self.assertEqual(result.source, "node-c")
        self.assertIsNone(result.migration_target)
        self.assertEqual(result.target_role, "decode")
        self.assertEqual(
            [record.step for record in result.actions],
            [
                "router_drain_source",
                "pause_source_admission",
                "wait_source_idle",
                "set_source_runtime_role",
                "refresh_router_source_role",
                "resume_source_admission",
                "router_undrain_source",
            ],
        )
        self.assertIn(
            (
                "http://node-c:30000",
                "/pd_flip/runtime_role/set",
                {"role": "decode", "force": False},
            ),
            client.posts,
        )

    def test_execute_failure_resumes_admission_and_undrains_source(self):
        client = ExecutingFakeClient(fail_target_prepare=True)
        controller = self.script.PDFlipController(self.config, client)

        result = controller.execute(direction="d_to_p", source_name="node-a")

        self.assertFalse(result.success)
        self.assertIn("target prepare failed", result.message)
        self.assertEqual(result.source, "node-a")
        self.assertEqual(
            client.posts[-2:],
            [
                (
                    "http://node-a:30000",
                    "/pd_flip/runtime_role/admission",
                    {"paused": False},
                ),
                (
                    "http://router",
                    "/pd_flip/router/worker/drain",
                    {"worker_id": "node-a", "draining": False},
                ),
            ],
        )

    def test_main_execute_returns_nonzero_when_execution_fails(self):
        client = ExecutingFakeClient(fail_target_prepare=True)
        self.script.HttpClient = lambda api_key=None, timeout_seconds=10.0: client

        with redirect_stdout(io.StringIO()):
            rc = self.script.main(
                [
                    "--router-url",
                    "http://router",
                    "--node",
                    "name=node-a,worker_url=http://node-a:30000,router_worker_id=node-a,bootstrap_port=8997",
                    "--node",
                    "name=node-b,worker_url=http://node-b:30000,router_worker_id=node-b,bootstrap_port=8997",
                    "--node",
                    "name=node-c,worker_url=http://node-c:30000,router_worker_id=node-c,bootstrap_port=8997",
                    "execute",
                    "--direction",
                    "d_to_p",
                    "--source-name",
                    "node-a",
                ]
            )

        self.assertEqual(rc, 1)

    def test_main_monitor_returns_no_flip_when_slo_is_safe(self):
        client = MonitorFakeClient()
        self.script.HttpClient = lambda api_key=None, timeout_seconds=10.0: client

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = self.script.main(
                [
                    "--router-url",
                    "http://router",
                    "--node",
                    "name=node-a,worker_url=http://node-a:30000,router_worker_id=node-a,bootstrap_port=8997",
                    "--node",
                    "name=node-b,worker_url=http://node-b:30000,router_worker_id=node-b,bootstrap_port=8997",
                    "--node",
                    "name=node-c,worker_url=http://node-c:30000,router_worker_id=node-c,bootstrap_port=8997",
                    "monitor",
                    "--ttft-slo",
                    "0.2",
                    "--tpot-slo",
                    "0.02",
                    "--iterations",
                    "1",
                    "--poll-interval",
                    "0",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertIn("no flip decision", stdout.getvalue())
        self.assertEqual(client.posts, [])

    def test_monitor_aborts_two_phase_migration_when_slo_recovers(self):
        client = ExecutingFakeClient()
        controller = self.script.PDFlipController(self.config, client)
        slo_monitor = SequenceSLOMonitor(self.script, [0.80, 0.96])

        result = controller.monitor(
            slo_monitor=slo_monitor,
            enter_threshold=0.90,
            exit_threshold=0.95,
            commit_threshold=0.90,
            iterations=1,
            poll_interval_seconds=0.0,
        )

        self.assertTrue(result.success)
        self.assertIn("SLO recovered", result.message)
        paths = [path for _, path, _ in client.posts]
        self.assertIn("/pd_flip/migration/target/abort", paths)
        self.assertIn("/pd_flip/migration/abort", paths)
        self.assertNotIn("/pd_flip/migration/target/commit", paths)
        self.assertNotIn("/pd_flip/runtime_role/set", paths)
        self.assertEqual(result.actions[0].target_role, "decode")

    def test_monitor_commits_two_phase_migration_when_slo_remains_risky(self):
        client = ExecutingFakeClient()
        controller = self.script.PDFlipController(self.config, client)
        slo_monitor = SequenceSLOMonitor(self.script, [0.80, 0.80, 0.80])

        result = controller.monitor(
            slo_monitor=slo_monitor,
            enter_threshold=0.90,
            exit_threshold=0.95,
            commit_threshold=0.90,
            iterations=1,
            poll_interval_seconds=0.0,
        )

        self.assertTrue(result.success)
        self.assertIn("committed", result.message)
        self.assertEqual(result.actions[0].target_role, "prefill")
        target_prepare = [
            post for post in client.posts if post[1] == "/pd_flip/migration/target/prepare"
        ][0]
        self.assertIs(target_prepare[2]["prepare_only"], True)
        self.assertIs(target_prepare[2]["adopt_on_commit"], True)
        paths = [path for _, path, _ in client.posts]
        self.assertIn("/pd_flip/migration/target/commit", paths)
        self.assertIn("/pd_flip/migration/source/finish", paths)
        self.assertIn("/pd_flip/runtime_role/set", paths)


if __name__ == "__main__":
    unittest.main()
