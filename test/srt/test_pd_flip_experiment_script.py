import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_experiment.py"
)
DOCKER_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_docker"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("pd_flip_experiment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, server_infos):
        self.server_infos = list(server_infos)
        self.posts = []

    def get_json(self, url, path):
        if path == "/pd_flip/migration/status":
            return [
                {
                    "success": True,
                    "status": {
                        "state": "source_transferred",
                        "pending_reqs": 0,
                        "transferred_reqs": 1,
                        "failed_reqs": 0,
                    },
                    "manifests": [{"rid": "rid-1"}],
                }
            ]
        self.assert_path(path, "/server_info")
        if not self.server_infos:
            return {
                "internal_states": [
                    {
                        "pd_flip": {
                            "enabled": True,
                            "state": "safe",
                            "direction": "none",
                            "current_role": "decode",
                        }
                    }
                ]
            }
        return self.server_infos.pop(0)

    def post_json(self, url, path, payload):
        self.posts.append((url, payload))
        if path == "/pd_flip/migration/source/start":
            return [
                {
                    "success": True,
                    "status": {
                        "state": "source_started",
                        "pending_reqs": 1,
                    },
                    "manifests": [{"rid": "rid-1"}],
                }
            ]
        if path == "/pd_flip/migration/target/prepare":
            return [
                {
                    "success": True,
                    "status": {
                        "state": "target_prepared",
                        "transferred_reqs": 1,
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
                        "released_reqs": 1,
                    },
                }
            ]
        self.assert_path(path, "/set_internal_state")
        return {"updated": True, "server_args": payload["server_args"]}

    def assert_path(self, actual, expected):
        if actual != expected:
            raise AssertionError(f"expected path {expected}, got {actual}")


class TestPDFlipExperimentScript(unittest.TestCase):
    def setUp(self):
        self.script = load_script_module()

    def test_extracts_pd_flip_from_internal_state(self):
        server_info = {
            "internal_states": [
                {"other": "state"},
                {
                    "pd_flip": {
                        "enabled": True,
                        "state": "preparing",
                        "direction": "d_to_p",
                    }
                },
            ]
        }

        pd_flip = self.script.extract_pd_flip(server_info)

        self.assertEqual(pd_flip["state"], "preparing")
        self.assertEqual(pd_flip["direction"], "d_to_p")

    def test_run_once_acks_prepare_and_commit_after_idle(self):
        worker_url = "http://127.0.0.1:30000"
        server_infos = [
            self._server_info("preparing", idle=False),
            self._server_info("preparing", idle=True),
            self._server_info("flipping", idle=True),
            self._server_info("safe", idle=True, direction="none"),
        ]
        client = FakeClient(server_infos)

        result = self.script.run_once(
            client=client,
            worker_urls=[worker_url],
            timeout_seconds=5.0,
            poll_interval_seconds=0.0,
            restart_command=None,
            sleep_fn=lambda _: None,
            log_fn=lambda _: None,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            client.posts,
            [
                (worker_url, {"server_args": {"pd_flip_prepare_ack": True}}),
                (worker_url, {"server_args": {"pd_flip_commit_ack": True}}),
            ],
        )

    def test_trigger_sets_direction_specific_slo_inputs(self):
        client = FakeClient([])

        self.script.trigger_flip(
            client=client,
            worker_url="http://127.0.0.1:30000",
            direction="d_to_p",
            prefill_nodes=1,
            decode_nodes=2,
            threshold=0.9,
            window_seconds=0.0,
        )

        self.assertEqual(
            client.posts[0][1],
            {
                "server_args": {
                    "pd_flip_prefill_nodes": 1,
                    "pd_flip_decode_nodes": 2,
                    "pd_flip_slo_threshold": 0.9,
                    "pd_flip_window_seconds": 0.0,
                    "pd_flip_prefill_slo_attainment": 0.0,
                    "pd_flip_decode_slo_attainment": 1.0,
                }
            },
        )

    def test_run_once_with_restart_waits_for_target_role(self):
        worker_url = "http://127.0.0.1:30000"
        server_infos = [
            self._server_info("preparing", idle=True),
            self._server_info("flipping", idle=True),
            self._server_info("safe", idle=True, direction="none", current_role="prefill"),
        ]
        client = FakeClient(server_infos)
        commands = []
        old_runner = self.script.run_restart_command
        self.script.run_restart_command = lambda command, log_fn: commands.append(command)
        try:
            result = self.script.run_once(
                client=client,
                worker_urls=[worker_url],
                timeout_seconds=5.0,
                poll_interval_seconds=0.0,
                restart_command="docker compose restart decode0",
                sleep_fn=lambda _: None,
                log_fn=lambda _: None,
            )
        finally:
            self.script.run_restart_command = old_runner

        self.assertEqual(result["status"], "completed_by_restart")
        self.assertEqual(commands, ["docker compose restart decode0"])
        self.assertEqual(
            client.posts,
            [(worker_url, {"server_args": {"pd_flip_prepare_ack": True}})],
        )

    def test_run_once_drives_migration_before_prepare_ack(self):
        source_url = "http://127.0.0.1:30000"
        target_url = "http://127.0.0.1:30001"
        server_infos = [
            self._server_info("preparing", idle=True),
            self._server_info("flipping", idle=True),
            self._server_info("safe", idle=True, direction="none"),
        ]
        client = FakeClient(server_infos)

        result = self.script.run_once(
            client=client,
            worker_urls=[source_url],
            timeout_seconds=5.0,
            poll_interval_seconds=0.0,
            restart_command=None,
            migration_target_url=target_url,
            sleep_fn=lambda _: None,
            log_fn=lambda _: None,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            client.posts,
            [
                (
                    source_url,
                    {
                        "session_id": "pd-flip-migration",
                        "target_url": target_url,
                    },
                ),
                (
                    target_url,
                    {
                        "session_id": "pd-flip-migration",
                        "source_url": source_url,
                        "manifests": [{"rid": "rid-1"}],
                    },
                ),
                (
                    source_url,
                    {
                        "session_id": "pd-flip-migration",
                        "released_rids": ["rid-1"],
                    },
                ),
                (source_url, {"server_args": {"pd_flip_prepare_ack": True}}),
                (source_url, {"server_args": {"pd_flip_commit_ack": True}}),
            ],
        )

    def test_docker_env_default_uses_eight_gpus_per_worker(self):
        values = self._read_docker_env()

        self.assertEqual(int(values["TP_SIZE"]) * int(values["DP_SIZE"]), 8)

    def test_docker_env_declares_four_cloud_hosts_and_monitor_thresholds(self):
        values = self._read_docker_env()

        self.assertEqual(values["NODE0_HOST"], "cloud-099")
        self.assertEqual(values["NODE1_HOST"], "cloud-100")
        self.assertEqual(values["NODE2_HOST"], "cloud-101")
        self.assertEqual(values["NODE3_HOST"], "cloud-102")
        self.assertEqual(values["NODE0_ROLE"], "prefill")
        self.assertEqual(values["NODE1_ROLE"], "prefill")
        self.assertEqual(values["NODE2_ROLE"], "decode")
        self.assertEqual(values["NODE3_ROLE"], "decode")
        self.assertIn("TTFT_SLO_SECONDS", values)
        self.assertIn("TPOT_SLO_SECONDS", values)
        self.assertIn("PD_FLIP_ENTER_THRESHOLD", values)
        self.assertIn("PD_FLIP_EXIT_THRESHOLD", values)
        self.assertIn("PD_FLIP_COMMIT_THRESHOLD", values)

    def test_docker_controller_script_supports_execute_action(self):
        script = (DOCKER_DIR / "run_controller.sh").read_text()

        self.assertIn("execute)", script)
        self.assertIn("execute --direction", script)

    def test_docker_harness_declares_monitor_runner(self):
        controller_script = (DOCKER_DIR / "run_controller.sh").read_text()
        monitor_path = DOCKER_DIR / "run_monitor.sh"
        monitor_script = monitor_path.read_text()
        windows_runner = (DOCKER_DIR / "windows_four_node.ps1").read_text()

        self.assertIn("monitor)", controller_script)
        self.assertIn("run_monitor.sh", controller_script)
        self.assertTrue(monitor_path.stat().st_mode & 0o111)
        self.assertIn("pd_flip_controller.py", monitor_script)
        self.assertIn("monitor", monitor_script)
        self.assertIn("--ttft-slo", monitor_script)
        self.assertIn("--tpot-slo", monitor_script)
        self.assertIn("--commit-threshold", monitor_script)
        self.assertIn("cloud-099", windows_runner)
        self.assertIn("cloud-102", windows_runner)
        self.assertIn("start-monitor", windows_runner)
        self.assertIn("write-env", windows_runner)
        self.assertIn("sync-env", windows_runner)
        self.assertIn("sync-code", windows_runner)
        self.assertIn('"clone"', windows_runner)
        self.assertIn('"/root/sglang"', windows_runner)
        self.assertIn("sglang-pd-switch:tianciJ", windows_runner)
        self.assertIn("/models/deepseek_v3.1_terminus", windows_runner)
        self.assertIn("nohup bash -lc", windows_runner)
        self.assertIn(".pid", windows_runner)

    def _read_docker_env(self):
        values = {}
        for line in (DOCKER_DIR / "env.example").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value
        return values

    def _server_info(self, state, idle, direction="d_to_p", current_role="decode"):
        return {
            "internal_states": [
                {
                    "pd_flip": {
                        "enabled": True,
                        "state": state,
                        "direction": direction,
                        "current_role": current_role,
                        "target_role": "prefill" if direction == "d_to_p" else None,
                        "is_idle_for_flip": idle,
                        "admission_paused": state in ("preparing", "flipping"),
                    }
                }
            ]
        }


if __name__ == "__main__":
    unittest.main()
