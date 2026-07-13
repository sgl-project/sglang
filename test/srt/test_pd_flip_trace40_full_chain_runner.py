import os
import json
import subprocess
import shlex
import tempfile
import unittest
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "pd_flip_trace40_full_chain.sh"
)
ENV_EXAMPLE = RUNNER.with_suffix(".env.example")
PREPARE_TRACE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_prepare_trace.py"
)


def _bash_path(path):
    path = Path(path).resolve()
    if os.name != "nt":
        return str(path)
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def _run_dry_preflight(env_file):
    command = " ".join(
        [
            f"ENV_FILE={shlex.quote(_bash_path(env_file))}",
            "DRY_RUN=1",
            "RUN_ID=test-run",
            shlex.quote(_bash_path(RUNNER)),
            "preflight",
        ]
    )
    return subprocess.run(
        ["bash", "-lc", command], text=True, capture_output=True
    )


class Trace40FullChainRunnerTest(unittest.TestCase):
    def test_prepare_trace_creates_four_valid_interleaved_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "scheduled.jsonl"
            manifest = Path(directory) / "schedule.json"
            rows = [
                {
                    "request_id": f"req-{index}",
                    "prompt_kind": "long" if index % 2 == 0 else "short",
                    "prompt_chars": 10000 if index % 2 == 0 else 1000,
                    "ttft_slo_s": 2.0,
                    "tpot_slo_s": 0.1,
                }
                for index in range(40)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(PREPARE_TRACE),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--wave-size",
                    "10",
                    "--wave-gap-seconds",
                    "6",
                    "--intra-wave-interval-seconds",
                    "0.15",
                    "--ttft-slo-override-seconds",
                    "0.2",
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            scheduled = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(scheduled), 40)
            self.assertEqual([row["request_id"] for row in scheduled], [row["request_id"] for row in rows])
            self.assertEqual(scheduled[0]["arrival_offset_s"], 0.0)
            self.assertEqual(scheduled[10]["arrival_offset_s"], 6.0)
            self.assertEqual(scheduled[20]["arrival_offset_s"], 12.0)
            self.assertEqual(scheduled[30]["arrival_offset_s"], 18.0)
            self.assertAlmostEqual(scheduled[-1]["arrival_offset_s"], 19.35)
            self.assertTrue(all(row["ttft_slo_s"] == 0.2 for row in scheduled))
            self.assertEqual(json.loads(manifest.read_text())["request_count"], 40)

    def test_env_example_points_at_current_cluster_layout(self):
        source = ENV_EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("ADMIN_API_KEY=replace-with", source)
        self.assertIn("IMAGE=sglang-pd-switch:tianciJ", source)
        self.assertIn("SGLANG_REPO=/home/tiancij/sglang-pd-e9c4472c3", source)
        self.assertIn("trace_interleaved_long_decode.jsonl", source)
        self.assertIn("PD_FLIP_FIRST_MIGRATION_RATIO=0.5", source)
        self.assertIn("PD_FLIP_OBSERVATION_SECONDS=10", source)
        self.assertIn("TRACE_WAVE_SIZE=10", source)
        self.assertIn("TRACE_WAVE_GAP_SECONDS=6", source)

    def test_runner_declares_full_timeline_contract(self):
        source = RUNNER.read_text(encoding="utf-8")

        for value in (
            "192.168.0.42",
            "192.168.0.40",
            "192.168.0.39",
            "192.168.0.41",
            "monitor-progressive",
            "--trace-slo-ledger",
            "--source-name '${SOURCE_NAME}'",
            "--migration-target-name '${MIGRATION_TARGET_NAME}'",
            "--first-migration-ratio",
            "--observation-seconds",
            "--interval-seconds 0.05",
            "chronyc tracking",
            "target_hicache_restore",
            "fallback",
            "b64decode",
            "git-unavailable",
            "prepare-trace-in-container",
            "set -Eeuo pipefail",
            "--output-dir '${RUN_DIR}/workload'",
            "--ttft-slo-override-seconds",
            "len(rows) == 40",
            "prompt_chars",
            "ttft_slo_s",
            "tpot_slo_s",
            "trace40_scheduled.jsonl",
            "TRACE_WAVE_GAP_SECONDS",
            "python/sglang/srt/disaggregation/decode.py",
        ):
            self.assertIn(value, source)

    def test_router_is_restarted_after_workers_are_healthy(self):
        source = RUNNER.read_text(encoding="utf-8")
        worker_wait = source.index(
            'wait_http "${HOSTS[$i]}" "http://127.0.0.1:30000/health"'
        )
        router_restart = source.index("docker restart tiancij-pd-router")
        router_wait = source.index(
            'wait_http "${HOSTS[0]}" "http://127.0.0.1:8000/v1/models"'
        )

        self.assertLess(worker_wait, router_restart)
        self.assertLess(router_restart, router_wait)

    def test_controller_source_and_target_have_defaults_and_env_overrides(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'SOURCE_NAME="${PD_FLIP_SOURCE_NAME:-node2}"', source
        )
        self.assertIn(
            'MIGRATION_TARGET_NAME="${PD_FLIP_MIGRATION_TARGET_NAME:-node3}"',
            source,
        )
        self.assertIn('--source-name \'${SOURCE_NAME}\'', source)
        self.assertIn(
            '--migration-target-name \'${MIGRATION_TARGET_NAME}\'', source
        )

    def test_preflight_rejects_same_source_and_migration_target(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "trace40.env"
            env_file.write_text(
                "\n".join(
                    [
                        "ADMIN_API_KEY=not-a-placeholder",
                        "IMAGE=sglang-pd-switch:tianciJ",
                        "SGLANG_REPO=/home/tiancij/sglang-pd-e9c4472c3",
                        "TRACE_PATH=/home/tiancij/pd-artifacts/trace.jsonl",
                        "ARTIFACT_ROOT=/home/tiancij/pd-artifacts",
                        "PD_FLIP_SOURCE_NAME=node1",
                        "PD_FLIP_MIGRATION_TARGET_NAME=node1",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )
            result = _run_dry_preflight(env_file)

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(
            "PD_FLIP_SOURCE_NAME and PD_FLIP_MIGRATION_TARGET_NAME must differ",
            result.stderr,
        )

    def test_preflight_rejects_unknown_source_or_migration_target(self):
        for source_name, target_name, expected in (
            ("unknown", "node3", "unknown PD_FLIP_SOURCE_NAME: unknown"),
            ("node2", "unknown", "unknown PD_FLIP_MIGRATION_TARGET_NAME: unknown"),
        ):
            with self.subTest(source_name=source_name, target_name=target_name):
                with tempfile.TemporaryDirectory() as directory:
                    env_file = Path(directory) / "trace40.env"
                    env_file.write_text(
                        "\n".join(
                            [
                                "ADMIN_API_KEY=not-a-placeholder",
                                "IMAGE=sglang-pd-switch:tianciJ",
                                "SGLANG_REPO=/home/tiancij/sglang-pd-e9c4472c3",
                                "TRACE_PATH=/home/tiancij/pd-artifacts/trace.jsonl",
                                "ARTIFACT_ROOT=/home/tiancij/pd-artifacts",
                                f"PD_FLIP_SOURCE_NAME={source_name}",
                                f"PD_FLIP_MIGRATION_TARGET_NAME={target_name}",
                            ]
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                    result = _run_dry_preflight(env_file)

                self.assertEqual(
                    result.returncode, 2, result.stdout + result.stderr
                )
                self.assertIn(expected, result.stderr)
                self.assertNotIn("[dry-run] ssh", result.stdout)

    def test_preflight_dry_run_has_no_external_side_effects_or_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "trace40.env"
            secret = "do-not-print-this-secret"
            env_file.write_text(
                "\n".join(
                    [
                        "ADMIN_API_KEY=" + secret,
                        "IMAGE=sglang-pd-switch:tianciJ",
                        "SGLANG_REPO=/home/tiancij/sglang-pd-e9c4472c3",
                        "TRACE_PATH=/home/tiancij/pd-artifacts/trace.jsonl",
                        "ARTIFACT_ROOT=/home/tiancij/pd-artifacts",
                    ]
                ),
                encoding="utf-8",
                newline="\n",
            )
            result = _run_dry_preflight(env_file)

        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr
        self.assertNotIn(secret, output)
        self.assertNotIn("docker start", output)
        self.assertNotIn("docker stop", output)
        self.assertIn("preflight complete", output)


if __name__ == "__main__":
    unittest.main()
