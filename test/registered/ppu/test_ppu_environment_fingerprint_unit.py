"""环境指纹的纯标准库测试；直接执行时不导入 sglang 的硬件依赖。"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "python/sglang/test/kits/environment_fingerprint.py"
OBSERVER = ROOT / "scripts/ci/ppu/observe_pod_environment.sh"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = load_module(
    ROOT / "python/sglang/test/ci/ci_register.py", "fingerprint_registry"
)
register_cpu_ci = registry.register_cpu_ci
register_ppu_ci = registry.register_ppu_ci
register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=15, suite="stage-a-test-cpu-ppu")


def mapping_line(library):
    info = library.stat()
    return f"1000-2000 r-xp 0 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x} {info.st_ino} {library}\n"


class TestEnvironmentFingerprint(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "缺少环境指纹采集模块")
        self.mod = load_module(MODULE, "fingerprint_under_test")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_environment_is_an_explicit_allowlist(self):
        env = self.mod.safe_environment(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "NODE_NAME": "compute-1",
                "SGLANG_WARMUP_TIMEOUT": "300",
                "PPU_ARTIFACTORY_PASSWORD": "secret",
                "GITHUB_TOKEN": "secret",
                "SGLANG_UNKNOWN_SECRET": "secret",
                "HTTPS_PROXY": "http://user:secret@proxy",
                "UNRELATED": "secret",
            }
        )
        self.assertEqual(
            env, {"CUDA_VISIBLE_DEVICES": "0,1", "SGLANG_WARMUP_TIMEOUT": "300"}
        )

    def test_recursive_redaction_keeps_generation_token_budget(self):
        value = self.mod.redact(
            {
                "max_tokens": 2048,
                "api_key": "secret",
                "nested": {"password": "secret"},
                "endpoint": "https://user:secret@host/path",
            }
        )
        self.assertEqual(value["max_tokens"], 2048)
        self.assertNotIn("secret", json.dumps(value))

    def test_launch_arguments_redact_both_flag_forms(self):
        args = self.mod.redact_args(
            ["--api-key", "secret", "--token=secret", "--tp-size", "8"]
        )
        self.assertNotIn("secret", json.dumps(args))
        self.assertEqual(args[-2:], ["--tp-size", "8"])

    def test_hash_stable_across_dictionary_order(self):
        self.assertEqual(
            self.mod.digest({"a": 1, "b": 2}), self.mod.digest({"b": 2, "a": 1})
        )
        self.assertNotEqual(self.mod.digest({"a": 1}), self.mod.digest({"a": 2}))

    def test_file_identity_hashes_bytes_and_not_just_mtime(self):
        path = self.root / "config.json"
        path.write_bytes(b"first")
        identity = self.mod.file_identity(path, self.mod.ReadBudget())
        self.assertEqual(identity["sha256"], hashlib.sha256(b"first").hexdigest())
        path.write_bytes(b"other")
        self.assertNotEqual(
            identity["sha256"],
            self.mod.file_identity(path, self.mod.ReadBudget())["sha256"],
        )

    def test_file_budget_never_hashes_oversized_files(self):
        path = self.root / "large"
        path.write_bytes(b"x" * 32)
        result = self.mod.file_identity(path, self.mod.ReadBudget(max_file_bytes=8))
        self.assertEqual(result["status"], "size_limit")
        self.assertNotIn("sha256", result)

    def test_aggregate_read_budget(self):
        path = self.root / "small"
        path.write_bytes(b"1234")
        budget = self.mod.ReadBudget(max_bytes=6)
        self.assertEqual(self.mod.file_identity(path, budget)["status"], "ok")
        self.assertEqual(
            self.mod.file_identity(path, budget)["status"], "budget_exhausted"
        )

    def test_missing_and_nonregular_files_are_explicit(self):
        self.assertEqual(
            self.mod.file_identity(self.root / "missing", self.mod.ReadBudget())[
                "status"
            ],
            "missing",
        )
        self.assertEqual(
            self.mod.file_identity(self.root, self.mod.ReadBudget())["status"],
            "not_regular",
        )
        if hasattr(os, "mkfifo"):
            fifo = self.root / "pipe"
            os.mkfifo(fifo)
            self.assertEqual(
                self.mod.file_identity(fifo, self.mod.ReadBudget())["status"],
                "not_regular",
            )

    def test_config_environment_is_also_allowlisted(self):
        result = self.mod.redact(
            {
                "server": {
                    "environment": {"UNRELATED": "private", "CUDA_VISIBLE_DEVICES": "0"}
                }
            }
        )
        self.assertNotIn("private", json.dumps(result))
        self.assertIn("CUDA_VISIBLE_DEVICES", json.dumps(result))

    def test_command_output_limit_stops_producer_before_timeout(self):
        result = self.mod.command(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdout.write('x'*3000000); sys.stdout.flush(); time.sleep(5)",
            ],
            timeout=0.5,
        )
        self.assertEqual(result["status"], "output_limit")
        self.assertNotIn("output", result)

    def test_pod_observer_failure_reason_is_preserved(self):
        path = self.root / "pods.tsv"
        path.with_suffix(".status").write_text("timeout:query_failed\n")
        with mock.patch.dict(os.environ, {"SGLANG_PPU_POD_METADATA": str(path)}):
            result = self.mod.pod_snapshot(0)
        self.assertEqual(result.get("observer_status"), "timeout:query_failed")

    def test_pod_metadata_selects_current_rank_and_actual_image(self):
        rows = "pod-worker-0\tuid0\tnode0\timage:tag\tcontainerd://sha256:aaa\n"
        rows += "pod-worker-1\tuid1\tnode1\timage:tag\tcontainerd://sha256:bbb\n"
        result = self.mod.parse_pod_metadata(rows, 1)
        self.assertEqual(result["image_id"], "containerd://sha256:bbb")
        self.assertEqual(result["node_name"], "node1")
        self.assertNotEqual(result["image_id"], result["requested_image"])
        self.assertEqual(self.mod.parse_pod_metadata(rows, 2)["status"], "missing")

    def test_tag_only_and_duplicate_pods_are_not_verified(self):
        self.assertEqual(
            self.mod.parse_pod_metadata("p-worker-0\tu\tn\ttag\t\n", 0)["status"],
            "pending",
        )
        rows = "p-worker-0\tu\tn\ttag\tsha256:aa\nq-worker-0\tu\tn\ttag\tsha256:bb\n"
        self.assertEqual(self.mod.parse_pod_metadata(rows, 0)["status"], "ambiguous")

    def test_process_tree_reads_only_descendants_and_deduplicates_mappings(self):
        library = self.root / "libhggc.so"
        library.write_bytes(b"library")
        for pid, children in ((10, "11"), (11, ""), (99, "")):
            directory = self.root / "proc" / str(pid)
            (directory / "task" / str(pid)).mkdir(parents=True)
            (directory / "task" / str(pid) / "children").write_text(children)
            (directory / "maps").write_text(mapping_line(library))
            (directory / "environ").write_bytes(b"RANK=1\0TOKEN=secret\0")
        result = self.mod.process_snapshot(
            10, self.mod.ReadBudget(), proc_root=self.root / "proc"
        )
        self.assertEqual([p["pid"] for p in result["processes"]], [10, 11])
        self.assertEqual(len(result["libraries"]), 1)
        self.assertEqual(
            result["libraries"][0]["sha256"], hashlib.sha256(b"library").hexdigest()
        )
        self.assertNotIn("secret", json.dumps(result))

    def test_children_of_nonmain_threads_are_collected_with_mapping_links(self):
        library = self.root / "libcuda.so"
        library.write_bytes(b"library")
        for pid in (10, 11):
            base = self.root / "proc" / str(pid)
            (base / "task" / str(pid)).mkdir(parents=True)
            (base / "task" / str(pid) / "children").write_text("")
            (base / "maps").write_text(mapping_line(library))
            (base / "environ").write_bytes(b"RANK=0\0")
        thread = self.root / "proc/10/task/12"
        thread.mkdir()
        (thread / "children").write_text("11")
        result = self.mod.process_snapshot(
            10, self.mod.ReadBudget(), proc_root=self.root / "proc"
        )
        self.assertEqual([p["pid"] for p in result["processes"]], [10, 11])
        self.assertEqual(result["processes"][1].get("library_paths"), [str(library)])

    def test_large_elf_build_id_uses_only_bounded_note_reads(self):
        path = self.root / "libhggc.so"
        header = struct.pack(
            "<16sHHIQQQIHHHHHH",
            b"\x7fELF\x02\x01\x01" + b"\0" * 9,
            3,
            62,
            1,
            0,
            64,
            0,
            0,
            64,
            56,
            1,
            0,
            0,
            0,
        )
        note = struct.pack("<III", 4, 4, 3) + b"GNU\0" + b"\x12\x34\xab\xcd"
        program = struct.pack("<IIQQQQQQ", 4, 0, 128, 0, 0, len(note), len(note), 4)
        with path.open("wb") as stream:
            stream.write(header + program + b"\0" * 8 + note)
            stream.truncate(9 * 1024 * 1024)
        budget = self.mod.ReadBudget(max_bytes=1024)
        result = self.mod.file_identity(path, budget)
        self.assertEqual(result.get("build_id"), "1234abcd")
        self.assertEqual(result["status"], "build_id_only")
        self.assertNotIn("sha256", result)
        self.assertGreater(budget.remaining, 0)

    def test_replaced_mapped_file_is_not_attributed_to_running_process(self):
        library = self.root / "libcuda.so"
        library.write_bytes(b"replacement")
        base = self.root / "proc/10"
        (base / "task/10").mkdir(parents=True)
        (base / "task/10/children").write_text("")
        (base / "maps").write_text(f"1000-2000 r-xp 0 00:00 1 {library}\n")
        (base / "environ").write_bytes(b"")
        result = self.mod.process_snapshot(
            10, self.mod.ReadBudget(), proc_root=self.root / "proc"
        )
        self.assertEqual(result["libraries"][0]["status"], "mapped_file_replaced")
        self.assertNotIn("sha256", result["libraries"][0])

    def test_hardware_query_falls_back_without_optional_firmware(self):
        answers = [
            {"status": "command_failed"},
            {"status": "ok", "output": "0, GPU-a, 0000:01:00.0, PPU, 1.0"},
        ]
        with mock.patch.object(self.mod, "command", side_effect=answers):
            result = self.mod.hardware_snapshot()
        self.assertEqual(result.get("devices", [{}])[0].get("uuid"), "GPU-a")
        self.assertEqual(result["status"], "partial")

    def test_package_versions_read_bounded_metadata_without_import(self):
        metadata = self.root / "torch-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Name: torch\nVersion: 1.0\n\nprivate-metadata"
        )
        with mock.patch.object(
            self.mod.importlib.metadata,
            "distribution",
            return_value=SimpleNamespace(_path=metadata),
        ):
            result = self.mod.package_snapshot(self.mod.ReadBudget())
        self.assertEqual(result["packages"]["torch"]["version"], "1.0")
        self.assertNotIn("private-metadata", json.dumps(result))

    def test_stage_failure_preserves_partial_evidence(self):
        request = {
            "output_dir": str(self.root),
            "phase": "server_ready",
            "config": {},
            "node_rank": 0,
            "server_pid": None,
            "server_args": [],
            "server_env": {},
        }
        self.mod.capture_to_file(request)
        request["failure_status"] = "timeout"
        self.mod.capture_to_file(request)
        snapshot = json.loads((self.root / "environment.json").read_text())[
            "snapshots"
        ]["server_ready"]
        self.assertEqual(snapshot["status"], "timeout")
        self.assertIn("source", snapshot["groups"])

    def test_unwritable_output_does_not_block_original_test(self):
        path = self.root / "not-a-directory"
        path.write_text("existing")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(
                self.mod.EnvironmentRecorder(path, {}).capture("setup_failed")
            )
        self.assertEqual(path.read_text(), "existing")

    def test_command_budget_covers_cumulative_output(self):
        budget = self.mod.ReadBudget(max_bytes=32)
        command = [sys.executable, "-c", "print('a' * 19)"]
        self.assertEqual(self.mod.command(command, budget=budget)["status"], "ok")
        self.assertEqual(
            self.mod.command(command, budget=budget)["status"], "budget_exhausted"
        )
        self.assertGreaterEqual(budget.remaining, 0)

    def test_total_budget_covers_proc_text_and_file_count(self):
        path = self.root / "small"
        path.write_text("1234")
        budget = self.mod.ReadBudget(max_bytes=6)
        self.assertEqual(self.mod.read_text(path, budget=budget), "1234")
        self.assertEqual(
            self.mod.file_identity(path, budget)["status"], "budget_exhausted"
        )
        budget = self.mod.ReadBudget(max_files=1)
        self.assertEqual(self.mod.file_identity(path, budget)["status"], "ok")
        self.assertEqual(
            self.mod.file_identity(path, budget)["status"], "file_count_limit"
        )

    def test_no_model_weight_or_shared_cache_directory_scan(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "weights.safetensors").write_text("do-not-read")
        with mock.patch.object(
            Path, "rglob", side_effect=AssertionError("禁止递归扫描")
        ):
            result = self.mod.model_snapshot(str(model), self.mod.ReadBudget())
        self.assertNotIn("weights.safetensors", json.dumps(result))
        self.assertIn("config.json", json.dumps(result))
        self.assertEqual(result["weights_verification"], "not_collected")

    def test_phase_snapshots_are_preserved_and_pid_is_not_fingerprinted(self):
        for phase, pid in (("before_start", None), ("server_ready", 987654321)):
            self.mod.capture_to_file(
                {
                    "output_dir": str(self.root),
                    "phase": phase,
                    "config": {},
                    "node_rank": 0,
                    "server_pid": pid,
                    "server_args": [],
                    "server_env": {},
                },
                collect=False,
            )
        report = json.loads((self.root / "environment.json").read_text())
        self.assertEqual(set(report["snapshots"]), {"before_start", "server_ready"})
        self.assertEqual(report["schema_version"], "ppu-environment/v1")

    def test_collector_failure_does_not_change_test_exit(self):
        recorder = self.mod.EnvironmentRecorder(self.root, {}, node_rank=0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(
            self.mod.subprocess, "Popen", side_effect=OSError("secret")
        ):
            self.assertFalse(recorder.capture("before_start"))
        self.assertNotIn("secret", output.getvalue())

    def test_collector_wall_timeout_is_bounded(self):
        recorder = self.mod.EnvironmentRecorder(self.root, {}, node_rank=0, timeout=0.1)
        sleeper = self.root / "sleeper.py"
        sleeper.write_text("import sys,time; sys.stdin.read(); time.sleep(60)\n")
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            self.mod, "__file__", str(sleeper)
        ):
            start = time.monotonic()
            self.assertFalse(recorder.capture("before_start"))
        self.assertLess(time.monotonic() - start, 3)

    def test_subprocess_capture_works_without_hardware_packages(self):
        recorder = self.mod.EnvironmentRecorder(
            self.root, {"model": {"path": str(self.root)}}, node_rank=2
        )
        self.assertTrue(recorder.capture("before_start"))
        report = json.loads((self.root / "environment.json").read_text())
        snapshot = report["snapshots"]["before_start"]
        self.assertEqual(report["node_rank"], 2)
        self.assertIn("source", snapshot["groups"])
        self.assertIn("hardware", snapshot["groups"])
        self.assertIn("jit", snapshot["groups"])
        self.assertNotIn("PPU_ARTIFACTORY_PASSWORD", json.dumps(report))

    def test_launch_settings_survive_later_phase_capture(self):
        recorder = self.mod.EnvironmentRecorder(self.root, {}, node_rank=0)
        with mock.patch.object(recorder, "_run", return_value="ok") as run:
            recorder.capture(
                "before_start",
                server_args=["--tp-size", "8"],
                server_env={"SGLANG_WARMUP_TIMEOUT": "300"},
            )
            recorder.capture("server_ready", server_pid=42)
        self.assertEqual(run.call_args.args[0]["server_args"], ["--tp-size", "8"])
        self.assertEqual(
            run.call_args.args[0]["server_env"], {"SGLANG_WARMUP_TIMEOUT": "300"}
        )

    def test_permission_denied_is_not_an_empty_file_hash(self):
        with mock.patch.object(
            self.mod.os, "open", side_effect=PermissionError("secret")
        ):
            result = self.mod.file_identity(self.root / "file", self.mod.ReadBudget())
        self.assertEqual(result["status"], "permission_denied")
        self.assertNotIn("sha256", result)
        self.assertNotIn("secret", json.dumps(result))

    def test_jit_hash_tracks_mapped_content_not_cache_location(self):
        request = {
            "output_dir": str(self.root),
            "phase": "server_ready",
            "config": {},
            "node_rank": 0,
            "server_pid": 100,
            "server_args": [],
            "server_env": {},
        }
        hashes = []
        for cache, content in (("pid-100", "aa"), ("pid-200", "aa"), ("pid-200", "bb")):
            process = {
                "status": "ok",
                "processes": [],
                "libraries": [
                    {
                        "path": f"/tmp/{cache}/kernel.so",
                        "status": "ok",
                        "sha256": content,
                        "is_jit": True,
                    }
                ],
            }
            with mock.patch.dict(
                os.environ, {"TVM_FFI_CACHE_DIR": f"/tmp/{cache}"}
            ), mock.patch.object(self.mod, "process_snapshot", return_value=process):
                self.mod.capture_to_file(request)
            groups = json.loads((self.root / "environment.json").read_text())[
                "snapshots"
            ]["server_ready"]["groups"]
            hashes.append((groups["jit"]["sha256"], groups["configuration"]["sha256"]))
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[1][0], hashes[2][0])

    def test_unknown_group_does_not_get_an_equality_hash(self):
        self.assertIsNone(self.mod.make_group({"status": "missing"})["sha256"])

    def test_mapped_jit_file_can_live_outside_dot_cache(self):
        cache = self.root / "local-jit"
        cache.mkdir()
        library = cache / "kernel.so"
        library.write_bytes(b"jit")
        base = self.root / "proc/10"
        (base / "task/10").mkdir(parents=True)
        (base / "task/10/children").write_text("")
        (base / "maps").write_text(mapping_line(library))
        (base / "environ").write_bytes(b"")
        with mock.patch.dict(os.environ, {"TVM_FFI_CACHE_DIR": str(cache)}):
            result = self.mod.process_snapshot(
                10, self.mod.ReadBudget(), proc_root=self.root / "proc"
            )
        self.assertEqual(len(result["libraries"]), 1)
        self.assertTrue(result["libraries"][0].get("is_jit"))

    def test_group_hash_excludes_process_and_pod_identity(self):
        request = {
            "output_dir": str(self.root),
            "phase": "server_ready",
            "config": {},
            "node_rank": 0,
            "server_pid": 100,
            "server_args": [],
            "server_env": {},
        }
        with mock.patch.object(
            self.mod, "hardware_snapshot", return_value={"status": "missing"}
        ), mock.patch.object(
            self.mod,
            "process_snapshot",
            return_value={"status": "ok", "processes": [{"pid": 100}], "libraries": []},
        ), mock.patch.object(
            self.mod,
            "pod_snapshot",
            return_value={"status": "ok", "pod_uid": "first", "image_id": "sha256:a"},
        ):
            self.mod.capture_to_file(request)
        first = json.loads((self.root / "environment.json").read_text())["snapshots"][
            "server_ready"
        ]
        request["server_pid"] = 200
        with mock.patch.object(
            self.mod, "hardware_snapshot", return_value={"status": "missing"}
        ), mock.patch.object(
            self.mod,
            "process_snapshot",
            return_value={"status": "ok", "processes": [{"pid": 200}], "libraries": []},
        ), mock.patch.object(
            self.mod,
            "pod_snapshot",
            return_value={"status": "ok", "pod_uid": "second", "image_id": "sha256:a"},
        ):
            self.mod.capture_to_file(request)
        second = json.loads((self.root / "environment.json").read_text())["snapshots"][
            "server_ready"
        ]
        self.assertEqual(first["groups"], second["groups"])
        self.assertNotEqual(first["observation"], second["observation"])


class TestPodObserver(unittest.TestCase):
    def setUp(self):
        self.assertTrue(OBSERVER.is_file(), "缺少编排侧 Pod 元数据观察器")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_observer(self, body, expected="2", job="test_job"):
        executable = self.root / "kubectl"
        executable.write_text("#!/bin/sh\n" + body)
        executable.chmod(0o755)
        output = self.root / "pods.tsv"
        environment = {
            **os.environ,
            "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
            "PPU_ENV_JOB": job,
            "GITHUB_REPOSITORY_OWNER": "Owner",
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "PPU_ENV_EXPECTED_PODS": expected,
            # Bash SECONDS 只有整秒精度，为首次状态写入留出一秒余量。
            "PPU_ENV_OBSERVER_LIMIT": "2",
            "PPU_ENV_OBSERVER_INTERVAL": "0.01",
        }
        result = subprocess.run(
            ["bash", str(OBSERVER), "watch", str(output)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def test_observer_uses_current_job_selector_and_stores_actual_identity(self):
        output = self.run_observer(
            'case "$*" in *ppu-job=ppu-owner-123-2-test-job*) ;; *) exit 9;; esac\n'
            "printf 'ppu-owner-123-2-test-job-worker-0\\tu0\\tn0\\timage:tag\\tsha256:aaa\\nppu-owner-123-2-test-job-worker-1\\tu1\\tn1\\timage:tag\\tsha256:bbb\\n'\n"
        )
        self.assertIn("sha256:bbb", output.read_text())
        self.assertEqual(output.with_suffix(".status").read_text().strip(), "ok")

    def test_observer_explicit_matrix_suffix_matches_action(self):
        output = self.run_observer(
            'case "$*" in *ppu-job=ppu-owner-123-2-ans16-1*) ;; *) exit 9;; esac\n'
            "printf 'ppu-owner-123-2-ans16-1-worker-0\\tu0\\tn0\\timage:tag\\tsha256:aaa\\n'\n",
            expected="1",
            job="ans16-1",
        )
        self.assertEqual(output.with_suffix(".status").read_text().strip(), "ok")

    def test_pr_observer_uses_resolved_name_and_records_timestamps(self):
        with mock.patch.dict(
            os.environ,
            {"PPU_ENV_JOB_NAME": "ppu-pr-exact-hash", "PPU_ENV_PR_TIMINGS": "true"},
        ):
            output = self.run_observer(
                'case "$*" in *ppu-job=ppu-pr-exact-hash*) ;; *) exit 9;; esac\n'
                'case "$*" in *creationTimestamp*) printf "2026-01-01T00:00:00Z\\t2026-01-01T00:00:03Z\\t2026-01-01T00:00:11Z\\n";;\n'
                '*) printf "ppu-pr-exact-hash-worker-0\\tu0\\tn0\\timage:tag\\tsha256:aaa\\n";; esac\n',
                expected="1",
            )
        self.assertEqual(output.with_suffix(".status").read_text().strip(), "ok")
        self.assertEqual(
            len(output.with_suffix(".timing.tsv").read_text().strip().split("\t")), 3
        )

    def test_observer_never_archives_other_job_rows(self):
        output = self.run_observer(
            "printf 'other-worker-0\\tu0\\tn0\\timage:tag\\tsha256:aaa\\n'\n",
            expected="1",
        )
        self.assertFalse(output.exists())
        self.assertNotEqual(output.with_suffix(".status").read_text().strip(), "ok")

    def test_observer_failure_is_bounded_and_does_not_fail_job(self):
        output = self.run_observer("echo secret >&2\nexit 1\n")
        self.assertFalse(output.exists())
        self.assertIn("query_failed", output.with_suffix(".status").read_text())

    def test_observer_requires_all_expected_pods_with_image_ids(self):
        output = self.run_observer(
            "printf 'ppu-owner-123-2-test-job-worker-0\\tu0\\tn0\\timage:tag\\tsha256:aaa\\n'\n"
        )
        self.assertNotEqual(output.with_suffix(".status").read_text().strip(), "ok")


class TestEnvironmentIntegration(unittest.TestCase):
    def test_evidence_collectors_archive_fingerprints_without_removing_nas(self):
        for suite in ("answer", "perf", "accuracy"):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "nas"
                source.mkdir()
                names = [
                    "environment.json",
                    "environment-pods.tsv",
                    "environment-pods.status",
                ]
                if suite != "accuracy":
                    (source / "ranks/rank-1").mkdir(parents=True)
                    names.append("ranks/rank-1/environment.json")
                else:
                    (source / "evalscope/cache").mkdir(parents=True)
                    (source / "evalscope/cache/private").write_text("not-for-artifact")
                for name in names:
                    (source / name).write_text("evidence")
                result = subprocess.run(
                    ["bash", str(ROOT / f"scripts/ci/ppu/collect_{suite}_evidence.sh")],
                    env={
                        **os.environ,
                        "ENTRY": "test",
                        "GITHUB_WORKSPACE": temp,
                        f"{suite.upper()}_RESULTS_ON_RUNNER": str(source),
                    },
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                destination = root / f"ppu-{suite}-artifacts"
                for name in names:
                    self.assertTrue((destination / name).is_file(), (suite, name))
                    self.assertEqual((source / name).read_text(), "evidence")
                if suite == "accuracy":
                    self.assertFalse((destination / "evalscope/cache").exists())

    def test_runtime_lifecycle_preserves_failures_and_builds_launch_args_once(self):
        # 执行原 mixin 类体，仅替换硬件/服务边界；不导入 torch/msgspec。
        for name in ("answer", "perf", "pd_perf", "accuracy"):
            for scenario in ("success", "launch_failure", "args_failure"):
                with self.subTest(
                    suite=name, scenario=scenario
                ), tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)
                    (directory / "config.json").write_text('{"model_type":"test"}')
                    config = {
                        "model": {
                            "path": temp,
                            "checkpoint_name": directory.name,
                            "accepted_model_types": ["test"],
                        },
                        "server": {"parameters": {}, "startup_timeout_seconds": 1},
                        "request": {},
                        "hardware": {"visible_devices": [0]},
                        "disaggregation": {"prefill_port": 8000, "router_port": 8001},
                        "prefill": {"startup_timeout_seconds": 1},
                    }
                    events = []
                    mod = load_module(MODULE, "fingerprint_lifecycle")

                    class Recorder(mod.EnvironmentRecorder):
                        def capture(self, phase, **kwargs):
                            events.append(phase)
                            return super().capture(phase, **kwargs)

                    def launch(**kwargs):
                        events.append("launch")
                        if scenario == "launch_failure":
                            raise RuntimeError("original-launch-error")
                        return SimpleNamespace(pid=42)

                    def args(*unused, **kwargs):
                        events.append("args")
                        if scenario == "args_failure":
                            raise RuntimeError("original-args-error")
                        return ["--tp-size", "1"]

                    tree = ast.parse(
                        (
                            ROOT / f"python/sglang/test/kits/{name}_suite_kit.py"
                        ).read_text()
                    )
                    cls_node = next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.ClassDef) and n.name.endswith("SuiteMixin")
                    )
                    namespace = {
                        "Path": Path,
                        "os": os,
                        "EnvironmentRecorder": Recorder,
                        "RESULTS_DIR_ENV": "TEST_ENV_RESULTS",
                        "DEFAULT_URL_FOR_TEST": "http://localhost",
                        "torch": SimpleNamespace(
                            cuda=SimpleNamespace(device_count=lambda: 1)
                        ),
                        "load_json": lambda p: json.loads(Path(p).read_text()),
                        "resolve_evaluation_paths": lambda *a: (
                            directory / "config.json",
                        )
                        * 2,
                        "resolve_distributed_runtime": lambda *a: None,
                        "resolve_pd_runtime": lambda *a: {
                            "node_rank": 0,
                            "role": "prefill",
                        },
                        "resolve_measurement_plan": lambda *a: [],
                        "resolve_evaluation_plan": lambda *a: {"dataset_dir": temp},
                        "DATASET_DIR_ENV": "TEST_ENV_DATASET",
                        "get_local_ip_auto": lambda: "127.0.0.1",
                        "pd_endpoint_url": lambda *a: "http://localhost",
                        "popen_launch_server": launch,
                        "popen_launch_pd_server": launch,
                        "wait_for_http_ready": lambda **kw: events.append(
                            "health_ready"
                        ),
                        "kill_process_tree": lambda pid: events.append("kill"),
                        "build_evalscope_command": lambda *a, **kw: [],
                    }
                    for builder in ("answer", "perf", "pd", "accuracy"):
                        namespace[f"build_{builder}_server_args"] = args
                    exec(
                        compile(
                            ast.Module(body=[cls_node], type_ignores=[]),
                            "<mixin>",
                            "exec",
                        ),
                        namespace,
                    )
                    case = type(
                        "LifecycleCase",
                        (namespace[cls_node.name], unittest.TestCase),
                        {},
                    )
                    case.data_root = directory
                    case._load_test_config = classmethod(
                        lambda cls: (config, directory / "test.json")
                    )
                    case._resolve_rank_dir = classmethod(
                        lambda cls: directory / "ranks"
                    )
                    case._server_environment = classmethod(lambda cls: None)
                    case._resolve_evalscope = classmethod(lambda cls: "evalscope")
                    for method in (
                        "_preflight",
                        "_open_server_logs",
                        "_resolve_metric_resources",
                        "_publish_endpoint",
                        "_write_node_inventory",
                        "_await_peer_endpoints",
                        "_urls_by_role",
                        "_await_peer_servers",
                        "_launch_router",
                    ):
                        setattr(case, method, classmethod(lambda cls, *a: None))
                    case._write_setup_failure = classmethod(
                        lambda cls, *a: events.append("failure_report")
                    )
                    case._release_worker_nodes = classmethod(
                        lambda cls: events.append("release")
                    )
                    with mock.patch.dict(
                        os.environ, {"TEST_ENV_RESULTS": temp}
                    ), mock.patch.object(
                        Recorder, "_run", return_value="timeout"
                    ), contextlib.redirect_stdout(
                        io.StringIO()
                    ):
                        if scenario == "success":
                            case.setUpClass()
                            case.tearDownClass()
                            self.assertLess(
                                events.index("before_start"), events.index("launch")
                            )
                            self.assertLess(
                                events.index("server_ready"),
                                events.index("before_stop"),
                            )
                            self.assertLess(
                                events.index("before_stop"), events.index("kill")
                            )
                            if name == "pd_perf":
                                self.assertLess(
                                    events.index("health_ready"),
                                    events.index("server_ready"),
                                )
                            self.assertEqual(events.count("args"), 1)
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "original-.*-error"
                            ):
                                case.setUpClass()
                            self.assertIn("setup_failed", events)
                            self.assertIn("failure_report", events)
                            self.assertNotIn("server_ready", events)

    def test_all_four_mixins_capture_before_launch_ready_failure_and_before_kill(self):
        for name in ("answer", "perf", "pd_perf", "accuracy"):
            with self.subTest(suite=name):
                text = (
                    ROOT / f"python/sglang/test/kits/{name}_suite_kit.py"
                ).read_text()
                tree = ast.parse(text)
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                phases = {
                    node.args[0].value: node.lineno
                    for node in calls
                    if isinstance(node.func, ast.Attribute)
                    and node.func.attr == "capture"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                }
                self.assertEqual(
                    set(phases),
                    {"before_start", "server_ready", "setup_failed", "before_stop"},
                )
                launch = next(
                    node.lineno
                    for node in calls
                    if isinstance(node.func, ast.Name)
                    and node.func.id
                    in {"popen_launch_server", "popen_launch_pd_server"}
                )
                self.assertLess(phases["before_start"], launch)
                self.assertGreater(phases["server_ready"], launch)
                teardown = next(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "tearDownClass"
                )
                kills = [
                    node.lineno
                    for node in ast.walk(teardown)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "kill_process_tree"
                ]
                self.assertLess(phases["before_stop"], min(kills))

    def test_yaml_structure_paths_and_job_identities(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("YAML 结构验证由安装 PyYAML 的本地验证环境执行")
        names = (
            "answer",
            "answer-16",
            "answer-32",
            "perf",
            "perf-16",
            "perf-32",
            "pd-perf-glm52",
            "pd-perf-qwen35",
            "accuracy",
        )
        identities, paths = set(), set()
        for name in names:
            data = yaml.safe_load(
                (ROOT / f".github/workflows/test-ppu-{name}.yml").read_text()
            )
            for job_name, job in data["jobs"].items():
                for index, step in enumerate(job["steps"]):
                    if not step.get("uses", "").startswith(
                        "flytiger-eco/ppu-distributed-action@"
                    ):
                        continue
                    observer = job["steps"][index - 1]
                    environment = observer["env"]
                    inputs = step["with"]
                    self.assertTrue(observer["continue-on-error"])
                    self.assertEqual(
                        int(environment["PPU_ENV_EXPECTED_PODS"]),
                        max(1, inputs["nnodes"]),
                    )
                    extra = inputs["extra_env"]
                    metadata = re.search(
                        r"SGLANG_PPU_POD_METADATA=([^,]+)", extra
                    ).group(1)
                    results = (
                        re.search(
                            r"SGLANG_PPU_(?:ANSWER|PERF|PD_PERF|ACCURACY)_RESULTS_DIR=(.*)",
                            extra,
                        )
                        .group(1)
                        .split("${{ inputs.dataset_dir", 1)[0]
                        .split(",", 1)[0]
                    )
                    self.assertEqual(metadata, results + "/environment-pods.tsv")
                    expected_runner = metadata.replace(
                        "/mnt/wl_nas/", "/wl_nas/"
                    ).replace("_RESULTS_IN_POD", "_RESULTS_ON_RUNNER")
                    self.assertEqual(environment["PPU_ENV_OUTPUT"], expected_runner)
                    self.assertEqual(
                        environment["PPU_ENV_JOB"],
                        inputs.get("job_suffix", "${{ github.job }}"),
                    )
                    entries = (
                        job.get("strategy", {}).get("matrix", {}).get("include", [{}])
                    )
                    for matrix_index, entry in enumerate(entries):
                        values = {
                            "github.job": job_name,
                            "github.run_id": "35950825084",
                            "github.run_attempt": "1",
                            "strategy.job-index": str(matrix_index),
                        }
                        values.update({"matrix." + k: str(v) for k, v in entry.items()})
                        values.update(
                            {
                                "env." + k: str(v)
                                for k, v in {
                                    **data.get("env", {}),
                                    **job.get("env", {}),
                                }.items()
                            }
                        )

                        def expand(text):
                            for _ in range(3):
                                text = re.sub(
                                    r"\$\{\{\s*([^}]+?)\s*\}\}",
                                    lambda m: values.get(m.group(1), m.group(0)),
                                    text,
                                )
                            return text

                        suffix = (
                            re.sub(
                                "[^a-z0-9-]",
                                "-",
                                expand(environment["PPU_ENV_JOB"]).lower(),
                            )
                            .strip("-")[:20]
                            .rstrip("-")
                        )
                        identity = ("ppu-flytiger-eco-35950825084-1-" + suffix)[
                            :52
                        ].rstrip("-")
                        output = expand(expected_runner)
                        self.assertNotIn(identity, identities, (name, job_name, entry))
                        self.assertNotIn(output, paths, (name, job_name, entry))
                        identities.add(identity)
                        paths.add(output)

    def test_matrix_metadata_and_pod_suffix_are_isolated(self):
        text = (ROOT / ".github/workflows/test-ppu-answer-16.yml").read_text()
        self.assertIn("job_suffix: ans16-${{ strategy.job-index }}", text)
        self.assertIn("PPU_ENV_JOB: ans16-${{ strategy.job-index }}", text)
        self.assertIn("${{ env.ANSWER_RESULTS_ON_RUNNER }}/environment-pods.tsv", text)
        self.assertIn(
            "SGLANG_PPU_POD_METADATA=${{ env.ANSWER_RESULTS_IN_POD }}/environment-pods.tsv",
            text,
        )

    def test_nine_workflows_inject_provenance(self):
        names = (
            "answer",
            "answer-16",
            "answer-32",
            "perf",
            "perf-16",
            "perf-32",
            "pd-perf-glm52",
            "pd-perf-qwen35",
            "accuracy",
        )
        for name in names:
            with self.subTest(workflow=name):
                text = (ROOT / f".github/workflows/test-ppu-{name}.yml").read_text()
                count = text.count("uses: flytiger-eco/ppu-distributed-action@")
                self.assertGreater(count, 0)
                # action 可使用 main；这里只检查来源信息注入，不限制版本。
                self.assertEqual(text.count("observe_pod_environment.sh start"), count)
                self.assertEqual(
                    len(re.findall(r"SGLANG_PPU_DISTRIBUTED_ACTION_SHA=[^,\s]+", text)),
                    count,
                )
                self.assertEqual(text.count("SGLANG_PPU_POD_METADATA="), count)
                self.assertEqual(
                    text.count("SGLANG_PPU_WORKFLOW_SHA=${{ github.workflow_sha }}"),
                    count,
                )
                steps = re.split(r"(?m)^      - name: ", text)
                for index, step in enumerate(steps):
                    if "uses: flytiger-eco/ppu-distributed-action@" not in step:
                        continue
                    observer = steps[index - 1]
                    expected = max(1, int(re.search(r"nnodes: (\d+)", step).group(1)))
                    self.assertIn(f'PPU_ENV_EXPECTED_PODS: "{expected}"', observer)
                    self.assertIn("continue-on-error: true", observer)
                    self.assertIn(
                        'observe_pod_environment.sh start "$PPU_ENV_OUTPUT"', observer
                    )
                    self.assertIn("/environment-pods.tsv", observer)


if __name__ == "__main__":
    unittest.main()
