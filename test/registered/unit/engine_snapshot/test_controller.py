import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec

import sglang.srt.engine_snapshot.controller as controller
from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotRuntimeFailure,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import (
    MANIFEST_FORMAT,
    SnapshotCanary,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotModelFile,
    load_manifest,
    publish_manifest,
    write_json_atomic,
)
from sglang.srt.engine_snapshot.runtime import EngineInventory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Every cache root the controller must own for its child.
CACHE_VARIABLES = (
    "SGLANG_CACHE_DIR",
    "SGLANG_JIT_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    "DG_JIT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "TILELANG_CACHE_DIR",
    "HUMMING_CACHE_DIR",
    "HUMMING_TMP_DIR",
    "CUTE_DSL_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "TMPDIR",
)


def identity(**overrides):
    built = SnapshotIdentity(
        code="code",
        model=[SnapshotModelFile("config.json", 3, 1)],
        host_id="host",
        gpu_name="NVIDIA H20",
        gpu_uuid="GPU-1",
        driver_version="580.1",
        python="3.12.0",
        kernel="5.10.0",
        libc=["glibc", "2.35"],
        torch="2.13.0",
        cuda_runtime="13.0",
        sglang_kernel="0.4.6",
        transformers="5.12.1",
        triton="3.5.0",
        criu="Version: 4.2",
        cuda_checkpoint="abc",
        environment=[],
    )
    return msgspec.structs.replace(built, **overrides) if overrides else built


class TestSnapshotController(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact_path = Path(self.directory.name)
        self.artifact_path.chmod(0o700)
        (self.artifact_path / "shm").mkdir()

    def mock_runtime(self):
        runtime = Mock()
        runtime.SHM_DIR = self.artifact_path / "shm"
        runtime.current_identity.return_value = identity()
        runtime.stdio_resources.return_value = ["pipe:[1]", "pipe:[2]"]
        return runtime

    def engine_info(self):
        return control.EngineInfo(
            gpu_uuid="GPU-1", model_path="/model", host="127.0.0.1", port=30184
        )

    def engine_inventory(self):
        return EngineInventory(
            root_pid=99999991,
            pids=[99999991],
            cuda_pids=[99999991],
            gpu_uuid="GPU-1",
            shared_paths=set(),
        )

    def manifest(self, **overrides):
        fields = dict(
            format=MANIFEST_FORMAT,
            directory=str(self.artifact_path / "artifact"),
            created_at="2026-09-15T00:00:00+00:00",
            artifact_bytes=1,
            model_path="/model",
            host="127.0.0.1",
            port=30184,
            identity=identity(),
            root_pid=99999991,
            pids=[99999991],
            cuda_pids=[99999991],
            files=[],
            dev_shm=[],
            canary=SnapshotCanary("The capital of France is", 42),
        )
        fields.update(overrides)
        return SnapshotManifest(**fields)

    def publish(self, **overrides):
        artifact = self.artifact_path / "artifact"
        artifact.mkdir(mode=0o700, exist_ok=True)
        (artifact / "control").mkdir(mode=0o700, exist_ok=True)
        publish_manifest(artifact, self.manifest(**overrides))
        return artifact

    def create(self, runtime, output=None):
        with patch.object(controller, "validate_server_args"):
            return controller.create_snapshot(
                str(output or self.artifact_path / "artifact"),
                ["--model-path", "/model"],
                runtime=runtime,
            )

    # ------------------------------------------------------------------ #
    # create
    # ------------------------------------------------------------------ #
    def test_create_runs_the_steps_and_publishes(self):
        runtime = self.mock_runtime()
        runtime.launch_child.return_value = 99999991
        runtime.wait_ready.return_value = self.engine_info()
        inventory = self.engine_inventory()
        runtime.inventory.return_value = inventory

        def dump(artifact_path, captured, timeout):
            write_json_atomic(
                artifact_path / "control" / control.CANARY,
                msgspec.to_builtins(SnapshotCanary("The capital of France is", 42)),
            )
            return captured

        runtime.dump.side_effect = dump

        manifest = self.create(runtime)

        artifact = self.artifact_path / "artifact"
        runtime.preflight.assert_called_once_with("create", artifact)
        runtime.wait_ready.assert_called_once_with(artifact, 99999991, 600)
        runtime.inventory.assert_called_once_with(artifact, 99999991, "/model", "GPU-1")
        runtime.verify_dead.assert_called_once_with(inventory)
        runtime.abort_create.assert_not_called()
        self.assertEqual(manifest.root_pid, 99999991)
        self.assertEqual(manifest.canary.token_id, 42)
        self.assertTrue((artifact / "manifest.json").is_file())
        self.assertEqual(load_manifest(artifact, identity=identity()).pids, [99999991])

        runtime = self.mock_runtime()
        runtime.launch_child.side_effect = OSError("spawn failed")
        with self.assertRaises(OSError):
            self.create(runtime, output=self.artifact_path / "isolated")
        environment = runtime.launch_child.call_args.args[2]
        for name in CACHE_VARIABLES:
            self.assertTrue(
                environment[name].startswith(
                    str(self.artifact_path / "isolated/runtime")
                ),
                name,
            )
        self.assertEqual(environment["USE_LIBUV"], "0")
        self.assertEqual(environment["GLOO_SOCKET_IFNAME"], "lo")

        # Importing sglang in the controller's own process redirects several
        # caches and writes DG_JIT_CACHE_DIR; none of that may reach the child.
        environment = controller._child_environment(
            {
                "PATH": "/usr/bin",
                "TRITON_CACHE_DIR": "/artifact_path/.cache/sglang/triton",
                "DG_JIT_CACHE_DIR": "/artifact_path/.cache/sglang/deep_gemm",
                "SGLANG_CACHE_DIR": "/artifact_path/.cache/sglang",
            },
            self.artifact_path / "artifact",
        )
        self.assertEqual(environment["PATH"], "/usr/bin")
        for name in ("TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR", "SGLANG_CACHE_DIR"):
            self.assertTrue(
                environment[name].startswith(
                    str(self.artifact_path / "artifact/runtime")
                ),
                name,
            )

    def test_create_failure_aborts_and_explains_the_artifact(self):
        runtime = self.mock_runtime()
        runtime.launch_child.return_value = 99999991
        runtime.wait_ready.side_effect = SnapshotRuntimeFailure("engine died")

        with self.assertRaisesRegex(SnapshotRuntimeFailure, "engine died"):
            self.create(runtime)

        artifact = self.artifact_path / "artifact"
        runtime.abort_create.assert_called_once()
        self.assertFalse((artifact / "manifest.json").exists())
        self.assertIn("engine died", (artifact / "failure.json").read_text())
        self.assertTrue((artifact / "control" / control.ABORT).exists())

        (self.artifact_path / "existing").mkdir(mode=0o700)
        with self.assertRaisesRegex(SnapshotUsageError, "already exists"):
            self.create(self.mock_runtime(), output=self.artifact_path / "existing")

    # ------------------------------------------------------------------ #
    # restore
    # ------------------------------------------------------------------ #
    def test_restore_runs_the_steps_and_carries_the_override(self):
        artifact = self.publish()
        runtime = self.mock_runtime()
        runtime.restore.return_value = 99999991

        outcome = controller.restore_snapshot(
            str(artifact), runtime=runtime, host="0.0.0.0", port=31111
        )

        manifest = load_manifest(artifact)
        self.assertEqual(
            (outcome.root_pid, outcome.host, outcome.port), (99999991, "0.0.0.0", 31111)
        )
        runtime.preflight.assert_called_once_with("restore", artifact)
        runtime.restore.assert_called_once_with(artifact, manifest, 300)
        runtime.verify_restorable.assert_called_once_with(manifest, "0.0.0.0", 31111)
        runtime.wait_listener.assert_called_once_with(
            artifact, manifest, "0.0.0.0", 31111, 300
        )
        runtime.complete_restore.assert_called_once_with(99999991)
        release = control.read_json(
            artifact / "control", control.RELEASE, control.ReleaseInfo
        )
        self.assertEqual((release.host, release.port), ("0.0.0.0", 31111))
        runtime.stop_restored_tree.assert_not_called()

        # Without an override the captured address is used.
        runtime = self.mock_runtime()
        runtime.restore.return_value = 99999991
        outcome = controller.restore_snapshot(str(artifact), runtime=runtime)
        self.assertEqual((outcome.host, outcome.port), ("127.0.0.1", 30184))

    def test_restore_validates_the_artifact(self):
        artifact = self.publish()
        runtime = self.mock_runtime()
        runtime.current_identity.return_value = identity(gpu_uuid="GPU-2")

        with self.assertRaises(SnapshotCompatibilityError) as error:
            controller.restore_snapshot(str(artifact), runtime=runtime)
        self.assertIn("gpu_uuid", str(error.exception))
        runtime.restore.assert_not_called()

        (artifact / "control").rmdir()
        with self.assertRaisesRegex(SnapshotCompatibilityError, "control"):
            controller.restore_snapshot(str(artifact), runtime=self.mock_runtime())

    def test_restore_contains_failures(self):
        artifact = self.publish()
        runtime = self.mock_runtime()
        runtime.restore.return_value = 99999991
        runtime.wait_listener.side_effect = SnapshotRuntimeFailure("not healthy")

        def stop(root_pid, artifact_path, failures):
            failures.append("restore cleanup: incomplete")

        runtime.stop_restored_tree.side_effect = stop
        with self.assertRaisesRegex(
            SnapshotRuntimeFailure, "not healthy; restore cleanup: incomplete"
        ):
            controller.restore_snapshot(str(artifact), runtime=runtime)
        self.assertTrue((artifact / "control" / control.ABORT).exists())

        runtime = self.mock_runtime()
        runtime.verify_restorable.side_effect = SnapshotUsageError("pid taken")
        with self.assertRaisesRegex(SnapshotUsageError, "pid taken"):
            controller.restore_snapshot(str(artifact), runtime=runtime)
        runtime.stop_restored_tree.assert_not_called()

    # ------------------------------------------------------------------ #
    # CLI
    # ------------------------------------------------------------------ #

    def test_cli_surface(self):
        from sglang.cli.snapshot import snapshot

        artifact = self.publish()
        with patch.object(controller, "restore_snapshot") as restore:
            restore.return_value = controller.RestoreOutcome(7, "0.0.0.0", 31111)
            with self.assertRaises(SystemExit) as exit_code:
                snapshot(
                    None,
                    [
                        "restore",
                        "--artifact",
                        str(artifact),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "31111",
                    ],
                )
        self.assertEqual(exit_code.exception.code, 0)
        self.assertEqual(restore.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(restore.call_args.kwargs["port"], 31111)

        for argv in (
            ["restore", "--artifact", "/artifact", "--port", "0"],
            ["restore", "--artifact", "/artifact", "--port", "70000"],
            ["restore", "--artifact", "/artifact", "--timeout", "nan"],
            ["restore", "--artifact", "/snapshot", "--device-map", "value"],
            ["restore", "--artifact", "/snapshot", "--model-path", "value"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                snapshot(None, argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
