import unittest
from unittest.mock import Mock, patch

import msgspec
from snapshot_fixtures import SnapshotArtifacts, artifact_manifest, identity

import sglang.srt.engine_snapshot.controller as controller
from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotRuntimeFailure,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import (
    publish_manifest,
    write_json_atomic,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Every cache root the controller must own for its child.
CACHE_VARIABLES = (
    "SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR"
    " CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE DG_JIT_CACHE_DIR SGLANG_DG_CACHE_DIR"
    " TILELANG_CACHE_DIR HUMMING_CACHE_DIR HUMMING_TMP_DIR CUTE_DSL_CACHE_DIR"
    " FLASH_ATTENTION_CUTE_DSL_CACHE_DIR TMPDIR"
).split()


class TestSnapshotController(SnapshotArtifacts, CustomTestCase):
    DIRECTORIES = ("shm",)

    def mock_runtime(self):
        runtime = Mock()
        runtime.SHM_DIR = self.artifact_path / "shm"
        runtime.current_identity.return_value = identity()
        runtime.stdio_resources.return_value = ["pipe:[1]", "pipe:[2]"]
        return runtime

    def manifest(self, **overrides):
        return artifact_manifest(self.artifact_path / "artifact", **overrides)

    def engine_info(self):
        return control.EngineInfo(
            gpu_uuid="GPU-1", model_path="/model", host="127.0.0.1", port=30184
        )

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

    def test_restore_judges_the_engines_canary_report(self):
        artifact = self.publish()
        resumed = artifact / "control" / control.RESUMED

        def report(token_id=42, logprob=-0.5):
            write_json_atomic(
                resumed,
                msgspec.to_builtins(
                    control.ResumedInfo(token_id=token_id, logprob=logprob)
                ),
                overwrite=True,
            )

        runtime = self.mock_runtime()
        runtime.restore.return_value = 99999991
        runtime.wait_listener.side_effect = lambda *arguments, **kwargs: report()

        outcome = controller.restore_snapshot(str(artifact), runtime=runtime)
        self.assertEqual(
            (outcome.root_pid, outcome.host, outcome.port),
            (99999991, "127.0.0.1", 30184),
        )
        runtime.complete_restore.assert_called_once_with(99999991)

        # A reload that shifted the logits is rejected here even though the
        # token id still matches, and the half-restored tree is torn down.
        runtime.wait_listener.side_effect = lambda *arguments, **kwargs: report(
            logprob=-2.0
        )
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "canary mismatch"):
            controller.restore_snapshot(str(artifact), runtime=runtime)
        runtime.stop_restored_tree.assert_called()

    def test_restore_rejects_an_ipv6_host(self):
        artifact = self.publish()
        runtime = self.mock_runtime()
        with self.assertRaisesRegex(SnapshotUsageError, "IPv6"):
            controller.restore_snapshot(str(artifact), runtime=runtime, host="::1")
        runtime.restore.assert_not_called()

    # ------------------------------------------------------------------ #
    # inspect and CLI
    # ------------------------------------------------------------------ #
    def test_inspect_reports_identity_and_resources(self):
        from sglang.cli.snapshot import snapshot

        artifact = self.publish()
        runtime = self.mock_runtime()
        runtime.occupied_pids.return_value = []

        manifest, checks = controller.inspect_snapshot(str(artifact), runtime=runtime)
        self.assertEqual(manifest.root_pid, 99999991)
        self.assertEqual(
            checks,
            {
                "identity": "match",
                "occupied_pids": [],
                "listen_address": "free",
            },
        )

        runtime.current_identity.return_value = identity(criu="Version: 4.3")
        runtime.occupied_pids.return_value = [99999991]
        runtime.check_port_free.side_effect = SnapshotUsageError("address in use")
        _, checks = controller.inspect_snapshot(str(artifact), runtime=runtime)
        self.assertIn("criu", checks["identity"])
        self.assertEqual(checks["occupied_pids"], [99999991])
        self.assertEqual(checks["listen_address"], "address in use")

        # The CLI exits 1 on a mismatch, so a deployment can gate on it.
        runtime.current_identity.return_value = identity(gpu_uuid="GPU-2")
        runtime.occupied_pids.return_value = []
        runtime.check_port_free.side_effect = None
        with self.assertRaises(SystemExit) as exit_code:
            snapshot(None, ["inspect", "--artifact", str(artifact)])
        self.assertEqual(exit_code.exception.code, 1)

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
