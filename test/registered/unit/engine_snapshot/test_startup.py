import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
from snapshot_fixtures import SnapshotArtifacts, artifact_manifest

from sglang.srt.engine_snapshot import control, startup
from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotRuntimeFailure,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import (
    SnapshotCanary,
    publish_manifest,
    write_json_atomic,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSnapshotStartup(SnapshotArtifacts, CustomTestCase):
    DIRECTORIES = ("control",)

    def setUp(self):
        super().setUp()
        self.control = self.artifact_path / "control"

    def args(self, **overrides):
        fields = dict(
            tp_size=1,
            pp_size=1,
            dp_size=1,
            nnodes=1,
            tokenizer_worker_num=1,
            detokenizer_worker_num=1,
            disaggregation_mode="null",
            weight_cache_mode="off",
            speculative_algorithm=None,
            quantization=None,
            enable_memory_saver=True,
            enable_lora=False,
            enable_hierarchical_cache=False,
            use_ray=False,
            encoder_only=False,
            smg_grpc_mode=False,
            grpc_port=None,
            model_path=str(self.artifact_path),
            load_format="auto",
            device="cuda",
            base_gpu_id=0,
            ssl_keyfile=None,
            ssl_certfile=None,
            port=30184,
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    @staticmethod
    def canary(token_id=42, logprob=-0.5):
        return SnapshotCanary(startup.CANARY_PROMPT, token_id, logprob)

    def write_manifest(self, token_id=42):
        publish_manifest(
            self.artifact_path,
            artifact_manifest(self.artifact_path, canary=self.canary(token_id)),
        )

    def barrier_scheduler(self):
        scheduler = Mock()
        scheduler.server_args = SimpleNamespace(model_path="/model", load_format="auto")
        scheduler.device_module.device_count.return_value = 1
        scheduler.device_module.get_device_properties.return_value.uuid = "1"
        scheduler.weight_updater.update_weights_from_disk.return_value = (
            SimpleNamespace(success=True)
        )
        return scheduler

    # ------------------------------------------------------------------ #
    # configuration validation
    # ------------------------------------------------------------------ #
    def test_validate_startup_accepts_supported_configurations(self):
        startup.validate_startup(self.args())

        resolved = self.args(device=None)
        resolved._resolved_overrides = [("device", {"device": "cuda"})]
        startup.validate_startup(resolved)
        startup.validate_startup(self.args(enable_lora=None))

    def test_validate_startup_rejects_unsupported_configurations(self):
        cases = (
            ({"tp_size": 2}, "tp_size"),
            ({"pp_size": 2}, "pp_size"),
            ({"dp_size": 2}, "dp_size"),
            ({"nnodes": 2}, "nnodes"),
            ({"enable_memory_saver": False}, "enable_memory_saver"),
            ({"use_ray": True}, "use_ray"),
            ({"weight_cache_mode": "daemon"}, "weight_cache_mode"),
            ({"quantization": "fp8"}, "quantization"),
            ({"encoder_only": True}, "encoder_only"),
            ({"load_format": "dummy"}, "load_format"),
            ({"device": "cpu"}, "CUDA"),
            ({"base_gpu_id": 1}, "CUDA"),
            ({"port": 0}, "listen port"),
            ({"port": 70000}, "listen port"),
            ({"ssl_certfile": "/cert.pem"}, "plain HTTP"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(SnapshotUsageError, expected):
                    startup.validate_startup(self.args(**overrides))

    # ------------------------------------------------------------------ #
    # barriers
    # ------------------------------------------------------------------ #
    def test_active_artifact_path_needs_the_engine_marker(self):
        with patch.dict(os.environ, {"SGLANG_SNAPSHOT_DIR": str(self.artifact_path)}):
            os.environ.pop("SGLANG_SNAPSHOT_ENGINE", None)
            self.assertIsNone(startup.active_artifact_path())
            os.environ["SGLANG_SNAPSHOT_ENGINE"] = "1"
            self.assertEqual(startup.active_artifact_path(), str(self.artifact_path))

    def test_scheduler_barrier_rehearses_then_parks_released(self):
        scheduler = self.barrier_scheduler()
        self.write_manifest()
        observed = {}

        def note_park(control_dir, name, type_, timeout_seconds=None):
            observed["scheduler_json"] = json.loads(
                (self.control / control.SCHEDULER).read_text()
            )
            observed["resumed"] = (self.control / control.RESUMED).exists()
            observed["calls"] = (
                scheduler.weight_updater.release_memory_occupation.call_count,
                scheduler.weight_updater.resume_memory_occupation.call_count,
                scheduler.weight_updater.update_weights_from_disk.call_count,
            )
            return control.ReleaseInfo()

        with (
            patch.object(startup.control, "wait_and_read", side_effect=note_park),
            patch.object(startup, "_run_canary", return_value=self.canary()),
        ):
            startup.scheduler_barrier(scheduler, str(self.artifact_path))

        self.assertEqual(observed["scheduler_json"]["gpu_uuid"], "GPU-1")
        self.assertEqual(
            observed["scheduler_json"]["canary"],
            {"prompt": startup.CANARY_PROMPT, "token_id": 42, "logprob": -0.5},
        )
        # The rehearsal already released, reloaded and re-verified, so the park
        # happens released with one more release than resume behind it.
        self.assertEqual(observed["calls"], (2, 1, 1))
        self.assertFalse(observed["resumed"])
        self.assertEqual(
            json.loads((self.control / control.RESUMED).read_text()),
            {"token_id": 42, "logprob": -0.5},
        )
        request = scheduler.weight_updater.update_weights_from_disk.call_args.args[0]
        self.assertEqual((request.model_path, request.load_format), ("/model", "auto"))

    def test_scheduler_barrier_applies_the_release_address(self):
        scheduler = self.barrier_scheduler()
        self.write_manifest()
        release = control.ReleaseInfo(host="0.0.0.0", port=31111)

        with (
            patch.object(startup, "_run_canary", return_value=self.canary()),
            patch.object(startup.control, "wait_and_read", return_value=release),
            patch("sglang.srt.runtime_context.get_context") as context,
        ):
            startup.scheduler_barrier(scheduler, str(self.artifact_path))

        context.return_value.override.assert_called_once_with(
            "snapshot-restore", host="0.0.0.0", port=31111
        )

    def test_scheduler_barrier_reports_failures(self):
        canary = self.canary()
        cases = (
            ("abort", False, (canary, canary), SnapshotRuntimeFailure, "aborted"),
            ("reload", True, (canary, canary), SnapshotRuntimeFailure, "bad weights"),
            (
                "canary",
                True,
                (canary, self.canary(token_id=7)),
                SnapshotRuntimeFailure,
                "canary mismatch",
            ),
            (
                "missing manifest",
                False,
                (canary, canary),
                SnapshotCompatibilityError,
                "manifest",
            ),
        )
        for name, with_manifest, runs, error, expected in cases:
            with self.subTest(case=name):
                for marker in (
                    control.RESUMED,
                    control.ERROR,
                    control.ABORT,
                    control.SCHEDULER,
                    control.READY,
                    control.RELEASE,
                ):
                    (self.control / marker).unlink(missing_ok=True)
                (self.artifact_path / "manifest.json").unlink(missing_ok=True)
                if with_manifest:
                    self.write_manifest()
                if name == "abort":
                    control.write_abort(self.control)
                else:
                    # The engine parks on the release marker before it resumes.
                    control.write_release(self.control)

                scheduler = self.barrier_scheduler()
                if name == "reload":
                    scheduler.weight_updater.update_weights_from_disk.return_value = (
                        SimpleNamespace(success=False, message="bad weights")
                    )
                with patch.object(startup, "_run_canary", side_effect=list(runs)):
                    with self.assertRaisesRegex(error, expected):
                        startup.scheduler_barrier(scheduler, str(self.artifact_path))

                self.assertFalse((self.control / control.RESUMED).exists())
                # The engine-side failure reaches the controller as a message.
                self.assertIn(expected, (self.control / control.ERROR).read_text())

    def test_scheduler_barrier_rejects_multiple_devices_before_release(self):
        scheduler = Mock()
        scheduler.device_module.device_count.return_value = 2
        with (
            patch.object(startup, "_run_canary") as forward,
            self.assertRaisesRegex(SnapshotUsageError, "one visible"),
        ):
            startup.scheduler_barrier(scheduler, str(self.artifact_path))
        scheduler.weight_updater.release_memory_occupation.assert_not_called()
        forward.assert_not_called()

    def test_server_barrier_publishes_and_applies_overrides(self):
        write_json_atomic(
            self.control / control.SCHEDULER,
            msgspec.to_builtins(
                control.SchedulerInfo(gpu_uuid="GPU-1", canary=self.canary())
            ),
        )
        args = SimpleNamespace(model_path="/model", host="127.0.0.1", port=30184)

        with patch("sglang.srt.runtime_context.get_context") as context:
            control.write_release(self.control)
            startup.server_barrier(args, str(self.artifact_path))
        self.assertEqual(
            json.loads((self.control / control.READY).read_text()),
            {
                "gpu_uuid": "GPU-1",
                "model_path": "/model",
                "host": "127.0.0.1",
                "port": 30184,
            },
        )
        context.return_value.override.assert_not_called()

        for release in (
            {"host": "0.0.0.0"},
            {"port": 31111},
            {"host": "0.0.0.0", "port": 31111},
        ):
            with self.subTest(release=release):
                (self.control / control.READY).unlink()
                control.write_release(self.control, **release)
                with patch("sglang.srt.runtime_context.get_context") as context:
                    startup.server_barrier(args, str(self.artifact_path))
                context.return_value.override.assert_called_once_with(
                    "snapshot-restore", **release
                )


# The canary's forward pass (PrefillAdder admission, KV release, a real GPU
# step) is exercised end to end by `sglang snapshot create`, which also fails
# when the rehearsal cannot reproduce the canary; the mismatch path is covered
# above.


if __name__ == "__main__":
    unittest.main()
