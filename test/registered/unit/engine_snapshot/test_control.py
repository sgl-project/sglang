import json
import unittest

import msgspec
from snapshot_fixtures import SnapshotArtifacts

from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import SnapshotRuntimeFailure
from sglang.srt.engine_snapshot.manifest import write_json_atomic
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestControlProtocol(SnapshotArtifacts, CustomTestCase):
    DIRECTORIES = ("control",)

    def setUp(self):
        super().setUp()
        self.control = self.artifact_path / "control"

    def test_wait_for_returns_when_the_marker_appears(self):
        (self.control / control.RELEASE).touch()
        control.wait_for(self.control, control.RELEASE, timeout_seconds=1)

    def test_wait_for_stops_on_abort_or_engine_failure(self):
        # The engine reports its own failure through error.json; the waiter must
        # surface that message rather than a timeout or a JSON envelope.
        cases = (
            ("abort", lambda: control.write_abort(self.control), "aborted"),
            (
                "error",
                lambda: control.write_error(
                    self.control, SnapshotRuntimeFailure("reload failed")
                ),
                "^reload failed$",
            ),
        )
        for name, prepare, expected in cases:
            with self.subTest(case=name):
                control.clear_handshake(self.control)
                prepare()
                with self.assertRaisesRegex(SnapshotRuntimeFailure, expected):
                    control.wait_for(self.control, control.RELEASE, timeout_seconds=30)

    def test_wait_for_times_out(self):
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "within 0.1s"):
            control.wait_for(self.control, control.RELEASE, timeout_seconds=0.1)

    def test_release_round_trip_and_handshake_reset(self):
        write_json_atomic(
            self.control / control.SCHEDULER,
            msgspec.to_builtins(control.SchedulerInfo(gpu_uuid="GPU-1")),
        )
        info = control.wait_and_read(
            self.control, control.SCHEDULER, control.SchedulerInfo, timeout_seconds=1
        )
        self.assertEqual(info.gpu_uuid, "GPU-1")

        control.write_release(self.control, host="0.0.0.0", port=31111)
        release = control.read_json(self.control, control.RELEASE, control.ReleaseInfo)
        self.assertEqual((release.host, release.port), ("0.0.0.0", 31111))

        # Releasing without an override keeps the captured listen address.
        control.write_release(self.control)
        release = control.read_json(self.control, control.RELEASE, control.ReleaseInfo)
        self.assertIsNone(release.host)
        self.assertIsNone(release.port)

        (self.control / control.RESUMED).touch()
        self.assertIsNone(control.read_error(self.control))
        control.clear_handshake(self.control)
        for marker in (control.RELEASE, control.ABORT, control.RESUMED, control.ERROR):
            self.assertFalse((self.control / marker).exists(), marker)

    def test_read_json_and_read_error_report_bad_input(self):
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "did not write"):
            control.read_json(self.control, control.READY, control.EngineInfo)

        write_json_atomic(
            self.control / control.READY, {"gpu_uuid": "GPU-1"}, overwrite=True
        )
        with self.assertRaisesRegex(SnapshotRuntimeFailure, "invalid ready.json"):
            control.read_json(self.control, control.READY, control.EngineInfo)

        (self.control / control.ERROR).write_text("{")
        self.assertIn("unreadable", control.read_error(self.control))
        (self.control / control.ERROR).write_text(json.dumps({"detail": "x"}))
        self.assertIn("malformed", control.read_error(self.control))


if __name__ == "__main__":
    unittest.main()
