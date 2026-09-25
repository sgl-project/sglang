import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import msgspec
from snapshot_fixtures import (
    ROOT_PID,
    SnapshotArtifacts,
    artifact_manifest,
    identity,
)

from sglang.srt.engine_snapshot import manifest
from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotSecurityError,
    SnapshotUsageError,
)
from sglang.srt.engine_snapshot.manifest import (
    SnapshotFile,
    load_manifest,
    locked,
    publish_manifest,
    record_failure,
    resolve_artifact_path,
    validate_identity,
    write_json_atomic,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSnapshotManifest(SnapshotArtifacts, CustomTestCase):
    DIRECTORIES = ("control", "files", "runtime", "images", "work")

    def manifest(self, **overrides):
        fields = dict(
            artifact_bytes=1024,
            pids=[ROOT_PID, ROOT_PID + 1],
            cuda_pids=[ROOT_PID + 1],
            files=[SnapshotFile("cache/kernel.bin", "a" * 64)],
        )
        fields.update(overrides)
        return artifact_manifest(
            fields.pop("artifact_path", self.artifact_path), **fields
        )

    def write(self, manifest_obj):
        write_json_atomic(
            self.artifact_path / manifest.MANIFEST_NAME,
            msgspec.to_builtins(manifest_obj),
            overwrite=True,
        )

    def test_publish_round_trip_and_owner_only_files(self):
        manifest_obj = self.manifest()
        publish_manifest(self.artifact_path, manifest_obj)
        loaded = load_manifest(self.artifact_path, identity=manifest_obj.identity)
        self.assertEqual(loaded, manifest_obj)
        self.assertEqual(loaded.canary.token_id, 42)

        # manifest.json is the completion marker and is written exactly once.
        with self.assertRaises(SnapshotUsageError):
            publish_manifest(self.artifact_path, manifest_obj)

        old_umask = os.umask(0o022)
        try:
            write_json_atomic(
                self.artifact_path / "control/ready.json", {"ready": True}
            )
        finally:
            os.umask(old_umask)
        self.assertEqual(
            (self.artifact_path / "control/ready.json").stat().st_mode & 0o777, 0o600
        )

        # A predictable temporary name must not be followed through a symlink.
        outside = self.artifact_path / "outside"
        outside.write_text("keep")
        (self.artifact_path / "control/other.tmp").symlink_to(outside)
        write_json_atomic(self.artifact_path / "control/other.json", {"ready": True})
        self.assertEqual(outside.read_text(), "keep")

    def test_rejects_malformed_or_inconsistent_manifests(self):
        for name, overrides in (
            ("format", {"format": 2}),
            ("artifact_path", {"artifact_path": "/elsewhere"}),
            ("root_pid", {"root_pid": 1}),
            ("duplicate pids", {"pids": [ROOT_PID, ROOT_PID]}),
            ("cuda outside tree", {"cuda_pids": [1]}),
            ("no cuda pids", {"cuda_pids": []}),
        ):
            with self.subTest(case=name):
                self.write(self.manifest(**overrides))
                with self.assertRaises(SnapshotCompatibilityError):
                    load_manifest(self.artifact_path)

        # A required or unknown field fails at parse time rather than being
        # defaulted, so an artifact from another format never loads.
        for name, edit in (
            ("missing file", None),
            ("missing canary", lambda data: data.pop("canary")),
            ("unknown field", lambda data: data.update(device_map="GPU-1=GPU-2")),
        ):
            with self.subTest(case=name):
                (self.artifact_path / manifest.MANIFEST_NAME).unlink(missing_ok=True)
                if edit is not None:
                    data = msgspec.to_builtins(self.manifest())
                    edit(data)
                    write_json_atomic(
                        self.artifact_path / manifest.MANIFEST_NAME,
                        data,
                        overwrite=True,
                    )
                with self.assertRaises(SnapshotCompatibilityError):
                    load_manifest(self.artifact_path)

    def test_identity_mismatch_names_the_fields(self):
        self.write(self.manifest())
        with self.assertRaises(SnapshotCompatibilityError) as error:
            load_manifest(
                self.artifact_path, identity=identity(gpu_uuid="GPU-2", criu="4.3")
            )
        self.assertIn("gpu_uuid", str(error.exception))
        self.assertIn("criu", str(error.exception))

        # Without an identity to compare against the artifact loads, so inspect
        # can report the difference instead of failing on it.
        loaded = load_manifest(self.artifact_path)
        with self.assertRaises(SnapshotCompatibilityError):
            validate_identity(loaded.identity, identity(gpu_uuid="GPU-2"))

    def test_environment_identity_hashes_values_and_skips_secrets(self):
        entries = manifest._environment_entries(
            {
                "SGLANG_HOST_IP": "127.0.0.1",
                "SGLANG_API_KEY": "secret",
                "SGLANG_SNAPSHOT_DIR": "/artifact",
                "CUDA_VISIBLE_DEVICES": "4",
                "HF_HOME": "/hf",
                "PATH": "/usr/bin",
            }
        )
        self.assertEqual(
            [entry.name for entry in entries],
            ["CUDA_VISIBLE_DEVICES", "SGLANG_HOST_IP"],
        )
        self.assertTrue(all(len(entry.digest) == 64 for entry in entries))
        self.assertNotIn("127.0.0.1", json.dumps(msgspec.to_builtins(entries)))

    def test_build_identity_describes_the_model_and_rejects_a_missing_one(self):
        captured = dict(
            gpu_uuid="GPU-1",
            gpu_name="NVIDIA GPU",
            driver_version="580.1",
            criu_version="Version: 4.2",
            cuda_checkpoint_sha256="abc",
        )
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            with (
                patch.object(manifest, "_code_digest", return_value="code"),
                patch.object(manifest, "_host_id", return_value="host"),
                patch.object(manifest, "_package_version", return_value="1.0"),
                patch.object(manifest, "_cuda_runtime", return_value="13.0"),
            ):
                built = manifest.build_identity(
                    model_path=str(model), environment={"SGLANG_A": "1"}, **captured
                )
            self.assertEqual([entry.path for entry in built.model], ["config.json"])
            self.assertEqual(built.criu, "Version: 4.2")

            with patch.object(manifest, "_code_digest", return_value="code"):
                with self.assertRaisesRegex(
                    SnapshotCompatibilityError, "model files are missing"
                ):
                    manifest.build_identity(model_path=tempfile.mkdtemp(), **captured)

    def test_artifact_path_security_and_lock(self):
        for path in ("../outside", "/etc/passwd", "a/../../outside", ""):
            with self.subTest(path=path), self.assertRaises(SnapshotSecurityError):
                resolve_artifact_path(self.artifact_path, path)
        (self.artifact_path / "link").symlink_to("/tmp")
        with self.assertRaises(SnapshotSecurityError):
            resolve_artifact_path(self.artifact_path, "link/file")

        with locked(self.artifact_path):
            with self.assertRaisesRegex(SnapshotUsageError, "already running"):
                with locked(self.artifact_path):
                    self.fail("concurrent operation acquired the artifact")

        for mode, expected in ((0o777, "writable ancestor"), (0o755, "owner-only")):
            with self.subTest(mode=oct(mode)):
                self.artifact_path.chmod(mode)
                with self.assertRaisesRegex(SnapshotSecurityError, expected):
                    with locked(self.artifact_path):
                        self.fail("untrusted directory accepted")
        self.artifact_path.chmod(0o700)

        untrusted = self.artifact_path / "untrusted"
        untrusted.mkdir(mode=0o777)
        untrusted.chmod(0o777)
        (untrusted / "artifact").mkdir(mode=0o700)
        with self.assertRaises(SnapshotSecurityError), locked(untrusted / "artifact"):
            pass

        alias = self.artifact_path / "alias"
        alias.symlink_to(self.artifact_path / "runtime", target_is_directory=True)
        (self.artifact_path / "runtime/artifact").mkdir(mode=0o700)
        with self.assertRaises(SnapshotSecurityError), locked(alias / "artifact"):
            pass

    def test_record_failure_explains_an_artifact(self):
        (self.artifact_path / "files/data").write_bytes(b"x" * 10)
        self.assertEqual(manifest.artifact_bytes(self.artifact_path), 10)
        record_failure(
            self.artifact_path, SnapshotUsageError("engine died"), ["sweep: nope"]
        )
        payload = json.loads((self.artifact_path / manifest.FAILURE_NAME).read_text())
        self.assertEqual(payload["error"], "engine died")
        self.assertEqual(payload["cleanup"], ["sweep: nope"])
        record_failure(self.artifact_path / "gone", SnapshotUsageError("x"))


if __name__ == "__main__":
    unittest.main()
