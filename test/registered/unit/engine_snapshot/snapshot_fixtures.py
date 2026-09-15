import tempfile
from pathlib import Path

import msgspec

from sglang.srt.engine_snapshot.manifest import (
    MANIFEST_FORMAT,
    SnapshotCanary,
    SnapshotFile,
    SnapshotIdentity,
    SnapshotManifest,
    SnapshotModelFile,
    sha256_file,
)
from sglang.test.ci.ci_register import register_cpu_ci

# This module holds no tests of its own; it only supplies the registry entry
# that every file under test/registered/ must declare.
register_cpu_ci(
    est_time=0,
    suite="base-a-test-cpu",
    disabled="Builders shared by the engine-snapshot unit tests",
)

# Nothing in the suite may depend on a number that is actually in use.
ROOT_PID = 99999991
CANARY_PROMPT = "The first month of the year is"


def identity(**overrides):
    """A complete captured identity; the manifest requires every field."""
    built = SnapshotIdentity(
        sglang_code="code",
        model=[SnapshotModelFile("config.json", 3, 1)],
        host_id="host",
        gpu_name="NVIDIA GPU",
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


def artifact_manifest(artifact_path, **overrides):
    fields = dict(
        format=MANIFEST_FORMAT,
        artifact_path=str(artifact_path),
        created_at="2026-09-15T00:00:00+00:00",
        artifact_bytes=1,
        model_path="/model",
        host="127.0.0.1",
        port=30184,
        identity=identity(),
        root_pid=ROOT_PID,
        pids=[ROOT_PID],
        cuda_pids=[ROOT_PID],
        stdio=["pipe:[99999991]", "pipe:[99999992]"],
        files=[],
        dev_shm=[],
        canary=SnapshotCanary(CANARY_PROMPT, 42),
    )
    fields.update(overrides)
    return SnapshotManifest(**fields)


class SnapshotArtifacts:
    """Mixin giving a test case an owner-only artifact tree.

    Deliberately not a TestCase: this module defines no tests of its own.
    """

    DIRECTORIES = ()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact_path = Path(self.directory.name)
        self.artifact_path.chmod(0o700)
        for directory in self.DIRECTORIES:
            (self.artifact_path / directory).mkdir()

    def manifest(self, **overrides):
        return artifact_manifest(self.artifact_path, **overrides)

    def record(self, name, contents=b"snapshot"):
        path = self.artifact_path / "dev_shm" / name
        path.write_bytes(contents)
        return SnapshotFile(name, sha256_file(path))
