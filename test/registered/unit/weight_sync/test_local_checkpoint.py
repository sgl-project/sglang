import json
import os
import shutil
import struct
import tempfile
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import zstandard
from safetensors.numpy import load_file, save_file

from sglang.srt.weight_sync import local_checkpoint
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _adler32(data: bytes) -> str:
    return f"{zlib.adler32(data, 1):08x}"


def _write_safetensors(path: str, tensors: dict, metadata: dict = None) -> None:
    header, blobs, offset = {}, [], 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def _read_tensors(ckpt_dir: str) -> dict:
    out = {}
    for name, (path, offset, nbytes) in local_checkpoint._tensor_locations(
        ckpt_dir
    ).items():
        with open(path, "rb") as f:
            f.seek(offset)
            out[name] = f.read(nbytes)
    return out


def _write_full_version(version_dir: str, tensors: dict) -> None:
    os.makedirs(version_dir)
    _write_safetensors(os.path.join(version_dir, "model.safetensors"), tensors)


def _write_delta_version(
    version_dir: str, version: int, old: dict, new: dict, encoding: str
) -> None:
    """Publish `new` as a delta against `old` the way the trainer side does."""
    os.makedirs(version_dir)
    compressor = zstandard.ZstdCompressor()
    payloads, checksums = {}, {}
    for name in new:
        o = np.frombuffer(old[name], dtype=np.uint8)
        n = np.frombuffer(new[name], dtype=np.uint8)
        if encoding == "xor":
            raw = (o ^ n).tobytes()
        else:
            positions = np.nonzero(o != n)[0].astype("<u4")
            raw = (
                len(positions).to_bytes(4, "little")
                + positions.tobytes()
                + n[positions].tobytes()
            )
        payloads[name] = compressor.compress(raw)
        checksums[name] = _adler32(new[name])
    _write_safetensors(
        os.path.join(version_dir, "model.safetensors"), payloads, checksums
    )
    with open(os.path.join(version_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {
                "metadata": {
                    "delta_encoding": encoding,
                    "version": str(version),
                    "base_version": str(version - 1),
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {name: "model.safetensors" for name in new},
            },
            f,
        )


class TestLocalCheckpointPull(CustomTestCase):
    V0 = {"a": bytes(range(16)), "b": b"\x10" * 8}
    V1 = {"a": bytes(range(16, 32)), "b": b"\x10" * 8}
    V2 = {"a": bytes(range(16, 32)), "b": b"\x10\x11\x10\x10\x10\x10\x22\x10"}

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.base = os.path.join(self.root, "base")
        self.source = os.path.join(self.root, "published")
        self.local = os.path.join(self.root, "local")
        os.makedirs(self.base)
        os.makedirs(self.source)
        _write_safetensors(os.path.join(self.base, "model.safetensors"), self.V0)
        with open(os.path.join(self.base, "config.json"), "w") as f:
            f.write("{}")

    def _vdir(self, version: int) -> str:
        return local_checkpoint._version_dir(self.source, version)

    def _pull(self, target: int) -> None:
        local_checkpoint.pull(
            local_checkpoint_dir=self.local,
            base_dir=self.base,
            source_dir=self.source,
            target_version=target,
        )

    def test_seeds_from_base_then_applies_xor_delta(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")

        self._pull(1)

        self.assertEqual(_read_tensors(self.local), self.V1)
        self.assertTrue(os.path.exists(os.path.join(self.local, "config.json")))
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 1)

    def test_chains_overwrite_delta_and_repeated_pull_is_a_noop(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_delta_version(self._vdir(2), 2, self.V1, self.V2, "overwrite")

        self._pull(1)
        self._pull(2)
        self._pull(2)

        self.assertEqual(_read_tensors(self.local), self.V2)
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 2)

    def test_fresh_host_seeds_from_newest_full_version_and_prunes_stale_files(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_full_version(self._vdir(2), self.V2)
        os.makedirs(self.local)
        with open(os.path.join(self.local, "stale.safetensors"), "w") as f:
            f.write("x")

        self._pull(2)

        self.assertEqual(_read_tensors(self.local), self.V2)
        self.assertFalse(os.path.exists(os.path.join(self.local, "stale.safetensors")))
        # seeded from v2 directly, so base-only files never came along
        self.assertFalse(os.path.exists(os.path.join(self.local, "config.json")))

    def test_a_failed_xor_apply_reseeds_on_the_next_pull(self):
        """xor is not idempotent, so retrying over half-patched bytes must reseed instead."""
        # publish a v1 whose payload was computed against the wrong base
        wrong_base = {"a": bytes(16), "b": self.V0["b"]}
        _write_delta_version(self._vdir(1), 1, wrong_base, self.V1, "xor")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self._pull(1)
        self.assertIsNone(local_checkpoint._read_applied_version(self.local))

        shutil.rmtree(self._vdir(1))
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        self._pull(1)

        self.assertEqual(_read_tensors(self.local), self.V1)
        self.assertEqual(local_checkpoint._read_applied_version(self.local), 1)

    def test_pulling_a_version_below_the_local_state_raises(self):
        """A pull must never report success while the host serves a different version."""
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_delta_version(self._vdir(2), 2, self.V1, self.V2, "xor")
        self._pull(2)

        with self.assertRaisesRegex(RuntimeError, "past the requested v1"):
            self._pull(1)
        self.assertEqual(_read_tensors(self.local), self.V2)

    def test_snapshot_reader_keeps_the_same_shards_across_the_next_pull(self):
        # A loader opens different shards at different times. Pause between two
        # opens while a later full pull replaces the active checkpoint.
        for version in (1, 2):
            directory = Path(self._vdir(version))
            directory.mkdir()
            (directory / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {name: f"{name}.safetensors" for name in ("a", "b")}}
                )
            )
            for name in ("a", "b"):
                save_file(
                    {name: np.array([version], dtype=np.uint8)},
                    directory / f"{name}.safetensors",
                )

        def snapshot(version):
            local_checkpoint.pull(
                self.local, self.base, self.source, version, snapshot=True
            )
            return Path(local_checkpoint.snapshot_path(self.local, version))

        first = snapshot(1)
        opened, resume = Event(), Event()

        def read_across_pull():
            a = load_file(first / "a.safetensors")["a"].item()
            opened.set()
            assert resume.wait(10)
            return a, load_file(first / "b.safetensors")["b"].item()

        with ThreadPoolExecutor(1) as executor:
            reader = executor.submit(read_across_pull)
            try:
                self.assertTrue(opened.wait(10))
                self._pull(2)
                second = snapshot(2)
            finally:
                resume.set()
            a, b = reader.result()
        self.assertEqual((a, b), (1, 1))
        self.assertEqual(_read_tensors(second), {"a": b"\x02", "b": b"\x02"})
        self.assertEqual(snapshot(1), first)

    def test_relative_snapshot_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute local_checkpoint_dir"):
            local_checkpoint.snapshot_path("relative", 1)

    def test_incomplete_full_publication_can_be_completed_and_retried(self):
        directory = Path(self._vdir(1))
        directory.mkdir()
        final = Path(local_checkpoint.snapshot_path(self.local, 1))

        def snapshot():
            local_checkpoint.pull(self.local, self.base, self.source, 1, snapshot=True)

        with self.assertRaisesRegex(
            ValueError, "checkpoint index or model.safetensors"
        ):
            snapshot()
        self.assertFalse(final.exists())

        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}})
        )
        save_file({"a": np.array([1], dtype=np.uint8)}, directory / "a.safetensors")
        # A legacy in-place pull may already have marked this incomplete version applied.
        self._pull(1)
        with self.assertRaisesRegex(ValueError, "Incomplete snapshot seed"):
            snapshot()
        self.assertFalse(final.exists())
        save_file({"b": np.array([2], dtype=np.uint8)}, directory / "b.safetensors")
        snapshot()
        self.assertEqual(_read_tensors(final), {"a": b"\x01", "b": b"\x02"})

    def test_missing_delta_payload_does_not_publish_a_snapshot(self):
        directory = Path(self._vdir(1))
        _write_delta_version(directory, 1, self.V0, self.V1, "xor")
        shard = directory / "model.safetensors"
        payload = shard.read_bytes()
        shard.unlink()
        final = Path(local_checkpoint.snapshot_path(self.local, 1))
        with self.assertRaisesRegex(ValueError, "Incomplete snapshot delta"):
            local_checkpoint.pull(self.local, self.base, self.source, 1, snapshot=True)
        self.assertFalse(final.exists())
        shard.write_bytes(payload)
        local_checkpoint.pull(self.local, self.base, self.source, 1, snapshot=True)
        self.assertEqual(_read_tensors(final), self.V1)

    def test_failed_snapshot_copy_is_not_published_and_can_retry(self):
        self._pull(0)
        final = Path(local_checkpoint.snapshot_path(self.local, 0))
        copy = shutil.copy2

        def fail_copy(src, dst):
            copy(src, dst)
            self.assertFalse(final.exists())
            raise OSError("copy interrupted")

        with patch.object(shutil, "copy2", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                local_checkpoint.pull(
                    self.local, self.base, self.source, 0, snapshot=True
                )
        self.assertFalse(final.exists())
        # Also recover leftover private staging from a terminated process.
        scratch = final.parent / ".staging"
        scratch.mkdir()
        (scratch / "stale").touch()
        local_checkpoint.pull(self.local, self.base, self.source, 0, snapshot=True)
        self.assertEqual(_read_tensors(final), self.V0)
        self.assertFalse((final / "stale").exists())

    def test_out_of_order_delta_raises(self):
        _write_delta_version(self._vdir(1), 1, self.V0, self.V1, "xor")
        _write_delta_version(self._vdir(2), 2, self.V1, self.V2, "xor")
        self._pull(1)
        # v2 now claims to build on a version the host never applied
        index = os.path.join(self._vdir(2), "model.safetensors.index.json")
        with open(index) as f:
            meta = json.load(f)
        meta["metadata"]["base_version"] = "5"
        with open(index, "w") as f:
            json.dump(meta, f)

        with self.assertRaisesRegex(RuntimeError, "out-of-order delta"):
            self._pull(2)


if __name__ == "__main__":
    unittest.main()
