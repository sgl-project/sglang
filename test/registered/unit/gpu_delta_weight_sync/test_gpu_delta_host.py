"""Shared publication bytes, host-scoped deduplication, and failure lifetimes."""

import hashlib
import json
import multiprocessing
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard as zstd
from sglang.srt.weight_sync import gpu_delta_host as host
from sglang.srt.weight_sync.gpu_delta_payload import OuterZstdPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture(directory):
    expected = {
        "dense": bytes(range(251)) * 100,
        "expert": b"snappy-bytes" * 50,
        "raw": b"abcd",
    }
    blob, tensors = bytearray(), []
    for name, data in expected.items():
        blob.extend(bytes((-len(blob)) % 16))
        start = len(blob)
        encoded = (
            data
            if name == "raw"
            else zstd.ZstdCompressor(write_checksum=True).compress(data)
        )
        blob.extend(encoded)
        entry = {
            "name": name,
            "encoding": "raw_bytes" if name == "raw" else "xor_bytes",
            "nbytes": len(data),
            "changed_bytes": len(data),
            "frames": [] if name == "raw" else [{}],
        }
        if name == "raw":
            entry["raw"] = {
                "file": "owner.bin",
                "encoded_offset": start,
                "encoded_bytes": len(data),
            }
        else:
            entry["outer"] = {
                "file": "owner.bin",
                "encoded_offset": start,
                "encoded_bytes": len(encoded),
                "decoded_bytes": len(data),
                "frames": [
                    {
                        "encoded_offset": 0,
                        "encoded_bytes": len(encoded),
                        "decoded_offset": 0,
                        "decoded_bytes": len(data),
                    }
                ],
            }
        tensors.append(entry)
    # A foreign EP tensor's compressed contents are intentionally invalid. It is
    # authenticated with the owner file but must not be decoded on this host.
    blob.extend(b"foreign-not-zstd")
    tensors.append(
        {
            "name": "foreign",
            "encoding": "xor_bytes",
            "nbytes": 10,
            "frames": [{}],
            "outer": {
                "file": "owner.bin",
                "encoded_offset": len(blob) - 16,
                "encoded_bytes": 16,
                "decoded_bytes": 10,
                "frames": [],
            },
        }
    )
    (directory / "owner.bin").write_bytes(blob)
    manifest = {
        "files": [
            {
                "name": "owner.bin",
                "nbytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        ],
        "tensors": tensors,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), manifest, expected


def fake_fallocate(fd, offset, length):
    os.ftruncate(fd, offset + length)


def _child(root, path, digest, manifest, names, barrier, output):
    pool = OuterZstdPool(2)
    try:
        with patch.object(host, "_cache_root", return_value=Path(root)):
            identity = host.host_cache_id()
            barrier.wait(timeout=10)
            metrics = {}
            snapshot = host.HostDecodedSnapshot(
                path, digest, manifest, names, pool, metrics
            )
            values = {
                name: bytes(
                    snapshot.mapping[row["offset"] : row["offset"] + row["nbytes"]]
                )
                for name, row in snapshot.index["tensors"].items()
            }
            output.put((identity, metrics, values, snapshot.index["arena_identity"]))
            snapshot.close()  # Closing before successful global resume retains it.
    finally:
        pool.close()


class TestSharedHostSnapshot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.root_patch = patch.object(host, "_cache_root", return_value=self.cache)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.fallocate = patch.object(
            os, "posix_fallocate", side_effect=fake_fallocate, create=True
        )
        self.fallocate.start()
        self.addCleanup(self.fallocate.stop)

    def test_two_engine_processes_verify_and_decode_once_with_retained_shared_bytes(
        self,
    ):
        path, digest, manifest, expected = fixture(self.root)
        context = multiprocessing.get_context("fork")
        barrier, output = context.Barrier(2), context.Queue()
        children = [
            context.Process(
                target=_child,
                args=(
                    str(self.cache),
                    path,
                    digest,
                    manifest,
                    sorted(expected),
                    barrier,
                    output,
                ),
            )
            for _ in range(2)
        ]
        for child in children:
            child.start()
        records = [output.get(timeout=15) for _ in children]
        for child in children:
            child.join(timeout=15)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(len({row[0] for row in records}), 1)
        self.assertEqual(records[0][3], records[1][3])
        self.assertEqual(
            sum(row[1]["host_payload_cache_created"] for row in records), 1
        )
        self.assertEqual(sum(row[1]["host_payload_cache_reused"] for row in records), 1)
        self.assertEqual(sum(row[1]["host_payload_hash_files"] for row in records), 1)
        self.assertEqual(sum(row[1]["host_outer_zstd_tensors"] for row in records), 2)
        self.assertTrue(all(row[2] == expected for row in records))
        # Reuse consumes retained verified bytes, never rereads mutable originals.
        (self.root / "owner.bin").write_bytes(b"corrupted original")
        pool = OuterZstdPool(2)
        self.addCleanup(pool.close)
        first = host.HostDecodedSnapshot(
            path, digest, manifest, sorted(expected), pool, {}
        )
        second = host.HostDecodedSnapshot(
            path, digest, manifest, sorted(expected), pool, {}
        )
        first.close(discard=True)
        self.assertFalse(first.directory.exists())
        dense = second.index["tensors"]["dense"]
        self.assertEqual(
            bytes(second.mapping[dense["offset"] : dense["offset"] + dense["nbytes"]]),
            expected["dense"],
        )
        second.close(discard=True)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            host.HostDecodedSnapshot(path, digest, manifest, sorted(expected), pool, {})

    def test_failed_tensor_waits_for_other_decode_before_returning(self):
        path, digest, manifest, _ = fixture(self.root)
        pool = OuterZstdPool(2)
        self.addCleanup(pool.close)
        slow_entered, failure_raised, release_slow, finished = [
            threading.Event() for _ in range(4)
        ]
        errors = []
        decode = pool.decode

        def controlled(payload, chunks, destination):
            if len(destination) > 1000:
                assert slow_entered.wait(5)
                failure_raised.set()
                raise ValueError("injected tensor decode failure")
            slow_entered.set()
            assert release_slow.wait(5)
            return decode(payload, chunks, destination)

        def build():
            try:
                host.HostDecodedSnapshot(
                    path, digest, manifest, ["dense", "expert"], pool, {}
                )
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        with patch.object(pool, "decode", side_effect=controlled):
            builder = threading.Thread(target=build)
            builder.start()
            try:
                self.assertTrue(failure_raised.wait(5))
                self.assertFalse(finished.is_set())
            finally:
                release_slow.set()
                builder.join(5)
        self.assertTrue(finished.is_set())
        self.assertEqual(str(errors[0]), "injected tensor decode failure")
        self.assertFalse(
            any(p.is_dir() and not p.name.startswith(".") for p in self.cache.iterdir())
        )

    def test_union_binding_and_corrupt_decode_never_publish_ready(self):
        path, digest, manifest, expected = fixture(self.root)
        pool = OuterZstdPool(2)
        self.addCleanup(pool.close)
        subset = host.HostDecodedSnapshot(path, digest, manifest, ["dense"], pool, {})
        complete = host.HostDecodedSnapshot(
            path, digest, manifest, sorted(expected), pool, {}
        )
        self.assertNotEqual(subset.directory, complete.directory)
        self.assertEqual(set(subset.index["tensors"]), {"dense"})
        subset.close(discard=True)
        complete.close(discard=True)
        blob = bytearray((self.root / "owner.bin").read_bytes())
        outer = manifest["tensors"][0]["outer"]
        blob[outer["encoded_offset"] + outer["encoded_bytes"] - 1] ^= 1
        (self.root / "owner.bin").write_bytes(blob)
        manifest["files"][0]["sha256"] = hashlib.sha256(blob).hexdigest()
        path.write_text(json.dumps(manifest))
        with self.assertRaises(zstd.ZstdError):
            host.HostDecodedSnapshot(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest,
                sorted(expected),
                pool,
                {},
            )
        self.assertFalse(
            any(p.is_dir() and not p.name.startswith(".") for p in self.cache.iterdir())
        )
        self.assertTrue(list(self.cache.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
