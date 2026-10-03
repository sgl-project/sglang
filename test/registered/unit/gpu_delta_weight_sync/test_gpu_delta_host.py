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


def metadata(version=1):
    return dict(
        stream_id="stream",
        session_id=f"session-{version}",
        base_version=version - 1,
        target_version=version,
        cohort=[{"engine_id": "a"}, {"engine_id": "b"}],
    )


def _child(root, path, digest, manifest, names, barrier, output):
    pool, arena = OuterZstdPool(2), host.HostArena()
    try:
        with patch.object(host, "_cache_root", return_value=Path(root)):
            identity = host.host_cache_id()
            barrier.wait(timeout=10)
            metrics = {}
            snapshot = arena.prepare(
                path, digest, manifest, names, pool, metrics, metadata()
            )
            values = {
                name: bytes(
                    arena.mapping[row["offset"] : row["offset"] + row["nbytes"]]
                )
                for name, row in snapshot.index["tensors"].items()
            }
            output.put(
                (identity, metrics, values, snapshot.index["shared"]["identity"])
            )
            snapshot.close()  # Abort/ordinary close never authorizes overwrite.
    finally:
        arena.close()
        pool.close()


class TestSharedHostSnapshot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        for replacement in (
            patch.object(host, "_cache_root", return_value=self.cache),
            patch.object(host, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(
                os, "posix_fallocate", side_effect=fake_fallocate, create=True
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.pool = OuterZstdPool(2)
        self.addCleanup(self.pool.close)

    def arena(self):
        arena = host.HostArena()
        self.addCleanup(arena.close)
        return arena

    def test_decode_overlaps_hash_but_ready_waits_for_verified_snapshot(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, metrics = self.arena(), {}
        hash_entered, decode_completed, release_hash, finished = [
            threading.Event() for _ in range(4)
        ]
        snapshots, errors = [], []
        original_hash, original_decode = host._hash_payloads, self.pool.decode

        def delayed_hash(files, definitions):
            hash_entered.set()
            assert release_hash.wait(5)
            return original_hash(files, definitions)

        def decode(payload, chunks, destination):
            assert hash_entered.wait(5)
            result = original_decode(payload, chunks, destination)
            decode_completed.set()
            return result

        def build():
            try:
                snapshots.append(
                    arena.prepare(
                        path,
                        digest,
                        manifest,
                        sorted(expected),
                        self.pool,
                        metrics,
                        metadata(),
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with (
            patch.object(host, "_hash_payloads", side_effect=delayed_hash),
            patch.object(self.pool, "decode", side_effect=decode),
        ):
            builder = threading.Thread(target=build)
            builder.start()
            try:
                self.assertTrue(decode_completed.wait(5))
                self.assertFalse(finished.is_set())
                state = json.loads(next(self.cache.glob("*/state.json")).read_text())
                self.assertEqual(state["state"], "BUILDING")
                # Both jobs consume the stable copied bytes, even if the source
                # file changes after its read/fstat checks completed.
                (self.root / "owner.bin").write_bytes(b"changed after snapshot")
            finally:
                release_hash.set()
                builder.join(5)
        self.assertFalse(builder.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(metrics["host_payload_hash_files"], 1)
        snapshot = snapshots[0]
        self.assertEqual(
            {
                name: bytes(
                    arena.mapping[row["offset"] : row["offset"] + row["nbytes"]]
                )
                for name, row in snapshot.index["tensors"].items()
            },
            expected,
        )
        self.assertEqual(
            json.loads((snapshot.directory / "state.json").read_text())["state"],
            "READY",
        )
        snapshot.close()

    def test_hash_and_decode_failures_drain_the_other_job_before_poison_return(self):
        path, digest, manifest, expected = fixture(self.root)
        original_hash, original_decode = host._hash_payloads, self.pool.decode
        for failure in ("hash", "decode"):
            with self.subTest(failure=failure):
                arena = self.arena()
                other_entered, failure_raised, release_other, finished = [
                    threading.Event() for _ in range(4)
                ]
                errors = []
                request = metadata() | {"stream_id": failure}
                current = json.loads(json.dumps(manifest))
                if failure == "hash":
                    # Real hash rejection with valid Zstd bytes isolates integrity
                    # failure from decoder failure.
                    current["files"][0]["sha256"] = "0" * 64
                current_path = self.root / f"manifest-{failure}.json"
                content = json.dumps(current).encode()
                current_path.write_bytes(content)
                current_digest = hashlib.sha256(content).hexdigest()

                def controlled_hash(files, definitions):
                    if failure == "hash":
                        assert other_entered.wait(5)
                        failure_raised.set()
                        return original_hash(files, definitions)
                    other_entered.set()
                    assert release_other.wait(5)
                    return original_hash(files, definitions)

                def controlled_decode(payload, chunks, destination):
                    if failure == "decode":
                        assert other_entered.wait(5)
                        failure_raised.set()
                        raise ValueError("injected decode failure")
                    other_entered.set()
                    assert release_other.wait(5)
                    return original_decode(payload, chunks, destination)

                def build():
                    try:
                        arena.prepare(
                            current_path,
                            current_digest,
                            current,
                            sorted(expected),
                            self.pool,
                            {},
                            request,
                        )
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        finished.set()

                with (
                    patch.object(host, "_hash_payloads", side_effect=controlled_hash),
                    patch.object(self.pool, "decode", side_effect=controlled_decode),
                ):
                    builder = threading.Thread(target=build)
                    builder.start()
                    try:
                        self.assertTrue(failure_raised.wait(5))
                        self.assertFalse(finished.is_set())
                    finally:
                        release_other.set()
                        builder.join(5)
                self.assertFalse(builder.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIn(
                    "SHA256" if failure == "hash" else "injected decode failure",
                    str(errors[0]),
                )
                with self.assertRaisesRegex(ValueError, "failed or already released"):
                    arena.prepare(
                        current_path,
                        current_digest,
                        current,
                        sorted(expected),
                        self.pool,
                        {},
                        request,
                    )

    def test_two_engine_processes_verify_once_and_never_reread_retained_bytes(self):
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
        self.assertEqual(sum(row[1]["host_payload_hash_files"] for row in records), 1)
        self.assertEqual(sum(row[1]["host_outer_zstd_tensors"] for row in records), 2)
        self.assertTrue(all(row[2] == expected for row in records))
        (self.root / "owner.bin").write_bytes(b"corrupted original")
        arena, timings = self.arena(), {}
        snapshot = arena.prepare(
            path, digest, manifest, sorted(expected), self.pool, timings, metadata()
        )
        self.assertEqual(timings["host_payload_hash_files"], 0)
        row = snapshot.index["tensors"]["dense"]
        self.assertEqual(
            bytes(arena.mapping[row["offset"] : row["offset"] + row["nbytes"]]),
            expected["dense"],
        )
        snapshot.close()
        with self.assertRaisesRegex(ValueError, "global APPLIED release"):
            arena.prepare(
                path, digest, manifest, sorted(expected), self.pool, {}, metadata(2)
            )

    def test_capacity_reuse_growth_and_late_release_do_not_alias_publications(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, first_metrics = self.arena(), {}
        first = arena.prepare(
            path,
            digest,
            manifest,
            sorted(expected),
            self.pool,
            first_metrics,
            metadata(),
        )
        original_inode = first.index["shared"]["identity"]
        self.assertGreaterEqual(
            first.index["shared"]["capacity"], 2 * first.index["arena_bytes"]
        )
        first.mark_reusable()  # Oracle substitutes the production global proof.
        first.close()
        warm = {}
        second = arena.prepare(
            path, digest, manifest, sorted(expected), self.pool, warm, metadata(2)
        )
        self.assertEqual(second.index["shared"]["identity"], original_inode)
        self.assertEqual(warm["host_shared_allocation_calls"], 0)
        self.assertEqual(warm["host_encoded_allocation_calls"], 0)
        self.assertEqual(warm["host_payload_hash_files"], 1)
        first.mark_reusable()  # Late cleanup must not release generation 2.
        with self.assertRaisesRegex(ValueError, "global APPLIED release"):
            arena.prepare(
                path, digest, manifest, sorted(expected), self.pool, {}, metadata(3)
            )
        second.mark_reusable()
        second.close()
        # Force growth through a larger raw target; old inode is never resized.
        before_capacity = second.index["shared"]["capacity"]
        large = bytes(before_capacity * 2 + 1)
        payload = (self.root / "owner.bin").read_bytes() + large
        raw = manifest["tensors"][2]
        raw.update(nbytes=len(large), changed_bytes=len(large))
        raw["raw"].update(
            encoded_offset=len(payload) - len(large), encoded_bytes=len(large)
        )
        (self.root / "owner.bin").write_bytes(payload)
        manifest["files"][0].update(
            nbytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        path.write_text(json.dumps(manifest))
        growth = {}
        third = arena.prepare(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest,
            sorted(expected),
            self.pool,
            growth,
            metadata(3),
        )
        self.assertNotEqual(third.index["shared"]["identity"], original_inode)
        self.assertEqual(third.index["shared"]["generation"], 2)
        self.assertEqual(growth["host_shared_allocation_calls"], 1)
        self.assertGreaterEqual(
            third.index["shared"]["capacity"], 2 * third.index["arena_bytes"]
        )
        self.assertEqual(original_inode[-1], before_capacity)
        self.assertLessEqual(
            third.index["arena_bytes"], third.index["shared"]["capacity"]
        )
        third.close()  # Aborted third update cannot release its slot.
        with self.assertRaisesRegex(ValueError, "global APPLIED release"):
            arena.prepare(
                path, digest, manifest, sorted(expected), self.pool, {}, metadata(4)
            )

    def test_failed_tensor_drains_other_workers_and_poisoned_slot_cannot_reuse(self):
        path, digest, manifest, _ = fixture(self.root)
        arena = self.arena()
        slow_entered, failure_raised, release_slow, finished = [
            threading.Event() for _ in range(4)
        ]
        errors, decode = [], self.pool.decode

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
                arena.prepare(
                    path,
                    digest,
                    manifest,
                    ["dense", "expert"],
                    self.pool,
                    {},
                    metadata(),
                )
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        with patch.object(self.pool, "decode", side_effect=controlled):
            builder = threading.Thread(target=build)
            builder.start()
            try:
                self.assertTrue(failure_raised.wait(5))
                self.assertFalse(finished.is_set())
            finally:
                release_slow.set()
                builder.join(5)
        self.assertEqual(str(errors[0]), "injected tensor decode failure")
        with self.assertRaisesRegex(ValueError, "global APPLIED release"):
            arena.prepare(
                path, digest, manifest, ["dense", "expert"], self.pool, {}, metadata(2)
            )

    def test_host_union_binding_and_corrupt_zstd_do_not_publish_ready(self):
        path, digest, manifest, expected = fixture(self.root)
        arena = self.arena()
        subset = arena.prepare(
            path, digest, manifest, ["dense"], self.pool, {}, metadata()
        )
        self.assertEqual(set(subset.index["tensors"]), {"dense"})
        with self.assertRaisesRegex(ValueError, "tensor union changed"):
            arena.prepare(
                path, digest, manifest, sorted(expected), self.pool, {}, metadata()
            )
        blob = bytearray((self.root / "owner.bin").read_bytes())
        outer = manifest["tensors"][0]["outer"]
        blob[outer["encoded_offset"] + outer["encoded_bytes"] - 1] ^= 1
        (self.root / "owner.bin").write_bytes(blob)
        manifest["files"][0]["sha256"] = hashlib.sha256(blob).hexdigest()
        path.write_text(json.dumps(manifest))
        with self.assertRaises(zstd.ZstdError):
            self.arena().prepare(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest,
                sorted(expected),
                self.pool,
                {},
                metadata(),
            )
        states = [
            json.loads(p.read_text())["state"] for p in self.cache.glob("*/state.json")
        ]
        self.assertEqual(sorted(states), ["BUILDING", "READY"])


if __name__ == "__main__":
    unittest.main()
