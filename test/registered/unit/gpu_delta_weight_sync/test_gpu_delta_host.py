"""Shared verified encoded bytes, rank-local decode, and failure lifetimes."""

import hashlib
import json
import multiprocessing
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
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
            "frames": []
            if name == "raw"
            else [
                {
                    "encoded_offset": 0,
                    "encoded_bytes": len(data),
                    "decoded_offset": 0,
                    "decoded_bytes": len(data),
                }
            ],
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
    # authenticated in its own owner file but must not be decoded on this host.
    foreign = b"foreign-not-zstd"
    tensors.append(
        {
            "name": "foreign",
            "encoding": "xor_bytes",
            "nbytes": 10,
            "changed_bytes": 1,
            "frames": [
                {
                    "encoded_offset": 0,
                    "encoded_bytes": 10,
                    "decoded_offset": 0,
                    "decoded_bytes": 10,
                }
            ],
            "outer": {
                "file": "foreign.bin",
                "encoded_offset": 0,
                "encoded_bytes": 16,
                "decoded_bytes": 10,
                "frames": [
                    {
                        "encoded_offset": 0,
                        "encoded_bytes": 16,
                        "decoded_offset": 0,
                        "decoded_bytes": 10,
                    }
                ],
            },
        }
    )
    (directory / "owner.bin").write_bytes(blob)
    (directory / "foreign.bin").write_bytes(foreign)
    manifest = {
        "frame_bytes": 1 << 20,
        "files": [
            {
                "name": "owner.bin",
                "nbytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            },
            {
                "name": "foreign.bin",
                "nbytes": len(foreign),
                "sha256": hashlib.sha256(foreign).hexdigest(),
            },
        ],
        "tensors": tensors,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), manifest, expected


def local_entries(manifest, names):
    return [entry for entry in manifest["tensors"] if entry["name"] in names]


def fake_fallocate(fd, offset, length):
    os.ftruncate(fd, offset + length)


class FakeHostAllocation:
    """Rank-private CPU backing; CUDA allocator admission is tested separately."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.view = memoryview(bytearray(capacity))

    def close(self):
        self.view.release()


def metadata(version=1, engine="a"):
    return dict(
        stream_id="stream",
        session_id=f"session-{version}",
        base_version=version - 1,
        target_version=version,
        participants=[
            {"engine_id": engine, "rank": 0},
            {"engine_id": engine, "rank": 1},
        ],
    )


def _child(root, path, digest, manifest, names, barrier, output):
    pool, arena = OuterZstdPool(2), host.HostArena("a", 0)
    try:
        with patch.object(host, "_cache_base", return_value=Path(root)):
            identity = host.host_cache_id("a")
            barrier.wait(timeout=10)
            metrics = {}
            snapshot = arena.prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, names),
                pool,
                metrics,
                metadata(),
            )
            values = {
                name: bytes(
                    arena.mapping[row["offset"] : row["offset"] + row["nbytes"]]
                )
                for name, row in snapshot.index["tensors"].items()
            }
            output.put(
                (identity, metrics, values, snapshot.index["rank_arena"]["identity"])
            )
            snapshot.close()  # Abort/ordinary close never authorizes overwrite.
    finally:
        arena.close()
        pool.close()


class TestHostSnapshot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        for replacement in (
            patch.object(host, "_cache_base", return_value=self.cache),
            patch.object(host, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(host, "HostAllocation", FakeHostAllocation),
            patch.object(
                os, "posix_fallocate", side_effect=fake_fallocate, create=True
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.pool = OuterZstdPool(2)
        self.addCleanup(self.pool.close)

    def arena(self, engine="a"):
        arena = host.HostArena(engine, 0)
        self.addCleanup(arena.close)
        return arena

    def test_verified_cache_precedes_local_decode_and_retains_immutable_bytes(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, metrics = self.arena(), {}
        entered, release, finished = [threading.Event() for _ in range(3)]
        snapshots, errors = [], []
        peer_finished = threading.Event()
        original_read = host._read_verify_payload

        def delayed_read(source, destination, expected):
            result = original_read(source, destination, expected)
            if source.name == "owner.bin":
                entered.set()
                assert release.wait(5)
            else:
                peer_finished.set()
            return result

        def build():
            try:
                snapshots.append(
                    arena.prepare(
                        path,
                        digest,
                        manifest,
                        local_entries(manifest, expected),
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
            patch.object(host, "_read_verify_payload", side_effect=delayed_read),
            patch.object(self.pool, "decode", wraps=self.pool.decode) as decode,
        ):
            builder = threading.Thread(target=build)
            builder.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(peer_finished.wait(5))
                self.assertFalse(finished.is_set())
                decode.assert_not_called()
                self.assertIsNone(arena.allocation)
                state = json.loads(next(self.cache.glob("*/*/state.json")).read_text())
                self.assertEqual(state["state"], "BUILDING")
                (self.root / "owner.bin").write_bytes(b"changed after retained read")
            finally:
                release.set()
                builder.join(5)
        self.assertFalse(builder.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(metrics["host_encoded_cache_hash_files"], 2)
        snapshot = snapshots[0]
        self.assertEqual(
            {name: bytes(snapshot.get(name).numpy()) for name in expected}, expected
        )
        follower_metrics = {}
        follower = self.arena().prepare(
            path,
            digest,
            manifest,
            local_entries(manifest, ["dense"]),
            self.pool,
            follower_metrics,
            metadata(),
        )
        self.assertEqual(bytes(follower.get("dense").numpy()), expected["dense"])
        self.assertEqual(follower_metrics["host_encoded_cache_hash_files"], 0)
        self.assertEqual(follower_metrics["host_rank_outer_zstd_tensors"], 1)
        follower.close()
        snapshot.close()

    def test_hash_failure_poison_and_rank_decode_failure_never_release_cache(self):
        path, digest, manifest, expected = fixture(self.root)
        manifest["files"][0]["sha256"] = "0" * 64
        peer_entered, release, failed = [threading.Event() for _ in range(3)]
        finished, errors = threading.Event(), []
        original_read = host._read_verify_payload

        def blocked_peer(source, destination, record):
            if source.name == "foreign.bin":
                peer_entered.set()
                assert release.wait(5)
            try:
                return original_read(source, destination, record)
            except ValueError:
                failed.set()
                raise

        def build():
            try:
                self.arena().prepare(
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, expected),
                    self.pool,
                    {},
                    metadata(),
                )
            except ValueError as error:
                errors.append(str(error))
            finally:
                finished.set()

        with (
            patch.object(host, "_read_verify_payload", side_effect=blocked_peer),
            patch.object(self.pool, "decode", wraps=self.pool.decode) as decode,
        ):
            thread = threading.Thread(target=build)
            thread.start()
            try:
                self.assertTrue(peer_entered.wait(5))
                self.assertTrue(failed.wait(5))
                self.assertFalse(finished.is_set())
                decode.assert_not_called()
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["delta payload SHA256 mismatch"])
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            self.arena().prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(),
            )
        # A different engine's authenticated bytes can still have invalid Zstd.
        # Encoded READY proves hash/metadata only; local failure cannot release it.
        manifest["files"][0]["sha256"] = hashlib.sha256(
            (self.root / "owner.bin").read_bytes()
        ).hexdigest()
        arena = self.arena("bad-decode")
        with self.assertRaisesRegex(ValueError, "standard Zstd frame"):
            arena.prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, ["foreign"]),
                self.pool,
                {},
                metadata(engine="bad-decode"),
            )
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            arena.prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, ["foreign"]),
                self.pool,
                {},
                metadata(2, "bad-decode"),
            )

    def test_engine_ranks_verify_once_but_decode_only_their_local_tensors(self):
        path, digest, manifest, expected = fixture(self.root)
        context = multiprocessing.get_context("fork")
        barrier, output = context.Barrier(2), context.Queue()
        names = [["dense", "raw"], ["expert", "raw"]]
        children = [
            context.Process(
                target=_child,
                args=(
                    str(self.cache),
                    path,
                    digest,
                    manifest,
                    selected,
                    barrier,
                    output,
                ),
            )
            for selected in names
        ]
        for child in children:
            child.start()
        records = [output.get(timeout=15) for _ in children]
        for child in children:
            child.join(timeout=15)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(len({row[0] for row in records}), 1)
        self.assertNotEqual(records[0][3], records[1][3])
        for field in ("created", "hash_files", "frames_validations"):
            self.assertEqual(
                sum(row[1]["host_encoded_cache_" + field] for row in records),
                2 if field == "hash_files" else 1,
            )
        self.assertTrue(
            all(row[1]["host_rank_outer_zstd_tensors"] == 1 for row in records)
        )
        self.assertEqual(
            {tuple(sorted(row[2])) for row in records},
            {tuple(sorted(n)) for n in names},
        )
        for row in records:
            self.assertEqual(row[2], {name: expected[name] for name in row[2]})
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            self.arena().prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(2),
            )

    def test_invalid_foreign_metadata_fails_before_allocation_read_or_decode(self):
        path, _, manifest, expected = fixture(self.root)
        manifest["tensors"][-1]["outer"]["frames"][0]["decoded_bytes"] = 11
        content = json.dumps(manifest).encode()
        path.write_bytes(content)
        with (
            patch.object(host, "_reserve", wraps=host._reserve) as allocate,
            patch.object(
                host, "_read_verify_payload", wraps=host._read_verify_payload
            ) as read,
            patch.object(self.pool, "decode", wraps=self.pool.decode) as decode,
            self.assertRaisesRegex(ValueError, "outer Zstd chunk range"),
        ):
            self.arena().prepare(
                path,
                hashlib.sha256(content).hexdigest(),
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(),
            )
        allocate.assert_not_called()
        read.assert_not_called()
        decode.assert_not_called()
        self.assertEqual(list(self.cache.glob("*/*/state.json")), [])

    def test_local_decode_does_not_hold_engine_encoded_cache_lock(self):
        path, digest, manifest, expected = fixture(self.root)
        first, follower, other = self.arena(), self.arena(), self.arena("b")
        slow_pool = OuterZstdPool(2)
        self.addCleanup(slow_pool.close)
        entered, release = threading.Event(), threading.Event()
        snapshots, errors = [], []
        decode = slow_pool.decode

        def blocked_decode(*args):
            entered.set()
            assert release.wait(5)
            return decode(*args)

        def build_first():
            try:
                snapshots.append(
                    first.prepare(
                        path,
                        digest,
                        manifest,
                        local_entries(manifest, ["dense"]),
                        slow_pool,
                        {},
                        metadata(),
                    )
                )
            except BaseException as error:
                errors.append(error)

        with patch.object(slow_pool, "decode", side_effect=blocked_decode):
            thread = threading.Thread(target=build_first)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                metrics = {}
                second = follower.prepare(
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, ["expert"]),
                    self.pool,
                    metrics,
                    metadata(),
                )
                third = other.prepare(
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, ["expert"]),
                    self.pool,
                    {},
                    metadata(engine="b"),
                )
                self.assertEqual(metrics["host_encoded_cache_reused"], 1)
                self.assertFalse(release.is_set())
                self.assertNotEqual(second.directory, third.directory)
                second.close()
                third.close()
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        snapshots[0].close()

    def test_rank_capacity_reuse_growth_and_late_encoded_release(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, first_metrics = self.arena(), {}
        first = arena.prepare(
            path,
            digest,
            manifest,
            local_entries(manifest, expected),
            self.pool,
            first_metrics,
            metadata(),
        )
        original = first.index["rank_arena"]
        self.assertEqual(
            original["capacity"], (first.index["arena_bytes"] + 1023) // 1024 * 1024
        )
        self.assertEqual(
            first.index["encoded"]["capacity"],
            (sum(file["nbytes"] for file in manifest["files"]) + 1023) // 1024 * 1024,
        )
        first.mark_reusable()
        first.close()
        warm = {}
        second = arena.prepare(
            path,
            digest,
            manifest,
            local_entries(manifest, expected),
            self.pool,
            warm,
            metadata(2),
        )
        self.assertEqual(second.index["rank_arena"], original)
        self.assertEqual(warm["host_rank_allocation_calls"], 0)
        self.assertEqual(warm["host_encoded_cache_allocation_calls"], 0)
        self.assertEqual(warm["host_rank_mapping_reused"], 1)
        first.mark_reusable()
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            arena.prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(3),
            )
        second.mark_reusable()
        second.close()
        manifest = json.loads(path.read_text())
        large = bytes(original["capacity"] * 2 + 1)
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
            local_entries(manifest, expected),
            self.pool,
            growth,
            metadata(3),
        )
        self.assertNotEqual(third.index["rank_arena"]["identity"], original["identity"])
        self.assertEqual(third.index["rank_arena"]["generation"], 2)
        self.assertEqual(growth["host_rank_allocation_calls"], 1)
        self.assertEqual(
            third.index["rank_arena"]["capacity"],
            (2 * third.index["arena_bytes"] + 1023) // 1024 * 1024,
        )
        self.assertEqual(
            third.index["encoded"]["capacity"],
            (2 * sum(file["nbytes"] for file in manifest["files"]) + 1023)
            // 1024
            * 1024,
        )
        third.close()
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            arena.prepare(
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(4),
            )


def decode_fixture():
    blob, entries, expected = bytearray(), [], {}
    for i in range(97):
        value = bytes([i % 251]) * (31 + i * 31)
        encoded = zstd.ZstdCompressor(write_checksum=True).compress(value)
        blob.extend(bytes(-len(blob) % 16))
        start = len(blob)
        blob.extend(encoded)
        name = f"t{i}"
        expected[name] = value
        entries.append(
            {
                "name": name,
                "encoding": "xor_bytes",
                "nbytes": len(value),
                "changed_bytes": len(value),
                "frames": [{}],
                "outer": {
                    "file": "payload",
                    "encoded_offset": start,
                    "encoded_bytes": len(encoded),
                    "decoded_bytes": len(value),
                    "frames": [
                        {
                            "encoded_offset": 0,
                            "encoded_bytes": len(encoded),
                            "decoded_offset": 0,
                            "decoded_bytes": len(value),
                        }
                    ],
                },
            }
        )
    start = len(blob)
    blob.extend(b"raw-target-value")
    expected["raw"] = b"raw-target-value"
    entries.append(
        {
            "name": "raw",
            "encoding": "raw_bytes",
            "nbytes": 16,
            "changed_bytes": 16,
            "frames": [],
            "raw": {"file": "payload", "encoded_offset": start, "encoded_bytes": 16},
        }
    )
    layout, size = host._tensor_layout(entries)
    return entries, layout, size, {"payload": memoryview(bytes(blob))}, expected


def test_bounded_jobs_decode_raw_and_compressed_bytes_exactly():
    entries, layout, size, files, expected = decode_fixture()
    destination = bytearray(size)
    metrics = {name: 0 for name in host._DECODE_METRICS}
    pool = OuterZstdPool(2)
    try:
        with patch.object(
            pool.executor, "submit", wraps=pool.executor.submit
        ) as submit:
            host._decode_arena(
                memoryview(destination), layout, files, entries, pool, metrics
            )
        assert submit.call_count == 4 * pool.workers
        assert metrics["host_rank_outer_zstd_tensors"] == 97
        for name, row in layout.items():
            assert (
                destination[row["offset"] : row["offset"] + row["nbytes"]]
                == expected[name]
            )
    finally:
        pool.close()


def test_failed_tensor_drains_other_groups_before_releasing_views():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    attempted, errors = [], []

    def decode(payload, chunks, destination):
        index = payload[0]
        attempted.append(index)
        if index == 0:
            raise ValueError("intentional bad frame")
        if index == 19:
            entered.set()
            assert release.wait(5)
        destination[:] = payload
        return 0, 0

    entries = [
        {
            "name": str(i),
            "encoding": "xor_bytes",
            "outer": {
                "file": "p",
                "encoded_offset": i,
                "encoded_bytes": 1,
                "decoded_bytes": 1,
                "frames": [{}],
            },
        }
        for i in range(20)
    ]
    layout = {str(i): {"offset": i, "nbytes": 1} for i in range(20)}
    pool = SimpleNamespace(
        workers=2, executor=ThreadPoolExecutor(max_workers=2), decode=decode
    )

    def run():
        try:
            host._decode_arena(
                memoryview(bytearray(20)),
                layout,
                {"p": memoryview(bytes(range(20)))},
                entries,
                pool,
                {name: 0 for name in host._DECODE_METRICS},
            )
        except ValueError as error:
            errors.append(str(error))
        finally:
            finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    assert not finished.is_set()
    release.set()
    thread.join(5)
    pool.executor.shutdown(wait=True)
    assert finished.is_set() and errors == ["intentional bad frame"]
    assert sorted(attempted) == [i for i in range(20) if i not in {8, 16}]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
