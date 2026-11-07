"""Engine-host sharing, DE-local preparation, and failure lifetimes."""

import hashlib
import json
import mmap
import multiprocessing
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta import host as host
from sglang.srt.weight_sync.gpu_delta.host import HostPayloadPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture(directory, raw_repeat=1):
    expected = {
        "dense": bytes(range(251)) * 100,
        "expert": b"snappy-bytes" * 50,
        "raw": b"abcd" * raw_repeat,
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
    # Every tensor is prepared once, including an expert this rank never copies.
    foreign_plain = b"foreign-inner-frame"
    foreign = zstd.ZstdCompressor(write_checksum=True).compress(foreign_plain)
    tensors.append(
        {
            "name": "foreign",
            "encoding": "xor_bytes",
            "nbytes": len(foreign_plain),
            "changed_bytes": len(foreign_plain),
            "frames": [
                dict(
                    encoded_offset=0,
                    encoded_bytes=len(foreign_plain),
                    decoded_offset=0,
                    decoded_bytes=len(foreign_plain),
                )
            ],
            "outer": {
                "file": "foreign.bin",
                "encoded_offset": 0,
                "encoded_bytes": len(foreign),
                "decoded_bytes": len(foreign_plain),
                "frames": [
                    dict(
                        encoded_offset=0,
                        encoded_bytes=len(foreign),
                        decoded_offset=0,
                        decoded_bytes=len(foreign_plain),
                    )
                ],
            },
        }
    )
    (directory / "owner.bin").write_bytes(blob)
    (directory / "foreign.bin").write_bytes(foreign)
    manifest = {
        "codec": "snappy-zstd",
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


def prepare_snapshot(
    arena, path, digest, manifest, entries, pool, timings, metadata, cancelled=None
):
    if arena.identity["tp_rank"] == 0:
        arena._pool = pool
    index = arena.prepare_encoded(
        path, digest, manifest, timings, metadata, cancelled=cancelled
    )
    return arena.prepare_local(index, entries, timings)


def fake_fallocate(fd, offset, length):
    os.ftruncate(fd, offset + length)


class FakeHostAllocation:
    """Rank-private CPU backing; CUDA allocator admission is tested separately."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.view = memoryview(bytearray(capacity))

    def close(self):
        self.view.release()


class FakeHostMapping:
    """Keep real shared mmap semantics without registering it with CUDA."""

    def __init__(self, mapping, pinned):
        self.registered = pinned
        self.mapping = mapping
        self.capacity = len(mapping)
        self.view = memoryview(mapping)

    def close(self):
        self.view.release()
        self.mapping = None


def shared_publication(directory, raw_repeat=1):
    path, _, manifest, expected = fixture(directory, raw_repeat)
    manifest["codec"] = "bitcomp-zstd"
    path.write_text(json.dumps(manifest))
    return (
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        manifest,
        expected | {"foreign": b"foreign-inner-frame"},
    )


def shared_participants(engine="a"):
    return [
        dict(
            engine_id=engine,
            host_cache_id="shared-host-" + engine,
            tp_rank=rank,
            dp_rank=0,
            pid=100 + rank,
            rank_id=f"{engine}-original-{rank}",
        )
        for rank in range(2)
    ]


def _shared_child(rank, publications, barrier, output):
    participants = shared_participants()
    arena = host.HostArena("a", 0, participants[rank])
    records = []
    try:
        with (
            patch.object(host, "configured_cpu_workers", return_value=2),
            patch("torch.cuda.set_device") as set_device,
        ):
            for version, (path, digest, manifest, expected) in enumerate(
                publications, 1
            ):
                hardware = version == 4
                barrier.wait(timeout=10)
                metrics = {}
                index = arena.prepare_encoded(
                    path, digest, manifest, metrics, metadata(version), hardware
                )
                names = ["dense", "raw"] if rank == 0 else ["dense", "expert"]
                snapshot = arena.prepare_local(
                    index, local_entries(manifest, names), metrics, hardware
                )
                assert set(snapshot.records) == set(names if hardware else expected)
                assert {name: bytes(snapshot.get(name)) for name in names} == {
                    name: expected[name] for name in names
                }
                assert arena.shared_allocation.registered is not hardware
                records.append(
                    dict(
                        metrics=metrics,
                        capacity=arena.capacity.copy(),
                        shared_capacity=arena.shared_capacity.copy(),
                        shared_records={
                            name: record.copy()
                            for name, record in index["prepared_records"].items()
                        },
                        layout_id=id(index["prepared_records"]),
                        record_ids={
                            name: id(record)
                            for name, record in index["prepared_records"].items()
                        },
                        pool_id=id(arena._pool) if arena._pool else None,
                        set_device_calls=set_device.call_count,
                    )
                )
                barrier.wait(timeout=10)
                if rank == 0:
                    snapshot.mark_reusable()
                barrier.wait(timeout=10)
                snapshot.close()
        output.put((rank, records))
    finally:
        arena.close()


def metadata(version=1, engine="a", validate_payload_hash=True):
    return dict(
        stream_id="stream",
        session_id=f"session-{version}",
        base_version=version - 1,
        target_version=version,
        validate_payload_hash=validate_payload_hash,
        participants=shared_participants(engine),
    )


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
            patch.object(host, "HostMapping", FakeHostMapping),
            patch("torch.cuda.set_device"),
            patch.object(
                os, "posix_fallocate", side_effect=fake_fallocate, create=True
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.pool = HostPayloadPool(2)
        self.addCleanup(self.pool.close)

    def arena(self, engine="a", rank=0):
        arena = host.HostArena(engine, 0, shared_participants(engine)[rank])
        self.addCleanup(arena.close)
        return arena

    def test_verified_cache_precedes_local_decode_and_retains_immutable_bytes(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, metrics = self.arena(), {}
        entered, release, finished = [threading.Event() for _ in range(3)]
        snapshots, errors = [], []
        peer_finished = threading.Event()
        original_read = host._read_verify_payload

        def delayed_read(source, destination, expected, skip_payload_hash):
            if source.name == "foreign.bin":
                assert entered.wait(5)
            result = original_read(source, destination, expected, skip_payload_hash)
            if source.name == "owner.bin":
                entered.set()
                assert release.wait(5)
            else:
                peer_finished.set()
            return result

        def build():
            try:
                snapshots.append(
                    prepare_snapshot(
                        arena,
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
            patch.object(
                self.pool, "decode_zstd", wraps=self.pool.decode_zstd
            ) as decode,
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
        self.assertEqual(metrics["host_encoded_cache_skip_payload_hash"], 0)
        self.assertGreaterEqual(
            metrics["host_prepared_cache_build_s"],
            metrics["host_rank_outer_zstd_decode_s"],
        )
        self.assertGreaterEqual(
            metrics["host_encoded_cache_access_s"], metrics["host_encoded_cache_wait_s"]
        )
        snapshot = snapshots[0]
        self.assertIs(snapshot.get("dense").obj, arena.mapping.obj)
        self.assertEqual(
            {name: bytes(snapshot.get(name)) for name in expected}, expected
        )
        follower_metrics = {}
        follower = prepare_snapshot(
            self.arena(rank=1),
            path,
            digest,
            manifest,
            local_entries(manifest, ["dense"]),
            self.pool,
            follower_metrics,
            metadata(),
        )
        self.assertEqual(bytes(follower.get("dense")), expected["dense"])
        self.assertEqual(follower_metrics["host_encoded_cache_hash_files"], 0)
        self.assertEqual(follower_metrics["host_rank_outer_zstd_tensors"], 0)
        follower.close()
        snapshot.close()

    def test_payload_hash_policy_follows_publication_and_requires_peer_agreement(self):
        path, digest, manifest, expected = fixture(self.root)
        strict = self.arena("strict")
        peer = self.arena("strict")
        for version, validate in enumerate((True, False, True), 1):
            metrics = {}
            # A legacy inherited environment cannot override the request policy.
            with patch.dict(
                os.environ, {"GPU_DELTA_SKIP_PAYLOAD_HASH": str(int(validate))}
            ):
                snapshot = prepare_snapshot(
                    strict,
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, expected),
                    self.pool,
                    metrics,
                    metadata(version, "strict", validate),
                )
            self.assertEqual(metrics["host_encoded_cache_created"], 1)
            self.assertEqual(
                metrics["host_encoded_cache_skip_payload_hash"], int(not validate)
            )
            self.assertEqual(
                metrics["host_encoded_cache_hash_files"], 2 if validate else 0
            )
            self.assertEqual(
                {name: bytes(snapshot.get(name)) for name in expected}, expected
            )
            with self.assertRaisesRegex(
                ValueError, "requires prior engine APPLIED release"
            ):
                prepare_snapshot(
                    peer,
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, expected),
                    self.pool,
                    {},
                    metadata(version, "strict", not validate),
                )
            snapshot.mark_reusable()
            snapshot.close()

        # The manifest itself still has an exact digest; only payload SHA is skipped.
        manifest["payload_checksum_format"] = "none"
        for record in manifest["files"]:
            record["sha256"] = None
        path.write_text(json.dumps(manifest))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for arena, created in ((self.arena(), 1), (self.arena(rank=1), 0)):
            metrics = {}
            snapshot = prepare_snapshot(
                arena,
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                metrics,
                metadata(validate_payload_hash=False),
            )
            self.assertEqual(
                {name: bytes(snapshot.get(name)) for name in expected}, expected
            )
            self.assertEqual(metrics["host_encoded_cache_skip_payload_hash"], 1)
            self.assertEqual(metrics["host_encoded_cache_created"], created)
            for field in ("hash_files", "hash_bytes", "sha256_worker_sum_s"):
                self.assertEqual(metrics["host_encoded_cache_" + field], 0)
            snapshot.close()
        with self.assertRaisesRegex(ValueError, "payload checksum omitted"):
            prepare_snapshot(
                self.arena(),
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(),
            )

    def test_hash_and_shared_decode_failure_never_release_cache(self):
        path, digest, manifest, expected = fixture(self.root)
        manifest["files"][0]["sha256"] = "0" * 64
        peer_entered, release, failed = [threading.Event() for _ in range(3)]
        finished, errors = threading.Event(), []
        original_read = host._read_verify_payload

        def blocked_peer(source, destination, record, skip_payload_hash):
            if source.name == "foreign.bin":
                peer_entered.set()
                assert release.wait(5)
            try:
                return original_read(source, destination, record, skip_payload_hash)
            except ValueError:
                failed.set()
                raise

        def build():
            try:
                prepare_snapshot(
                    self.arena(),
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
            patch.object(
                self.pool, "decode_zstd", wraps=self.pool.decode_zstd
            ) as decode,
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
            prepare_snapshot(
                self.arena(),
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(),
            )
        # Verification alone cannot publish READY when shared outer decoding fails.
        manifest["files"][0]["sha256"] = hashlib.sha256(
            (self.root / "owner.bin").read_bytes()
        ).hexdigest()
        invalid = b"not-zstd".ljust(manifest["files"][1]["nbytes"], b" ")
        (self.root / "foreign.bin").write_bytes(invalid)
        manifest["files"][1]["sha256"] = hashlib.sha256(invalid).hexdigest()
        arena = self.arena("bad-decode")
        with self.assertRaises(zstd.ZstdError):
            prepare_snapshot(
                arena,
                path,
                digest,
                manifest,
                local_entries(manifest, ["foreign"]),
                self.pool,
                {},
                metadata(engine="bad-decode"),
            )
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(
                arena,
                path,
                digest,
                manifest,
                local_entries(manifest, ["foreign"]),
                self.pool,
                {},
                metadata(2, "bad-decode"),
            )

    def test_shared_preparation_has_one_owner_and_reuses_or_grows_one_inode(self):
        publications = []
        for version, repeat in enumerate((1, 1, 20000, 1, 1), 1):
            directory = self.root / f"version-{version}"
            directory.mkdir()
            publication = shared_publication(directory, repeat)
            if version in (2, 4):
                path, _, manifest, expected = publication
                if version == 2:
                    foreign = manifest["tensors"][-1]
                    foreign["frames"] = []
                    foreign["changed_bytes"] = 0
                    expected.pop("foreign")
                else:
                    manifest["codec"] = "snappy-zstd"
                path.write_text(json.dumps(manifest))
                publication = (
                    path,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    manifest,
                    expected,
                )
            publications.append(publication)
        context = multiprocessing.get_context("fork")
        barrier, output = context.Barrier(2), context.Queue()
        children = [
            context.Process(
                target=_shared_child,
                args=(rank, publications, barrier, output),
            )
            for rank in range(2)
        ]
        for child in children:
            child.start()
        try:
            records = dict(output.get(timeout=30) for _ in children)
            for child in children:
                child.join(timeout=10)
                self.assertEqual(child.exitcode, 0)
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=10)
        for version in range(5):
            owner, peer = records[0][version], records[1][version]
            self.assertEqual(owner["shared_capacity"], peer["shared_capacity"])
            self.assertEqual(owner["shared_records"], peer["shared_records"])
            self.assertEqual(owner["metrics"]["host_prepared_cache_created"], 1)
            self.assertEqual(peer["metrics"]["host_prepared_cache_reused"], 1)
            self.assertEqual(
                owner["metrics"]["host_rank_outer_zstd_tensors"],
                2 if version == 1 else 3,
            )
            self.assertEqual(peer["metrics"]["host_rank_outer_zstd_tensors"], 0)
            self.assertEqual(owner["metrics"]["host_encoded_cache_hash_files"], 2)
            self.assertEqual(peer["metrics"]["host_encoded_cache_hash_files"], 0)
            self.assertEqual(owner["metrics"]["host_rank_cpu_workers"], 2)
            self.assertEqual(peer["metrics"]["host_rank_cpu_workers"], 0)
            if version != 3:
                self.assertEqual(owner["capacity"], peer["capacity"])
                for row in (owner, peer):
                    self.assertEqual(
                        row["metrics"]["host_rank_private_capacity_bytes"], 0
                    )
                    self.assertEqual(row["metrics"]["host_rank_staging_copy_bytes"], 0)
        self.assertEqual(len({row["pool_id"] for row in records[0]}), 1)
        self.assertTrue(all(row["pool_id"] is None for row in records[1]))
        for previous, current in zip(records[0], records[0][1:]):
            self.assertNotEqual(previous["layout_id"], current["layout_id"])
            for name in ("dense", "expert", "raw"):
                self.assertEqual(
                    previous["record_ids"][name], current["record_ids"][name]
                )
        self.assertNotIn("foreign", records[0][1]["shared_records"])
        self.assertIn("foreign", records[0][2]["shared_records"])
        de_owner, de_peer = records[0][3], records[1][3]
        self.assertNotEqual(
            de_owner["capacity"]["identity"], de_peer["capacity"]["identity"]
        )
        for row, names in (
            (de_owner, ["dense", "raw"]),
            (de_peer, ["dense", "expert"]),
        ):
            self.assertGreater(row["metrics"]["host_rank_private_capacity_bytes"], 0)
            self.assertEqual(
                row["metrics"]["host_rank_staging_copy_bytes"],
                sum(len(publications[3][3][name]) for name in names),
            )
        for rows in records.values():
            first, warm, grown, de, restored = rows
            self.assertEqual([row["set_device_calls"] for row in rows], [1, 1, 2, 2, 3])
            self.assertEqual(first["shared_capacity"], warm["shared_capacity"])
            self.assertEqual(warm["metrics"]["host_rank_mapping_reused"], 1)
            self.assertEqual(warm["metrics"]["host_rank_allocation_calls"], 0)
            self.assertNotEqual(
                first["shared_capacity"]["identity"],
                grown["shared_capacity"]["identity"],
            )
            self.assertEqual(grown["shared_capacity"]["generation"], 2)
            self.assertGreater(
                grown["shared_capacity"]["capacity"],
                first["shared_capacity"]["capacity"],
            )
            self.assertEqual(grown["shared_capacity"], de["shared_capacity"])
            self.assertEqual(grown["shared_capacity"], restored["shared_capacity"])

    def test_failed_shared_creator_cannot_publish_or_release_prepared_bytes(self):
        path, digest, manifest, _ = shared_publication(self.root)
        participants = shared_participants()
        owner, peer = [host.HostArena("a", 0, identity) for identity in participants]
        self.addCleanup(owner.close)
        self.addCleanup(peer.close)
        owner._pool = self.pool
        request = metadata() | dict(participants=participants)
        prior_index = owner.prepare_encoded(path, digest, manifest, {}, request, False)
        prior = owner.prepare_local(
            prior_index, local_entries(manifest, ["dense"]), {}, False
        )
        prior.mark_reusable()
        prior.close()
        raw_offset = prior_index["prepared_records"]["raw"]["offset"]
        manifest["tensors"][-1]["frames"] = []
        manifest["tensors"][-1]["changed_bytes"] = 0
        path = path.with_name("failed-manifest.json")
        path.write_text(json.dumps(manifest))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        request = metadata(2) | dict(participants=participants)
        with patch.object(
            host, "_prepare_arena", side_effect=RuntimeError("decode failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                owner.prepare_encoded(path, digest, manifest, {}, request, False)
        state = json.loads((owner.directory / "state.json").read_text())
        self.assertEqual(state["state"], "BUILDING")
        self.assertNotEqual(
            prior_index["prepared_records"]["raw"]["offset"], raw_offset
        )
        prior.mark_reusable()
        self.assertEqual(
            json.loads((owner.directory / "state.json").read_text()), state
        )
        prepared_path = owner.directory / owner.shared_capacity["file"]
        owner.close()
        self.assertTrue(prepared_path.exists())
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            peer.prepare_encoded(path, digest, manifest, {}, request, False)
        self.assertIsNone(peer.allocation)
        self.assertIsNone(peer._pool)
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            owner.prepare_encoded(
                path,
                digest,
                manifest,
                {},
                metadata(3) | dict(participants=participants),
                False,
            )

    def test_payload_reader_handles_short_reads_and_failure(self):
        path = self.root / "payload.bin"

        class ShortReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def readinto(self, destination):
                with destination[:64] as part:
                    return self.stream.readinto(part)

            def read(self, count):
                return self.stream.read(count)

        for data in (b"", bytes(range(251)) * 9, b"x" * 256):
            with self.subTest(bytes=len(data)):
                path.write_bytes(data)
                reader = ShortReader(path.open("rb", buffering=0))
                with (
                    patch.object(Path, "open", return_value=reader),
                    memoryview(bytearray(len(data))) as destination,
                ):
                    times = host._read_verify_payload(
                        path,
                        destination,
                        {
                            "nbytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        },
                        False,
                    )
                    self.assertEqual(destination, data)
                    self.assertTrue(all(value >= 0 for value in times))
                self.assertTrue(reader.stream.closed)
        path.write_bytes(b"a" * 80)
        with memoryview(bytearray(100)) as destination:
            with self.assertRaisesRegex(ValueError, "truncated delta payload"):
                host._read_verify_payload(
                    path, destination, {"nbytes": 80, "sha256": "0" * 64}, True
                )
        with mmap.mmap(-1, 80) as mapping, memoryview(mapping) as destination:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                host._read_verify_payload(
                    path, destination, {"nbytes": 80, "sha256": "0" * 64}, False
                )
            _, hash_s = host._read_verify_payload(
                path, destination, {"nbytes": 80, "sha256": "0" * 64}, True
            )
            self.assertEqual(destination, b"a" * 80)
            self.assertEqual(hash_s, 0)
            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                host._read_verify_payload(
                    path, destination, {"nbytes": 79, "sha256": "0" * 64}, True
                )

    def test_followers_wait_for_shared_ready_without_blocking_other_engines(self):
        path, digest, manifest, expected = fixture(self.root)
        first, follower, other = self.arena(), self.arena(rank=1), self.arena("b")
        previous = prepare_snapshot(
            first,
            path,
            digest,
            manifest,
            local_entries(manifest, ["dense"]),
            self.pool,
            {},
            metadata(),
        )
        previous.mark_reusable()
        previous.close()
        slow_pool = HostPayloadPool(2)
        self.addCleanup(slow_pool.close)
        entered, release, follower_started, follower_finished = [
            threading.Event() for _ in range(4)
        ]
        snapshots, errors, metrics = [], [], {}
        decode = slow_pool.decode_zstd
        stale_checked, cancelled = threading.Event(), threading.Event()
        reads, locks = [], []
        read_bytes, flock = Path.read_bytes, host.fcntl.flock

        def observed_read(path):
            current = threading.current_thread()
            if current in threads:
                reads.append((current, path.name))
                if reads.count((threads[1], "state.json")) >= 3:
                    stale_checked.set()
            return read_bytes(path)

        def observed_lock(file, mode):
            current = threading.current_thread()
            if current in threads and mode != host.fcntl.LOCK_UN:
                locks.append((current, mode))
            return flock(file, mode)

        def blocked_decode(*args):
            entered.set()
            assert release.wait(5)
            return decode(*args)

        def build(arena, pool, selected, finished=None):
            try:
                if finished is not None:
                    follower_started.set()
                snapshots.append(
                    prepare_snapshot(
                        arena,
                        path,
                        digest,
                        manifest,
                        local_entries(manifest, selected),
                        pool,
                        metrics if finished is not None else {},
                        metadata(2),
                        cancelled,
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                if finished is not None:
                    finished.set()

        threads = [
            threading.Thread(target=build, args=(first, slow_pool, ["dense"])),
            threading.Thread(
                target=build,
                args=(follower, self.pool, ["expert"], follower_finished),
            ),
        ]
        with (
            patch.object(slow_pool, "decode_zstd", side_effect=blocked_decode),
            patch.object(Path, "read_bytes", new=observed_read),
            patch.object(host.fcntl, "flock", new=observed_lock),
        ):
            threads[1].start()
            try:
                self.assertTrue(follower_started.wait(5))
                self.assertTrue(stale_checked.wait(5))
                self.assertNotIn((threads[1], "index.json"), reads)
                threads[0].start()
                self.assertTrue(entered.wait(5))
                third = prepare_snapshot(
                    other,
                    path,
                    digest,
                    manifest,
                    local_entries(manifest, ["expert"]),
                    self.pool,
                    {},
                    metadata(engine="b"),
                )
                self.assertEqual(bytes(third.get("expert")), expected["expert"])
                self.assertFalse(release.is_set())
                self.assertFalse(follower_finished.is_set())
                self.assertNotEqual(first.directory, third.directory)
                third.close()
            finally:
                release.set()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(5)
                if any(thread.is_alive() for thread in threads):
                    cancelled.set()
                    for thread in threads:
                        if thread.ident is not None:
                            thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(metrics["host_prepared_cache_reused"], 1)
        self.assertEqual(metrics["host_rank_outer_zstd_tensors"], 0)
        self.assertEqual(metrics["host_rank_cpu_workers"], 0)
        self.assertEqual(reads.count((threads[1], "index.json")), 1)
        self.assertNotIn((threads[0], "index.json"), reads)
        for thread, mode in locks:
            self.assertEqual(
                mode,
                (host.fcntl.LOCK_EX if thread is threads[0] else host.fcntl.LOCK_SH)
                | host.fcntl.LOCK_NB,
            )
        for snapshot in snapshots:
            snapshot.close()

    def test_rank_capacity_reuse_growth_and_late_encoded_release(self):
        path, digest, manifest, expected = fixture(self.root)
        arena, first_metrics = self.arena(), {}
        first = prepare_snapshot(
            arena,
            path,
            digest,
            manifest,
            local_entries(manifest, expected),
            self.pool,
            first_metrics,
            metadata(),
        )
        with patch.object(
            Path, "read_bytes", autospec=True, side_effect=Path.read_bytes
        ) as read:
            reused = arena.prepare_encoded(path, digest, manifest, {}, metadata())
        self.assertIs(reused, arena._index)
        self.assertEqual(
            [call.args[0] for call in read.call_args_list],
            [arena.directory / "state.json"],
        )
        original = arena.capacity.copy()
        self.assertEqual(
            original["capacity"],
            (first_metrics["host_rank_arena_bytes"] + 1023) // 1024 * 1024,
        )
        self.assertEqual(
            first_metrics["host_encoded_cache_capacity_bytes"],
            (sum(file["nbytes"] for file in manifest["files"]) + 1023) // 1024 * 1024,
        )
        first.mark_reusable()
        first.close()
        warm = {}
        second = prepare_snapshot(
            arena,
            path,
            digest,
            manifest,
            local_entries(manifest, expected),
            self.pool,
            warm,
            metadata(2),
        )
        self.assertEqual(arena.capacity, original)
        self.assertEqual(warm["host_rank_allocation_calls"], 0)
        self.assertEqual(warm["host_encoded_cache_allocation_calls"], 0)
        self.assertEqual(warm["host_rank_mapping_reused"], 1)
        first.mark_reusable()
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(
                arena,
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
        third = prepare_snapshot(
            arena,
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest,
            local_entries(manifest, expected),
            self.pool,
            growth,
            metadata(3),
        )
        self.assertNotEqual(arena.capacity["identity"], original["identity"])
        self.assertEqual(arena.capacity["generation"], 2)
        self.assertEqual(growth["host_rank_allocation_calls"], 1)
        self.assertEqual(
            arena.capacity["capacity"],
            (2 * growth["host_rank_arena_bytes"] + 1023) // 1024 * 1024,
        )
        self.assertEqual(
            growth["host_encoded_cache_capacity_bytes"],
            (2 * sum(file["nbytes"] for file in manifest["files"]) + 1023)
            // 1024
            * 1024,
        )
        third.close()
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            prepare_snapshot(
                arena,
                path,
                digest,
                manifest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(4),
            )
        arena.close()
        self.assertIsNone(arena._index)


def decode_fixture(codec):
    blobs = {"payload": bytearray(b"first-file"), "other": bytearray(b"second-file")}
    entries, expected = [], {}
    for i in range(97):
        filename = "other" if i % 3 == 0 else "payload"
        blob = blobs[filename]
        value = bytes([i % 251]) * (31 + i * 31)
        if i == 33:
            value = bytes(j % 251 for j in range(len(value)))
        encoded, outer_frames = value, []
        if codec != "lz4":
            parts = (
                (value[: len(value) // 2], value[len(value) // 2 :])
                if i == 33
                else (value,)
            )
            encoded, decoded_offset = bytearray(), 0
            for part in parts:
                frame = zstd.ZstdCompressor(write_checksum=True).compress(part)
                outer_frames.append(
                    {
                        "encoded_offset": len(encoded),
                        "encoded_bytes": len(frame),
                        "decoded_offset": decoded_offset,
                        "decoded_bytes": len(part),
                    }
                )
                encoded.extend(frame)
                decoded_offset += len(part)
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
                    "file": filename,
                    "encoded_offset": start,
                    "encoded_bytes": len(encoded),
                    "decoded_bytes": len(value),
                    "frames": outer_frames,
                },
            }
        )
    for name, value in (
        ("raw", b"raw-target-value"),
        ("raw-large", bytes(range(256)) * 256),
    ):
        filename = "payload" if name == "raw" else "other"
        blob = blobs[filename]
        start = len(blob)
        blob.extend(value)
        expected[name] = value
        entries.append(
            {
                "name": name,
                "encoding": "raw_bytes",
                "nbytes": len(value),
                "changed_bytes": len(value),
                "frames": [],
                "raw": {
                    "file": filename,
                    "encoded_offset": start,
                    "encoded_bytes": len(value),
                },
            }
        )
    layout, size = host._tensor_layout(entries)
    file_spans, position = {}, 0
    for name, blob in blobs.items():
        file_spans[name] = (position, len(blob))
        position += len(blob)
    backing = memoryview(b"".join(blobs.values()))
    return entries, layout, size, backing, file_spans, expected


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4-zstd", "lz4"])
def test_bounded_jobs_decode_raw_and_compressed_bytes_exactly(codec):
    entries, layout, size, encoded, file_spans, expected = decode_fixture(codec)
    backing = bytearray(b"left" + bytes(size) + b"right")
    destination = memoryview(backing)[4:-5]
    metrics = {name: 0 for name in host._PREPARE_METRICS}
    pool = HostPayloadPool(2)
    try:
        with (
            patch.object(pool.executor, "submit", wraps=pool.executor.submit) as submit,
            patch.object(pool, "decode_zstd", wraps=pool.decode_zstd) as decode,
        ):
            host._prepare_arena(
                memoryview(destination),
                layout,
                encoded,
                file_spans,
                entries,
                pool,
                metrics,
                codec,
            )
        assert submit.call_count == 4 * pool.workers
        assert metrics["host_rank_outer_zstd_tensors"] == (0 if codec == "lz4" else 97)
        assert metrics["host_rank_encoded_copy_tensors"] == (
            97 if codec == "lz4" else 0
        )
        if codec == "lz4":
            decode.assert_not_called()
            assert metrics["host_rank_encoded_copy_bytes"] == sum(
                len(v) for k, v in expected.items() if not k.startswith("raw")
            )
        else:
            assert decode.call_count == submit.call_count
        for name, row in layout.items():
            assert (
                destination[row["offset"] : row["offset"] + row["nbytes"]]
                == expected[name]
            )
        assert backing[:4] == b"left" and backing[-5:] == b"right"
        # A span cannot read the following logical file in the shared backing.
        for name in ("t1", "raw"):
            entry = next(entry for entry in entries if entry["name"] == name)
            payload = entry["raw"] if name == "raw" else entry["outer"]
            original_offset = payload["encoded_offset"]
            payload["encoded_offset"] = file_spans[payload["file"]][1]
            with pytest.raises(ValueError, match="span escapes declared file"):
                host._prepare_arena(
                    destination,
                    layout,
                    encoded,
                    file_spans,
                    entries,
                    pool,
                    metrics,
                    codec,
                )
            payload["encoded_offset"] = original_offset
    finally:
        pool.close()


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4"])
def test_failed_tensor_drains_other_groups_before_releasing_views(codec):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    attempted, errors = [], []

    def decode_one(payload, destination):
        index = payload[0]
        attempted.append(index)
        if index == 0:
            raise ValueError("intentional bad frame")
        if index == 19:
            entered.set()
            assert release.wait(5)
        destination[:] = payload

    def decode(payloads, sizes, offsets, destination):
        for payload, size, offset in zip(payloads, sizes, offsets, strict=True):
            decode_one(memoryview(payload), destination[offset : offset + size])
        return 0.0

    entries = [
        {
            "name": str(i),
            "encoding": "xor_bytes",
            "outer": {
                "file": "p",
                "encoded_offset": i,
                "encoded_bytes": 1,
                "decoded_bytes": 1,
                "frames": [
                    dict(
                        encoded_offset=0,
                        encoded_bytes=1,
                        decoded_offset=0,
                        decoded_bytes=1,
                    )
                ],
            },
        }
        for i in range(20)
    ]
    layout = {str(i): {"offset": i, "nbytes": 1} for i in range(20)}
    pool = SimpleNamespace(
        workers=2, executor=ThreadPoolExecutor(max_workers=2), decode_zstd=decode
    )

    def run():
        try:
            host._prepare_arena(
                memoryview(bytearray(20)),
                layout,
                memoryview(bytes(range(20))),
                {"p": (0, 20)},
                entries,
                pool,
                {name: 0 for name in host._PREPARE_METRICS},
                codec,
            )
        except ValueError as error:
            errors.append(str(error))
        finally:
            finished.set()

    def copy(destination, source):
        decode_one(memoryview(source), memoryview(destination))

    copy_patch = patch.object(host, "copy_bytes", side_effect=copy)
    copy_patch.start()
    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(5)
    assert not finished.is_set()
    release.set()
    thread.join(5)
    pool.executor.shutdown(wait=True)
    copy_patch.stop()
    assert finished.is_set() and errors == ["intentional bad frame"]
    assert sorted(attempted) == [i for i in range(20) if i not in {8, 16}]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
