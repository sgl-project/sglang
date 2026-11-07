"""Engine-host sharing, DE-local preparation, and failure lifetimes."""

import hashlib
import json
import mmap
import multiprocessing
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta import host as host
from sglang.srt.weight_sync.gpu_delta import host_worker as worker
from sglang.srt.weight_sync.gpu_delta.host_worker import HostPayloadPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TENSOR_IDS = {name: i for i, name in enumerate(("dense", "expert", "foreign", "raw"))}


def write_manifest(path, manifest, version=1):
    manifest.update(
        base_version=version - 1, target_version=version, plan_digest="fixture-plan"
    )
    path = path.with_name(f"manifest-v{version}.json")
    content = json.dumps(manifest).encode()
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def fixture(directory, raw_repeat=1):
    expected = {
        "dense": bytes(range(251)) * 100 + b"d" * (raw_repeat - 1) * 4,
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
    for entry in tensors:
        size = 1 << 20 if entry["name"] == "dense" else entry["nbytes"]
        shape = [size] if entry["encoding"] == "raw_bytes" else [1, size]
        entry.update(
            dtype="U8",
            shape=shape,
            nbytes=size,
            byte_order="little",
            views=[{"id": entry["name"], "slices": [[0, n] for n in shape]}],
        )
    path, digest = write_manifest(directory / "manifest.json", manifest)
    return path, digest, manifest, expected


def local_entries(manifest, names):
    return [entry for entry in manifest["tensors"] if entry["name"] in names]


def prepare_snapshot(
    arena, path, digest, entries, pool, timings, metadata, cancelled=None
):
    if arena._worker is not None:
        arena._worker.builder.pool = pool
    publication = arena.prepare_publication(
        metadata | dict(manifest_path=str(path), manifest_sha256=digest),
        timings,
        cancelled,
    )
    return arena.prepare_local(
        publication, [TENSOR_IDS[e["name"]] for e in entries], timings
    )


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

    def pin(self):
        self.registered = True

    def close(self):
        self.view.release()
        self.mapping = None


class InlineCpuWorker:
    """Use the real CPU builder for deterministic unit fault injection."""

    def __init__(self, directory, device, pool):
        with patch.object(worker, "HostPayloadPool", return_value=pool):
            self.builder = worker.HostBuilder(
                directory,
                pool.workers,
                {
                    name: host.decoder_backend(device, name)
                    for name in ("snappy", "lz4", "bitcomp")
                },
            )

    def prepare(self, expected, namespace, metrics, cancelled):
        result = self.builder.prepare(expected, namespace)
        metrics.update(result["metrics"])
        return result["index"]

    def release(self, ready_state):
        return self.builder.release(ready_state)

    def close(self):
        self.builder.close()


def shared_publication(directory, raw_repeat=1, version=1):
    path, _, manifest, expected = fixture(directory, raw_repeat)
    manifest["codec"] = "bitcomp-zstd"
    path, digest = write_manifest(path, manifest, version)
    return path, digest, manifest, expected | {"foreign": b"foreign-inner-frame"}


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
    records, reads = [], []
    read_bytes = Path.read_bytes

    def observed_read(path):
        reads.append(path.name)
        return read_bytes(path)

    try:
        with (
            patch.object(host, "configured_cpu_workers", return_value=2),
            patch("torch.cuda.set_device") as set_device,
            patch.object(Path, "read_bytes", new=observed_read),
        ):
            for version, (path, digest, manifest, expected) in enumerate(
                publications, 1
            ):
                hardware = version == 4
                barrier.wait(timeout=10)
                metrics = {}
                reads.clear()
                publication = arena.prepare_publication(
                    metadata(version)
                    | dict(manifest_path=str(path), manifest_sha256=digest),
                    metrics,
                )
                assert all(
                    not array.flags.writeable
                    for array in (
                        publication.tensors,
                        publication.frames,
                        publication.gaps,
                    )
                )
                if version == 1:
                    assert [
                        e["name"] for e in publication.read_catalog()["tensors"]
                    ] == list(TENSOR_IDS)
                names = ["dense", "raw"] if rank == 0 else ["dense", "expert"]
                snapshot = arena.prepare_local(
                    publication, [TENSOR_IDS[name] for name in names], metrics, hardware
                )
                assert {
                    name: bytes(snapshot.get(TENSOR_IDS[name])) for name in names
                } == {name: expected[name] for name in names}
                assert arena.shared_allocation.registered is not hardware
                records.append(
                    dict(
                        metrics=metrics,
                        capacity=arena.capacity.copy(),
                        shared_capacity=arena.shared_capacity.copy(),
                        metadata_capacity=publication.index["metadata"].copy(),
                        tensors=publication.tensors.copy(),
                        frames=publication.frames.copy(),
                        gaps=publication.gaps.copy(),
                        offsets=snapshot.offsets.copy(),
                        pool_id=id(arena._worker.builder.pool)
                        if arena._worker
                        else None,
                        set_device_calls=set_device.call_count,
                        manifest_reads=reads.count(path.name),
                        catalog_reads=reads.count("catalog.json"),
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
        plan_digest="fixture-plan",
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
        self.pool = HostPayloadPool(2)
        self.addCleanup(self.pool.close)
        for replacement in (
            patch.object(host, "_cache_base", return_value=self.cache),
            patch.object(host, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(worker, "_CAPACITY_ALIGNMENT", 1024),
            patch.object(
                host,
                "CpuWorker",
                side_effect=lambda directory, device: InlineCpuWorker(
                    directory, device, self.pool
                ),
            ),
            patch.object(host, "HostAllocation", FakeHostAllocation),
            patch.object(host, "HostMapping", FakeHostMapping),
            patch("torch.cuda.set_device"),
            patch.object(
                host,
                "decoder_backend",
                side_effect=lambda device, codec: (
                    "cuda" if codec == "bitcomp" else "hardware"
                ),
            ),
            patch.object(
                os, "posix_fallocate", side_effect=fake_fallocate, create=True
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

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
        original_read = worker._read_verify_payload

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
            patch.object(worker, "_read_verify_payload", side_effect=delayed_read),
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
        self.assertIs(snapshot.get(TENSOR_IDS["dense"]).obj, arena.mapping.obj)
        self.assertEqual(
            {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected}, expected
        )
        follower_metrics = {}
        follower = prepare_snapshot(
            self.arena(rank=1),
            path,
            digest,
            local_entries(manifest, ["dense"]),
            self.pool,
            follower_metrics,
            metadata(),
        )
        self.assertEqual(bytes(follower.get(TENSOR_IDS["dense"])), expected["dense"])
        self.assertEqual(follower_metrics["host_encoded_cache_hash_files"], 0)
        self.assertEqual(follower_metrics["host_rank_outer_zstd_tensors"], 0)
        follower.close()
        publication = snapshot.publication
        snapshot.close()
        catalog = publication.directory / publication.index["catalog"]["file"]
        catalog.write_bytes(b"changed catalog")
        with self.assertRaisesRegex(ValueError, "shared canonical catalog changed"):
            publication.read_catalog()

    def test_payload_hash_policy_follows_publication_and_requires_peer_agreement(self):
        path, digest, manifest, expected = fixture(self.root)
        strict = self.arena("strict")
        peer = self.arena("strict")
        for version, validate in enumerate((True, False, True), 1):
            path, digest = write_manifest(path, manifest, version)
            metrics = {}
            # A legacy inherited environment cannot override the request policy.
            with patch.dict(
                os.environ, {"GPU_DELTA_SKIP_PAYLOAD_HASH": str(int(validate))}
            ):
                snapshot = prepare_snapshot(
                    strict,
                    path,
                    digest,
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
                {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected},
                expected,
            )
            with self.assertRaisesRegex(
                ValueError, "requires prior engine APPLIED release"
            ):
                prepare_snapshot(
                    peer,
                    path,
                    digest,
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
        path, digest = write_manifest(path, manifest)
        for arena, created in ((self.arena(), 1), (self.arena(rank=1), 0)):
            metrics = {}
            snapshot = prepare_snapshot(
                arena,
                path,
                digest,
                local_entries(manifest, expected),
                self.pool,
                metrics,
                metadata(validate_payload_hash=False),
            )
            self.assertEqual(
                {name: bytes(snapshot.get(TENSOR_IDS[name])) for name in expected},
                expected,
            )
            self.assertEqual(metrics["host_encoded_cache_skip_payload_hash"], 1)
            self.assertEqual(metrics["host_encoded_cache_created"], created)
            for field in ("hash_files", "hash_bytes", "sha256_worker_sum_s"):
                self.assertEqual(metrics["host_encoded_cache_" + field], 0)
            snapshot.close()
        with self.assertRaisesRegex(ValueError, "payload checksum omitted"):
            prepare_snapshot(
                self.arena("require-checksum"),
                path,
                digest,
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(engine="require-checksum"),
            )

    def test_hash_and_shared_decode_failure_never_release_cache(self):
        path, digest, manifest, expected = fixture(self.root)
        with self.assertRaisesRegex(
            ValueError, "immutable delta manifest SHA256 mismatch"
        ):
            self.arena("bad-manifest").prepare_publication(
                metadata(engine="bad-manifest")
                | dict(manifest_path=str(path), manifest_sha256="0" * 64),
                {},
            )
        manifest["files"][0]["sha256"] = "0" * 64
        path, digest = write_manifest(path, manifest)
        peer_entered, release, failed = [threading.Event() for _ in range(3)]
        finished, errors = threading.Event(), []
        original_read = worker._read_verify_payload

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
            patch.object(worker, "_read_verify_payload", side_effect=blocked_peer),
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
        path, digest = write_manifest(path, manifest)
        arena = self.arena("bad-decode")
        with self.assertRaises(zstd.ZstdError):
            prepare_snapshot(
                arena,
                path,
                digest,
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
            publication = shared_publication(directory, repeat, version)
            if version in (2, 4):
                path, _, manifest, expected = publication
                if version == 2:
                    foreign = manifest["tensors"][-1]
                    foreign["frames"] = []
                    foreign["changed_bytes"] = 0
                    expected.pop("foreign")
                    manifest["tensors"].reverse()
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
            self.assertEqual(owner["metadata_capacity"], peer["metadata_capacity"])
            for key in ("tensors", "frames", "gaps"):
                np.testing.assert_array_equal(owner[key], peer[key])
            self.assertEqual(owner["manifest_reads"], 1)
            self.assertEqual(peer["manifest_reads"], 0)
            for row in (owner, peer):
                self.assertEqual(row["catalog_reads"], int(version == 0))
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
        self.assertEqual(
            records[0][1]["tensors"][TENSOR_IDS["foreign"]]["prepared_nbytes"], 0
        )
        self.assertGreater(
            records[0][2]["tensors"][TENSOR_IDS["foreign"]]["prepared_nbytes"], 0
        )
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
        request = metadata() | dict(participants=participants)
        prior_metrics = {}
        prior_publication = owner.prepare_publication(
            request | dict(manifest_path=str(path), manifest_sha256=digest),
            prior_metrics,
        )
        prior = owner.prepare_local(
            prior_publication, [TENSOR_IDS["dense"]], prior_metrics, False
        )
        prior.mark_reusable()
        prior.close()
        manifest["tensors"][-1]["frames"] = []
        manifest["tensors"][-1]["changed_bytes"] = 0
        path, digest = write_manifest(path, manifest, 2)
        request = metadata(2) | dict(participants=participants)
        with patch.object(
            worker, "_prepare_arena", side_effect=RuntimeError("decode failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                owner.prepare_publication(
                    request | dict(manifest_path=str(path), manifest_sha256=digest), {}
                )
        state = json.loads((owner.directory / "state.json").read_text())
        self.assertEqual(state["state"], "BUILDING")
        prior.mark_reusable()
        self.assertEqual(
            json.loads((owner.directory / "state.json").read_text()), state
        )
        prepared_path = owner.directory / owner.shared_capacity["file"]
        owner.close()
        self.assertTrue(prepared_path.exists())
        with self.assertRaisesRegex(ValueError, "failed or already released"):
            peer.prepare_publication(
                request | dict(manifest_path=str(path), manifest_sha256=digest), {}
            )
        self.assertIsNone(peer.allocation)
        self.assertIsNone(peer._worker)
        with self.assertRaisesRegex(ValueError, "engine APPLIED release"):
            owner.prepare_publication(
                metadata(3)
                | dict(
                    participants=participants,
                    manifest_path=str(path),
                    manifest_sha256=digest,
                ),
                {},
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
                    times = worker._read_verify_payload(
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
                worker._read_verify_payload(
                    path, destination, {"nbytes": 80, "sha256": "0" * 64}, True
                )
        with mmap.mmap(-1, 80) as mapping, memoryview(mapping) as destination:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                worker._read_verify_payload(
                    path, destination, {"nbytes": 80, "sha256": "0" * 64}, False
                )
            _, hash_s = worker._read_verify_payload(
                path, destination, {"nbytes": 80, "sha256": "0" * 64}, True
            )
            self.assertEqual(destination, b"a" * 80)
            self.assertEqual(hash_s, 0)
            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                worker._read_verify_payload(
                    path, destination, {"nbytes": 79, "sha256": "0" * 64}, True
                )

    def test_followers_wait_for_shared_ready_without_blocking_other_engines(self):
        path, digest, manifest, expected = fixture(self.root)
        first, follower, other = self.arena(), self.arena(rank=1), self.arena("b")
        previous = prepare_snapshot(
            first,
            path,
            digest,
            local_entries(manifest, ["dense"]),
            self.pool,
            {},
            metadata(),
        )
        previous.mark_reusable()
        previous.close()
        path, digest = write_manifest(path, manifest, 2)
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
                    local_entries(manifest, ["expert"]),
                    self.pool,
                    {},
                    metadata(2, engine="b"),
                )
                self.assertEqual(
                    bytes(third.get(TENSOR_IDS["expert"])), expected["expert"]
                )
                self.assertFalse(release.is_set())
                self.assertFalse(follower_finished.is_set())
                self.assertNotEqual(first.directory, other.directory)
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
                host.fcntl.LOCK_EX
                if thread is threads[0]
                else host.fcntl.LOCK_SH | host.fcntl.LOCK_NB,
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
            local_entries(manifest, expected),
            self.pool,
            first_metrics,
            metadata(),
        )
        with patch.object(
            Path, "read_bytes", autospec=True, side_effect=Path.read_bytes
        ) as read:
            reused = arena.prepare_publication(
                metadata() | dict(manifest_path=str(path), manifest_sha256=digest), {}
            )
        self.assertEqual(reused.header["target_version"], 1)
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
        path, digest = write_manifest(path, manifest, 2)
        warm = {}
        second = prepare_snapshot(
            arena,
            path,
            digest,
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
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(3),
            )
        second.mark_reusable()
        second.close()
        manifest = json.loads(path.read_text())
        large = hashlib.shake_256(b"growth").digest(original["capacity"] * 2 + 1)
        encoded = zstd.ZstdCompressor(write_checksum=True).compress(large)
        payload = (self.root / "owner.bin").read_bytes() + encoded
        dense = next(entry for entry in manifest["tensors"] if entry["name"] == "dense")
        dense["changed_bytes"] = len(large)
        dense["frames"][0].update(encoded_bytes=len(large), decoded_bytes=len(large))
        dense["outer"].update(
            encoded_offset=len(payload) - len(encoded),
            encoded_bytes=len(encoded),
            decoded_bytes=len(large),
            frames=[
                dict(
                    encoded_offset=0,
                    encoded_bytes=len(encoded),
                    decoded_offset=0,
                    decoded_bytes=len(large),
                )
            ],
        )
        (self.root / "owner.bin").write_bytes(payload)
        manifest["files"][0].update(
            nbytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        path, digest = write_manifest(path, manifest, 3)
        growth = {}
        third = prepare_snapshot(
            arena,
            path,
            digest,
            local_entries(manifest, expected),
            self.pool,
            growth,
            metadata(3),
        )
        self.assertEqual(bytes(third.get(TENSOR_IDS["dense"])), large)
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
                local_entries(manifest, expected),
                self.pool,
                {},
                metadata(4),
            )
        arena.close()
        self.assertIsNone(arena.mapping)


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
    ordinals = {entry["name"]: i for i, entry in enumerate(entries)}
    tensors = np.zeros(len(entries), dtype=worker.TENSOR_DTYPE)
    size = 0
    for name, ordinal in ordinals.items():
        size = (size + 15) // 16 * 16
        tensors[ordinal]["prepared_offset"] = size
        tensors[ordinal]["prepared_nbytes"] = len(expected[name])
        size += len(expected[name])
    file_spans, position = {}, 0
    for name, blob in blobs.items():
        file_spans[name] = (position, len(blob))
        position += len(blob)
    backing = memoryview(b"".join(blobs.values()))
    return entries, tensors, ordinals, size, backing, file_spans, expected


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4-zstd", "lz4"])
def test_bounded_jobs_decode_raw_and_compressed_bytes_exactly(codec):
    entries, tensors, ordinals, size, encoded, file_spans, expected = decode_fixture(
        codec
    )
    backing = bytearray(b"left" + bytes(size) + b"right")
    destination = memoryview(backing)[4:-5]
    metrics = {name: 0 for name in worker._PREPARE_METRICS}
    pool = HostPayloadPool(2)
    try:
        with (
            patch.object(pool.executor, "submit", wraps=pool.executor.submit) as submit,
            patch.object(pool, "decode_zstd", wraps=pool.decode_zstd) as decode,
        ):
            worker._prepare_arena(
                memoryview(destination),
                tensors,
                ordinals,
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
        for name, ordinal in ordinals.items():
            row = tensors[ordinal]
            start, count = int(row["prepared_offset"]), int(row["prepared_nbytes"])
            assert destination[start : start + count] == expected[name]
        assert backing[:4] == b"left" and backing[-5:] == b"right"
        # A span cannot read the following logical file in the shared backing.
        for name in ("t1", "raw"):
            entry = next(entry for entry in entries if entry["name"] == name)
            payload = entry["raw"] if name == "raw" else entry["outer"]
            original_offset = payload["encoded_offset"]
            payload["encoded_offset"] = file_spans[payload["file"]][1]
            with pytest.raises(ValueError, match="span escapes declared file"):
                worker._prepare_arena(
                    destination,
                    tensors,
                    ordinals,
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
    ordinals = {str(i): i for i in range(20)}
    tensors = np.zeros(20, dtype=worker.TENSOR_DTYPE)
    tensors["prepared_offset"] = np.arange(20)
    tensors["prepared_nbytes"] = 1
    pool = SimpleNamespace(
        workers=2, executor=ThreadPoolExecutor(max_workers=2), decode_zstd=decode
    )

    def run():
        try:
            worker._prepare_arena(
                memoryview(bytearray(20)),
                tensors,
                ordinals,
                memoryview(bytes(range(20))),
                {"p": (0, 20)},
                entries,
                pool,
                {name: 0 for name in worker._PREPARE_METRICS},
                codec,
            )
        except ValueError as error:
            errors.append(str(error))
        finally:
            finished.set()

    def copy(destination, source):
        decode_one(memoryview(source), memoryview(destination))

    copy_patch = patch.object(worker, "copy_bytes", side_effect=copy)
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


@pytest.mark.parametrize("finish", ["close", "failure"])
def test_cpu_process_reuses_exact_publications_without_torch(
    tmp_path, monkeypatch, finish
):
    monkeypatch.setattr(host, "configured_cpu_workers", lambda: 2)
    monkeypatch.setattr(host, "decoder_backend", lambda device, codec: "cuda")
    cache = tmp_path / "cache"
    cache.mkdir()
    guard = tmp_path / "import-guard"
    launches, popen = [], subprocess.Popen

    def launch(command, **kwargs):
        # Bootstrap the actual isolated entry file; do not import this test module.
        code = f"""
import runpy, sys
from pathlib import Path
def guard_import(event, args):
    if event == 'import' and args[0].split('.')[0] in {{'torch', 'sglang', 'flashinfer'}}:
        raise ImportError('GPU/model imports are forbidden in the CPU worker')
sys.addaudithook(guard_import)
Path({str(guard)!r}).touch()
entry = sys.argv.pop(1)
sys.argv[0] = entry
runpy.run_path(entry, run_name='__main__')
"""
        entry = command.index(str(Path(host.__file__).with_name("host_worker.py")))
        process = popen(command[:entry] + ["-c", code] + command[entry:], **kwargs)
        launches.append(process)
        return process

    monkeypatch.setattr(host.subprocess, "Popen", launch)
    client = host.CpuWorker(cache, 0)
    previous, identities, metrics_by_version = None, [], []
    try:
        for version in (1, 2):
            directory = tmp_path / str(version)
            directory.mkdir()
            path, digest, _, expected = shared_publication(directory, version=version)
            expected_publication = {
                "manifest_path": str(path),
                "manifest_sha256": digest,
                "session_id": f"publication-{version}",
                "base_version": version - 1,
                "target_version": version,
                "plan_digest": "fixture-plan",
                "skip_payload_hash": False,
            }
            metrics = {}
            current = client.prepare(expected_publication, {}, metrics, None)
            assert current["cpu_worker"]["pid"] == client.process.pid != os.getpid()
            identities.append(current["cpu_worker"])
            catalog = json.loads((cache / current["catalog"]["file"]).read_bytes())
            numeric = (cache / current["metadata"]["file"]).read_bytes()
            tensors, frames, gaps = worker._metadata_views(
                numeric, current["metadata_counts"]
            )
            prepared = (cache / current["prepared"]["file"]).read_bytes()
            for ordinal, entry in enumerate(catalog["tensors"]):
                row = tensors[ordinal]
                start, size = int(row["prepared_offset"]), int(row["prepared_nbytes"])
                assert prepared[start : start + size] == expected[entry["name"]]
            assert frames.shape[0] == 4 and gaps.shape[1] == 2
            if previous is not None:
                assert all(
                    current[k] == previous[k]
                    for k in ("encoded", "prepared", "metadata")
                )
                assert all(
                    metrics[k] == 0
                    for k in (
                        "host_encoded_cache_allocation_calls",
                        "host_prepared_cache_allocation_calls",
                        "host_metadata_cache_allocation_calls",
                        "host_cpu_worker_spawn_s",
                    )
                )
            metrics_by_version.append(metrics)
            ready_state = json.loads((cache / "state.json").read_text())
            assert ready_state["state"] == "READY"
            client.release(ready_state)
            assert json.loads((cache / "state.json").read_text())["state"] == "REUSABLE"
            previous = current
        assert identities[0] == identities[1] and guard.exists()
        assert metrics_by_version[0]["host_cpu_worker_spawn_s"] > 0
        if finish == "failure":
            bad = expected_publication | {
                "session_id": "bad-publication",
                "manifest_sha256": "0" * 64,
                "base_version": 2,
                "target_version": 3,
            }
            with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
                client.prepare(bad, {}, {}, None)
            with pytest.raises(RuntimeError, match="cannot restart"):
                client.prepare(expected_publication, {}, {}, None)
        client.close()
        assert client.process.poll() is not None and len(launches) == 1
        with pytest.raises(ChildProcessError):
            os.waitpid(client.process.pid, os.WNOHANG)
    finally:
        if client.process.poll() is None:
            client.close(terminate=True)
        client.close()


@pytest.mark.parametrize("abort", ["cancel", "child-exit"])
def test_cpu_process_failure_stops_writer_and_never_releases_publication(
    tmp_path, monkeypatch, abort
):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(host, "_cache_base", lambda: cache)
    monkeypatch.setattr(host, "configured_cpu_workers", lambda: 2)
    monkeypatch.setattr(host, "decoder_backend", lambda device, codec: "cuda")
    ready_r, ready_w = os.pipe()
    release_r, release_w = os.pipe()
    cancelled, finished = threading.Event(), threading.Event()
    launches, errors, popen = [], [], subprocess.Popen

    def launch(command, **kwargs):
        code = f"""
import fcntl, json, os, sys
from multiprocessing.connection import Connection
from pathlib import Path
fd = int(sys.argv[sys.argv.index('--socket-fd') + 1])
connection = Connection(fd)
request = json.loads(connection.recv_bytes())
directory = Path(request['configuration']['directory'])
with (directory / '.lock').open('a+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    (directory / 'state.json').write_text(json.dumps({{'state': 'BUILDING'}}))
    with (directory / 'inflight.bin').open('wb') as pending:
        pending.write(b'started')
        pending.flush()
        os.write({ready_w}, b'1')
        os.read({release_r}, 1)
        pending.write(b'finished')
        pending.flush()
    os._exit(7)
"""
        entry = command.index(str(Path(host.__file__).with_name("host_worker.py")))
        kwargs["pass_fds"] += (ready_w, release_r)
        process = popen(command[:entry] + ["-c", code] + command[entry:], **kwargs)
        launches.append(process)
        return process

    monkeypatch.setattr(host.subprocess, "Popen", launch)
    path, digest, _, _ = shared_publication(tmp_path)
    arena = host.HostArena("a", 0, shared_participants()[0])

    def build():
        try:
            arena.prepare_publication(
                metadata() | dict(manifest_path=str(path), manifest_sha256=digest),
                {},
                cancelled,
            )
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=build)
    thread.start()
    try:
        assert select.select([ready_r], [], [], 10)[0] and os.read(ready_r, 1) == b"1"
        assert not finished.is_set()
        with (arena.directory / ".lock").open("a+b") as lock:
            with pytest.raises(BlockingIOError):
                host.fcntl.flock(lock, host.fcntl.LOCK_EX | host.fcntl.LOCK_NB)
            assert (
                json.loads((arena.directory / "state.json").read_text())["state"]
                == "BUILDING"
            )
            if abort == "cancel":
                cancelled.set()
            else:
                os.write(release_w, b"1")
            thread.join(10)
            assert not thread.is_alive() and len(errors) == 1
            host.fcntl.flock(lock, host.fcntl.LOCK_EX | host.fcntl.LOCK_NB)
            host.fcntl.flock(lock, host.fcntl.LOCK_UN)
            assert (
                json.loads((arena.directory / "state.json").read_text())["state"]
                == "BUILDING"
            )
        assert launches[0].poll() is not None
        assert (arena.directory / "inflight.bin").read_bytes() == (
            b"started" if abort == "cancel" else b"startedfinished"
        )
        with pytest.raises(ChildProcessError):
            os.waitpid(launches[0].pid, os.WNOHANG)
        client = arena._worker
        arena.close()
        assert arena._worker is client
        with pytest.raises(RuntimeError, match="cannot restart"):
            client.prepare({}, {}, {}, None)
        assert len(launches) == 1
    finally:
        cancelled.set()
        thread.join(10)
        arena.close()
        for fd in (ready_r, ready_w, release_r, release_w):
            os.close(fd)


@pytest.mark.parametrize("reason", ["eof", "parent-death"])
def test_cpu_process_stops_on_eof_or_exact_parent_death(reason):
    # Keep the socket peer alive for parent-death, so EOF cannot explain exit.
    parent = subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    local, child = socket.socketpair()
    parent_fd = os.pidfd_open(parent.pid)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            worker.__file__,
            "--socket-fd",
            str(child.fileno()),
            "--parent-fd",
            str(parent_fd),
        ],
        pass_fds=(child.fileno(), parent_fd),
    )
    child.close()
    child_fd = os.pidfd_open(process.pid)
    try:
        if reason == "eof":
            local.close()
        else:
            parent.terminate()
            parent.wait(timeout=10)
            assert local.fileno() >= 0
        assert select.select([child_fd], [], [], 10)[0]
        assert process.wait(timeout=10) == (0 if reason == "eof" else 1)
    finally:
        local.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if parent.poll() is None:
            parent.terminate()
        parent.wait(timeout=10)
        parent.stdin.close()
        os.close(parent_fd)
        os.close(child_fd)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
