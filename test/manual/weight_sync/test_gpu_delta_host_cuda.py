"""Two engines each share one pinned arena between two independent CUDA ranks.

Manual-only: Linux shared tmpfs, two CUDA GPUs, Torch and zstandard are required.
The test barriers coordinate the oracle only; production preparation has no
collectives. Both 4- and 8-worker creator pools exercise the same exact bytes.
"""

import hashlib
import json
import multiprocessing
import os
import tempfile
from pathlib import Path

import pytest
import zstandard as zstd


def _publication(directory, version, repeat):
    expected = {
        f"tensor-{i}": bytes((j + i + version) % 256 for j in range(256)) * repeat
        for i in range(8)
    }
    expected["raw"] = b"unaligned-raw-target"
    blob, entries = bytearray(), []
    for name, value in expected.items():
        blob.extend(bytes((-len(blob)) % 16))
        start = len(blob)
        encoded = (
            value
            if name == "raw"
            else zstd.ZstdCompressor(write_checksum=True).compress(value)
        )
        blob.extend(encoded)
        entry = {
            "name": name,
            "encoding": "raw_bytes" if name == "raw" else "xor_bytes",
            "nbytes": len(value),
            "changed_bytes": len(value),
            "frames": []
            if name == "raw"
            else [
                {
                    "encoded_offset": 0,
                    "encoded_bytes": len(value),
                    "decoded_offset": 0,
                    "decoded_bytes": len(value),
                }
            ],
        }
        if name == "raw":
            entry["raw"] = {
                "file": "owner.bin",
                "encoded_offset": start,
                "encoded_bytes": len(value),
            }
        else:
            entry["outer"] = {
                "file": "owner.bin",
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
            }
        entries.append(entry)
    (directory / "owner.bin").write_bytes(blob)
    manifest = {
        "frame_bytes": 1 << 20,
        "files": [
            {
                "name": "owner.bin",
                "nbytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        ],
        "tensors": entries,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), expected


def _consumer(rank, engine, workers, publications, cache, barrier, output):
    import torch

    from sglang.srt.weight_sync import gpu_delta_host as host
    from sglang.srt.weight_sync.gpu_delta_payload import OuterZstdPool

    os.environ["GPU_DELTA_HOST_CACHE_DIR"] = cache
    # Exercise the exact capacity-growth algorithm with small oracle tensors.
    # Production's coarser alignment is not a wire/codec requirement.
    host._CAPACITY_ALIGNMENT = 1 << 20
    torch.cuda.set_device(rank % 2)
    device = torch.device("cuda", rank % 2)
    pool, arena = OuterZstdPool(workers), host.HostArena(engine)
    stream = torch.cuda.Stream(device=device)
    records = []
    try:
        identity = host.host_cache_id(engine)
        for version, (path, digest, expected) in enumerate(publications, 1):
            metadata = dict(
                stream_id="native-shared-stream",
                session_id=f"update-{version}",
                base_version=version - 1,
                target_version=version,
                participants=[
                    {"engine_id": engine, "rank": 0},
                    {"engine_id": engine, "rank": 1},
                ],
            )
            barrier.wait(timeout=90)
            metrics = {}
            snapshot = arena.prepare(
                path,
                digest,
                json.loads(Path(path).read_text()),
                sorted(expected),
                pool,
                metrics,
                metadata,
            )
            arena.register(device, metrics)
            assert arena.tensor.is_pinned() and arena.registered
            source = arena.tensor[: snapshot.index["arena_bytes"]]
            with torch.cuda.stream(stream):
                copied = source.to(device=device, non_blocking=True)
                complete = torch.cuda.Event()
                complete.record(stream)
            complete.synchronize()
            host_copy = copied.cpu()
            for name, value in expected.items():
                record = snapshot.index["tensors"][name]
                actual = host_copy[
                    record["offset"] : record["offset"] + record["nbytes"]
                ]
                assert bytes(actual.numpy()) == value
            # Substitute the production all-original-engine-rank APPLIED certificate with
            # an explicit two-process completion barrier in this isolated oracle.
            barrier.wait(timeout=90)
            snapshot.mark_reusable()
            snapshot.close()
            source = None
            assert arena.registered  # Per-update disposal retains registration.
            records.append(
                dict(
                    version=version,
                    host_cache_id=identity,
                    arena_identity=arena.identity,
                    mapping_pointer=arena.tensor.data_ptr(),
                    metrics=metrics,
                    exact_h2d_bytes=True,
                )
            )
        stream.synchronize()
        arena.close()
        assert not arena.registered and arena.mapping is None
        output.put(
            dict(rank=rank, engine_id=engine, updates=records, final_unregistered=True)
        )
    finally:
        stream.synchronize()
        arena.close()
        pool.close()


@pytest.mark.parametrize("workers", [4, 8])
def test_two_engines_reuse_registered_capacity_and_grow_on_new_inode(workers):
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gpu-delta-native-", dir="/dev/shm"
    ) as directory:
        root = Path(directory)
        publications = []
        for version, repeat in enumerate((1024, 512, 4096), 1):
            version_dir = root / str(version)
            version_dir.mkdir()
            path, digest, expected = _publication(version_dir, version, repeat)
            publications.append((str(path), digest, expected))
        barriers, output = [context.Barrier(2), context.Barrier(2)], context.Queue()
        processes = [
            context.Process(
                target=_consumer,
                args=(
                    rank,
                    f"engine-{rank // 2}",
                    workers,
                    publications,
                    str(root / "cache"),
                    barriers[rank // 2],
                    output,
                ),
            )
            for rank in range(4)
        ]
        try:
            for process in processes:
                process.start()
            records = [output.get(timeout=240) for _ in processes]
            for process in processes:
                process.join(timeout=90)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=15)
        for engine in ("engine-0", "engine-1"):
            engine_records = [
                record for record in records if record["engine_id"] == engine
            ]
            assert len(engine_records) == 2
            for version in range(3):
                rows = [record["updates"][version] for record in engine_records]
                assert rows[0]["arena_identity"] == rows[1]["arena_identity"]
                assert (
                    sum(row["metrics"]["host_payload_cache_created"] for row in rows)
                    == 1
                )
                assert (
                    sum(row["metrics"]["host_payload_hash_files"] for row in rows) == 1
                )
                assert (
                    sum(row["metrics"]["host_frames_validations"] for row in rows) == 1
                )
                assert (
                    sum(row["metrics"]["host_outer_zstd_tensors"] for row in rows) == 8
                )
                assert sum(
                    row["metrics"]["host_shared_allocation_calls"] for row in rows
                ) == (0 if version == 1 else 1)
                assert all(
                    row["metrics"]["host_shared_register_calls"]
                    == (0 if version == 1 else 1)
                    for row in rows
                )
                assert all(
                    row["metrics"]["host_shared_registration_reused"]
                    == int(version == 1)
                    for row in rows
                )
        assert len({record["updates"][0]["host_cache_id"] for record in records}) == 2
        assert (
            len({tuple(record["updates"][0]["arena_identity"]) for record in records})
            == 2
        )
        for record in records:
            a, b, c = record["updates"]
            assert a["arena_identity"] == b["arena_identity"] != c["arena_identity"]
            assert a["mapping_pointer"] == b["mapping_pointer"] != c["mapping_pointer"]
            assert [
                row["metrics"]["host_shared_capacity_generation"]
                for row in record["updates"]
            ] == [1, 1, 2]
            assert b["metrics"]["host_shared_registered_bytes"] == 0
            assert all(
                row["metrics"]["host_shared_arena_bytes"]
                <= row["metrics"]["host_shared_capacity_bytes"]
                for row in record["updates"]
            )
            assert record["final_unregistered"]
        print(
            json.dumps(
                dict(
                    status="PASS",
                    cpu_workers=workers,
                    logical_engines=2,
                    ranks_per_engine=2,
                    updates=3,
                    records=records,
                ),
                sort_keys=True,
            )
        )
