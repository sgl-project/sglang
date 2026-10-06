"""Two engines verify encoded files once and decode into private rank DE arenas.

Manual-only: Linux, two Blackwell CUDA GPUs, Torch, Snappy and Zstandard are required.
The test barriers coordinate the oracle only; production preparation has no
collectives. Both 4- and 8-worker rank pools exercise the same exact bytes.
"""

import ctypes
import hashlib
import json
import multiprocessing
import os
import random
import tempfile
from pathlib import Path

import pytest
import snappy
import zstandard as zstd


def _publication(directory, version, repeat):
    expected = {
        f"tensor-{i}": random.Random(i + version).randbytes(256 * repeat)
        for i in range(8)
    }
    expected["raw"] = b"unaligned-raw-target"
    blob, entries = bytearray(), []
    for name, value in expected.items():
        blob.extend(bytes((-len(blob)) % 16))
        start = len(blob)
        inner = value if name == "raw" else snappy.compress(value)
        encoded, outer_frames = bytearray(), []
        if name == "raw":
            encoded.extend(value)
        else:
            for offset in range(0, len(inner), 1 << 20):
                chunk = inner[offset : offset + (1 << 20)]
                encoded.extend(bytes((-len(encoded)) % 16))
                compressed = zstd.ZstdCompressor(write_checksum=True).compress(chunk)
                outer_frames.append(
                    dict(
                        encoded_offset=len(encoded),
                        encoded_bytes=len(compressed),
                        decoded_offset=offset,
                        decoded_bytes=len(chunk),
                    )
                )
                encoded.extend(compressed)
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
                    "encoded_bytes": len(inner),
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
                "decoded_bytes": len(inner),
                "frames": outer_frames,
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

    from sglang.srt.weight_sync.gpu_delta import host as host
    from sglang.srt.weight_sync.gpu_delta import memory as memory
    from sglang.srt.weight_sync.gpu_delta.codec import DecodeFrame, NvcompDecoder
    from sglang.srt.weight_sync.gpu_delta.payload import OuterZstdPool

    os.environ["GPU_DELTA_HOST_CACHE_DIR"] = cache
    # Exercise the exact capacity-growth algorithm with small oracle tensors.
    # Production's coarser alignment is not a wire/codec requirement.
    host._CAPACITY_ALIGNMENT = 1 << 20
    torch.cuda.set_device(rank % 2)
    device = torch.device("cuda", rank % 2)
    pool, arena = OuterZstdPool(workers), host.HostArena(engine, device.index)
    stream = torch.cuda.Stream(device=device)
    records = []
    try:
        identity = host.host_cache_id(engine)
        names = {f"tensor-{i}" for i in range(4 * (rank % 2), 4 * (rank % 2 + 1))} | {
            "raw"
        }
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
            manifest = json.loads(Path(path).read_text())
            snapshot = arena.prepare(
                path,
                digest,
                manifest,
                [entry for entry in manifest["tensors"] if entry["name"] in names],
                pool,
                metrics,
                metadata,
            )
            source = arena.tensor
            frames, offsets, size = [], {}, 0
            for entry in manifest["tensors"]:
                name = entry["name"]
                if name not in names:
                    continue
                if name == "raw":
                    assert bytes(snapshot.get(name).numpy()) == expected[name]
                    continue
                size = (size + 15) // 16 * 16
                offsets[name] = size
                frames.append(
                    DecodeFrame(
                        snapshot.index["tensors"][name]["offset"],
                        entry["frames"][0]["encoded_bytes"],
                        size,
                        len(expected[name]),
                    )
                )
                size += len(expected[name])
            decoder = NvcompDecoder(device, "snappy-zstd")
            with torch.cuda.stream(stream):
                workspace = decoder.allocate_workspace([frames])
            plan = decoder.prepare_batches([frames], source, workspace, stream)
            decoded = [
                torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)
            ]
            stream.wait_stream(torch.cuda.current_stream(device))
            plans = plan.bind_outputs(decoded)
            with torch.cuda.stream(stream):
                plans[0].enqueue()
                complete = torch.cuda.Event()
                complete.record(stream)
            complete.synchronize()
            assert plans[0].statuses.tolist() == [0] * len(frames)
            assert plans[0].actual_sizes.tolist() == [
                frame.decoded_bytes for frame in frames
            ]
            host_copy = decoded[0].cpu()
            for name, offset in offsets.items():
                assert (
                    bytes(host_copy[offset : offset + len(expected[name])].numpy())
                    == expected[name]
                )
            # Substitute the Miles all-original-engine-rank completion barrier with
            # an explicit two-process completion barrier in this isolated oracle.
            barrier.wait(timeout=90)
            snapshot.mark_reusable()
            snapshot.close()
            source = None
            assert arena.allocation is not None
            del plan, plans, workspace, decoded
            properties, granularity = memory._AllocationProperties(), ctypes.c_size_t()
            driver = arena.allocation.driver
            memory._check(
                driver.cuMemGetAllocationPropertiesFromHandle(
                    ctypes.byref(properties), arena.allocation.handle.value
                ),
                "native allocation properties",
            )
            memory._check(
                driver.cuMemGetAllocationGranularity(
                    ctypes.byref(granularity), ctypes.byref(properties), 0
                ),
                "native allocation granularity",
            )
            records.append(
                dict(
                    version=version,
                    host_cache_id=identity,
                    arena_identity=arena.capacity["identity"],
                    local_names=sorted(names),
                    mapping_pointer=arena.tensor.data_ptr(),
                    metrics=metrics,
                    exact_host_de_bytes=True,
                    allocation_granularity=granularity.value,
                )
            )
        stream.synchronize()
        arena.close()
        assert arena.allocation is None and arena.mapping is None
        output.put(
            dict(rank=rank, engine_id=engine, updates=records, final_unmapped=True)
        )
    finally:
        stream.synchronize()
        arena.close()
        pool.close()


@pytest.mark.parametrize("workers", [4, 8])
def test_two_engines_reuse_host_de_capacity_and_grow(workers):
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
                assert rows[0]["arena_identity"] != rows[1]["arena_identity"]
                assert set(rows[0]["local_names"]) & set(rows[1]["local_names"]) == {
                    "raw"
                }
                assert (
                    sum(row["metrics"]["host_encoded_cache_created"] for row in rows)
                    == 1
                )
                assert (
                    sum(row["metrics"]["host_encoded_cache_hash_files"] for row in rows)
                    == 1
                )
                assert (
                    sum(
                        row["metrics"]["host_encoded_cache_frames_validations"]
                        for row in rows
                    )
                    == 1
                )
                assert (
                    sum(row["metrics"]["host_rank_outer_zstd_tensors"] for row in rows)
                    == 8
                )
                assert sum(
                    row["metrics"]["host_rank_allocation_calls"] for row in rows
                ) == (0 if version == 1 else 2)
                assert all(
                    row["metrics"]["host_rank_mapping_reused"] == int(version == 1)
                    for row in rows
                )
        assert len({record["updates"][0]["host_cache_id"] for record in records}) == 2
        assert len({record["updates"][0]["arena_identity"] for record in records}) == 4
        for record in records:
            a, b, c = record["updates"]
            for row, multiplier in ((a, 1), (c, 2)):
                alignment = max(1 << 20, row["allocation_granularity"])
                assert row["metrics"]["host_rank_capacity_bytes"] == (
                    (
                        multiplier * row["metrics"]["host_rank_arena_bytes"]
                        + alignment
                        - 1
                    )
                    // alignment
                    * alignment
                )
            assert a["arena_identity"] == b["arena_identity"] != c["arena_identity"]
            assert a["mapping_pointer"] == b["mapping_pointer"] != c["mapping_pointer"]
            assert [
                row["metrics"]["host_rank_capacity_generation"]
                for row in record["updates"]
            ] == [1, 1, 2]
            assert all(
                row["metrics"]["host_rank_arena_bytes"]
                <= row["metrics"]["host_rank_capacity_bytes"]
                for row in record["updates"]
            )
            assert record["final_unmapped"]
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
