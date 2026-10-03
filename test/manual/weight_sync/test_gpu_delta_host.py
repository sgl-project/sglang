"""Two independent CUDA processes consume one host-shared decoded arena.

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


def _publication(directory):
    expected = {
        f"tensor-{i}": bytes((j + i) % 256 for j in range(256)) * 1024 for i in range(8)
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
            "frames": [] if name == "raw" else [{}],
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


def _consumer(rank, workers, path, digest, expected, cache, barrier, removed, output):
    import torch
    from sglang.srt.weight_sync.gpu_delta_host import HostDecodedSnapshot, host_cache_id
    from sglang.srt.weight_sync.gpu_delta_payload import OuterZstdPool

    os.environ["WEIGHT_DELTA_HOST_CACHE_DIR"] = cache
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    pool, snapshot = OuterZstdPool(workers), None
    try:
        identity = host_cache_id()
        barrier.wait(timeout=90)
        metrics = {}
        snapshot = HostDecodedSnapshot(
            path,
            digest,
            json.loads(Path(path).read_text()),
            sorted(expected),
            pool,
            metrics,
        )
        snapshot.register(device, metrics)
        assert snapshot.tensor.is_pinned() and snapshot.registered
        source = snapshot.tensor
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            copied = source.to(device=device, non_blocking=True)
            complete = torch.cuda.Event()
            complete.record(stream)
        complete.synchronize()
        host_copy = copied.cpu()
        for name, value in expected.items():
            record = snapshot.index["tensors"][name]
            actual = host_copy[record["offset"] : record["offset"] + record["nbytes"]]
            assert bytes(actual.numpy()) == value
        # All readers and H2D transfers are complete before normal disposal.
        # One engine can unlink names while another retains its own valid map.
        barrier.wait(timeout=90)
        inode = snapshot.index["arena_identity"]
        if rank == 0:
            source = None
            snapshot.close(discard=True)
            removed.set()
        else:
            assert removed.wait(timeout=90)
            assert not snapshot.directory.exists()
            for name, value in expected.items():
                assert bytes(snapshot.get(name).numpy()) == value
            source = None
            snapshot.close(discard=True)
        assert not snapshot.registered and snapshot.mapping is None
        output.put(
            {
                "rank": rank,
                "host_cache_id": identity,
                "arena_identity": inode,
                "metrics": metrics,
                "exact_h2d_bytes": True,
                "registered_and_unregistered": True,
            }
        )
    finally:
        if snapshot is not None:
            stream = locals().get("stream")
            if stream is not None:
                stream.synchronize()
            snapshot.close()
        pool.close()


@pytest.mark.parametrize("workers", [4, 8])
def test_two_engines_share_decode_and_register_independent_mappings(workers):
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gpu-delta-native-", dir="/dev/shm"
    ) as directory:
        root = Path(directory)
        path, digest, expected = _publication(root)
        barrier, removed, output = context.Barrier(2), context.Event(), context.Queue()
        processes = [
            context.Process(
                target=_consumer,
                args=(
                    rank,
                    workers,
                    str(path),
                    digest,
                    expected,
                    str(root / "cache"),
                    barrier,
                    removed,
                    output,
                ),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            records = [output.get(timeout=180) for _ in processes]
            for process in processes:
                process.join(timeout=90)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=15)
        assert len({record["host_cache_id"] for record in records}) == 1
        assert records[0]["arena_identity"] == records[1]["arena_identity"]
        assert (
            sum(record["metrics"]["host_payload_cache_created"] for record in records)
            == 1
        )
        assert (
            sum(record["metrics"]["host_payload_cache_reused"] for record in records)
            == 1
        )
        assert (
            sum(record["metrics"]["host_payload_hash_files"] for record in records) == 1
        )
        assert (
            sum(record["metrics"]["host_outer_zstd_tensors"] for record in records) == 8
        )
        assert all(
            record["metrics"]["host_shared_register_calls"] == 1 for record in records
        )
        assert not list((root / "cache").glob("*/arena.bin"))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "cpu_workers": workers,
                    "logical_engines": 2,
                    "records": records,
                },
                sort_keys=True,
            )
        )
