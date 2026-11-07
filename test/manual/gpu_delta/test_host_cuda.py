"""Native host-arena and complete receiver checks.

Manual-only: Linux, CUDA, nvCOMP 6, Torch, Snappy, LZ4 and Zstandard are required.
Bitcomp uses one GPU and the Miles producer, without requiring DE. The two-engine
DE test requires two SM100/SM103 GPUs; its barriers coordinate the oracle, not
production preparation.
"""

import hashlib
import json
import mmap
import multiprocessing
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import lz4.block
import numpy as np
import pytest
import snappy
import zstandard as zstd


@pytest.mark.parametrize("codec", ["bitcomp", "bitcomp-zstd"])
def test_bitcomp_receiver_double_buffer_exact_updates_and_stale_target(
    codec, monkeypatch
):
    """Real producer/receiver, five batches, omitted frames and raw replacement."""
    import torch
    from miles.utils.gpu_delta.encoder import GpuBatchEncoder
    from miles.utils.gpu_delta.publication import PublicationWriter

    from sglang.srt.weight_sync.gpu_delta import checkpoint
    from sglang.srt.weight_sync.gpu_delta.bindings import _digest
    from sglang.srt.weight_sync.gpu_delta.host import host_cache_id
    from sglang.srt.weight_sync.gpu_delta.receiver import GpuDeltaBackend

    device = torch.device("cuda", torch.cuda.current_device())
    frame_bytes = 1 << 16
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"], num_hidden_layers=5
    )
    model.num_fused_shared_experts = 0
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
    for layer in model.model.layers:
        layer.weight = torch.nn.Parameter(
            torch.zeros((1, 2 * frame_bytes + 140), dtype=torch.uint8, device=device),
            requires_grad=False,
        )
    model.model.raw = torch.nn.Parameter(
        torch.zeros(17, dtype=torch.uint8, device=device), requires_grad=False
    )
    inventory = {
        name: {"dtype": "U8", "shape": list(value.shape)}
        for name, value in model.named_parameters()
    }
    monkeypatch.setattr(
        checkpoint, "read_canonical_checkpoint_inventory", lambda _: inventory
    )
    monkeypatch.setenv("GPU_DELTA_CPU_WORKERS", "4")
    monkeypatch.setenv("GPU_DELTA_LAYERS_PER_BATCH", "1")
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "0")
    encoder = GpuBatchEncoder(device, codec=codec, frame_bytes=frame_bytes)
    with (
        tempfile.TemporaryDirectory(
            prefix="gpu-delta-bitcomp-", dir="/dev/shm"
        ) as directory,
        ThreadPoolExecutor(max_workers=1) as prepare_executor,
    ):
        root = Path(directory)
        monkeypatch.setenv("GPU_DELTA_HOST_CACHE_DIR", str(root / "cache"))
        identity = dict(
            engine_id=directory,
            host_cache_id=host_cache_id(directory),
            tp_rank=0,
            dp_rank=0,
            pid=os.getpid(),
            rank_id="bitcomp-original-rank",
        )
        backend = GpuDeltaBackend(SimpleNamespace(model=model), identity)
        definitions = backend.describe()["tensors"]
        plan_digest = _digest(sorted(definitions, key=lambda row: row["name"]))
        prepared = None
        try:
            for version in range(1, 4):
                before = {
                    name: value.detach().cpu().flatten().pin_memory()
                    for name, value in model.named_parameters()
                }
                after = {
                    name: value.clone().pin_memory() for name, value in before.items()
                }
                for name, value in after.items():
                    if name == "model.raw":
                        value.bitwise_xor_(version)
                    else:
                        value[:frame_bytes:4096].bitwise_xor_(version)
                        value[-140:].bitwise_xor_(version + 7)
                matrices = [
                    row for row in definitions if row["encoding"] == "xor_bytes"
                ]
                encoded = encoder.finish_device(
                    encoder.encode_device(
                        [(before[row["name"]], after[row["name"]]) for row in matrices]
                    )
                )
                writer = PublicationWriter(
                    root / str(version),
                    "tiny-receiver",
                    version - 1,
                    version,
                    plan_digest,
                    codec,
                    frame_bytes=frame_bytes,
                )
                try:
                    for row, result in zip(matrices, encoded, strict=True):
                        writer.add_encoded_tensor(
                            row["name"],
                            *result[:4],
                            dtype=row["dtype"],
                            shape=row["shape"],
                            views=row["views"],
                        )
                    raw = next(row for row in definitions if row["name"] == "model.raw")
                    writer.add_raw_tensor(
                        raw["name"],
                        before[raw["name"]].numpy(),
                        after[raw["name"]].numpy(),
                        raw["dtype"],
                        raw["shape"],
                        raw["views"],
                    )
                    descriptor = writer.finish()
                finally:
                    writer.close()
                metadata = dict(
                    descriptor,
                    session_id="tiny-receiver",
                    validate_payload_hash=True,
                    participants=[identity],
                )
                # Match the scheduler's persistent preparation worker: CUDA
                # allocations on this foreground thread do not bind its context.
                prepared = prepare_executor.submit(
                    backend.prepare,
                    descriptor["manifest_path"],
                    descriptor["manifest_sha256"],
                    metadata,
                ).result()
                assert not prepared.hardware
                assert (
                    not prepared.decode_plan.decoded_slots
                    and not prepared.decode_plan.input_slots
                )
                assert backend.host_arena.tensor.is_pinned()
                assert prepared.timings["compressed_batches"] == 5
                assert prepared.timings["decoded_zero_bytes"] == 5 * frame_bytes
                assert prepared.timings["host_batch_plan_reused"] == int(version > 1)
                for name, value in model.named_parameters():
                    assert torch.equal(value.cpu().flatten(), before[name])
                if version == 3:
                    model.model.layers[0].weight = torch.nn.Parameter(
                        model.model.layers[0].weight.clone(), requires_grad=False
                    )
                    with pytest.raises(
                        RuntimeError, match="parameter identity changed"
                    ):
                        prepared.apply()
                    for name, value in model.named_parameters():
                        assert torch.equal(value.cpu().flatten(), before[name])
                    prepared.close()
                else:
                    result = prepared.apply()
                    assert result["applied"] and result["target_version"] == version
                    assert result["timings"]["compressed_input_buffers"] == 2
                    assert result["timings"]["decoded_buffers"] == 2
                    assert result["timings"]["compressed_input_h2d_bytes"] > 0
                    assert result["timings"]["de_host_input_bytes"] == 0
                    for name, value in model.named_parameters():
                        assert torch.equal(value.cpu().flatten(), after[name])
                    prepared.release_and_close()
                prepared = None
        finally:
            if prepared is not None:
                prepared.close()
            backend.close()


def _pinned_mapping_consumer(rank, path, expected, barrier, output):
    import torch

    from sglang.srt.weight_sync.gpu_delta.memory import HostMapping

    torch.cuda.set_device(rank)
    stream = torch.cuda.Stream()
    with Path(path).open("r+b") as source:
        inode = (os.fstat(source.fileno()).st_dev, os.fstat(source.fileno()).st_ino)
        mapping = mmap.mmap(source.fileno(), 0)
    allocation = HostMapping(mapping, True)
    tensor = torch.frombuffer(allocation.view, dtype=torch.uint8)
    try:
        assert tensor.is_pinned()
        with torch.cuda.stream(stream):
            copied = tensor.to("cuda", non_blocking=True)
        stream.synchronize()
        assert bytes(copied.cpu().numpy()) == expected
        barrier.wait(timeout=90)
        if rank == 0:
            tensor = None
            allocation.close()
        barrier.wait(timeout=90)
        if rank == 1:
            # Unregistering one rank must leave its peer's registration usable.
            with torch.cuda.stream(stream):
                copied = tensor.to("cuda", non_blocking=True)
            stream.synchronize()
            assert bytes(copied.cpu().numpy()) == expected
        output.put((rank, inode))
    finally:
        stream.synchronize()
        tensor = None
        allocation.close()
        mapping.close()


def test_shared_pinned_mapping_survives_peer_unregister():
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gpu-delta-shared-pin-", dir="/dev/shm"
    ) as directory:
        path = Path(directory) / "prepared.bin"
        expected = random.Random(189).randbytes(1 << 20)
        path.write_bytes(expected)
        barrier, output = context.Barrier(2), context.Queue()
        children = [
            context.Process(
                target=_pinned_mapping_consumer,
                args=(rank, str(path), expected, barrier, output),
            )
            for rank in range(2)
        ]
        try:
            for child in children:
                child.start()
            rows = [output.get(timeout=120) for _ in children]
            for child in children:
                child.join(timeout=30)
                assert child.exitcode == 0
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=15)
        assert {row[0] for row in rows} == {0, 1}
        assert len({row[1] for row in rows}) == 1
        assert path.read_bytes() == expected


def _publication(directory, version, repeat, codec):
    expected = {
        f"tensor-{i}": random.Random(i + version).randbytes(256 * repeat)
        for i in range(8)
    }
    expected["raw"] = b"unaligned-raw-target"
    blob, entries = bytearray(), []
    for name, value in expected.items():
        blob.extend(bytes((-len(blob)) % 16))
        start = len(blob)
        inner = (
            value
            if name == "raw"
            else (
                snappy.compress(value)
                if codec == "snappy-zstd"
                else lz4.block.compress(value, store_size=False)
            )
        )
        encoded, outer_frames = bytearray(), []
        if name == "raw" or codec == "lz4":
            encoded.extend(inner)
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
            "frames": (
                []
                if name == "raw"
                else [
                    {
                        "encoded_offset": 0,
                        "encoded_bytes": len(inner),
                        "decoded_offset": 0,
                        "decoded_bytes": len(value),
                    }
                ]
            ),
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
    for entry in entries:
        size = len(expected["raw"]) if entry["name"] == "raw" else 1 << 20
        shape = [size] if entry["encoding"] == "raw_bytes" else [1, size]
        entry.update(
            dtype="U8",
            shape=shape,
            nbytes=size,
            byte_order="little",
            views=[{"id": entry["name"], "slices": [[0, n] for n in shape]}],
        )
    manifest = {
        "codec": codec,
        "base_version": version - 1,
        "target_version": version,
        "plan_digest": "native-host-plan",
        "frame_bytes": (1 << 19) if version < 3 else (1 << 20),
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
    from sglang.srt.weight_sync.gpu_delta.codec import NvcompDecoder

    os.environ["GPU_DELTA_HOST_CACHE_DIR"] = cache
    # Exercise the exact capacity-growth algorithm with small oracle tensors.
    # Production's coarser alignment is not a wire/codec requirement.
    host._CAPACITY_ALIGNMENT = 1 << 20
    torch.cuda.set_device(rank % 2)
    device = torch.device("cuda", rank % 2)
    os.environ["GPU_DELTA_CPU_WORKERS"] = str(workers)
    identity = host.host_cache_id(engine)
    participants = [
        dict(
            engine_id=engine,
            host_cache_id=identity,
            tp_rank=local,
            dp_rank=0,
            pid=100 + local,
            rank_id=f"{engine}-original-{local}",
        )
        for local in range(2)
    ]
    arena = host.HostArena(engine, device.index, participants[rank % 2])
    stream = torch.cuda.Stream(device=device)
    records = []
    try:
        names = {f"tensor-{i}" for i in range(4 * (rank % 2), 4 * (rank % 2 + 1))} | {
            "raw"
        }
        for version, (path, digest, expected) in enumerate(publications, 1):
            metadata = dict(
                stream_id="native-shared-stream",
                session_id=f"update-{version}",
                base_version=version - 1,
                target_version=version,
                plan_digest="native-host-plan",
                validate_payload_hash=True,
                participants=participants,
            )
            barrier.wait(timeout=90)
            metrics = {}
            publication = arena.prepare_publication(path, digest, metadata, metrics)
            codec = publication.header["codec"]
            if version == 1:
                ordinals = {
                    entry["name"]: i
                    for i, entry in enumerate(publication.read_catalog()["tensors"])
                }
            selected = [ordinals[name] for name in sorted(names)]
            snapshot = arena.prepare_local(publication, selected, metrics)
            assert metrics["host_rank_staging_copy_bytes"] == int(
                publication.tensors["prepared_nbytes"][selected].sum()
            )
            source = arena.tensor
            frames, offsets, size = [], {}, 0
            for name in sorted(names):
                ordinal = ordinals[name]
                if name == "raw":
                    assert bytes(snapshot.get(ordinal)) == expected[name]
                    continue
                size = (size + 15) // 16 * 16
                offsets[name] = size
                row = publication.tensors[ordinal]
                assert row["frame_count"] == 1
                frame = publication.frames[:, row["frame_start"]]
                frames.append(
                    (
                        int(snapshot.offsets[ordinal] + frame[0]),
                        int(frame[1]),
                        int(frame[2]),
                        size + int(frame[3]),
                    )
                )
                size += len(expected[name])
            decoder = NvcompDecoder(device, codec.removesuffix("-zstd"))
            table = np.asarray(frames, dtype=np.int64).T.copy()
            plan = decoder.prepare_batches(table, [len(frames)], source, stream)
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
            assert plans[0].actual_sizes.tolist() == [frame[2] for frame in frames]
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
            del plan, plans, decoded
            driver = arena.allocation.driver
            properties = memory._check(
                driver.cuMemGetAllocationPropertiesFromHandle(arena.allocation.handle),
                "native allocation properties",
            )
            granularity = memory._check(
                driver.cuMemGetAllocationGranularity(
                    properties,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
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
                    allocation_granularity=granularity,
                )
            )
        stream.synchronize()
        arena.close()
        assert arena.allocation is None and arena.mapping is None
        assert arena.shared_allocation is None
        output.put(
            dict(rank=rank, engine_id=engine, updates=records, final_unmapped=True)
        )
    finally:
        stream.synchronize()
        arena.close()


@pytest.mark.parametrize("workers", [4, 8])
def test_two_engines_reuse_host_de_capacity_and_grow(workers):
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gpu-delta-native-", dir="/dev/shm"
    ) as directory:
        root = Path(directory)
        publications = []
        for version, repeat in enumerate((2048, 1024, 4096), 1):
            version_dir = root / str(version)
            version_dir.mkdir()
            path, digest, expected = _publication(
                version_dir, version, repeat, ("lz4", "snappy-zstd", "lz4")[version - 1]
            )
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
                assert (
                    sum(row["metrics"]["host_rank_cpu_workers"] for row in rows)
                    == workers
                )
                assert (
                    sum(row["metrics"]["host_prepared_cache_created"] for row in rows)
                    == 1
                )
                assert (
                    sum(row["metrics"]["host_prepared_cache_reused"] for row in rows)
                    == 1
                )
                assert all(
                    row["metrics"]["host_rank_staging_copy_bytes"] > 0 for row in rows
                )
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
                assert sum(
                    row["metrics"]["host_rank_outer_zstd_tensors"] for row in rows
                ) == (8 if version == 1 else 0)
                assert sum(
                    row["metrics"]["host_rank_encoded_copy_tensors"] for row in rows
                ) == (0 if version == 1 else 8)
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
