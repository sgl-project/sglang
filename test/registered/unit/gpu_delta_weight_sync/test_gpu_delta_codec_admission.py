"""CPU codec admission, outer payload integrity and prepared metadata ownership."""

import ctypes
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta.payload import HostPayloadPool, validate_codec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta/codec.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_codec_under_test", _path)
codec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = codec
_spec.loader.exec_module(codec)


def _frame_table(batches):
    rows = [row for batch in batches for row in batch]
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T.copy()
    return table, list(map(len, batches))


@pytest.mark.parametrize(
    "name,algorithm,options_type,reserved_offset",
    [
        ("snappy", "Snappy", codec._SnappyOptions, 8),
        ("lz4", "LZ4", codec._Lz4Options, 16),
    ],
)
@pytest.mark.parametrize("sorting", ["0", "1"])
def test_codec_abi_and_frozen_hardware_options(
    monkeypatch, name, algorithm, options_type, reserved_offset, sorting
):
    # Match nvCOMP 5.3 shared_types.h, snappy.h and lz4.h. In particular, LZ4
    # inserts two enums before reserved bytes despite both structs being 64B.
    assert ctypes.sizeof(options_type) == 64
    assert ctypes.alignment(options_type) == ctypes.alignment(ctypes.c_int)
    assert options_type.backend.offset == 0
    assert options_type.sort_before_hw_decompress.offset == 4
    assert options_type.reserved.offset == reserved_offset
    if name == "lz4":
        assert options_type.data_type.offset == 8
        assert options_type.bitshuffle_mode.offset == 12
    monkeypatch.setenv("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", sorting)
    calls, device_queries = [], []
    attributes = {136: {"Snappy": 2, "LZ4": 4}[algorithm], 137: 4 << 20}

    def device_attribute(attribute, device):
        device_queries.append((attribute, device))
        return 0, attributes[attribute]

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.gpu_delta.memory",
        SimpleNamespace(
            _driver=lambda: SimpleNamespace(
                CUdevice_attribute=SimpleNamespace(
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK=136,
                    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH=137,
                ),
                CUmemDecompressAlgorithm=SimpleNamespace(
                    CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY=2,
                    CU_MEM_DECOMPRESS_ALGORITHM_LZ4=4,
                ),
                cuDeviceGetAttribute=device_attribute,
            )
        ),
    )

    class Function:
        def __init__(self, suffix):
            self.suffix = suffix

        def __call__(self, *arguments):
            options = arguments[0 if self.suffix == "GetRequiredAlignments" else 2]
            assert type(options) is options_type
            calls.append((self.suffix, bytes(options)))
            if self.suffix == "GetRequiredAlignments":
                ctypes.cast(arguments[1], ctypes.POINTER(codec._Alignments))[0] = (
                    codec._Alignments(16, 16, 16)
                )
            else:
                ctypes.cast(arguments[3], ctypes.POINTER(ctypes.c_size_t))[0] = 64
            return 0

    functions = {
        "nvcompBatched" + algorithm + "Decompress" + suffix: Function(suffix)
        for suffix in ("GetRequiredAlignments", "GetTempSizeAsync", "Async")
    }
    distribution = SimpleNamespace(version="5.3.0.16", locate_file=lambda path: path)
    monkeypatch.setattr(
        codec.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(codec.ctypes, "CDLL", lambda _: SimpleNamespace(**functions))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    for allocator_api in (
        "get_allocator_backend",
        "_snapshot",
        "_set_allocator_settings",
    ):
        monkeypatch.setattr(
            torch.cuda.memory,
            allocator_api,
            lambda *a: pytest.fail("global allocator access"),
        )
    decoder = codec.NvcompDecoder(torch.device("cuda", 0), name)
    assert decoder.inner_codec == name and decoder.backend == "hardware"
    assert decoder.maximum_chunk_bytes == 4 << 20
    assert decoder._options.backend == 1
    assert decoder._options.sort_before_hw_decompress == int(sorting)
    assert bytes(decoder._options)[8:] == bytes(56)
    align_function = functions[
        f"nvcompBatched{algorithm}DecompressGetRequiredAlignments"
    ]
    assert align_function.argtypes == [options_type, ctypes.POINTER(codec._Alignments)]
    assert decoder._temporary.argtypes == [
        ctypes.c_size_t,
        ctypes.c_size_t,
        options_type,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    assert decoder._decode.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        options_type,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    assert decoder._decode.restype is ctypes.c_int
    # Alignment and workspace queries must receive the same admitted options.
    geometry = (1, 64, 64)
    assert decoder.temporary_bytes(geometry) == 64
    assert [suffix for suffix, _ in calls] == [
        "GetRequiredAlignments",
        "GetTempSizeAsync",
    ]
    assert calls[0][1] == calls[1][1]
    assert decoder.temporary_bytes(geometry) == 64
    assert device_queries == [(136, 0), (137, 0)]
    # Allocation capability alone does not prove this codec exists. Frame
    # lengths are checked against the cached device limit during preparation.
    attributes[136] = 0
    with pytest.raises(RuntimeError, match="no hardware"):
        codec.NvcompDecoder(decoder.device, name)
    attributes[136] = {"Snappy": 2, "LZ4": 4}[algorithm]
    attributes[137] = 1 << 20
    assert codec.NvcompDecoder(decoder.device, name).maximum_chunk_bytes == 1 << 20
    attributes[137] = 0
    with pytest.raises(RuntimeError, match="positive hardware decompression limit"):
        codec.NvcompDecoder(decoder.device, name)


def test_workspace_query_cache_uses_exact_geometry_and_preserves_failure(monkeypatch):
    # Exercise the real query/cache code without a CUDA allocation or library.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cuda", 0)
    decoder.backend = "hardware"
    decoder.inner_codec = "snappy"
    decoder._options = codec._SnappyOptions()
    decoder._temporary_sizes = {}
    calls, fail = [], True

    def query(count, maximum, options, size, total):
        calls.append((count, maximum, total))
        if total == 31 and fail:
            return 1
        # Zero is also a valid cache entry; it must not trigger another query.
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    decoder._temporary = query
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    assert decoder.temporary_bytes((0, 0, 0)) == 0
    assert decoder.temporary_bytes((2, 20, 30)) == 0
    assert decoder.temporary_bytes((2, 20, 30)) == 0
    assert calls == [(2, 20, 30)]
    # Same count and maximum, different total: the native query must run again.
    with pytest.raises(RuntimeError, match="status=1"):
        decoder.temporary_bytes((2, 20, 31))
    fail = False
    assert decoder.temporary_bytes((2, 20, 31)) == 0
    assert decoder.temporary_bytes((2, 20, 31)) == 0
    assert calls == [(2, 20, 30), (2, 20, 31), (2, 20, 31)]


def test_prepared_stream_scratch_and_slot_status_ownership(monkeypatch):
    # Real tensor views/pointers exercise nonzero, non-aligned native workspace
    # requirements without a device. Native tests cover actual DE and ring reuse.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.alignments = codec._Alignments(16, 16, 16)
    decoder._options = codec._SnappyOptions()
    decoder.temporary_bytes = lambda geometry: geometry[0] * 13
    recorded = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.gpu_delta.memory",
        SimpleNamespace(
            device_empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
            require_de_capable=lambda _: None,
            record_device_stream=lambda tensor, stream: recorded.append(
                (tensor.untyped_storage().data_ptr(), stream.cuda_stream)
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    stream = SimpleNamespace(device=decoder.device, cuda_stream=1)
    counts = [1, 2, 0, 3] * 3
    workspace = decoder._allocate_workspace(
        [(count, 32, count * 32) for count in counts]
    )
    metadata = torch.zeros((4, sum(counts)), dtype=torch.int64)
    host_metadata = metadata.clone()
    plan = codec.PreparedDecodePlan(
        decoder,
        counts,
        torch.empty(128, dtype=torch.uint8),
        workspace,
        host_metadata,
        metadata,
        stream,
        np.zeros(sum(counts), dtype=np.int64),
        [128, 128],
        [0, 0],
    )
    assert workspace.temporary.shape == (39,)
    assert workspace.temporary.data_ptr() % decoder.alignments.temp == 0
    assert workspace.statuses.shape == workspace.actual_sizes.shape == (2, 3)
    for index, batch in enumerate(plan.batches):
        slot = index % 2
        assert batch.stream is stream
        assert batch._arguments[-1] == stream.cuda_stream
        assert batch._arguments[5:7] == (
            workspace.temporary.data_ptr(),
            workspace.temporary.numel(),
        )
        assert batch.statuses.storage_offset() == slot * 3
        assert batch.actual_sizes.storage_offset() == slot * 3
        if not counts[index]:
            batch.enqueue()  # Empty metadata must not call the native decoder.
    outputs = [torch.empty(128, dtype=torch.uint8) for _ in range(2)]
    assert plan.bind_outputs(outputs) is plan.batches
    consumer = SimpleNamespace(device=decoder.device, cuda_stream=2)
    plan.record_stream(consumer)
    with pytest.raises(ValueError, match="consumer stream/device mismatch"):
        plan.record_stream(SimpleNamespace(device=torch.device("cuda", 0)))
    for tensor in (
        metadata,
        workspace.temporary,
        workspace.statuses,
        workspace.actual_sizes,
        *outputs,
    ):
        assert {
            (tensor.untyped_storage().data_ptr(), s.cuda_stream)
            for s in (stream, consumer)
        } <= set(recorded)
    offset = 0
    for index, count in enumerate(counts):
        assert (
            metadata[3, offset : offset + count].tolist()
            == [outputs[index % 2].data_ptr()] * count
        )
        offset += count


def test_host_input_plans_reject_invalid_geometry_before_device_work(monkeypatch):
    # Native tests exercise successful slab construction/slot reuse. Keep the
    # vector arithmetic and output-lease failures here without a fake CUDA API.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.inner_codec = "snappy"
    decoder.maximum_chunk_bytes = 4 << 20
    decoder.alignments = codec._Alignments(16, 16, 16)
    table, counts = _frame_table([[(0, 20, 64, 0), (32, 17, 32, 128)]])
    geometry, bounds, remainders = decoder._frame_geometry(table, counts, 0, 512)
    assert (geometry, bounds, remainders) == (
        [(2, 64, 96)],
        [160, 0],
        [0, None],
    )
    for rows, error in (
        ([(496, 32, 64, 0)], "outside input"),
        ([(np.iinfo(np.int64).max, 32, 64, 0)], "outside input"),
        ([(0, 32, 64, np.iinfo(np.int64).max)], "Output arithmetic"),
        ([(1, 32, 64, 0)], "Misaligned encoded"),
        ([(0, 32, 64, 0), (32, 32, 64, 32)], "Overlapping"),
        ([(0, 32, 64, 0), (32, 32, 64, 65)], "Misaligned decoded"),
    ):
        with pytest.raises(ValueError, match=error):
            decoder._frame_geometry(*_frame_table([rows]), 0, 512)
    for invalid in (table.astype(np.int32), table[:, ::-1], table[:, :-1]):
        with pytest.raises(ValueError, match="contiguous int64"):
            decoder._frame_geometry(invalid, counts, 0, 512)
    maximum = decoder.maximum_chunk_bytes
    # A compressible full 4 MiB output is legal; either actual length exceeding
    # hardware's bound is not. This is independent of the inner algorithm.
    assert decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512)[
        1
    ] == [maximum, 0]
    for encoded, decoded in ((maximum + 1, 64), (20, maximum + 1)):
        with pytest.raises(ValueError, match="device limit"):
            decoder._frame_geometry(*_frame_table([[(0, encoded, decoded, 0)]]), 0, 512)
    decoder.maximum_chunk_bytes = 1 << 20
    with pytest.raises(ValueError, match="device limit"):
        decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512)
    assert decoder._frame_geometry(*_frame_table([[], []]), 0, 512) == (
        [(0, 0, 0), (0, 0, 0)],
        [0, 0],
        [None, None],
    )

    plan = object.__new__(codec.PreparedDecodePlan)
    plan.decoder = decoder
    plan.output_bounds, plan.output_remainders = [128, 128], [0, 0]
    plan.decoded_slots = []
    # Every malformed output must fail before CUDA context, allocation or upload.
    monkeypatch.setattr(torch.cuda, "device", lambda _: pytest.fail("entered CUDA"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.gpu_delta.memory",
        SimpleNamespace(
            device_empty=lambda *a, **kw: pytest.fail("allocated device storage"),
            require_de_capable=lambda _: pytest.fail("queried device"),
            record_device_stream=lambda *a: pytest.fail("recorded device"),
        ),
    )
    with pytest.raises(ValueError, match="Decoder stream/device mismatch"):
        decoder.prepare_batches(
            table,
            counts,
            torch.empty(512, dtype=torch.uint8),
            SimpleNamespace(device=torch.device("cuda", 0)),
        )
    a, b = [torch.empty(256, dtype=torch.uint8) for _ in range(2)]
    for slots, error in (
        ((a, b[:64]), "Out-of-bounds"),
        ((a, b[1:]), "Misaligned decoded"),
        ((a, a[16:]), "must not overlap"),
        ((a,), "count differs"),
        ((a, b, a), "count differs"),
    ):
        with pytest.raises(ValueError, match=error):
            plan.bind_outputs(slots)
        assert not plan.decoded_slots


class TestOuterZstd(unittest.TestCase):
    def test_manifest_codec_and_frame_geometry_admission(self):
        admitted = dict(codec="snappy-zstd", frame_bytes=1 << 20)
        for codec in ("snappy-zstd", "lz4-zstd", "lz4"):
            for size in (1 << 16, 192 << 10, 1 << 19, 1 << 20, 2 << 20, 4 << 20):
                with self.subTest(codec=codec, size=size):
                    validate_codec(admitted | {"codec": codec, "frame_bytes": size})
        for patch_value in (
            {"codec": "zstd"},
            {"codec": None},
            {"frame_bytes": 0},
            {"frame_bytes": -1},
            {"frame_bytes": (4 << 20) + 1},
            {"frame_bytes": 1.5},
            {"frame_bytes": True},
        ):
            with (
                self.subTest(patch=patch_value),
                self.assertRaisesRegex(ValueError, "codec"),
            ):
                validate_codec(admitted | patch_value)

    def test_gpu_outer_chunks_decode_directly_to_one_destination(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for known_size in (False, True):
            payload, chunks = bytearray(), []
            for index, value in enumerate(values):
                payload.extend(bytes((-len(payload)) % 16))
                compressed = zstd.ZstdCompressor(
                    level=1, write_content_size=known_size
                ).compress(value)
                chunks.append(
                    dict(
                        encoded_offset=len(payload),
                        encoded_bytes=len(compressed),
                        decoded_offset=index * (1 << 20),
                        decoded_bytes=len(value),
                    )
                )
                payload.extend(compressed)
            target = bytearray(sum(map(len, values)))
            pool.executor.submit(
                pool.decode_zstd, memoryview(payload), chunks, memoryview(target)
            ).result()
            self.assertEqual(target, b"".join(values))

    def test_invalid_frames_fail_without_publishing_a_decoded_result(self):
        raw = bytes(range(64)) * 128
        encoded = bytearray(zstd.ZstdCompressor(write_checksum=True).compress(raw))
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        chunk = dict(
            encoded_offset=0,
            encoded_bytes=len(encoded),
            decoded_offset=0,
            decoded_bytes=len(raw) + 1,
        )
        with self.assertRaisesRegex(ValueError, "truncated"):
            pool.executor.submit(
                pool.decode_zstd,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw) + 1)),
            ).result()
        chunk["decoded_bytes"] = len(raw)
        encoded[-1] ^= 1
        with self.assertRaises(zstd.ZstdError):
            pool.executor.submit(
                pool.decode_zstd,
                memoryview(encoded),
                [chunk],
                memoryview(bytearray(len(raw))),
            ).result()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
