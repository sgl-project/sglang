"""nvCOMP ABI/options, hardware admission and prepared metadata ownership."""

import ctypes
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
    "backend,settings,segments,error",
    [
        ("cudaMallocAsync", False, [], "native CUDA"),
        ("native", True, [], "expandable_segments"),
        ("native", None, [], "expandable_segments"),
        ("native", False, [{"device": 0, "is_expandable": True}], "existing PyTorch"),
        ("native", False, [{"device": 1, "is_expandable": True}], None),
    ],
)
def test_hardware_allocator_uses_effective_settings(
    monkeypatch, backend, settings, segments, error
):
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: backend)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: {
            "allocator_settings": {"expandable_segments": settings},
            "segments": segments,
        },
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            codec._require_hardware_allocator(torch.device("cuda", 0))
    else:
        codec._require_hardware_allocator(torch.device("cuda", 0))


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
    calls, admissions, device_queries = [], [], []
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
    monkeypatch.setattr(codec, "_require_hardware_allocator", admissions.append)
    decoder = codec.NvcompDecoder(torch.device("cuda", 0), name)
    assert decoder.inner_codec == name and decoder.backend == "hardware"
    assert decoder.maximum_chunk_bytes == 4 << 20
    assert admissions == [decoder.device]
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


@pytest.mark.parametrize("name", ["snappy", "lz4"])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_host_input_plans_reject_invalid_geometry_before_device_work(
    monkeypatch, name, stages
):
    # Native tests exercise successful slab construction/slot reuse. Keep the
    # vector arithmetic and output-lease failures here without a fake CUDA API.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.inner_codec = name
    decoder.maximum_chunk_bytes = 4 << 20
    decoder.alignments = codec._Alignments(16, 16, 16)
    table, counts = _frame_table([[(0, 20, 64, 0), (32, 17, 32, 128)]])
    geometry, bounds, remainders = decoder._frame_geometry(
        table, counts, 0, 512, stages
    )
    assert (geometry, bounds, remainders) == (
        [(2, 64, 96)],
        [160] + [0] * (stages - 1),
        [0] + [None] * (stages - 1),
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
            decoder._frame_geometry(*_frame_table([rows]), 0, 512, stages)
    for invalid in (table.astype(np.int32), table[:, ::-1], table[:, :-1]):
        with pytest.raises(ValueError, match="contiguous int64"):
            decoder._frame_geometry(invalid, counts, 0, 512, stages)
    maximum = decoder.maximum_chunk_bytes
    # A compressible full 4 MiB output is legal; either actual length exceeding
    # hardware's bound is not. This is independent of the inner algorithm.
    assert decoder._frame_geometry(
        *_frame_table([[(0, 20, maximum, 0)]]), 0, 512, stages
    )[1] == [maximum] + [0] * (stages - 1)
    for encoded, decoded in ((maximum + 1, 64), (20, maximum + 1)):
        with pytest.raises(ValueError, match="device limit"):
            decoder._frame_geometry(
                *_frame_table([[(0, encoded, decoded, 0)]]), 0, 512, stages
            )
    decoder.maximum_chunk_bytes = 1 << 20
    with pytest.raises(ValueError, match="device limit"):
        decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512, stages)
    assert decoder._frame_geometry(*_frame_table([[], []]), 0, 512, stages) == (
        [(0, 0, 0), (0, 0, 0)],
        [0] * stages,
        [None] * stages,
    )

    plan = object.__new__(codec.PreparedDecodePlan)
    plan.decoder, plan.slot_count = decoder, stages
    plan.output_bounds, plan.output_remainders = [128] * stages, [0] * stages
    plan.decoded_slots = []
    # Every malformed output must fail before CUDA context, allocation or upload.
    monkeypatch.setattr(torch.cuda, "device", lambda _: pytest.fail("entered CUDA"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.gpu_delta.memory",
        SimpleNamespace(require_de_capable=lambda _: pytest.fail("queried device")),
    )
    outputs = [torch.empty(256, dtype=torch.uint8) for _ in range(stages)]
    a, b, *rest = outputs
    for slots, error in (
        ((a, b[:64], *rest), "Out-of-bounds"),
        ((a, b[1:], *rest), "Misaligned decoded"),
        ((a, a[16:], *rest), "must not overlap"),
        ((a,), "count differs"),
    ):
        with pytest.raises(ValueError, match=error):
            plan.bind_outputs(slots)
        assert not plan.decoded_slots


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
