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
def test_host_input_plans_split_output_and_status_slots(monkeypatch, name, stages):
    # CPU-backed device views exercise the actual slab construction and C ABI
    # arguments without requiring CUDA, nvCOMP or a host-DE allocation locally.
    device = torch.device("cuda", 0)

    class DeviceView:
        def __init__(self, value):
            self.value = value
            self.device = device
            self.pointer_queries = self.size_queries = 0

        def data_ptr(self):
            self.pointer_queries += 1
            return self.value.data_ptr()

        def numel(self):
            self.size_queries += 1
            return self.value.numel()

        def __getattr__(self, name):
            return getattr(self.value, name)

        def __getitem__(self, key):
            return DeviceView(self.value[key])

        def copy_(self, source, non_blocking):
            assert non_blocking
            uploads.append(tuple(self.value.shape))
            self.value.copy_(source)
            return self

    empty = torch.empty
    uploads, admitted = [], []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.weight_sync.gpu_delta.memory",
        SimpleNamespace(require_de_capable=admitted.append),
    )

    def allocate(shape, dtype, device, **kwargs):
        value = empty(shape, dtype=dtype, device="cpu")
        return value if str(device) == "cpu" else DeviceView(value)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = device
    decoder.backend = "hardware"
    decoder.inner_codec = name
    decoder.maximum_chunk_bytes = 4 << 20
    decoder._algorithm, options_type = codec._DECOMPRESS_OPTIONS[name]
    decoder._options = options_type()
    decoder.alignments = codec._Alignments(16, 16, 16)
    decoder.temporary_bytes = lambda geometry: 64 if geometry[0] else 0
    launches = []
    decoder._decode = lambda *arguments: launches.append(arguments) or 0
    stream = SimpleNamespace(device=device, cuda_stream=1234)
    host = allocate(512, torch.uint8, "cpu")
    batches = [
        [(0, 20, 64, 0), (32, 17, 32, 128)],
        [(64, 25, 128, 16)],
        [(96, 30, 64, 32)],
    ]
    batches = [batches[index % len(batches)] for index in range(7)]
    count = sum(map(len, batches))
    table, counts = _frame_table(batches)
    original_table = table.copy()
    prepared_plan = decoder.prepare_batches(table, counts, host, stream, stages)
    workspace = prepared_plan.workspace
    np.testing.assert_array_equal(table, original_table)
    # The caller can release/reuse its numeric table after preparation; pinned
    # input/size rows and delayed output offsets are owned by the plan.
    table.fill(-1)
    # Preparation uploads input/size metadata without any output allocation,
    # output pointer upload or decompression. Paused binding fills just row 3.
    assert uploads == [(3, count)] and not launches
    original_batches = tuple(prepared_plan.batches)
    assert not prepared_plan.decoded_slots
    outputs = tuple(allocate(256, torch.uint8, device) for _ in range(stages))
    plans = prepared_plan.bind_outputs(outputs)
    assert tuple(plans) == original_batches
    assert all(plan.decoded_slots is prepared_plan.decoded_slots for plan in plans)
    assert all(output.pointer_queries == output.size_queries == 1 for output in outputs)
    assert uploads == [(3, count), (count,)] and not launches
    assert workspace.statuses.shape == workspace.actual_sizes.shape == (stages, 2)
    assert plans[0].metadata.stride(0) == count
    assert plans[0].metadata.untyped_storage().data_ptr() == (
        plans[1].metadata.untyped_storage().data_ptr()
    )
    assert plans[0].host_metadata.untyped_storage().data_ptr() == (
        plans[1].host_metadata.untyped_storage().data_ptr()
    )
    for index, (plan, frames) in enumerate(zip(plans, batches)):
        slot = index % stages
        assert plan.host_input is host
        assert plan.decoded_slots[slot] is outputs[slot]
        assert plan.stream is stream
        assert plan.metadata.value.tolist() == [
            [host.data_ptr() + frame[0] for frame in frames],
            [frame[1] for frame in frames],
            [frame[2] for frame in frames],
            [outputs[slot].data_ptr() + frame[3] for frame in frames],
        ]
        plan.enqueue()
        arguments = launches[-1]
        assert arguments[:4] == (
            plan.metadata[0].data_ptr(),
            plan.metadata[1].data_ptr(),
            plan.expected_sizes.data_ptr(),
            workspace.actual_sizes[slot].data_ptr(),
        )
        assert arguments[4:8] == (
            len(frames),
            workspace.temporary.data_ptr(),
            workspace.temporary.numel(),
            plan.metadata[3].data_ptr(),
        )
        assert arguments[9:] == (workspace.statuses[slot].data_ptr(), 1234)
        assert type(arguments[8]) is options_type
    # The first wrap reuses slot zero; slot one's delayed apply can still
    # read its status and size result. A single DE stream owns the temp buffer.
    assert plans[0].statuses.data_ptr() == plans[stages].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() == plans[stages].actual_sizes.data_ptr()
    assert plans[0].statuses.data_ptr() != plans[1].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() != plans[1].actual_sizes.data_ptr()
    plans[1].statuses.value.fill_(7)
    plans[1].actual_sizes.value.fill_(128)
    plans[stages].statuses.value.zero_()
    plans[stages].actual_sizes.value.fill_(64)
    assert plans[1].statuses.value.tolist() == [7]
    assert plans[1].actual_sizes.value.tolist() == [128]
    assert admitted == [
        workspace.temporary.data_ptr(),
        workspace.actual_sizes.data_ptr(),
        workspace.statuses.data_ptr(),
        plans[0].metadata.untyped_storage().data_ptr(),
        *(output.data_ptr() for output in outputs),
    ]

    # Overlapping pairs are rejected even when other slots are disjoint,
    # before an output-pointer upload or native submission.
    for slots, error in (
        ((outputs[0], outputs[1][:64], *outputs[2:]), "Out-of-bounds"),
        ((outputs[0], outputs[1][1:], *outputs[2:]), "Misaligned decoded"),
        ((outputs[0], outputs[0][16:], *outputs[2:]), "must not overlap"),
        (outputs[:1], "count differs"),
    ):
        with pytest.raises(ValueError, match=error):
            prepared_plan.bind_outputs(slots)
    # Numeric admission must reject bad rows before any allocation or upload,
    # including arithmetic which would otherwise wrap int64 vector operations.
    for rows, error in (
        ([(496, 32, 64, 0)], "outside input"),
        ([(np.iinfo(np.int64).max, 32, 64, 0)], "outside input"),
        ([(0, 32, 64, np.iinfo(np.int64).max)], "Output arithmetic"),
        ([(1, 32, 64, 0)], "Misaligned encoded"),
        ([(0, 32, 64, 0), (32, 32, 64, 32)], "Overlapping"),
        ([(0, 32, 64, 0), (32, 32, 64, 65)], "Misaligned decoded"),
    ):
        with pytest.raises(ValueError, match=error):
            decoder.prepare_batches(*_frame_table([rows]), host, stream, stages)
    for invalid in (
        original_table.astype(np.int32),
        original_table[:, ::-1],
        original_table[:, :-1],
    ):
        with pytest.raises(ValueError, match="contiguous int64"):
            decoder.prepare_batches(invalid, counts, host, stream, stages)
    assert len(uploads) == 2 and len(launches) == 7 and len(admitted) == 4 + stages

    # Compressible 4 MiB output is admitted without allocating output slots;
    # both actual lengths must fit the device, including an expanded input.
    maximum = decoder.maximum_chunk_bytes
    large = decoder.prepare_batches(
        *_frame_table([[(0, 20, maximum, 0)]]), host, stream, stages
    )
    assert large.output_bounds == [maximum] + [0] * (stages - 1)
    assert not large.decoded_slots and len(launches) == 7
    for encoded, decoded in ((maximum + 1, 64), (20, maximum + 1)):
        with pytest.raises(ValueError, match="device limit"):
            decoder.prepare_batches(
                *_frame_table([[(0, encoded, decoded, 0)]]), host, stream, stages
            )
    decoder.maximum_chunk_bytes = 1 << 20
    with pytest.raises(ValueError, match="device limit"):
        decoder.prepare_batches(
            *_frame_table([[(0, 20, maximum, 0)]]), host, stream, stages
        )
    assert len(uploads) == 3 and len(launches) == 7
    empty_plan = decoder.prepare_batches(*_frame_table([[], []]), host, stream, stages)
    assert empty_plan.workspace.temporary.numel() == 0
    assert empty_plan.workspace.statuses.shape == (stages, 0)
    assert empty_plan.output_bounds == [0] * stages
    for batch in empty_plan.batches:
        batch.enqueue()
    assert len(uploads) == 3 and len(launches) == 7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
