"""nvCOMP ABI/options, hardware admission and prepared metadata ownership."""

import ctypes
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta_codec.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_codec_under_test", _path)
codec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = codec
_spec.loader.exec_module(codec)


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
        ("snappy-zstd", "Snappy", codec._SnappyOptions, 8),
        ("lz4-zstd", "LZ4", codec._Lz4Options, 16),
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
    if name == "lz4-zstd":
        assert options_type.data_type.offset == 8
        assert options_type.bitshuffle_mode.offset == 12
    monkeypatch.setenv("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", sorting)
    calls, admissions = [], []

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
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(codec, "_require_hardware_allocator", admissions.append)
    decoder = codec.NvcompDecoder(torch.device("cuda", 0), name)
    assert decoder.codec == name and decoder.backend == "hardware"
    assert admissions == [decoder.device]
    assert decoder._options.backend == 1
    assert decoder._options.sort_before_hw_decompress == int(sorting)
    assert bytes(decoder._options)[8:] == bytes(56)
    assert decoder._align.argtypes == [options_type, ctypes.POINTER(codec._Alignments)]
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
    frame = [codec.DecodeFrame(0, 16, 0, 64)]
    assert decoder.temporary_bytes(frame) == 64
    assert [suffix for suffix, _ in calls] == [
        "GetRequiredAlignments",
        "GetTempSizeAsync",
    ]
    assert calls[0][1] == calls[1][1]


def test_workspace_query_cache_uses_exact_geometry_and_preserves_failure(monkeypatch):
    # Exercise the real query/cache code without a CUDA allocation or library.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cuda", 0)
    decoder.backend = "hardware"
    decoder.codec = "snappy-zstd"
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

    def frames(*lengths):
        return [codec.DecodeFrame(0, 1, 0, length) for length in lengths]

    assert decoder.temporary_bytes([]) == 0
    assert decoder.temporary_bytes(frames(10, 20)) == 0
    assert decoder.temporary_bytes(frames(20, 10)) == 0
    assert calls == [(2, 20, 30)]
    # Same count and maximum, different total: the native query must run again.
    with pytest.raises(RuntimeError, match="status=1"):
        decoder.temporary_bytes(frames(11, 20))
    fail = False
    assert decoder.temporary_bytes(frames(11, 20)) == 0
    assert decoder.temporary_bytes(frames(11, 20)) == 0
    assert calls == [(2, 20, 30), (2, 20, 31), (2, 20, 31)]


@pytest.mark.parametrize("name", ["snappy-zstd", "lz4-zstd"])
def test_host_input_plans_split_output_and_status_slots(monkeypatch, name):
    # CPU-backed device views exercise the actual slab construction and C ABI
    # arguments without requiring CUDA, nvCOMP or a host-DE allocation locally.
    device = torch.device("cuda", 0)

    class DeviceView:
        def __init__(self, value):
            self.value = value
            self.device = device

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
        "sglang.srt.weight_sync.gpu_delta_memory",
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
    decoder.codec = name
    decoder._algorithm, options_type = codec._DECOMPRESS_OPTIONS[name]
    decoder._options = options_type()
    decoder.alignments = codec._Alignments(16, 16, 16)
    decoder.temporary_bytes = lambda frames: 64
    launches = []
    decoder._decode = lambda *arguments: launches.append(arguments) or 0
    stream = SimpleNamespace(device=device, cuda_stream=1234)
    host = allocate(512, torch.uint8, "cpu")
    batches = [
        [codec.DecodeFrame(0, 20, 0, 64), codec.DecodeFrame(32, 17, 128, 32)],
        [codec.DecodeFrame(64, 25, 16, 128)],
        [codec.DecodeFrame(96, 30, 32, 64)],
    ]
    workspace = decoder.allocate_workspace(batches)
    prepared_plan = decoder.prepare_batches(batches, host, workspace, stream)
    # Preparation uploads input/size metadata without any output allocation,
    # output pointer upload or decompression. Paused binding fills just row 3.
    assert uploads == [(3, 4)] and not launches
    outputs = tuple(allocate(256, torch.uint8, device) for _ in range(2))
    plans = prepared_plan.bind_outputs(outputs)
    assert uploads == [(3, 4), (4,)] and not launches
    assert workspace.statuses.shape == workspace.actual_sizes.shape == (2, 2)
    assert plans[0].metadata.stride(0) == 4
    assert plans[0].metadata.untyped_storage().data_ptr() == (
        plans[1].metadata.untyped_storage().data_ptr()
    )
    assert plans[0].host_metadata.untyped_storage().data_ptr() == (
        plans[1].host_metadata.untyped_storage().data_ptr()
    )
    for index, (plan, frames) in enumerate(zip(plans, batches)):
        slot = index % 2
        assert plan.host_input is host
        assert plan.decoded is outputs[slot]
        assert plan.stream is stream
        assert plan.metadata.value.tolist() == [
            [host.data_ptr() + frame.input_offset for frame in frames],
            [frame.encoded_bytes for frame in frames],
            [frame.decoded_bytes for frame in frames],
            [outputs[slot].data_ptr() + frame.output_offset for frame in frames],
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
    # The third batch reuses only slot zero; slot one's delayed apply can still
    # read its status and size result. A single DE stream owns the temp buffer.
    assert plans[0].statuses.data_ptr() == plans[2].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() == plans[2].actual_sizes.data_ptr()
    assert plans[0].statuses.data_ptr() != plans[1].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() != plans[1].actual_sizes.data_ptr()
    plans[1].statuses.value.fill_(7)
    plans[1].actual_sizes.value.fill_(128)
    plans[2].statuses.value.zero_()
    plans[2].actual_sizes.value.fill_(64)
    assert plans[1].statuses.value.tolist() == [7]
    assert plans[1].actual_sizes.value.tolist() == [128]
    assert admitted == [
        workspace.temporary.data_ptr(),
        workspace.actual_sizes.data_ptr(),
        workspace.statuses.data_ptr(),
        plans[0].metadata.untyped_storage().data_ptr(),
        outputs[0].data_ptr(),
        outputs[1].data_ptr(),
    ]

    # Both output slots are bounds checked, and aliased output slots are rejected
    # before an output-pointer upload or native submission.
    for slots, error in (
        ((outputs[0], outputs[1][:64]), "Out-of-bounds"),
        ((outputs[0], outputs[1][1:]), "Misaligned decoded"),
        ((outputs[0], outputs[0]), "must not overlap"),
        (outputs[:1], "two decoded"),
    ):
        with pytest.raises(ValueError, match=error):
            prepared_plan.bind_outputs(slots)
    with pytest.raises(ValueError, match="outside input"):
        decoder.prepare_batches(
            [[codec.DecodeFrame(496, 32, 0, 64)]], host, workspace, stream
        )
    assert len(uploads) == 2 and len(launches) == 3 and len(admitted) == 6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
