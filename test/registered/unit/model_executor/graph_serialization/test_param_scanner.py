"""CPU tests for ``graph_serialization.codec``: the pure ``ParamScanner``,
``EventTable``, the guarded driver calls of ``LoadedCudaGraph`` and the
``GraphCodec`` stubs. No CUDA: the region registry, events and the driver
binding are fakes.
"""

import struct
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.graph_serialization import codec
from sglang.srt.model_executor.graph_serialization.codec import (
    DEFAULT_HOST_GARBAGE_RANGE,
    EventTable,
    GraphCodec,
    LoadedCudaGraph,
    ParamScanner,
    ScanResult,
    driver_is_device_pointer,
)
from sglang.srt.model_executor.graph_serialization.format import RegionRef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WEIGHT_ID, WEIGHT_BASE, WEIGHT_NBYTES = "weight:storage:0", 0x7F00_0000_0000, 4096
KV_ID, KV_BASE, KV_NBYTES = "kv:k_buffer:0", 0x7F10_0000_0000, 8192

# A live device address no provider registered (an allocator block nobody
# enumerated). Inside the default host-garbage range on purpose: the device
# pointer check must run before the garbage check.
STRAY_DEVICE_PTR = 0x7E00_0000_0000
# Dead host garbage (fact 14): inside the default range, not a device pointer.
HOST_GARBAGE = 0x5F3A_1234_5678


class FakeRegistry:
    """``RegionRegistry.classify`` over two live ranges (range based, fact 9)."""

    def __init__(self):
        self.regions = [
            (WEIGHT_ID, WEIGHT_BASE, WEIGHT_NBYTES),
            (KV_ID, KV_BASE, KV_NBYTES),
        ]
        self.queries = []

    def classify(self, word):
        self.queries.append(word)
        for region_id, base, nbytes in self.regions:
            if base <= word < base + nbytes:
                return RegionRef(region_id, word - base)
        return None


class RecordingDevicePointer:
    def __init__(self, live=()):
        self.live = set(live)
        self.queries = []

    def __call__(self, word):
        self.queries.append(word)
        return word in self.live


def words(*values):
    return b"".join(struct.pack("<Q", v) for v in values)


def make_scanner(is_device_pointer=None, **kwargs):
    registry = FakeRegistry()
    probe = is_device_pointer or RecordingDevicePointer()
    return ParamScanner(registry, probe, **kwargs), registry, probe


def test_scan_classifies_region_words_including_interior_offsets():
    scanner, _, _ = make_scanner()
    raw = words(WEIGHT_BASE, KV_BASE + 256, KV_BASE + KV_NBYTES - 8)

    result = scanner.scan(raw)

    assert isinstance(result, ScanResult)
    assert result.refs == [
        RegionRef(WEIGHT_ID, 0),
        RegionRef(KV_ID, 256),
        RegionRef(KV_ID, KV_NBYTES - 8),
    ]
    assert result.unknown_device_words == []
    assert result.sanitized == raw


def test_scan_one_past_region_end_is_not_a_region_word():
    scanner, _, _ = make_scanner()
    raw = words(WEIGHT_BASE + WEIGHT_NBYTES)

    result = scanner.scan(raw)

    assert result.refs == [None]
    # Not a device pointer per the probe, inside the garbage range -> zeroed.
    assert result.sanitized == words(0)


def test_scan_records_unknown_device_pointer_by_byte_offset_and_keeps_it():
    probe = RecordingDevicePointer(live={STRAY_DEVICE_PTR})
    scanner, _, _ = make_scanner(probe)
    raw = words(7, STRAY_DEVICE_PTR, WEIGHT_BASE + 16)

    result = scanner.scan(raw)

    assert result.unknown_device_words == [8]
    assert result.refs == [None, None, RegionRef(WEIGHT_ID, 16)]
    # An unknown live pointer is never zeroed even though it lies inside the
    # host-garbage range; the graph becomes needs_recapture instead.
    assert result.sanitized == raw


def test_scan_zeroes_host_garbage_word_only():
    scanner, _, _ = make_scanner()
    raw = words(HOST_GARBAGE, WEIGHT_BASE, HOST_GARBAGE + 8)

    result = scanner.scan(raw)

    assert result.sanitized == words(0, WEIGHT_BASE, 0)
    assert result.refs == [None, RegionRef(WEIGHT_ID, 0), None]
    assert result.unknown_device_words == []


def test_scan_preserves_scalars_verbatim():
    lo, hi = DEFAULT_HOST_GARBAGE_RANGE
    scalars = (0, 1, 0x400000, 4096, lo - 8, hi, 0xFFFF_FFFF_FFFF_FFFF)
    scanner, _, probe = make_scanner()
    raw = words(*scalars)

    result = scanner.scan(raw)

    assert result.sanitized == raw
    assert result.refs == [None] * len(scalars)
    assert result.unknown_device_words == []
    # The zero word is never sent to the driver probe.
    assert 0 not in probe.queries
    assert 0x400000 in probe.queries


def test_scan_region_check_runs_before_device_pointer_probe():
    probe = RecordingDevicePointer(live={WEIGHT_BASE + 8, STRAY_DEVICE_PTR})
    scanner, _, _ = make_scanner(probe)

    result = scanner.scan(words(WEIGHT_BASE + 8, STRAY_DEVICE_PTR))

    assert result.refs == [RegionRef(WEIGHT_ID, 8), None]
    assert result.unknown_device_words == [8]
    assert probe.queries == [STRAY_DEVICE_PTR]


def test_scan_preserves_partial_trailing_word():
    scanner, _, _ = make_scanner()
    tail = b"\x01\x02\x03\x04"
    raw = words(HOST_GARBAGE, WEIGHT_BASE) + tail

    result = scanner.scan(raw)

    assert len(result.refs) == 2
    assert result.sanitized == words(0, WEIGHT_BASE) + tail
    assert len(result.sanitized) == len(raw)


def test_scan_buffer_shorter_than_one_word_is_all_tail():
    scanner, registry, probe = make_scanner()
    raw = b"\xaa\xbb\xcc"

    result = scanner.scan(raw)

    assert result.refs == []
    assert result.sanitized == raw
    assert registry.queries == [] and probe.queries == []


def test_scan_empty_buffer():
    scanner, _, _ = make_scanner()

    result = scanner.scan(b"")

    assert result.refs == []
    assert result.unknown_device_words == []
    assert result.sanitized == b""


def test_scan_honours_custom_host_garbage_range():
    scanner, _, _ = make_scanner(host_garbage_range=(0x1000, 0x2000))
    raw = words(0x1800, HOST_GARBAGE, 0x2000)

    result = scanner.scan(raw)

    assert result.sanitized == words(0, HOST_GARBAGE, 0x2000)
    assert scanner.host_garbage_range == (0x1000, 0x2000)


def test_scanner_rejects_inverted_host_garbage_range():
    with pytest.raises(ValueError):
        make_scanner(host_garbage_range=(0x2000, 0x1000))


def test_scan_never_opens_proc_self_maps(monkeypatch):
    import builtins

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, str) and file.startswith("/proc/"):
            raise AssertionError(f"ParamScanner opened {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    scanner, _, _ = make_scanner()
    # 0x400000 is the slot_bytes scalar that equals the python3 ELF base
    # (fact 17); it must be kept as a scalar without consulting host maps.
    assert scanner.scan(words(0x400000)).sanitized == words(0x400000)


# --- EventTable --------------------------------------------------------------


def test_event_table_returns_live_handle():
    table = EventTable({"metadata_prep_done": SimpleNamespace(cuda_event=0x1234)})

    assert table.handle("metadata_prep_done") == 0x1234
    assert table.roles() == ("metadata_prep_done",)
    assert "metadata_prep_done" in table
    assert "other" not in table


def test_event_table_unknown_role_raises():
    table = EventTable({"metadata_prep_done": SimpleNamespace(cuda_event=1)})

    with pytest.raises(ValueError, match="unknown event role"):
        table.handle("nope")


def test_event_table_rejects_lazy_handle_zero():
    # torch creates the CUevent lazily: cuda_event is 0 until the first
    # record() and the driver rejects handle 0 (fact 18).
    table = EventTable({"metadata_prep_done": SimpleNamespace(cuda_event=0)})

    with pytest.raises(ValueError, match="handle 0"):
        table.handle("metadata_prep_done")


# --- LoadedCudaGraph and guarded driver calls ---------------------------------


class FakeDriver:
    def __init__(self, pointer_ok=()):
        self.calls = []
        self.pointer_ok = set(pointer_ok)
        self.CUpointer_attribute = SimpleNamespace(
            CU_POINTER_ATTRIBUTE_RANGE_START_ADDR="range_start"
        )

    def cuGraphLaunch(self, graph_exec, stream):
        self.calls.append(("launch", graph_exec, stream))
        return (0,)

    def cuGraphExecDestroy(self, graph_exec):
        self.calls.append(("exec_destroy", graph_exec))
        return (0,)

    def cuGraphDestroy(self, graph):
        self.calls.append(("graph_destroy", graph))
        return (0,)

    def cuPointerGetAttribute(self, attribute, ptr):
        self.calls.append(("pointer_attr", attribute, ptr))
        return (0, ptr) if ptr in self.pointer_ok else (1, None)


def fake_check(result):
    if result[0] != 0:
        raise RuntimeError(f"CUDA error {result[0]}")
    return None if len(result) == 1 else result[1]


@pytest.fixture
def fake_driver(monkeypatch):
    driver = FakeDriver(pointer_ok={STRAY_DEVICE_PTR})
    monkeypatch.setattr(codec, "cuda_drv", driver)
    monkeypatch.setattr(codec, "checkCudaErrors", fake_check)
    monkeypatch.setattr(
        codec,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(cuda_stream=0x77)
            )
        ),
    )
    return driver


def test_loaded_graph_replay_launches_exec_on_current_stream(fake_driver):
    graph = LoadedCudaGraph(raw_graph=0x10, exec=0x20)

    graph.replay()

    assert fake_driver.calls == [("launch", 0x20, 0x77)]
    assert graph.raw_cuda_graph() == 0x10


def test_loaded_graph_replay_without_own_exec_raises(fake_driver):
    graph = LoadedCudaGraph(raw_graph=0x10, exec=None)

    with pytest.raises(RuntimeError, match="registry"):
        graph.replay()
    assert fake_driver.calls == []


def test_loaded_graph_reset_destroys_exec_then_graph_and_is_idempotent(fake_driver):
    graph = LoadedCudaGraph(raw_graph=0x10, exec=0x20)

    graph.reset()
    graph.reset()

    assert fake_driver.calls == [("exec_destroy", 0x20), ("graph_destroy", 0x10)]
    assert graph.exec is None and graph.raw_graph == 0


def test_loaded_graph_reset_without_exec_destroys_graph_only(fake_driver):
    graph = LoadedCudaGraph(raw_graph=0x10)

    graph.reset()

    assert fake_driver.calls == [("graph_destroy", 0x10)]


def test_driver_is_device_pointer_uses_range_start_attribute(fake_driver):
    assert driver_is_device_pointer(STRAY_DEVICE_PTR) is True
    assert driver_is_device_pointer(HOST_GARBAGE) is False
    assert fake_driver.calls[0] == ("pointer_attr", "range_start", STRAY_DEVICE_PTR)


def test_driver_calls_raise_not_implemented_without_binding(monkeypatch):
    monkeypatch.setattr(codec, "cuda_drv", None)
    graph = LoadedCudaGraph(raw_graph=0x10, exec=0x20)

    with pytest.raises(NotImplementedError, match="section 6.6"):
        graph.replay()
    with pytest.raises(NotImplementedError, match="section 6.6"):
        graph.reset()
    with pytest.raises(NotImplementedError, match="section 6.6"):
        driver_is_device_pointer(STRAY_DEVICE_PTR)
    # Nothing to destroy: reset stays a no-op even without the binding.
    LoadedCudaGraph(raw_graph=0, exec=None).reset()


# --- GraphCodec ----------------------------------------------------------------


def test_graph_codec_builds_scanner_with_injected_probe():
    probe = RecordingDevicePointer()
    graph_codec = GraphCodec(is_device_pointer=probe, host_garbage_range=(1, 2))

    scanner = graph_codec.scanner_for(FakeRegistry())

    assert isinstance(scanner, ParamScanner)
    assert scanner.host_garbage_range == (1, 2)
    scanner.scan(words(5))
    assert probe.queries == [5]


def test_graph_codec_encode_and_materialize_are_stubs():
    graph_codec = GraphCodec(is_device_pointer=RecordingDevicePointer())
    common = dict(registry=FakeRegistry(), resolver=object(), policy=object())

    with pytest.raises(NotImplementedError, match="GraphCodec.encode.*section 6.6"):
        graph_codec.encode(0x10, event_roles={}, **common)
    with pytest.raises(NotImplementedError, match="GraphCodec.materialize.*6.6"):
        graph_codec.materialize(
            object(),
            kernels=(),
            reloc=object(),
            resolver=object(),
            events=EventTable({}),
            device_ctx=0,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
