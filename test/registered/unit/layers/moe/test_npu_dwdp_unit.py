"""NPU DWDP page coverage and failure cleanup without accelerator hardware."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.hardware_backend.npu.dwdp.manager import NPUDwdpManager
from sglang.srt.hardware_backend.npu.dwdp.vmm import NPUWeightVMM
from sglang.srt.layers.moe.dwdp.layout import WeightSpec
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_shared_prefetch_preserves_cuda_order_and_remote_slices(monkeypatch, rank):
    trace, events, layers = [], [], [2, 5, 8, 11]

    def stream(name):
        return NS(
            name=name,
            wait_event=lambda event: trace.append(("wait", name, event.index)),
        )

    copy_stream, compute_stream = stream("copy"), stream("compute")

    def event():
        result = NS(index=len(events))
        result.record = lambda stream: trace.append(
            ("record", stream.name, result.index)
        )
        events.append(result)
        return result

    monkeypatch.setattr(
        torch,
        "cuda",
        NS(
            Stream=lambda **kwargs: copy_stream,
            Event=event,
            current_stream=lambda *args: compute_stream,
            stream=lambda stream: nullcontext(),
        ),
    )
    weights = {li: torch.zeros(6, 2) for li in layers}
    peers = {
        (r, li, "w"): torch.full((2, 2), float(li * 10 + r))
        for r in range(3)
        for li in layers
    }
    for li in layers:
        weights[li][rank * 2 : (rank + 1) * 2].copy_(peers[rank, li, "w"])
    buffer = NS(
        device_id=0,
        buffer_index_for_layer=lambda li: layers.index(li) % 2,
        get_remote_slices=lambda li, name: [
            (weights[li][: rank * 2], 0, rank * 2),
            (weights[li][(rank + 1) * 2 :], (rank + 1) * 2, 6),
        ],
    )
    manager = DWDPWeightManager(
        buffer, peers, [(0, 2), (2, 4), (4, 6)], layers, ["w"], rank, 3
    )
    assert NPUDwdpManager.wait_prefetch is DWDPWeightManager.wait_prefetch
    assert (
        NPUDwdpManager.record_compute_and_prefetch_next
        is DWDPWeightManager.record_compute_and_prefetch_next
    )
    assert trace == [("record", "compute", 2), ("record", "compute", 3)]
    original_copy = manager._prefetch_layer_per_slice

    def copy(li):
        trace.append(("copy", li))
        original_copy(li)

    manager._prefetch_layer_per_slice = copy

    def expected_prefetch(pos):
        return [
            ("wait", "copy", 2 + pos % 2),
            ("copy", layers[pos]),
            ("record", "copy", pos % 2),
        ]

    for step in range(2):
        if step:
            for (peer, _, _), data in peers.items():
                if peer != rank:
                    data.add_(100)
        trace.clear()
        expected = expected_prefetch(0) + expected_prefetch(1)
        manager.prefetch_first_layers()
        for pos, li in enumerate(layers):
            manager.wait_prefetch(li)
            torch.testing.assert_close(
                weights[li], torch.cat([peers[r, li, "w"] for r in range(3)])
            )
            trace.append(("compute", li))
            manager.record_compute_and_prefetch_next(li)
            expected += [
                ("wait", "compute", pos % 2),
                ("compute", li),
                ("record", "compute", 2 + pos % 2),
            ]
            if pos + 2 < len(layers):
                expected += expected_prefetch(pos + 2)
        assert trace == expected


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("expert_bytes", [2, 64, 300, 1024])
def test_remote_pages_cover_exactly_remote_weights(world, expert_bytes):
    vmm = NPUWeightVMM.__new__(NPUWeightVMM)
    vmm.granularity = 1024
    spec = WeightSpec(
        world * 4, (4, expert_bytes), (world * 4, expert_bytes), torch.uint8
    )
    for rank in range(world):
        manager = NPUDwdpManager.__new__(NPUDwdpManager)
        manager.dwdp_size, manager.dwdp_rank = world, rank
        page = vmm.layout(spec, rank)
        start, end, total = (
            rank * spec.chunk_bytes,
            (rank + 1) * spec.chunk_bytes,
            world * spec.chunk_bytes,
        )
        assert page.page_start == start // 1024 * 1024
        assert page.page_end == (end + 1023) // 1024 * 1024
        assert page.total_size == (total + 1023) // 1024 * 1024
        manager._specs, manager._pages, manager._weights, peers = {}, {}, {}, {}
        for index, name in enumerate(("w13_weight", "w2_weight")):
            manager._specs[5, name], manager._pages[5, name] = spec, page
            manager._weights[5, name] = NS(data_ptr=lambda i=index: (i + 1) * 100000)
            peers.update(
                {(r, 5, name): (index + 1) * 1000000 + r * 10000 for r in range(world)}
            )
        covered = [set(), set()]
        for initial in (True, False):
            for peer, dst, src, size in manager._copy_plan(5, peers, initial=initial):
                index, offset = dst // 100000 - 1, dst % 100000
                assert peer != rank
                assert (
                    src
                    == (index + 1) * 1000000
                    + peer * 10000
                    + offset
                    - peer * spec.chunk_bytes
                )
                assert (page.page_start <= offset < page.page_end) == initial
                region = set(range(offset, offset + size))
                assert not covered[index] & region
                covered[index] |= region
        assert covered == [set(range(total)) - set(range(start, end))] * 2


@pytest.mark.parametrize("fail", [False, True])
def test_migration_releases_source_only_after_success(monkeypatch, fail):
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager.device, manager.dwdp_rank = torch.device("cpu"), 0
    manager._position = {5: 0}
    manager._specs = {(5, "w13_weight"): WeightSpec(4, (2, 3), (4, 3), torch.float16)}
    manager._weights, manager._pages, manager._local, manager._exports = {}, {}, {}, {}
    manager._vmm = NS(create_weight=lambda *args: (torch.empty(4, 3), 100, 1, object()))
    manager.acl = NS(
        rt=NS(
            memcpy=lambda *args: int(fail),
            mem_export_to_shareable_handle=lambda *args: (7, 0),
            mem_set_pid_to_shareable_handle=lambda *args: 0,
        )
    )
    calls = []
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            current_stream=lambda *args: NS(synchronize=lambda: None),
            synchronize=lambda *args: None,
            empty_cache=lambda: calls.append("cache"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.npu, "npu_format_cast", lambda data, fmt: data, raising=False
    )
    source = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float16))
    if fail:
        with pytest.raises(RuntimeError, match="copy local shard"):
            manager._migrate_weight((5, "w13_weight"), source, [123])
    else:
        manager._migrate_weight((5, "w13_weight"), source, [123])
    assert (source.untyped_storage().nbytes() > 0) == fail
    assert calls == ([] if fail else ["cache"])


@pytest.mark.parametrize(
    "state", ["new", "initializing", "ready", "closed", "import_error"]
)
def test_cleanup_order_and_terminal_state(monkeypatch, state):
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager._state = "ready" if state == "import_error" else state
    manager.device, manager.group = None, NS(cpu_group="cpu")
    manager._moe_layer_indices, manager._position, manager._specs, manager._weights = (
        [],
        {},
        {},
        {},
    )
    manager._pages, manager._local, manager._exports, manager._plans = {}, {}, {}, {}
    calls = []

    def close_imports():
        calls.append("imports")
        if state == "import_error":
            raise RuntimeError("import release failed")

    manager._vmm = NS(close_imports=close_imports, close=lambda: calls.append("local"))
    monkeypatch.setattr(
        torch, "npu", NS(synchronize=lambda *args: calls.append("sync")), raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.dwdp.manager.dist.barrier",
        lambda **kwargs: calls.append("barrier"),
    )
    if state == "import_error":
        with pytest.raises(RuntimeError, match="import release"):
            manager.cleanup()
        assert calls == ["sync", "barrier", "imports"]
        assert manager._state == "ready"
        return
    if state == "initializing":
        with pytest.raises(RuntimeError, match="reload"):
            manager.setup(None)
        with pytest.raises(RuntimeError, match="initialized"):
            manager.prefetch_first_layers()
    manager.cleanup()
    manager.cleanup()
    assert calls == (
        ["sync", "barrier", "imports", "barrier", "local"]
        if state in ("initializing", "ready")
        else []
    )
    with pytest.raises(RuntimeError, match="reload"):
        manager.setup(None)


def test_peer_layout_mismatch_before_device_allocation():
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager._state, manager._specs = "new", {}
    weight = torch.empty(2, 3, dtype=torch.float16)
    manager._collect_moe_layers = lambda model: [
        (2, NS(num_global_routed_experts=4, w13_weight=weight, w2_weight=weight))
    ]
    manager._validate = lambda layers: None
    manager._gather = lambda value: (
        [value, {}] if isinstance(value, dict) else [value, value]
    )
    with pytest.raises(ValueError, match="different expert weight layouts"):
        manager.setup(None)
    assert manager._state == "new"
