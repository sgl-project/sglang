"""NPU DWDP page coverage and failure cleanup without accelerator hardware."""

import ctypes
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.hardware_backend.npu.dwdp.manager import NPUDwdpManager
from sglang.srt.hardware_backend.npu.dwdp.vmm import (
    NPUWeightVMM,
    tensor_from_pointer,
)
from sglang.srt.layers.moe.dwdp.layout import WeightSpec, build_remote_slices
from sglang.srt.layers.moe.dwdp.page_pool import PagePool, compute_slot_sizes
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def isolate_scheduler_env():
    from sglang.srt.environ import envs

    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
        yield


@pytest.mark.parametrize("wrapped", [False, True])
def test_shared_layer_discovery_preserves_cuda_scope(monkeypatch, wrapped):
    from sglang.srt.layers.moe.dwdp.layer_utils import collect_moe_layers

    class MoE(torch.nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        NS(FusedMoE=MoE),
    )
    first, second, last = MoE(91), MoE(5), MoE(-7)
    decoder = torch.nn.Module()
    decoder.layers = torch.nn.ModuleList(
        [
            torch.nn.Identity(),
            torch.nn.Sequential(first, second),
            torch.nn.Identity(),
            last,
        ]
    )
    decoder.outside_layers = MoE(42)
    model = torch.nn.Module()
    model.model, model.outside_decoder = decoder, MoE(100)
    # Preserve decoder indices and the first MoE per layer, even when actual
    # layer_id values disagree. Do not expand discovery to other modules.
    assert collect_moe_layers(model if wrapped else decoder) == [
        (1, first),
        (3, last),
    ]
    assert NPUDwdpManager._collect_moe_layers is collect_moe_layers
    assert (
        collect_moe_layers(NS(layers=torch.nn.ModuleList([torch.nn.Identity()]))) == []
    )
    with pytest.raises(AttributeError):
        collect_moe_layers(torch.nn.ModuleList([first]))


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16])
def test_parameter_gather_preserves_order_and_bias_dtype(monkeypatch, weight_dtype):
    from sglang.srt.layers.moe.dwdp import layer_utils

    names = ("w13_weight_bias", "w13_weight_scale", "w2_weight_bias")
    parameters = {
        name: torch.arange(4, dtype=torch.float32).reshape(2, 2) + index * 20
        for index, name in enumerate(names)
    }
    trace, replaced = [], {}
    manager = NS(dwdp_size=2, group=NS(device_group=object()))

    def replace(name, full):
        trace.append(("replace", name))
        replaced[name] = full

    experts = NS(
        num_global_routed_experts=4,
        w13_weight=torch.empty(0, dtype=weight_dtype),
        named_per_expert_tensors=lambda count: list(parameters.items()),
        replace_expert_tensor=replace,
    )

    def all_gather(shards, data, *, group):
        assert group is manager.group.device_group
        assert len(shards) == 2
        trace.append(
            ("gather", next(name for name in names if parameters[name] is data))
        )
        for rank, shard in enumerate(shards):
            assert shard.shape == data.shape and shard.dtype == data.dtype
            shard.copy_(data + rank * 10)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    NPUDwdpManager._replicate_small_params(manager, experts)
    assert trace == [
        item for name in names for item in (("gather", name), ("replace", name))
    ]
    assert list(replaced) == list(names)
    for name, data in parameters.items():
        expected = torch.cat((data, data + 10))
        if name.endswith("_bias") and weight_dtype == torch.float16:
            expected = expected.to(torch.float16)
        torch.testing.assert_close(replaced[name], expected)
    # The common helper also preserves the caller's routed-expert truncation.
    gathered = list(
        layer_utils.gather_per_expert_tensors(experts, 2, 3, 2, manager.group)
    )
    assert all(full.shape[0] == 3 for _, full in gathered)


def test_shared_prefetch_waits_for_compute_before_slot_reuse():
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

    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager._state, manager._moe_layer_indices = "ready", layers
    manager._buffer_index_for_layer = lambda li: layers.index(li) % 2
    manager._prefetch_layer_per_slice = lambda li: trace.append(("copy", li))
    manager._init_prefetch(
        NS(
            Stream=lambda **kwargs: copy_stream,
            Event=event,
            current_stream=lambda *args: compute_stream,
            stream=lambda stream: nullcontext(),
        ),
        None,
    )
    assert NPUDwdpManager.wait_prefetch is DWDPWeightManager.wait_prefetch
    assert (
        NPUDwdpManager.record_compute_and_prefetch_next
        is DWDPWeightManager.record_compute_and_prefetch_next
    )
    assert trace == [("record", "compute", 2), ("record", "compute", 3)]

    def expected_prefetch(pos):
        return [
            ("wait", "copy", 2 + pos % 2),
            ("copy", layers[pos]),
            ("record", "copy", pos % 2),
        ]

    for _ in range(2):
        trace.clear()
        expected = expected_prefetch(0) + expected_prefetch(1)
        manager.prefetch_first_layers()
        for pos, li in enumerate(layers):
            manager.wait_prefetch(li)
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


@pytest.mark.parametrize("world, rank", [(2, 0), (3, 1), (4, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int8])
def test_copy_restores_all_remote_weights_without_changing_local(world, rank, dtype):
    assert build_remote_slices(torch.empty(2, 1), 0, 2, 2) == []
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager.dwdp_size, manager.dwdp_rank = world, rank
    manager._specs, manager._weights, manager._peer_views = {}, {}, {}
    manager._remote_slices, manager._peer_ranges = {}, {}
    layers, local = [2, 5, 8, 11], {}
    for pos, li in enumerate(layers):
        local_experts = 1 + pos % 2
        num_experts = world * local_experts
        for name, width in (("w13_weight", 10), ("w2_weight", 14)):
            width //= torch.tensor([], dtype=dtype).element_size()
            manager._specs[li, name] = WeightSpec(
                num_experts, (local_experts, width), (num_experts, width), dtype
            )
            manager._weights[li, name] = torch.empty(num_experts, width, dtype=dtype)
            for peer in range(world):
                value = (
                    torch.arange(local_experts * width).reshape(local_experts, width)
                    + pos * 20
                    + peer * 4
                ).to(dtype)
                if peer == rank:
                    start, end = peer * local_experts, (peer + 1) * local_experts
                    manager._weights[li, name][start:end].copy_(value)
                    local[li, name] = value.clone()
                else:
                    manager._peer_views[peer, li, name] = value
    manager._prepare_copy_views()
    for step in range(2):
        if step:
            for value in manager._peer_views.values():
                value.add_(3)
        for li in layers:
            # Poison every remote byte, including the parts in resident pages.
            for name in ("w13_weight", "w2_weight"):
                for dst, *_ in manager._get_remote_slices(li, name):
                    dst.fill_(-77)
            manager._prefetch_layer_per_slice(li)
            for name in ("w13_weight", "w2_weight"):
                expected = torch.cat(
                    [
                        local[li, name]
                        if peer == rank
                        else manager._peer_views[peer, li, name]
                        for peer in range(world)
                    ]
                )
                torch.testing.assert_close(manager._weights[li, name], expected)
                count = manager._specs[li, name].local_experts
                torch.testing.assert_close(
                    manager._weights[li, name][rank * count : (rank + 1) * count],
                    local[li, name],
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int8])
def test_npu_peer_tensor_wrap_preserves_shard_metadata(monkeypatch, dtype):
    calls, storage, pointer = [], object(), 0x100080

    def wrap_storage(ptr, device, nbytes):
        calls.append((ptr, device, nbytes))
        return storage

    def wrap_tensor(metadata, wrapped_storage):
        assert wrapped_storage is storage
        return metadata

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        NS(
            _C=NS(
                _construct_storage_from_data_pointer=wrap_storage,
                _construct_NPU_Tensor_From_Storage_And_Metadata=wrap_tensor,
            )
        ),
    )
    shape, device = (2, 3, 5), NS(type="npu", index=1)
    nbytes = 30 * torch.tensor([], dtype=dtype).element_size()
    metadata = tensor_from_pointer(
        pointer, nbytes, shape=shape, dtype=dtype, device=device
    )
    assert calls == [(pointer, device, nbytes)]
    assert metadata == dict(
        size=shape,
        stride=(15, 5, 1),
        nbytes=nbytes,
        dtype=dtype,
        data_ptr=pointer,
        storage_offset=0,
        device=device,
        npu_format=2,
        layout=torch.strided,
        requires_grad=False,
    )


@pytest.mark.parametrize("failure", [None, "copy", "export"])
def test_migration_releases_source_only_after_success(monkeypatch, failure):
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager.device, manager.dwdp_rank = torch.device("cpu"), 0
    manager._position = {5: 0}
    manager._specs = {(5, "w13_weight"): WeightSpec(4, (2, 3), (4, 3), torch.float16)}
    manager._weights, manager._exports = {}, {}
    calls = []

    def export(handle, peers):
        calls.append((handle, peers))
        if failure == "export":
            raise RuntimeError("export failed")
        return 7

    manager._vmm = NS(create_local=lambda *args: (100, 1), export_shard=export)
    manager.acl = NS(
        rt=NS(
            memcpy=lambda *args: int(failure == "copy"),
        )
    )
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
    if failure:
        with pytest.raises(RuntimeError, match="copy local shard|export failed"):
            manager._migrate_weight((5, "w13_weight"), source, [123])
    else:
        assert manager._migrate_weight((5, "w13_weight"), source, [123]) == 1
    assert (source.untyped_storage().nbytes() > 0) == bool(failure)
    assert calls == (
        [] if failure == "copy" else [(1, [123])] + ([] if failure else ["cache"])
    )
    assert manager._exports == ({} if failure else {(5, "w13_weight"): 7})


class _FakeACL:
    """Enforce whole-handle mapping and resource ownership without a device."""

    def __init__(self):
        self.allocations, self.reservations, self.mappings = {}, {}, {}
        self.freed, self.unmapped = [], []
        self.next_handle, self.next_pointer = 1, 0x100000
        self.fail_free_after = None
        self.fail_once, self.fail_after_allocations, self.fail_after_maps = (
            None,
            None,
            None,
        )

    def mem_get_allocation_granularity(self, *args):
        return 4, 0

    def malloc_physical(self, size, *args):
        if len(self.allocations) == self.fail_after_allocations:
            self.fail_after_allocations = None
            return None, 1
        handle = self.next_handle
        self.next_handle += 1
        self.allocations[handle] = size
        return handle, 0

    def reserve_mem_address(self, size, *args):
        ptr = self.next_pointer
        self.next_pointer += size + 0x1000
        self.reservations[ptr] = size
        return ptr, 0

    def map_mem(self, ptr, size, offset, handle, flags):
        if len(self.mappings) == self.fail_after_maps:
            self.fail_after_maps = None
            return 1
        assert offset == 0 and size == self.allocations[handle]
        assert any(
            base <= ptr and ptr + size <= base + span
            for base, span in self.reservations.items()
        )
        assert all(
            ptr + size <= base or ptr >= base + span
            for base, (span, _) in self.mappings.items()
        )
        self.mappings[ptr] = size, handle
        return 0

    def unmap_mem(self, ptr):
        assert ptr not in self.unmapped
        self.unmapped.append(ptr)
        del self.mappings[ptr]
        return 0

    def release_mem_address(self, ptr):
        if self.fail_once == "release_mem_address":
            self.fail_once = None
            return 1
        size = self.reservations[ptr]
        assert all(
            base + span <= ptr or base >= ptr + size
            for base, (span, _) in self.mappings.items()
        )
        del self.reservations[ptr]
        return 0

    def free_physical(self, handle):
        if len(self.freed) == self.fail_free_after:
            self.fail_free_after = None
            return 1
        assert handle not in self.freed
        assert all(
            mapped_handle != handle for _, mapped_handle in self.mappings.values()
        )
        self.freed.append(handle)
        del self.allocations[handle]
        return 0


def test_fabric_handle_v2_abi_and_errors(monkeypatch):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    payload, calls = bytes(range(128)), []

    def export(handle, flags, kind, share):
        assert (handle, flags, kind, ctypes.sizeof(share)) == (0x1234, 0, 2, 128)
        ctypes.memmove(share, payload, 128)
        calls.append("export")
        return 0

    def authorize(share, kind, pids, count):
        assert ctypes.string_at(share, 128) == payload
        assert (kind, count, list(pids)) == (2, 2, [17, 29])
        calls.append("authorize")
        return 0

    def import_handle(share, kind, flags, handle):
        assert ctypes.string_at(share, 128) == payload
        assert (kind, flags) == (2, 0)
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = 0x5678
        calls.append("import")
        return 0

    library = NS(
        aclrtMemExportToShareableHandleV2=export,
        aclrtMemSetPidToShareableHandleV2=authorize,
        aclrtMemImportFromShareableHandleV2=import_handle,
    )
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)
    api = module._FabricHandleAPI()
    assert export.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    assert authorize.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ]
    assert import_handle.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    assert all(
        function.restype is ctypes.c_int
        for function in (export, authorize, import_handle)
    )
    assert api.export_handle(0x1234, [17, 29]) == payload
    assert api.import_handle(payload) == 0x5678
    assert calls == ["export", "authorize", "import"]
    with pytest.raises(ValueError, match="handle size"):
        api.import_handle(payload[:-1])
    for name in vars(library):
        with monkeypatch.context() as patch:
            patch.setattr(library, name, lambda *args: 1)
            with pytest.raises(RuntimeError, match="ACL error 1"):
                if "Import" in name:
                    api.import_handle(payload)
                else:
                    api.export_handle(0x1234, [17, 29])


@pytest.mark.parametrize("missing", ["library", "symbol"])
def test_fabric_handle_api_requires_library_and_all_symbols(monkeypatch, missing):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    def load(path):
        assert path == "libascendcl.so"
        if missing == "library":
            raise OSError("missing library")
        return NS()

    monkeypatch.setattr(module.ctypes, "CDLL", load)
    with pytest.raises(OSError if missing == "library" else AttributeError):
        module._FabricHandleAPI()


@pytest.mark.parametrize(
    "local_error, peer_error, same_host",
    [
        (None, None, True),
        (None, None, False),
        ("export", None, True),
        ("allocate", None, True),
        (None, "peer failed", True),
        ("library", None, True),
        ("symbol", None, True),
        ("export", None, False),
        (None, "peer failed", False),
    ],
)
def test_fabric_selection_and_same_host_fallback(
    monkeypatch, vmm, local_error, peer_error, same_host
):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    vmm, rt = vmm
    calls, peers = [], [17, 29]
    rt.fail_after_allocations = 0 if local_error == "allocate" else None

    def export(handle, pids):
        calls.append(("fabric export", handle, pids))
        if local_error == "export":
            raise RuntimeError("export failed")
        return bytes([handle]) * 128

    def api_factory():
        if local_error in ("library", "symbol"):
            raise (OSError if local_error == "library" else AttributeError)(local_error)
        return api

    def legacy_export(handle, flags, kind):
        calls.append(("legacy export", handle, flags, kind))
        return handle + 100, 0

    def authorize(share, pids):
        calls.append(("legacy authorize", share, pids))
        return 0

    api = NS(export_handle=export, import_handle=lambda share: rt.malloc_physical(8)[0])
    monkeypatch.setattr(module, "_FabricHandleAPI", api_factory)
    monkeypatch.setattr(
        rt, "mem_export_to_shareable_handle", legacy_export, raising=False
    )
    monkeypatch.setattr(rt, "mem_set_pid_to_shareable_handle", authorize, raising=False)
    monkeypatch.setattr(
        rt,
        "mem_import_from_shareable_handle",
        lambda share, device: rt.malloc_physical(8),
        raising=False,
    )

    def gather(result):
        error, cleanup_error = result
        assert bool(error) == bool(local_error) and cleanup_error is None
        return [result, (peer_error, None)]

    if not same_host and (local_error or peer_error):
        with pytest.raises(RuntimeError, match="requires FABRIC on every rank"):
            vmm.select_transport(peers, gather, same_host)
        assert vmm._fabric is None
    else:
        vmm.select_transport(peers, gather, same_host)
        fabric = local_error is None and peer_error is None
        assert (vmm._fabric is api) == fabric
        assert not rt.allocations and not vmm._handles
        handle = vmm._allocate(8)
        share = vmm.export_shard(handle, peers)
        assert calls[-1 if fabric else -2] == (
            ("fabric export", handle, peers)
            if fabric
            else ("legacy export", handle, 0, 0)
        )
        assert (
            sum(call[0].endswith("export") and call[1] == handle for call in calls) == 1
        )
        assert vmm.import_shard(share, 8, 1) == vmm._imports[0][1] + 1
    vmm.close()
    assert not rt.allocations and not rt.reservations and not rt.mappings


def test_selected_fabric_does_not_retry_real_weight_exports_with_ipc(monkeypatch, vmm):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    vmm, rt = vmm
    exports = []

    def export(handle, peers):
        exports.append(handle)
        if len(exports) == 2:
            raise RuntimeError("real export failed")
        return b"x" * 128

    monkeypatch.setattr(module, "_FabricHandleAPI", lambda: NS(export_handle=export))
    monkeypatch.setattr(
        rt,
        "mem_export_to_shareable_handle",
        lambda *args: pytest.fail("cannot re-export through IPC"),
        raising=False,
    )
    vmm.select_transport([17], lambda result: [result, result], True)
    handle = vmm._allocate(8)
    with pytest.raises(RuntimeError, match="real export failed"):
        vmm.export_shard(handle, [17])
    assert exports == [1, handle] and handle != 1
    vmm.close()
    assert not rt.allocations


@pytest.mark.parametrize("peer_failure", [False, True])
def test_fabric_probe_cleanup_failure_is_fatal_and_retryable(
    monkeypatch, vmm, peer_failure
):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    vmm, rt = vmm
    rt.fail_free_after = None if peer_failure else 0
    api = NS(export_handle=lambda *args: b"x" * 128)
    monkeypatch.setattr(module, "_FabricHandleAPI", lambda: api)

    def gather(result):
        assert result[0] is None and bool(result[1]) != peer_failure
        return [result, (None, "peer cleanup failed" if peer_failure else None)]

    with pytest.raises(RuntimeError, match="probe cleanup failed"):
        vmm.select_transport([17], gather, True)
    assert vmm._fabric is None
    assert len(vmm._handles) == (0 if peer_failure else 1)
    vmm.close()
    assert not rt.allocations and not vmm._handles


@pytest.fixture
def vmm(monkeypatch):
    from sglang.srt.hardware_backend.npu.dwdp import vmm as module

    rt = _FakeACL()
    monkeypatch.setattr(
        rt,
        "mem_import_from_shareable_handle",
        lambda size, device: rt.malloc_physical(size),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "tensor_from_pointer",
        lambda pointer, nbytes, **metadata: NS(
            pointer=pointer, nbytes=nbytes, **metadata
        ),
    )
    return NPUWeightVMM(NS(rt=rt), NS(index=0)), rt


@pytest.mark.parametrize(
    "failure", ["unmap_mem", "release_mem_address", "free_physical"]
)
def test_import_cleanup_retries_only_remaining_steps(monkeypatch, vmm, failure):
    vmm, rt = vmm
    for _ in range(2):
        vmm.import_shard(8, 8, 1)
    (handle, pointer, _), (done_handle, _, _) = vmm._imports
    operations = ("unmap_mem", "release_mem_address", "free_physical")
    failed_at = operations.index(failure)
    target = handle if failure == "free_physical" else pointer
    original = getattr(rt, failure)
    pending = [target]

    def fail_once(value):
        if pending == [value]:
            pending.clear()
            return 1
        return original(value)

    monkeypatch.setattr(rt, failure, fail_once)
    with pytest.raises(RuntimeError, match="ACL error"):
        vmm.close_imports()
    assert rt.freed == [done_handle]
    assert vmm._imports == [
        [handle, None if failed_at == 2 else pointer, failed_at == 0]
    ]
    # Fake ACL rejects double-unmap, address release and physical free.
    vmm.close()
    vmm.close()
    assert not vmm._imports and not rt.allocations
    assert not rt.mappings and not rt.reservations


@pytest.mark.parametrize("failure", ["reserve_mem_address", "map_mem"])
def test_partial_import_cleanup_skips_unfinished_steps(monkeypatch, vmm, failure):
    vmm, rt = vmm
    original = getattr(rt, failure)
    monkeypatch.setattr(
        rt,
        failure,
        lambda *args: (None, 1) if failure == "reserve_mem_address" else 1,
    )
    with pytest.raises(RuntimeError, match="ACL error"):
        vmm.import_shard(8, 8, 1)
    monkeypatch.setattr(rt, failure, original)
    handle, pointer, mapped = vmm._imports[0]
    assert not mapped
    assert (pointer is None) == (failure == "reserve_mem_address")
    # Unfinished map/reserve operations must not be undone; Fake ACL rejects it.
    vmm.close()
    vmm.close()
    assert rt.freed == [handle]
    assert not vmm._imports and not rt.allocations
    assert not rt.mappings and not rt.reservations


def test_shared_page_pool_reuses_maximum_not_each_size():
    allocated, freed, mappings = [], [], []

    def allocate(size):
        allocated.append(size)
        return len(allocated)

    layouts = {
        li: {"w": NS(pre_size=size, post_size=0)}
        for li, size in zip((2, 7, 11), (6, 10, 6))
    }
    sizes = compute_slot_sizes(layouts, {li: 0 for li in layouts})
    assert sizes == [10, 0]
    pool = PagePool(
        sizes,
        0,
        granularity=1,
        page_size=2,
        allocate_page=allocate,
        release_page=freed.append,
    )
    reservation = NS(map_existing=lambda *args: mappings.append(args))
    for size in (6, 10, 6):
        mappings.clear()
        pool.map_pages(0, reservation, 0, size)
        assert mappings == [(2 * i, 2, i + 1) for i in range(size // 2)]
    assert allocated == [2] * 5
    assert pool.num_pages(0) == 5 and pool.slot_size(0) == 10
    pool.release()
    pool.release()
    assert freed == list(range(1, 6))


def _weight_specs():
    assignments = {2: 0, 4: 1, 7: 0, 9: 1}
    specs = {
        (li, name): WeightSpec(4, (1, width), (4, width), torch.uint8)
        for li, widths in zip(assignments, ((9, 13), (17, 19), (35, 29), (7, 11)))
        for name, width in zip(("w13_weight", "w2_weight"), widths)
    }
    return specs, assignments


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_npu_reuses_shared_pool_pages_across_weights_and_layers(vmm, rank):
    vmm, rt = vmm
    specs, assignments = _weight_specs()
    local = {key: vmm.create_local(spec, rank) for key, spec in specs.items()}
    assert len(rt.allocations) == len(specs)  # No pool before old weights are released.
    views = vmm.create_weights(
        specs, rank, {key: value[1] for key, value in local.items()}, assignments
    )
    pool = vmm._page_pool
    assert type(pool) is PagePool
    assert pool.page_size == 8 * vmm.granularity
    required = [0, 0]
    for li, slot in assignments.items():
        cursor = 0
        for key in (key for key in specs if key[0] == li):
            layout, view = vmm.layout(specs[key], rank), views[key]
            base = view.pointer - layout.pre_padding
            assert rt.mappings[base + layout.pre_size] == (
                layout.mnnvl_size,
                local[key][1],
            )
            assert (
                view.pointer + rank * specs[key].chunk_bytes
                == base + layout.pre_size + layout.data_offset
            )
            assert local[key][0] - layout.data_offset in rt.mappings
            for offset, size in (
                (0, layout.pre_size),
                (layout.pre_size + layout.mnnvl_size, layout.post_size),
            ):
                for page in range(size // pool.page_size):
                    assert rt.mappings[base + offset + page * pool.page_size] == (
                        pool.page_size,
                        pool._page_handles[slot][cursor],
                    )
                    cursor += 1
        required[slot] = max(required[slot], cursor)
    assert [pool.num_pages(slot) for slot in (0, 1)] == required
    if rank == 1:
        assert vmm.layout(specs[2, "w13_weight"], rank).pre_padding > 0
    handles = set(rt.allocations)
    vmm.close()
    vmm.close()
    assert set(rt.freed) == handles
    assert not rt.allocations and not rt.mappings and not rt.reservations


@pytest.mark.parametrize(
    "failure", ["allocate", "map", "release_mem_address", "free_physical"]
)
def test_npu_pool_partial_failure_cleanup_is_retryable(vmm, failure):
    vmm, rt = vmm
    specs, assignments = _weight_specs()
    local = {key: vmm.create_local(spec, 1)[1] for key, spec in specs.items()}
    if failure == "allocate":
        rt.fail_after_allocations = len(rt.allocations) + 1
    elif failure == "map":
        rt.fail_after_maps = len(rt.mappings) + 1
    if failure in ("allocate", "map"):
        with pytest.raises(RuntimeError, match="ACL error"):
            vmm.create_weights(specs, 1, local, assignments)
    else:
        vmm.create_weights(specs, 1, local, assignments)
        if failure == "free_physical":
            rt.fail_free_after = 1
        else:
            rt.fail_once = failure
        with pytest.raises(RuntimeError, match="ACL error"):
            vmm.close()
        if failure == "free_physical":
            assert len(rt.freed) == 1
            assert rt.freed[0] not in vmm._handles
    handles = set(rt.allocations) | set(rt.freed)
    vmm.close()
    vmm.close()
    assert set(rt.freed) == handles
    assert not rt.allocations and not rt.mappings and not rt.reservations


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
    manager._exports, manager._peer_views = {}, {}
    manager._remote_slices, manager._peer_ranges = {}, {}
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


def _dwdp_config(device, **kwargs):
    from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.arg_groups.parallel_hook import handle_dwdp
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy", device=device, tp_size=2, dwdp_size=2, **kwargs
    )
    handle_dwdp(args)
    parse_cuda_graph_config(args)
    return resolving_view(args)


@pytest.mark.parametrize(
    "kwargs, message",
    [({"enable_lora": True}, "LoRA"), ({"enable_pdmux": True}, "layer-split")],
)
def test_invalid_npu_dwdp_args(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _dwdp_config("npu", **kwargs)


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("disable_fusion", [False, True])
def test_dwdp_preserves_optional_features(device, shared, disable_fusion):
    from sglang.srt.environ import envs

    kwargs = dict(
        cpu_offload_gb=1,
        offload_group_size=2,
        enable_memory_saver=True,
        disable_shared_experts_fusion=disable_fusion,
    )
    with envs.SGLANG_SHARED_EXPERT_TP1.override(shared):
        cfg = _dwdp_config(device, **kwargs)
        assert envs.SGLANG_SHARED_EXPERT_TP1.get() == shared
    assert all(getattr(cfg, name) == value for name, value in kwargs.items())
    assert cfg.attn_dp_size == cfg.ep_size == 2
    assert cfg.dp_size == cfg.moe_dense_tp_size == 1
    assert cfg.moe_a2a_backend == "none" and cfg.enable_dp_lm_head


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ("disabled", "disabled")),
        ({"disable_cuda_graph": False}, ("disabled", "disabled")),
        ({"cuda_graph_backend_decode": "full"}, ("full", "disabled")),
        ({"cuda_graph_backend_prefill": "full"}, ("disabled", "full")),
        (
            {
                "cuda_graph_config": {
                    "decode": {"backend": "full"},
                    "prefill": {"backend": "full"},
                }
            },
            ("full", "full"),
        ),
        (
            {
                "cuda_graph_backend_decode": "full",
                "cuda_graph_config": {
                    "decode": {"backend": "disabled"},
                    "prefill": {"backend": "full"},
                },
            },
            ("disabled", "full"),
        ),
        (
            {
                "cuda_graph_config": {
                    "decode": {"backend": "breakable"},
                    "prefill": {"backend": "breakable"},
                }
            },
            ("breakable", "breakable"),
        ),
    ],
)
def test_dwdp_graph_config_matches_cuda(device, kwargs, expected):
    cfg = _dwdp_config(device, **kwargs)
    assert cfg.disable_cuda_graph
    assert (
        cfg.cuda_graph_config.decode.backend,
        cfg.cuda_graph_config.prefill.backend,
    ) == expected


@pytest.mark.parametrize("nnodes", [1, 2])
def test_npu_dwdp_full_resolution(tmp_path, monkeypatch, nnodes):
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    for module in ("sglang.srt.utils.common", "sglang.srt.hardware_backend.npu.utils"):
        monkeypatch.setattr(f"{module}.get_npu_memory_capacity", lambda: 64 * 1024)
    config = dict(
        architectures=["DeepseekV2ForCausalLM"],
        model_type="deepseek_v2",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        vocab_size=128,
        max_position_embeddings=2048,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    with envs.SGLANG_SHARED_EXPERT_TP1.override(False):
        args = ServerArgs(
            model_path=str(tmp_path),
            device="npu",
            tp_size=2,
            dwdp_size=2,
            nnodes=nnodes,
            dist_init_addr="127.0.0.1:29999",
        )
        args.resolve_once()
        cfg = resolved_view(args)
        assert not envs.SGLANG_SHARED_EXPERT_TP1.get()
        assert cfg.attn_dp_size == cfg.ep_size == 2 and cfg.dp_size == 1
        assert not cfg.disable_shared_experts_fusion
        assert (
            cfg.cuda_graph_config.prefill.backend
            == cfg.cuda_graph_config.decode.backend
            == "disabled"
        )
