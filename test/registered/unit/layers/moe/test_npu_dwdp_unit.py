"""NPU DWDP page coverage and failure cleanup without accelerator hardware."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
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


def test_cuda_manager_binds_shared_layer_discovery():
    from sglang.srt.layers.moe.dwdp import layer_utils

    # Inspect the CUDA class without importing its driver-only dependencies.
    source = Path(layer_utils.__file__).with_name("dwdp_manager.py").read_text()
    tree = ast.parse(source)
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    binding = next(
        node
        for node in manager.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_collect_moe_layers"
            for target in node.targets
        )
    )
    assert ast.unparse(binding.value) == "staticmethod(collect_moe_layers)"
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "sglang.srt.layers.moe.dwdp.layer_utils"
        and any(alias.name == "collect_moe_layers" for alias in node.names)
        for node in tree.body
    )
    buffer_tree = ast.parse(
        Path(layer_utils.__file__).with_name("weight_buffer.py").read_text()
    )
    setup = next(
        node
        for node in ast.walk(buffer_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_layer"
    )
    calls = {
        getattr(node.func, "id", getattr(node.func, "attr", None))
        for node in ast.walk(setup)
        if isinstance(node, ast.Call)
    }
    assert {"map_weight_pages", "build_remote_slices"} <= calls


@pytest.mark.parametrize(
    "backend, weight_dtype",
    [("cuda", torch.float16), ("npu", torch.float16), ("npu", torch.bfloat16)],
)
def test_shared_parameter_gather_preserves_caller_order(
    monkeypatch, backend, weight_dtype
):
    from sglang.srt.hardware_backend.npu.dwdp import manager as npu_manager
    from sglang.srt.layers.moe.dwdp import layer_utils

    names = ("w13_weight_bias", "w13_weight_scale", "w2_weight_bias")
    parameters = {
        name: torch.arange(4, dtype=torch.float32).reshape(2, 2) + index * 20
        for index, name in enumerate(names)
    }
    trace, replaced, helper_calls = [], {}, []
    num_total = 3 if backend == "cuda" else 4
    group = NS(device_group=object())

    def named(count):
        trace.append(("named", count))
        return list(parameters.items())

    def replace(name, full):
        trace.append(("replace", name))
        replaced[name] = full

    experts = NS(
        num_global_routed_experts=num_total,
        w13_weight=torch.empty(0, dtype=weight_dtype),
        named_per_expert_tensors=named,
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

    def gather(experts, local_experts, num_total, dwdp_size, group):
        helper_calls.append((experts, local_experts, num_total, dwdp_size, group))
        return layer_utils.gather_per_expert_tensors(
            experts, local_experts, num_total, dwdp_size, group
        )

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    manager = NS(dwdp_size=2, group=group)
    if backend == "npu":
        monkeypatch.setattr(npu_manager, "gather_per_expert_tensors", gather)
        NPUDwdpManager._replicate_small_params(manager, experts)
    else:
        source = Path(layer_utils.__file__).with_name("dwdp_manager.py").read_text()
        method = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_allgather_small_params"
        )
        namespace = {
            "gather_per_expert_tensors": gather,
            "logger": NS(debug=lambda *_: None),
        }
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        manager.layout = NS(num_experts_per_worker=2, num_routed_experts=num_total)
        namespace[method.name](manager, [(7, experts)], group)
    assert helper_calls == [(experts, 2, num_total, 2, group)]
    assert trace == [("named", 2)] + [
        item for name in names for item in (("gather", name), ("replace", name))
    ]
    assert list(replaced) == list(names)
    for name, data in parameters.items():
        expected = torch.cat((data, data + 10))[:num_total]
        if (
            backend == "npu"
            and name.endswith("_bias")
            and weight_dtype == torch.float16
        ):
            expected = expected.to(torch.float16)
        torch.testing.assert_close(replaced[name], expected)


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
def test_remote_slices_cover_all_remote_weights(monkeypatch, world, expert_bytes):
    import sglang.srt.hardware_backend.npu.dwdp.manager as npu_manager

    calls = []

    def slices(full, start, end, total):
        calls.append((start, end, total))
        return build_remote_slices(full, start, end, total)

    monkeypatch.setattr(npu_manager, "build_remote_slices", slices)
    assert build_remote_slices(torch.empty(2, 1), 0, 2, 2) == []
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
        assert page.total_size >= total + page.pre_padding
        manager._specs, manager._weights = {}, {}
        manager._remote_slices, manager._peer_ranges = {}, {}
        for name in ("w13_weight", "w2_weight"):
            manager._specs[5, name] = spec
            manager._weights[5, name] = torch.zeros(spec.full_shape, dtype=spec.dtype)
        manager._prepare_copy_views()
        assert calls[-2:] == [(rank * 4, (rank + 1) * 4, world * 4)] * 2
        for name in ("w13_weight", "w2_weight"):
            full, covered = manager._weights[5, name], set()
            for dst, expert_start, expert_end in manager._get_remote_slices(5, name):
                offset = dst.data_ptr() - full.data_ptr()
                size = dst.numel() * dst.element_size()
                assert offset == expert_start * expert_bytes
                assert size == (expert_end - expert_start) * expert_bytes
                region = set(range(offset, offset + size))
                assert not covered & region
                covered |= region
                dst.fill_(1)
            assert covered == set(range(total)) - set(range(start, end))
            assert full[rank * 4 : (rank + 1) * 4].count_nonzero() == 0
            assert full.count_nonzero() == total - spec.chunk_bytes


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int8])
def test_npu_uses_shared_copy_for_all_remote_edges(monkeypatch, rank, dtype):
    import sglang.srt.hardware_backend.npu.dwdp.manager as npu_manager

    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager.dwdp_size, manager.dwdp_rank = 4, rank
    manager._specs, manager._weights, manager._peer_views = {}, {}, {}
    manager._remote_slices, manager._peer_ranges = {}, {}
    layers, local = [2, 5, 8, 11], {}
    for pos, li in enumerate(layers):
        local_experts = 1 + pos % 2
        num_experts = 4 * local_experts
        for name, width in (("w13_weight", 10), ("w2_weight", 14)):
            width //= torch.tensor([], dtype=dtype).element_size()
            manager._specs[li, name] = WeightSpec(
                num_experts, (local_experts, width), (num_experts, width), dtype
            )
            manager._weights[li, name] = torch.empty(num_experts, width, dtype=dtype)
            for peer in range(4):
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
    calls, original = [], npu_manager.copy_remote_expert_slices

    def copy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(npu_manager, "copy_remote_expert_slices", copy)
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
                        for peer in range(4)
                    ]
                )
                torch.testing.assert_close(manager._weights[li, name], expected)
                count = manager._specs[li, name].local_experts
                torch.testing.assert_close(
                    manager._weights[li, name][rank * count : (rank + 1) * count],
                    local[li, name],
                )
    assert calls == layers * 2


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


@pytest.mark.parametrize("fail", [False, True])
def test_migration_releases_source_only_after_success(monkeypatch, fail):
    manager = NPUDwdpManager.__new__(NPUDwdpManager)
    manager.device, manager.dwdp_rank = torch.device("cpu"), 0
    manager._position = {5: 0}
    manager._specs = {(5, "w13_weight"): WeightSpec(4, (2, 3), (4, 3), torch.float16)}
    manager._weights, manager._exports = {}, {}
    manager._vmm = NS(create_local=lambda *args: (100, 1))
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
        assert manager._migrate_weight((5, "w13_weight"), source, [123]) == 1
    assert (source.untyped_storage().nbytes() > 0) == fail
    assert calls == ([] if fail else ["cache"])


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


def _fake_import_vmm(monkeypatch):
    rt = _FakeACL()
    monkeypatch.setattr(
        rt,
        "mem_import_from_shareable_handle",
        lambda size, device: rt.malloc_physical(size),
        raising=False,
    )
    return NPUWeightVMM(NS(rt=rt), NS(index=0)), rt


@pytest.mark.parametrize(
    "failure", ["unmap_mem", "release_mem_address", "free_physical"]
)
def test_import_cleanup_retries_only_remaining_steps(monkeypatch, failure):
    vmm, rt = _fake_import_vmm(monkeypatch)
    for _ in range(2):
        vmm.import_shard(8, 8, 1)
    (handle, pointer, _), (done_handle, done_pointer, _) = vmm._imports
    operations = ("unmap_mem", "release_mem_address", "free_physical")
    failed_at = operations.index(failure)
    values = (pointer, pointer, handle)
    done_values = (done_pointer, done_pointer, done_handle)
    pending = {(failure, values[failed_at])}
    calls = []
    for operation in operations:
        original = getattr(rt, operation)

        def invoke(value, operation=operation, original=original):
            calls.append((operation, value))
            if (operation, value) in pending:
                pending.remove((operation, value))
                return 1
            return original(value)

        monkeypatch.setattr(rt, operation, invoke)
    with pytest.raises(RuntimeError, match="ACL error"):
        vmm.close_imports()
    remaining = [entry.copy() for entry in vmm._imports]
    assert calls == list(zip(operations, done_values)) + list(
        zip(operations[: failed_at + 1], values[: failed_at + 1])
    )
    assert rt.freed == [done_handle]
    calls.clear()
    # The old implementation repeats an already successful unmap here when
    # address/handle release failed, which the fake ACL rejects.
    vmm.close_imports()
    assert remaining == [[handle, None if failed_at == 2 else pointer, failed_at == 0]]
    assert calls == list(zip(operations[failed_at:], values[failed_at:]))
    calls.clear()
    vmm.close_imports()
    vmm.close()
    vmm.close()
    assert calls == []
    assert not vmm._imports and not rt.allocations
    assert not rt.mappings and not rt.reservations


@pytest.mark.parametrize("failure", ["reserve_mem_address", "map_mem"])
def test_partial_import_cleanup_skips_unfinished_steps(monkeypatch, failure):
    vmm, rt = _fake_import_vmm(monkeypatch)
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
    calls = []
    for operation in ("unmap_mem", "release_mem_address", "free_physical"):
        original = getattr(rt, operation)

        def invoke(value, operation=operation, original=original):
            calls.append((operation, value))
            return original(value)

        monkeypatch.setattr(rt, operation, invoke)
    vmm.close_imports()
    vmm.close_imports()
    vmm.close()
    assert calls == ([] if pointer is None else [("release_mem_address", pointer)]) + [
        ("free_physical", handle)
    ]
    assert not vmm._imports and not rt.allocations
    assert not rt.mappings and not rt.reservations


@pytest.mark.parametrize("pre, post", [(0, 0), (4, 0), (0, 2), (4, 2)])
def test_shared_weight_mapping_order_and_cursor(pre, post):
    allocated = []

    def allocate(size):
        allocated.append(size)
        return len(allocated)

    pool = PagePool(
        [20, 0],
        0,
        granularity=1,
        page_size=2,
        allocate_page=allocate,
        release_page=lambda handle: None,
    )
    layout = NS(
        pre_size=pre,
        post_size=post,
        mnnvl_size=4,
        pre_pages=pre // 2,
        post_pages=post // 2,
    )
    trace, expected, cursor = [], [], 1
    reservation = NS(map_existing=lambda *args: trace.append(args))
    for handle in (100, 101):
        before = cursor
        cursor = pool.map_weight_pages(0, reservation, layout, handle, cursor)
        expected += [(i * 2, 2, before + i + 1) for i in range(pre // 2)]
        expected.append((pre, 4, handle))
        expected += [
            (pre + 4 + i * 2, 2, before + pre // 2 + i + 1) for i in range(post // 2)
        ]
    assert cursor == 1 + 2 * (pre + post) // 2
    assert trace == expected


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


def test_shared_page_pool_preserves_cuda_driver_order(monkeypatch):
    trace = []

    def granularity(device):
        trace.append(("granularity", device))
        return 2

    def prop(device, handle_types):
        trace.append(("prop", device, handle_types))
        return "prop"

    class Reservation:
        def __init__(self, size, prop, device, alignment):
            trace.append(("reserve", size, prop, device, alignment))

        def map(self, offset, size, retain_handle):
            handle = 100 + sum(item[0] == "map" for item in trace)
            trace.append(("map", offset, size, retain_handle, handle))
            return handle

        def close(self, release_handles):
            trace.append(("close", release_handles))

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.cuda_vmm_utils",
        NS(
            get_device_granularity=granularity,
            make_device_allocation_prop=prop,
            VmmReservation=Reservation,
            check_drv=lambda value, label: value,
        ),
    )
    bindings = NS(
        driver=NS(cuMemRelease=lambda handle: trace.append(("release", handle)))
    )
    monkeypatch.setitem(sys.modules, "cuda", NS(bindings=bindings))
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    pool = PagePool([17, 1], 3)
    reservation = NS(map_existing=lambda *args: trace.append(("existing", *args)))
    pool.map_pages(0, reservation, 32, 17)
    pool.release()
    pool.release()
    expected = [("granularity", 3), ("prop", 3, None)]
    for handle in (100, 101, 102):
        expected += [
            ("reserve", 16, "prop", 3, 2),
            ("map", 0, 16, True, handle),
            ("close", False),
        ]
    assert trace == expected + [
        ("existing", 32, 16, 100),
        ("existing", 48, 16, 101),
        ("release", 100),
        ("release", 101),
        ("release", 102),
    ]


def _fake_weight_vmm(monkeypatch):
    import sglang.srt.hardware_backend.npu.dwdp.vmm as module

    monkeypatch.setattr(
        module,
        "tensor_from_pointer",
        lambda pointer, nbytes, **metadata: NS(
            pointer=pointer, nbytes=nbytes, **metadata
        ),
    )
    rt = _FakeACL()
    vmm = NPUWeightVMM(NS(rt=rt), NS(index=0))
    assignments = {2: 0, 4: 1, 7: 0, 9: 1}
    specs = {
        (li, name): WeightSpec(4, (1, width), (4, width), torch.uint8)
        for li, widths in zip(assignments, ((9, 13), (17, 19), (35, 29), (7, 11)))
        for name, width in zip(("w13_weight", "w2_weight"), widths)
    }
    return vmm, rt, specs, assignments


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_npu_reuses_shared_pool_pages_across_weights_and_layers(monkeypatch, rank):
    vmm, rt, specs, assignments = _fake_weight_vmm(monkeypatch)
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
def test_npu_pool_partial_failure_cleanup_is_retryable(monkeypatch, failure):
    vmm, rt, specs, assignments = _fake_weight_vmm(monkeypatch)
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
