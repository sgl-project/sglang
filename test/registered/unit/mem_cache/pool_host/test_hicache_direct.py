import mmap
from dataclasses import replace
from functools import partial
from itertools import permutations
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import (
    MambaPool,
    MHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
)
from sglang.srt.mem_cache.pool_host.direct import (
    load_block_dma_module,
    transfer_layer_block,
    transfer_mamba_block,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("block_ends", [None, (1, 2)])
def test_hybrid_pool_order_reuses_direct_layout(monkeypatch, block_ends):
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host import direct
    from sglang.srt.mem_cache.pool_host.group import PoolEntry

    entries = {}
    for name in (PoolName.KV, PoolName.SWA, PoolName.DRAFT):
        host = object.__new__(MHATokenToKVPoolHost)
        host.kv_buffer = torch.empty(1)
        device = SimpleNamespace(k_buffer=[torch.empty(1)], v_buffer=[torch.empty(1)])
        entries[name] = PoolEntry(name, host, device, layer_mapper=lambda layer: layer)
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=entries[PoolName.KV], entry_map=entries
    )
    engine = object.__new__(L2TransferEngine)
    engine.batch_direct = True
    engine._direct_layouts = {}
    controller.l2_transfer_engine = engine
    prepared = []

    def prepare(transfers, block_ends):
        prepared.append((transfers, block_ends))
        return object()

    monkeypatch.setattr(direct, "prepare_direct_transfer", prepare)
    indices = torch.tensor([0])
    sidecars = [
        PoolTransfer(
            name=name,
            host_indices=torch.tensor([i]),
            device_indices=torch.tensor([i + 3]),
        )
        for i, name in enumerate((PoolName.SWA, PoolName.DRAFT), start=1)
    ]
    layouts = []
    for ordered in permutations(sidecars):
        transfers = controller._l2_transfers(indices, indices, list(ordered))
        layouts.append(engine._prepare_direct(transfers, block_ends))
        for sidecar in sidecars:
            transfer = next(
                t for t in transfers if t.host_pool is entries[sidecar.name].host_pool
            )
            assert transfer.host_indices is sidecar.host_indices
            assert transfer.device_indices is sidecar.device_indices
    assert all(layout is layouts[0] for layout in layouts)
    assert len(prepared) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("pages", [[], [5, 0, 3]])
def test_native_page_indices_preserve_page_order(page_size, pages):
    expected = torch.tensor(pages, dtype=torch.int64)
    tokens = (expected[:, None] * page_size + torch.arange(page_size)).flatten()
    result = load_block_dma_module().page_indices(
        [(tokens, page_size), (tokens.clone(), page_size)]
    )
    assert len(result) == 2
    for indices in result:
        assert torch.equal(torch.from_dlpack(indices), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("invalid", ["partial", "dtype", "strided", "rank", "cuda"])
def test_native_page_indices_reject_invalid_tokens(invalid):
    tokens = torch.arange(128)
    tokens = {
        "partial": lambda: tokens[:-1],
        "dtype": lambda: tokens.int(),
        "strided": lambda: torch.arange(256)[::2],
        "rank": lambda: tokens.view(1, 128),
        "cuda": lambda: tokens.cuda(),
    }[invalid]()
    with pytest.raises(ValueError, match="pages"):
        load_block_dma_module().page_indices([(torch.arange(128), 128), (tokens, 128)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_native_restore_combines_pools_and_preserves_unselected_slots():
    module = load_block_dma_module()
    host = torch.arange(4 * 7 * 8, dtype=torch.float32).reshape(4, 7, 8).pin_memory()
    device = torch.full_like(host, -1, device="cuda")
    mamba_host = (
        torch.arange(5 * 2 * 1 * 17, dtype=torch.float32)
        .reshape(5, 2, 1, 17)
        .pin_memory()
    )
    mamba_device = torch.full((2, 6, 17), -1.0, device="cuda")
    host_ids, device_ids = torch.tensor([0, 1, 4]), torch.tensor([3, 4, 1])
    mh, md = torch.tensor([1, 2, 4]), torch.tensor([2, 3, 5])
    counter = LayerDoneCounter(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    kv = [
        [
            (
                (
                    [host[2 * layer], host[2 * layer + 1]],
                    [device[2 * layer], device[2 * layer + 1]],
                    host,
                    [],
                    [],
                ),
                host_ids,
                device_ids,
            )
        ]
        for layer in range(2)
    ]
    mamba = [
        (([mamba_host[:, layer, 0]], [mamba_device[layer]], mamba_host, [], []), mh, md)
        for layer in range(2)
    ]
    producer = counter.events[counter.update_producer()]
    with torch.cuda.stream(stream):
        module.transfer(
            [block + [state] for block, state in zip(kv, mamba)],
            True,
            stream.cuda_stream,
            partial(producer.complete),
        )
    producer.finish_event.synchronize()
    expected = torch.full_like(host, -1)
    expected[:, device_ids] = host[:, host_ids]
    assert torch.equal(device.cpu(), expected)
    expected_mamba = torch.full((2, 6, 17), -1.0)
    expected_mamba[:, md] = mamba_host[mh, :, 0].transpose(0, 1)
    assert torch.equal(mamba_device.cpu(), expected_mamba)
    assert all(event.query() for event in producer.load_events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("invalid", ["late_index", "cross_pool_alias", "unpinned_host"])
def test_native_restore_validates_all_blocks_before_submitting(invalid):
    module = load_block_dma_module()
    host = torch.ones((2, 4, 8), pin_memory=True)
    device = torch.zeros_like(host, device="cuda")
    index = torch.tensor([1])
    first = (([host[0]], [device[0]], host, [], []), index, index)
    last_index = torch.tensor([4]) if invalid == "late_index" else index
    last_host = (
        torch.ones_like(host, pin_memory=False) if invalid == "unpinned_host" else host
    )
    last = (([last_host[1]], [device[1]], last_host, [], []), index, last_index)
    blocks = [[first], [last]]
    if invalid == "cross_pool_alias":
        blocks[1].append(last)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    completed = []
    with pytest.raises(
        RuntimeError, match="out of bounds|overlap|requires pinned host memory"
    ):
        module.transfer(blocks, True, stream.cuda_stream, completed.append)
    stream.synchronize()
    assert completed == []
    assert torch.count_nonzero(device).item() == 0


@pytest.fixture
def mxfp8_pools():
    device = MHATokenToKVPoolMXFP8(
        size=512,
        page_size=128,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=128,
        layer_num=2,
        device="cuda",
        enable_memory_saver=False,
    )
    host = MHATokenToKVPoolMXFP8Host(
        device,
        host_to_device_ratio=2,
        host_size=0,
        page_size=128,
        layout="page_first_direct",
    )
    yield host, device
    host.destroy()


@pytest.fixture(
    params=[(False, False), (True, False), (False, True), (True, True)],
    ids=["layer-first-conv", "page-first-conv", "layer-first-ssm", "page-first-ssm"],
)
def mamba_pools(request):
    envelope, has_ssm = request.param
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=8,
        n_groups=1,
        num_heads=2,
        head_dim=4,
        state_size=4,
        conv_kernel=3,
    )
    if not has_ssm:
        shape = replace(shape, temporal=(0, 0, 0))
    device = MambaPool(
        size=12,
        spec_state_size=4,
        cache_params=Mamba2CacheParams(shape=shape, layers=[0, 1, 2]),
        mamba_layer_ids=[0, 1, 2],
        envelope_layout=envelope,
        device="cuda",
        enable_memory_saver=False,
    )
    host = MambaPoolHost(
        device, host_to_device_ratio=2, host_size=0, layout="page_first_direct"
    )
    yield host, device
    host.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("host_to_device", [False, True])
def test_l2_prepares_physical_indices_under_shared_layout_lease(
    mxfp8_pools, monkeypatch, batched, host_to_device
):
    from sglang.srt.mem_cache.pool_host import direct

    host, device = mxfp8_pools
    engine = L2TransferEngine("direct")
    if batched:
        engine.enable_batch_direct()
    calls = []
    released = []

    def acquire():
        assert not calls
        calls.append("acquire")

    def translate(indices):
        assert calls[0] == "acquire" and not released
        calls.append("translate")
        return indices + 128

    def prepare(host_ids, device_ids, backend):
        assert calls[-1] == "translate" and not released
        assert backend == "direct"
        calls.append("prepare")
        return host_ids + 256, device_ids

    def release(event, key):
        assert calls.count("prepare") == 2
        released.append((event, key))

    domain = SimpleNamespace(acquire_layout=acquire, release_layout=release)
    monkeypatch.setattr(host, "shared_allocation_domain", domain)
    monkeypatch.setattr(host, "prepare_transfer_indices", prepare)
    monkeypatch.setattr(device, "host_transfer_translate", translate, raising=False)
    prepare_direct = direct.prepare_direct_transfer

    def prepare_layout(transfers, block_ends):
        assert calls.count("prepare") == 2 and not released
        return prepare_direct(transfers, block_ends)

    monkeypatch.setattr(direct, "prepare_direct_transfer", prepare_layout)
    pairs = [(host.k_data_refs, device.k_buffer), (host.v_data_refs, device.v_buffer)]
    for host_layers, device_layers in pairs:
        for host_rows, device_rows in zip(host_layers, device_layers):
            source, destination = (
                (host_rows, device_rows) if host_to_device else (device_rows, host_rows)
            )
            source.view(torch.uint8).random_(1, 128)
            destination.view(torch.uint8).zero_()
    transfers = [
        L2Transfer(
            host,
            device,
            torch.arange(first, first + 128),
            torch.arange(first, first + 128),
        )
        for first in (0, 128)
    ]
    if host_to_device:
        completion = engine.submit_host_to_device(transfers, transfer_layer_id_max=2)
    else:
        completion = engine.submit_device_to_host(transfers)
    direction = "host_to_device" if host_to_device else "device_to_host"
    assert calls == ["acquire", "translate", "prepare", "translate", "prepare"]
    assert released == [(completion.finish_event, (id(engine), direction))]
    completion.finish_event.synchronize()
    for host_layers, device_layers in pairs:
        for host_rows, device_rows in zip(host_layers, device_layers):
            cpu_host = host_rows.flatten(0, 1).view(torch.uint8)
            cpu_device = device_rows.cpu().view(torch.uint8)
            source, destination = (
                (cpu_host, cpu_device) if host_to_device else (cpu_device, cpu_host)
            )
            expected = torch.zeros_like(destination)
            if host_to_device:
                expected[128:384] = source[256:512]
            else:
                expected[256:512] = source[128:384]
            assert torch.equal(destination, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_combined_backup_preserves_mxfp8_scales_and_mamba_slots(
    mxfp8_pools, mamba_pools
):
    host, device = mxfp8_pools
    mh, md = mamba_pools
    hi, di = torch.arange(256, 512), torch.arange(128, 384)
    mhi, mdi = torch.tensor([1, 4]), torch.tensor([2, 5])
    pairs = [
        (
            host.k_data_refs,
            [buf.unflatten(0, (-1, 128)) for buf in device.k_buffer],
            torch.tensor([2, 3]),
            torch.tensor([1, 2]),
        ),
        (
            host.v_data_refs,
            [buf.unflatten(0, (-1, 128)) for buf in device.v_buffer],
            torch.tensor([2, 3]),
            torch.tensor([1, 2]),
        ),
        (
            host.k_scale_host_layers,
            host.k_scale_device_rows,
            torch.tensor([2, 3]),
            torch.tensor([1, 2]),
        ),
        (
            host.v_scale_host_layers,
            host.v_scale_device_rows,
            torch.tensor([2, 3]),
            torch.tensor([1, 2]),
        ),
    ]
    for hosts, devices, _, _ in pairs:
        for source, destination in zip(devices, hosts):
            source.view(torch.uint8).random_(1, 128)
            destination.view(torch.uint8).zero_()
    mamba_pairs = list(
        zip(
            [*mh.conv_buffer, mh.temporal_buffer],
            [*md.mamba_cache.conv, md.mamba_cache.temporal],
        )
    )
    for destination, source in mamba_pairs:
        source.random_(1, 50)
        destination.zero_()
    transfers = [L2Transfer(host, device, hi, di), L2Transfer(mh, md, mhi, mdi)]
    engine = L2TransferEngine("direct")
    engine.enable_batch_direct()
    ack = engine.submit_device_to_host(transfers)
    ack.finish_event.synchronize()
    layout = engine._prepare_direct(transfers)
    assert layout is not None
    assert engine._prepare_direct(transfers) is layout
    for hosts, devices, dst_ids, src_ids in pairs:
        for destination, source in zip(hosts, devices):
            actual = destination.view(torch.uint8)
            expected = torch.zeros_like(actual)
            expected[dst_ids] = source.view(torch.uint8).cpu()[src_ids]
            assert torch.equal(actual, expected)
    for destination, source in mamba_pairs:
        expected = torch.zeros_like(destination)
        expected[mhi, :, 0] = source.cpu()[:, mdi].transpose(0, 1)
        assert torch.equal(destination, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_backup_rejects_late_invalid_pool_before_any_write():
    module = load_block_dma_module()
    host = torch.zeros((2, 4, 8), pin_memory=True)
    device = torch.ones_like(host, device="cuda")
    index = torch.tensor([1])
    transfers = [
        (([host[0]], [device[0]], host, [], []), index, index),
        (([host[1]], [device[1]], host, [], []), torch.tensor([4]), index),
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="out of bounds"):
        module.transfer([transfers], False, stream.cuda_stream, None)
    stream.synchronize()
    assert torch.count_nonzero(host).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("notify_layers", [False, True])
def test_native_mxfp8_restore_moves_payload_and_scales(
    mxfp8_pools, notify_layers, monkeypatch
):
    from sglang.srt.mem_cache.pool_host import direct

    module = load_block_dma_module()
    copy_stats = []

    def record_transfer(*args):
        result = module.transfer(*args)
        copy_stats.append(tuple(result))
        return result

    monkeypatch.setattr(
        direct,
        "load_block_dma_module",
        lambda: SimpleNamespace(
            transfer=record_transfer,
            registration_bounds=module.registration_bounds,
            page_indices=module.page_indices,
        ),
    )
    host, device = mxfp8_pools
    host.kv_buffer.view(torch.uint8).random_(0, 128)
    host.k_scale_host.random_(1, 200)
    host.v_scale_host.random_(1, 200)
    host_ids = torch.cat((torch.arange(256, 384), torch.arange(0, 128)))
    device_ids = torch.arange(128, 384)
    transfer = L2Transfer(host, device, host_ids, device_ids)
    engine = L2TransferEngine("direct")
    engine.enable_batch_direct()
    counter = LayerDoneCounter(2)
    producer = counter.events[counter.update_producer()]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        ack = engine.submit_host_to_device(
            [transfer],
            transfer_layer_id_max=2,
            layer_block_ends=(1, 2),
            on_layer_done=producer.complete if notify_layers else None,
        )
        ack.finish_event.synchronize()
    assert "hicache.restore_dma_cpp_submit" in {
        event.name for event in profile.events()
    }
    # One descriptor per page/layer/component, including all 128 tokens.
    assert copy_stats == [(16, 128 * 128)]
    layout = engine._prepare_direct([transfer], (1, 2))
    assert layout is not None
    # Repeated restores reuse the same tensor views, including scale views.
    assert engine._prepare_direct([transfer], (1, 2)) is layout
    for host_rows, device_rows, source_ids, destination_ids in (
        (
            host.k_data_refs,
            [buf.unflatten(0, (-1, 128)) for buf in device.k_buffer],
            torch.tensor([2, 0]),
            torch.tensor([1, 2]),
        ),
        (
            host.v_data_refs,
            [buf.unflatten(0, (-1, 128)) for buf in device.v_buffer],
            torch.tensor([2, 0]),
            torch.tensor([1, 2]),
        ),
        (
            host.k_scale_host_layers,
            host.k_scale_device_rows,
            torch.tensor([2, 0]),
            torch.tensor([1, 2]),
        ),
        (
            host.v_scale_host_layers,
            host.v_scale_device_rows,
            torch.tensor([2, 0]),
            torch.tensor([1, 2]),
        ),
    ):
        for source, destination in zip(host_rows, device_rows):
            actual = destination.view(torch.uint8).cpu()
            expected = torch.zeros_like(actual)
            expected[destination_ids] = source.view(torch.uint8)[source_ids]
            assert torch.equal(actual, expected)
    if notify_layers:
        assert all(event.query() for event in producer.load_events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("replace_keys", [False, True])
def test_native_restore_rebuilds_views_after_device_buffer_replacement(
    mxfp8_pools, replace_keys
):
    host, device = mxfp8_pools
    host.kv_buffer.view(torch.uint8).fill_(17)
    indices = torch.arange(128)
    transfer = L2Transfer(host, device, indices, indices)
    engine = L2TransferEngine("direct")
    engine.enable_batch_direct()
    previous = engine._prepare_direct([transfer], (1, 2))
    assert previous is not None
    if replace_keys:
        device.k_buffer = [torch.zeros_like(buffer) for buffer in device.k_buffer]
        source, destination = host.k_data_refs, device.k_buffer
    else:
        device.v_buffer = [torch.zeros_like(buffer) for buffer in device.v_buffer]
        source, destination = host.v_data_refs, device.v_buffer
    completed = []
    ack = engine.submit_host_to_device(
        [transfer],
        transfer_layer_id_max=2,
        layer_block_ends=(1, 2),
        on_layer_done=completed.append,
    )
    ack.finish_event.synchronize()
    assert completed == [0, 1]
    assert engine._prepare_direct([transfer], (1, 2)) is not previous
    for host_layer, device_layer in zip(source, destination, strict=True):
        assert torch.equal(
            device_layer[:128].view(torch.uint8).cpu(),
            host_layer[0].view(torch.uint8),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_native_mxfp8_rejects_partial_pages_before_payload(mxfp8_pools):
    host, device = mxfp8_pools
    host.kv_buffer.view(torch.uint8).fill_(17)
    host_ids = torch.arange(127)
    device_ids = torch.arange(127)
    engine = L2TransferEngine("direct")
    engine.enable_batch_direct()
    completed = []
    with pytest.raises(ValueError, match="pages"):
        engine.submit_host_to_device(
            [L2Transfer(host, device, host_ids, device_ids)],
            transfer_layer_id_max=2,
            on_layer_done=completed.append,
        )
    engine.host_to_device_stream.synchronize()
    assert completed == []
    assert all(
        torch.count_nonzero(buf.view(torch.uint8)).item() == 0
        for buf in device.k_buffer + device.v_buffer
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("page_size", [16, 128])
def test_bf16_page_first_direct_restores_layer_blocks(page_size):
    device = MHATokenToKVPool(
        size=4 * page_size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        layer_num=3,
        device="cuda",
        enable_memory_saver=False,
    )
    host = MHATokenToKVPoolHost(device, 2, 0, page_size, "page_first_direct")
    try:
        host.kv_buffer.random_(1, 97)
        hi = torch.cat(
            (torch.arange(2 * page_size, 3 * page_size), torch.arange(page_size))
        )
        di = torch.arange(page_size, 3 * page_size)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            host.load_to_device_layer_block(device, hi, di, [0, 2], "direct")
        stream.synchronize()
        for host_layers, device_layers in (
            (host.k_data_refs, device.k_buffer),
            (host.v_data_refs, device.v_buffer),
        ):
            for layer, gpu in enumerate(device_layers):
                expected = torch.zeros_like(gpu, device="cpu")
                if layer in (0, 2):
                    expected[di] = host_layers[layer][[2, 0]].flatten(0, 1)
                assert torch.equal(gpu.cpu(), expected)
    finally:
        host.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("mamba", [False, True])
@pytest.mark.parametrize("host_to_device", [False, True])
def test_direct_rejects_unpinned_host_before_copy(mamba, host_to_device):
    host = torch.ones((4, 2, 1, 8) if mamba else (2, 4, 8))
    device = torch.zeros((2, 4, 8), device="cuda")
    indices = torch.tensor([1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        pytest.raises(RuntimeError, match="requires pinned host memory"),
    ):
        if mamba:
            transfer_mamba_block(
                [host], [device], indices, indices, [0, 1], host_to_device
            )
        else:
            source, destination = (host, device) if host_to_device else (device, host)
            transfer_layer_block(
                list(source), list(destination), indices, indices, host
            )
    stream.synchronize()
    assert torch.count_nonzero(device).item() == 0
    assert torch.all(host == 1).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("default_stream", [False, True])
def test_direct_layer_block_preserves_holes(default_stream):
    host = (
        torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16).pin_memory()
    )
    device = torch.full_like(host, -1, device="cuda")
    source = torch.tensor([0, 1, 4, 7, 8, 9])
    destination = torch.tensor([2, 3, 5, 7, 8, 9])
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        count, largest = transfer_layer_block(
            list(host), list(device), source, destination, host
        )
    stream.synchronize()
    expected = torch.full_like(host, -1)
    expected[:, destination] = host[:, source]
    assert torch.equal(device.cpu(), expected)
    assert count == 9
    assert largest == 3 * 16 * host.element_size()
    output = torch.full_like(host, -2, pin_memory=True)
    with torch.cuda.stream(stream):
        transfer_layer_block(list(device), list(output), destination, source, output)
    stream.synchronize()
    expected.fill_(-2)
    expected[:, source] = host[:, source]
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_direct_mamba_batch_handles_mixed_component_sizes():
    # Put the smallest component first to cover the driver-sensitive ordering.
    hosts = [
        torch.arange(5 * 3 * size).reshape(5, 3, 1, size).pin_memory()
        for size in (8, 128, 32)
    ]
    devices = [
        torch.full((3, 7, size), -1, dtype=torch.int64, device="cuda")
        for size in (8, 128, 32)
    ]
    source, destination = torch.tensor([1, 4]), torch.tensor([5, 2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        count, largest = transfer_mamba_block(
            hosts, devices, source, destination, [0, 2], True
        )
    stream.synchronize()
    assert count == 12
    assert largest == 128 * hosts[0].element_size()
    for host, device in zip(hosts, devices):
        expected = torch.full_like(device, -1, device="cpu")
        for layer in (0, 2):
            expected[layer, destination] = host[source, layer, 0]
        assert torch.equal(device.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("default_stream", [False, True])
@pytest.mark.parametrize("host_to_device", [False, True])
def test_direct_mamba_batch_preserves_layers_and_holes(default_stream, host_to_device):
    sizes = (8, 128, 32)
    hosts = [
        torch.arange(12 * 4 * size).reshape(12, 4, 1, size).pin_memory()
        for size in sizes
    ]
    devices = [
        torch.full((4, 14, size), -1, dtype=torch.int64, device="cuda")
        for size in sizes
    ]
    host_ids = torch.tensor([1, 2, 3, 7, 8, 10])
    device_ids = torch.tensor([4, 5, 6, 1, 2, 12])
    layers = [0, 2]
    if not host_to_device:
        for host, device in zip(hosts, devices):
            device.copy_(torch.arange(device.numel()).reshape(device.shape))
            host.fill_(-2)
    expected = [
        tensor.cpu().clone() for tensor in (devices if host_to_device else hosts)
    ]
    for host, device, output in zip(hosts, devices, expected):
        for layer in layers:
            if host_to_device:
                output[layer, device_ids] = host[host_ids, layer, 0]
            else:
                output[host_ids, layer, 0] = device[layer, device_ids].cpu()
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        count, largest = transfer_mamba_block(
            hosts, devices, host_ids, device_ids, layers, host_to_device
        )
    stream.synchronize()
    assert count == len(host_ids) * len(layers) * len(sizes)
    assert largest == max(sizes) * hosts[0].element_size()
    for actual, output in zip(devices if host_to_device else hosts, expected):
        assert torch.equal(actual.cpu(), output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("host_to_device", [False, True])
def test_direct_mamba_batch_splits_host_registration_boundaries(
    monkeypatch, host_to_device
):
    from sglang.srt.mem_cache.pool_host import direct

    host = (
        torch.arange(8 * 3 * 16, dtype=torch.float32).reshape(8, 3, 1, 16).pin_memory()
    )
    device = torch.arange(3 * 8 * 16, dtype=torch.float32, device="cuda").reshape(
        3, 8, 16
    )
    # One boundary falls between rows; another cuts a selected row in half.
    offsets = (0, 576, 992, host.numel() * host.element_size())
    monkeypatch.setattr(
        direct,
        "_registration_bounds",
        lambda buffer: (
            [buffer.data_ptr() + offset for offset in offsets[:-1]],
            [buffer.data_ptr() + offset for offset in offsets[1:]],
        ),
    )
    expected = device.cpu() if host_to_device else host.clone()
    if host_to_device:
        expected[0] = host[:, 0, 0]
    else:
        expected[:, 0, 0] = device[0].cpu()
    indices = torch.arange(8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        count, largest = transfer_mamba_block(
            [host], [device], indices, indices, [0], host_to_device
        )
    stream.synchronize()
    assert count == 9
    assert largest == 16 * host.element_size()
    assert torch.equal(device.cpu() if host_to_device else host, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_native_restore_rejects_overlapping_mamba_destinations_before_submitting():
    host = torch.ones((6, 2, 1, 32), pin_memory=True)
    device = torch.zeros((2, 6, 32), device="cuda")
    ids = torch.tensor([1, 2, 3])
    transfer = (
        (
            [host[:, layer, 0] for layer in range(2)],
            [device[layer] for layer in range(2)],
            host,
            [],
            [],
        ),
        ids,
        ids,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    completed = []
    with pytest.raises(RuntimeError, match="overlap"):
        load_block_dma_module().transfer(
            [[transfer, transfer]], True, stream.cuda_stream, completed.append
        )
    stream.synchronize()
    assert completed == []
    assert torch.count_nonzero(device).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_direct_splits_copies_at_host_registration_boundaries():
    chunk = mmap.PAGESIZE
    runtime = torch.cuda.cudart()
    with mmap.mmap(-1, 3 * chunk) as allocation:
        host = torch.frombuffer(allocation, dtype=torch.float32).reshape(3, chunk // 4)
        host.copy_(torch.arange(host.numel()).reshape(host.shape))
        registered = []
        try:
            for offset, size in ((0, chunk), (chunk, 2 * chunk)):
                ptr = host.data_ptr() + offset
                assert int(runtime.cudaHostRegister(ptr, size, 0)) == 0
                registered.append(ptr)
            device = torch.full_like(host, -1, device="cuda")
            indices = torch.arange(3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                count, largest = transfer_layer_block(
                    [host], [device], indices, indices, host
                )
            stream.synchronize()
            assert count == 2
            assert largest == 2 * chunk
            assert torch.equal(device.cpu(), host)
        finally:
            torch.cuda.synchronize()
            for ptr in reversed(registered):
                assert int(runtime.cudaHostUnregister(ptr)) == 0
            del host


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("host_to_device", [False, True])
@pytest.mark.parametrize("layers", [[0, 1, 2, 3], [3, 0, 1]])
def test_page_first_mamba_coalesces_adjacent_layers(
    monkeypatch, host_to_device, layers
):
    from sglang.srt.mem_cache.layout.token_major import (
        build_mamba_entry_views,
        mamba_entry_bytes,
    )
    from sglang.srt.mem_cache.pool_host import direct

    spec = dict(
        layer_num=4,
        conv_state_shapes=[(2, 8), (4, 16)],
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )
    raw = torch.zeros(7 * mamba_entry_bytes(**spec), dtype=torch.uint8, device="cuda")
    convs, temporal = build_mamba_entry_views(raw, **spec, max_slots=7)
    devices = [*convs, temporal]
    hosts = [
        torch.empty((8, 4, 1, *t.shape[2:]), dtype=t.dtype, pin_memory=True)
        for t in devices
    ]
    for host, device in zip(hosts, devices):
        host.copy_(torch.arange(host.numel()).reshape(host.shape) % 97)
        device.copy_(
            torch.arange(device.numel(), device="cuda").reshape(device.shape) % 71
        )
    raw_before = raw.clone()
    host_ids, device_ids = torch.tensor([1, 4]), torch.tensor([5, 2])
    expected = [t.cpu().clone() for t in (devices if host_to_device else hosts)]
    for host, device, result in zip(hosts, devices, expected):
        for layer in layers:
            if host_to_device:
                result[layer, device_ids] = host[host_ids, layer, 0]
            else:
                result[host_ids, layer, 0] = device[layer, device_ids].cpu()
    # Split at registration boundaries without touching other components or skipped layers.
    split = hosts[0].data_ptr() + 4 * 16 * 2 + 20
    monkeypatch.setattr(
        direct,
        "_registration_bounds",
        lambda t: (
            (
                [t.data_ptr(), split],
                [split, t.data_ptr() + t.numel() * t.element_size()],
            )
            if t is hosts[0]
            else ([], [])
        ),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        count, largest = transfer_mamba_block(
            hosts, devices, host_ids, device_ids, layers, host_to_device
        )
    stream.synchronize()
    layer_runs = 1 if len(layers) == 4 else 2
    assert count == len(devices) * len(host_ids) * layer_runs + 1
    assert largest == (4 if len(layers) == 4 else 2) * 128
    for actual, result in zip(devices if host_to_device else hosts, expected):
        torch.testing.assert_close(actual.cpu(), result, rtol=0, atol=0)
    if not host_to_device:
        assert torch.equal(raw, raw_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_page_first_mamba_rejects_alias_before_copy():
    host = torch.ones((6, 3, 1, 16), pin_memory=True)
    device = torch.zeros((6, 3, 16), device="cuda").transpose(0, 1)
    ids = torch.tensor([1, 4])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), pytest.raises(RuntimeError, match="overlap"):
        transfer_mamba_block([host, host], [device, device], ids, ids, [0, 1, 2], True)
    stream.synchronize()
    assert torch.count_nonzero(device).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("logical_layers", [(0, 1, 2), (1, 3, 5)])
@pytest.mark.parametrize("block_layers", [1, 2])
def test_mamba_layer_block_completion_and_slot_operations(
    mamba_pools, logical_layers, block_layers
):
    host, device = mamba_pools
    hosts = [*host.conv_buffer, host.temporal_buffer]
    devices = [*device.mamba_cache.conv, device.mamba_cache.temporal]
    for buf in hosts:
        buf.random_(1, 50)
    host_ids, device_ids = torch.tensor([1, 4]), torch.tensor([2, 5])
    num_layers = logical_layers[-1] + 1
    layer_map = {logical: local for local, logical in enumerate(logical_layers)}
    counter = LayerDoneCounter(num_layers, block_layers=block_layers)
    producer = counter.events[counter.update_producer()]
    engine = L2TransferEngine("direct")
    engine.enable_batch_direct()
    completed = []

    def complete(layer):
        producer.complete(layer)
        completed.append(layer)
        producer.load_events[counter.layer_to_block[layer]].synchronize()
        for src, dst in zip(hosts, devices):
            expected = torch.zeros_like(dst, device="cpu")
            ready = sum(logical <= layer for logical in logical_layers)
            expected[:ready, device_ids] = src[host_ids, :ready, 0].transpose(0, 1)
            torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)

    ack = engine.submit_host_to_device(
        [L2Transfer(host, device, host_ids, device_ids, layer_mapper=layer_map.get)],
        transfer_layer_id_max=num_layers,
        layer_block_ends=counter.block_ends,
        on_layer_done=complete,
    )
    ack.finish_event.synchronize()
    assert completed == [end - 1 for end in counter.block_ends]
    source = device_ids.cuda()
    destination = torch.tensor([7, 9], device="cuda")
    expected = [t.clone() for t in devices]
    device.copy_from(source, destination)
    for t in expected:
        t[:, destination] = t[:, source]
    device.clear_slots(source)
    for actual, result in zip(devices, expected):
        result[:, source] = 0
        torch.testing.assert_close(actual, result, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize(
    "unsupported", [None, "layout", "draft", "sharded", "hnd", "relocating"]
)
def test_batch_direct_keeps_unsupported_pools_on_existing_path(
    mxfp8_pools, monkeypatch, unsupported
):
    from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool

    host, device = mxfp8_pools
    if unsupported == "layout":
        monkeypatch.setattr(host, "layout", "page_first")
    elif unsupported == "draft":
        monkeypatch.setattr(host, "mtp_draft_device_pools", [device])
    elif unsupported == "sharded":
        monkeypatch.setattr(device, "layer_shard_enabled", True)
    elif unsupported == "hnd":
        monkeypatch.setattr(device, "kv_cache_layout", "hnd")
    elif unsupported == "relocating":
        monkeypatch.setattr(host, "device_pool", object.__new__(UnifiedKVPool))
    engine = L2TransferEngine("direct", host)
    assert engine.batch_direct is (unsupported is None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_mamba_side_state_preserves_existing_transfer_path(mamba_pools, monkeypatch):
    host, _ = mamba_pools
    assert L2TransferEngine("direct", host).batch_direct
    monkeypatch.setattr(host, "slot_state_buffers", (torch.empty(1),))
    assert not L2TransferEngine("direct", host).batch_direct


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_dynamic_sidecar_reselects_transfer_path(mxfp8_pools):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry

    host, device = mxfp8_pools
    group = HostPoolGroup([PoolEntry(PoolName.KV, host, device, None)])
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = group
    controller.extra_host_mem_release_queues = {}
    controller.enable_storage = False
    controller.l2_transfer_engine = L2TransferEngine("direct", group)
    assert controller.l2_transfer_engine.batch_direct
    # An unsupported sidecar added after the controller was created must not
    # enter the static-pool planner or leave cached layouts alive.
    controller.l2_transfer_engine._direct_layouts[True] = object()
    sidecar = SimpleNamespace(can_use_write_back_jit=False)
    controller.register_host_pool_entry(
        PoolEntry(PoolName.DRAFT, sidecar, device, None)
    )
    assert not controller.l2_transfer_engine.batch_direct
    assert not controller.l2_transfer_engine._direct_layouts
