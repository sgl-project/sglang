import mmap
from functools import partial

import pytest
import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host.direct import (
    load_block_dma_module,
    transfer_layer_block,
    transfer_mamba_block,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
