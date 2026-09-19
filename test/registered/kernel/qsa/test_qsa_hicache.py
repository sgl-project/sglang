"""Exercise real QSA host transfers and File L3 with reordered physical pages."""

import tempfile

import pytest
import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE = 64
RATIO = 4


def make_device_pool(layers, start_layer=0):
    return QSATokenToKVPool(
        size=PAGE * 8,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=1,
        head_dim=128,
        full_attention_layer_ids=layers,
        start_layer=start_layer,
        device="cuda",
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=128,
        num_request_slots=4,
    )


@pytest.fixture
def pools(request):
    layout, backend = request.param
    publish(ServerArgs(model_path="dummy"), role="test")
    device = make_device_pool([7, 11], start_layer=4)
    anchor = MHATokenToKVPoolHost(device.full_kv_pool, 2, 0, PAGE, layout)
    host = QSAIndexerPoolHost(device, anchor)
    try:
        yield device, anchor, host, backend
    finally:
        torch.cuda.synchronize()
        host.destroy()
        anchor.destroy()
        reset_context()


def slots(pages, device="cpu"):
    return (
        torch.tensor(pages, device=device)[:, None] * PAGE
        + torch.arange(PAGE, device=device)
    ).reshape(-1)


def compressed_slots(indices):
    return indices[::RATIO] // RATIO


@pytest.mark.parametrize(
    "pools",
    [
        ("layer_first", "kernel"),
        ("page_first", "kernel"),
        ("layer_first", "direct"),
        ("page_first_direct", "direct"),
    ],
    indirect=True,
)
def test_qsa_host_roundtrip_and_consumer_fence(pools):
    device, anchor, host, backend = pools
    source = slots([3, 1], "cuda")
    destination = slots([4, 2], "cuda")
    host_indices = slots([2, 0])
    if backend == "kernel" and not host.can_use_write_back_jit:
        host_indices = host_indices.cuda()
    torch.manual_seed(42)
    device.qsa_compressed_flat.normal_()
    expected = [
        b[compressed_slots(source)].clone() for b in device.qsa_compressed_k_buffer_pool
    ]
    engine = L2TransferEngine(backend)
    transfer = L2Transfer(
        host,
        device,
        host_indices,
        source.cpu() if backend == "direct" else source,
        {3: 0, 7: 1}.get,
    )
    engine.submit_device_to_host([transfer]).finish_event.synchronize()
    device.qsa_compressed_flat.fill_(-99)
    counter = LayerDoneCounter(8)
    producer = counter.update_producer()
    counter.set_consumer(producer)
    device.register_layer_transfer_counter(counter)
    with torch.cuda.stream(engine.host_to_device_stream):
        torch.cuda._sleep(20_000_000)
    engine.submit_host_to_device(
        [
            transfer._replace(
                host_indices=(
                    host_indices.cuda() if backend == "kernel" else host_indices
                ),
                device_indices=(
                    destination.cpu() if backend == "direct" else destination
                ),
            )
        ],
        layer_num=8,
        on_layer_done=counter.events[producer].complete,
    )
    # Read through the production accessor without a global synchronize. The
    # indexer must wait for the stage-local layer event before reading its keys.
    actual = [
        device.get_qsa_compressed_k_buffer(layer)[compressed_slots(destination)]
        for layer in (7, 11)
    ]
    for restored, saved in zip(actual, expected):
        torch.testing.assert_close(restored, saved, rtol=0, atol=0)
    assert host.size == anchor.size
    assert host.size_per_token * PAGE == sum(x[0].nbytes for x in host.device_buffers)


@pytest.mark.parametrize("pools", [("page_first_direct", "direct")], indirect=True)
def test_qsa_file_restart_and_missing_sidecar(pools):
    device, anchor, host, backend = pools
    source, host_indices = slots([1, 3], "cuda"), slots([0, 2])
    device.qsa_compressed_flat.normal_()
    expected = [
        b[compressed_slots(source)].clone() for b in device.qsa_compressed_k_buffer_pool
    ]
    engine = L2TransferEngine(backend)
    engine.submit_device_to_host(
        [L2Transfer(host, device, host_indices, source.cpu())]
    ).finish_event.synchronize()
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="qsa-roundtrip",
    )
    keys = ["first", "second"]
    transfer = PoolTransfer(PoolName.INDEXER, host_indices=host_indices, keys=keys)
    with tempfile.TemporaryDirectory() as directory:
        writer = HiCacheFile(config, file_path=directory)
        writer.register_mem_host_pool_v2(host, PoolName.INDEXER)
        for key in keys:
            assert writer.set(key, anchor.get_data_page(0))
        # Existing KV without compressed keys must never count as a usable hit.
        assert writer.batch_exists_v2(keys, [transfer]).kv_hit_pages == 0
        assert all(writer.batch_set_v2([transfer])[PoolName.INDEXER])
        reader = HiCacheFile(config, file_path=directory)
        reader.register_mem_host_pool_v2(host, PoolName.INDEXER)
        assert reader.batch_exists_v2(keys, [transfer]).kv_hit_pages == len(keys)
        host.kv_buffer.zero_()
        device.qsa_compressed_flat.fill_(-99)
        transfer.host_indices = slots([1, 3])
        assert all(reader.batch_get_v2([transfer])[PoolName.INDEXER])
        destination = slots([4, 2], "cuda")
        engine.submit_host_to_device(
            [L2Transfer(host, device, transfer.host_indices, destination.cpu())],
            layer_num=2,
        ).finish_event.synchronize()
        for restored, saved in zip(device.qsa_compressed_k_buffer_pool, expected):
            torch.testing.assert_close(
                restored[compressed_slots(destination)], saved, rtol=0, atol=0
            )
        pointers, sizes = host.get_page_buffer_meta(transfer.host_indices)
        assert len(pointers) == len(keys)
        assert sizes == [host.get_data_page(0).nbytes] * len(keys)


@pytest.mark.parametrize("pools", [("page_first_direct", "direct")], indirect=True)
def test_packed_mtp_indexer_roundtrip(pools):
    device, anchor, _, backend = pools
    draft = make_device_pool([0])
    host = QSAIndexerPoolHost(device, anchor, mtp_draft_device_pools=(draft,))
    try:
        source, destination, host_indices = slots([3, 1]), slots([4, 2]), slots([0, 2])
        all_buffers = [
            *device.qsa_compressed_k_buffer_pool,
            *draft.qsa_compressed_k_buffer_pool,
        ]
        for buffer in all_buffers:
            buffer.normal_()
        expected = [b[compressed_slots(source).cuda()].clone() for b in all_buffers]
        engine = L2TransferEngine(backend)
        engine.submit_device_to_host(
            [L2Transfer(host, device, host_indices, source)]
        ).finish_event.synchronize()
        # Whole-page storage payload must include the draft as well as target.
        pages = [host.get_data_page(i).clone() for i in (0, 2 * PAGE)]
        host.kv_buffer.zero_()
        for i, page in zip((0, 2 * PAGE), pages):
            host.set_from_flat_data_page(i, page)
        for buffer in all_buffers:
            buffer.fill_(-99)
        engine.submit_host_to_device(
            [
                L2Transfer(host, device, host_indices, destination, {3: 0, 7: 1}.get),
                L2Transfer(
                    host, draft, host_indices, destination, {0: 2}.get, is_draft=True
                ),
            ],
            layer_num=8,
        ).finish_event.synchronize()
        for buffer, saved in zip(all_buffers, expected):
            torch.testing.assert_close(
                buffer[compressed_slots(destination).cuda()], saved, rtol=0, atol=0
            )
    finally:
        host.destroy()
