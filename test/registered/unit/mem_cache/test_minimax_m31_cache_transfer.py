"""NVFP4 cache transport preserves every payload byte, block scale and cached prefix."""

import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.minimax import minimax_index_transfer_pages
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    get_kv_transfer_buf_infos,
    setup_state_kv_args,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_auto_size import auto_size_hicache
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.pool_host.base import (
    host_memory_budget_bytes,
    host_memory_budget_scope,
)
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.minimax_m3_nvfp4 import (
    MiniMaxM3Nvfp4IndexKPoolHost,
    MiniMaxM3Nvfp4MainKVPoolHost,
)
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def token_mapping(pages, length):
    return (torch.tensor(pages)[:, None] * 128 + torch.arange(128)).flatten()[:length]


def make_pool(start_layer=0, layer_num=2):
    layers = list(range(start_layer, start_layer + layer_num))
    return MiniMaxM3Nvfp4KVPool(
        size=768,
        page_size=128,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=layers,
        disable_value_sparse_layer_ids=layers,
        device="cpu",
        start_layer=start_layer,
        end_layer=start_layer + layer_num,
    )


_MAIN_HOST = (
    "sglang.srt.mem_cache.pool_host.minimax_m3_nvfp4.MiniMaxM3Nvfp4MainKVPoolHost"
)


def make_main_host(main_pool, layout):
    return MiniMaxM3Nvfp4MainKVPoolHost(
        main_pool.get_device_pool_infos()[0].buffer_info,
        2,
        0,
        128,
        layout,
        pin_memory=False,
        device_capacity_tokens=main_pool.size,
        start_layer=main_pool.start_layer,
        end_layer=main_pool.end_layer,
    )


def make_host_pools(pool, layout):
    main = make_main_host(pool.main_pool, layout)
    index = MiniMaxM3Nvfp4IndexKPoolHost(
        pool.index_k_pool.get_device_pool_infos()[0].buffer_info,
        main,
        layout,
        pin_memory=False,
        start_layer=pool.index_k_pool.start_layer,
        end_layer=pool.index_k_pool.end_layer,
    )
    return main, index


def fill_pool(pool):
    for sub_pool in (pool.main_pool, pool.index_k_pool):
        sub_pool.slab.random_(0, 256)


def test_page_rows_pack_components_at_aligned_offsets():
    pool = make_pool()
    for sub_pool, components in (
        (pool.main_pool, ("k_buffer", "k_scale_buffer", "v_buffer", "v_scale_buffer")),
        (pool.index_k_pool, ("k_buffer", "k_scale_buffer")),
    ):
        assert sub_pool.row_bytes % 128 == 0
        assert sub_pool.get_kv_size_bytes() == (sub_pool.slab.nbytes, 0)
        end = 0
        for name in components:
            for layer, view in enumerate(getattr(sub_pool, name)):
                row = sub_pool.rows[layer]
                offset = view.data_ptr() - row.data_ptr()
                assert view.stride(0) == sub_pool.row_bytes
                assert offset % 128 == 0 and offset >= end
                assert offset + view[0].nbytes <= sub_pool.row_bytes
            end = offset + view[0].nbytes
        # Components never overlap: filling each view sets exactly its own bytes.
        sub_pool.slab.zero_()
        for name in components:
            for view in getattr(sub_pool, name):
                view.add_(1)
        assert int(sub_pool.slab.max()) == 1
        assert int(sub_pool.slab.long().sum()) == sum(
            buf.numel() for name in components for buf in getattr(sub_pool, name)
        )


class CopyingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp
    get_mla_kv_ptrs_with_pp = CommonKVManager.get_mla_kv_ptrs_with_pp
    is_mla_backend = False
    is_hybrid_mla_backend = False
    enable_custom_mem_pool = False
    pp_size = 1
    max_transfer_batch_indices = 0
    kv_args = SimpleNamespace(prefill_start_layer=0)

    def _transfer_data(self, session, blocks):
        for src, dst, size in blocks:
            ctypes.memmove(dst, src, size)
        return 0


def static_storage(pool) -> StaticKVStorage:
    """The storage a static pool registers with PD peers through."""
    allocator = SimpleNamespace(storage=None, state=None, get_kvcache=lambda: pool)
    return StaticKVStorage(None, allocator)


@pytest.mark.parametrize("length,prefix", [(257, 0), (257, 128), (256, 256)])
def test_pd_transfer_sends_the_suffix_with_all_scales(length, prefix):
    src, dst = make_pool(), make_pool()
    fill_pool(src)
    for sub_pool in (dst.main_pool, dst.index_k_pool):
        sub_pool.slab.fill_(173)
    source_pages = minimax_index_transfer_pages(
        token_mapping([1, 3, 2], length), page_size=128, prefix_len=prefix
    )
    destination_slots = token_mapping([4, 5, 6], length)
    destination_slots[:prefix] = -1  # L2-hit slots may not be restored yet.
    destination_pages = minimax_index_transfer_pages(
        destination_slots, page_size=128, prefix_len=prefix
    )
    src_storage, dst_storage = static_storage(src), static_storage(dst)
    src_args, dst_args = KVArgs(), KVArgs()
    setup_state_kv_args(src_args, src_storage)
    setup_state_kv_args(dst_args, dst_storage)
    assert src_args.state_types == [StateType.MINIMAX_INDEX_K]

    def state(args):
        return args.state_data_ptrs[0], args.state_data_lens[0], args.state_item_lens[0]

    def kv_layer_ids(pool, storage):
        return build_kv_layer_ids(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=None,
            num_draft_entries=0,
            num_hidden_layers=2,
            storage=storage,
        )

    # Main rows pair by layer id like production; the index state is flat.
    channels = [
        (
            get_kv_transfer_buf_infos(src_storage),
            get_kv_transfer_buf_infos(dst_storage),
            False,
            kv_layer_ids(src, src_storage),
            kv_layer_ids(dst, dst_storage),
        ),
        (state(src_args), state(dst_args), True, None, None),
    ]
    assert channels[0][3] == [0, 1]
    for (sp, sl, si), (dp, dl, di), flat, src_ids, dst_ids in channels:
        assert si == di and sl == dl
        assert (
            MooncakeKVManager._send_kvcache_generic(
                CopyingKVManager(),
                mooncake_session_id="cpu-test",
                src_data_ptrs=sp,
                dst_data_ptrs=dp,
                item_lens=si,
                prefill_data_indices=source_pages,
                dst_data_indices=destination_pages,
                executor=None,
                force_flat=flat,
                src_layer_ids=src_ids,
                dst_layer_ids=dst_ids,
            )
            == 0
        )
    copied = dict(zip(destination_pages.tolist(), source_pages.tolist()))
    for sub in ("main_pool", "index_k_pool"):
        before, after = getattr(src, sub).slab, getattr(dst, sub).slab
        for page in range(after.shape[1]):
            if page in copied:
                assert torch.equal(before[:, copied[page]], after[:, page])
            else:
                assert torch.all(after[:, page] == 173)


def test_transfer_prefix_must_be_in_range_and_page_aligned():
    slots = torch.arange(256)
    for prefix in (-128, 127, 384):
        with pytest.raises(ValueError):
            minimax_index_transfer_pages(slots, page_size=128, prefix_len=prefix)


@pytest.mark.parametrize("length,prefix", [(256, 256), (385, 128)])
def test_final_chunk_skips_decode_hit_but_sends_aux_on_full_hit(length, prefix):
    sender = Mock(supports_host_tensor_indices=False)
    sender.should_send_kv_chunk.return_value = True
    sender.get_max_transfer_tokens.return_value = None
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=128, translate_kv_indices_for_transfer=lambda x: x
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=token_mapping([2, 4, 1, 3], length).unsqueeze(0)
        ),
        enable_staging=False,
        disagg_metadata_buffers=Mock(),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.MINIMAX_INDEX_K])
            )
        ),
        disagg_prefill_pending_chunk_rids=set(),
    )
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=0),
        origin_input_ids=list(range(length)),
        rid="prefix-hit",
        start_send_idx=prefix,
        disagg_decode_prefix_len=prefix,
        extend_end=length,
        disagg_kv_sender=sender,
    )
    send_chunk = SchedulerDisaggregationPrefillMixin._send_kv_chunk
    if length == 385:
        send_chunk(scheduler, req, last_chunk=False, end_idx=384)
        early_pages, early_state = sender.send.call_args.args
        np.testing.assert_array_equal(early_pages, [4, 1])
        assert early_state is None
        scheduler.disagg_metadata_buffers.set_buf.assert_not_called()
    send_chunk(scheduler, req, last_chunk=True)
    main_pages, state_pages = sender.send.call_args.args
    if length == 385:
        np.testing.assert_array_equal(main_pages, [3])
        np.testing.assert_array_equal(state_pages[0], [4, 1, 3])
    else:
        assert len(main_pages) == 0 and len(state_pages[0]) == 0
    scheduler.disagg_metadata_buffers.set_buf.assert_called_once()


@pytest.mark.parametrize("start_layer", [0, 4])
def test_retraction_round_trip_preserves_unaligned_rows_and_scales(start_layer):
    pool = make_pool(start_layer)
    fill_pool(pool)
    src = torch.tensor([127, 128, 131, 259, 514])
    dst = torch.tensor([520, 600, 621, 700, 750])
    for sub_pool in (pool.main_pool, pool.index_k_pool):
        sub_pool.cpu_offloading_chunk_size = 2
    with patch(
        "sglang.srt.mem_cache.minimax_m3_nvfp4_pool.current_platform.synchronize"
    ):
        snapshot = pool.get_cpu_copy(src, req_pool_index=0)
        pool.load_cpu_copy(snapshot, dst, req_pool_index=0)
    for sub_pool in (pool.main_pool, pool.index_k_pool):
        for layer in range(
            sub_pool.start_layer, sub_pool.start_layer + sub_pool.layer_num
        ):
            expected = sub_pool.get_quantized_rows(layer, src)
            actual = sub_pool.get_quantized_rows(layer, dst)
            for left, right in zip(expected, actual):
                assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("layout", ["layer_first", "page_first", "page_first_direct"])
@pytest.mark.parametrize("backend", ["kernel", "direct"])
def test_host_round_trip_preserves_pages_and_other_layers(layout, backend):
    pool = make_pool()
    fill_pool(pool)
    sub_pools = (pool.main_pool, pool.index_k_pool)
    originals = [sub.slab[:, [1, 3]].clone() for sub in sub_pools]
    main, index = make_host_pools(pool, layout)
    try:
        device_indices = torch.cat([torch.arange(128, 256), torch.arange(384, 512)])
        host_indices = torch.cat([torch.arange(256, 384), torch.arange(640, 768)])
        target_indices = torch.arange(512, 768)
        for host, device in ((main, pool.main_pool), (index, pool.index_k_pool)):
            host.backup_from_device_all_layer(
                device, host_indices, device_indices, backend
            )
        for sub in sub_pools:
            sub.slab.zero_()
        for host, device in ((main, pool.main_pool), (index, pool.index_k_pool)):
            load = host.prepare_layer_loader(
                device, host_indices, target_indices, backend
            )
            load(1)
            assert torch.count_nonzero(device.rows[0]) == 0
            load(0)
        # Whole rows, padding included, land on the target pages only.
        for expected, sub in zip(originals, sub_pools):
            assert torch.equal(expected, sub.slab[:, [4, 5]])
            assert torch.count_nonzero(sub.slab[:, [0, 1, 2, 3, 6]]) == 0
    finally:
        index.destroy()
        main.destroy()


def test_invalid_host_pages_fail_before_writing():
    pool = make_pool()
    host = make_main_host(pool.main_pool, "layer_first")
    try:
        valid = torch.arange(128)
        bad_indices = [
            torch.arange(127),
            torch.arange(1, 129),
            valid.flip(0),
            valid - 128,
            valid + host.size,
            valid.float(),
        ]
        for indices in bad_indices:
            with pytest.raises(ValueError):
                host.backup_from_device_all_layer(
                    pool.main_pool, indices, valid, "kernel"
                )
        assert torch.count_nonzero(pool.main_pool.slab) == 0
    finally:
        host.destroy()


def storage_config(model_name="m31-nvfp4"):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


@pytest.mark.parametrize("layout", ["layer_first", "page_first", "page_first_direct"])
def test_file_storage_round_trip_restores_main_and_index_rows(layout, tmp_path):
    pool = make_pool()
    fill_pool(pool)
    sub_pools = (pool.main_pool, pool.index_k_pool)
    originals = [sub.slab[:, [1, 3]].clone() for sub in sub_pools]
    keys = ["first", "second"]
    main, index = make_host_pools(pool, layout)
    try:
        device_indices = torch.cat([torch.arange(128, 256), torch.arange(384, 512)])
        host_indices = torch.cat([torch.arange(256, 384), torch.arange(640, 768)])
        for host, device in ((main, pool.main_pool), (index, pool.index_k_pool)):
            host.backup_from_device_all_layer(
                device, host_indices, device_indices, "kernel"
            )
        writer = HiCacheFile(storage_config(), file_path=str(tmp_path))
        writer.register_mem_host_pool_v2(index, PoolName.INDEXER)
        # The controller writes the KV anchor page by page and the KV-derived
        # index sidecar through the v2 pool transfers.
        for i, key in enumerate(keys):
            assert writer.set(key, main.get_data_page(host_indices[i * 128]))
        transfer = PoolTransfer(PoolName.INDEXER, host_indices=host_indices, keys=keys)
        assert all(writer.batch_set_v2([transfer])[PoolName.INDEXER])

        for host in (main, index):
            host.kv_buffer.zero_()
        for sub in sub_pools:
            sub.slab.zero_()

        reader = HiCacheFile(storage_config(), file_path=str(tmp_path))
        reader.register_mem_host_pool_v2(index, PoolName.INDEXER)
        assert reader.batch_exists_v2(keys, [transfer]).kv_hit_pages == len(keys)
        restore_indices = torch.cat([torch.arange(0, 128), torch.arange(384, 512)])
        for i, key in enumerate(keys):
            page = reader.get(key, main.get_dummy_flat_data_page())
            main.set_from_flat_data_page(restore_indices[i * 128], page)
        transfer.host_indices = restore_indices
        assert all(reader.batch_get_v2([transfer])[PoolName.INDEXER])

        target_indices = torch.arange(512, 768)
        for host, device in ((main, pool.main_pool), (index, pool.index_k_pool)):
            load = host.prepare_layer_loader(
                device, restore_indices, target_indices, "kernel"
            )
            for layer in range(host.layer_num):
                load(layer)
        # Whole rows, padding included, land on the target pages only.
        for expected, sub in zip(originals, sub_pools):
            assert torch.equal(expected, sub.slab[:, [4, 5]])
            assert torch.count_nonzero(sub.slab[:, [0, 1, 2, 3, 6]]) == 0
    finally:
        index.destroy()
        main.destroy()


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
def test_zero_copy_page_meta_splits_main_rows_into_kv_objects(layout):
    pool = make_pool()
    fill_pool(pool)
    main, index = make_host_pools(pool, layout)
    try:
        device_pages = [1, 3]
        host_indices = torch.cat([torch.arange(256, 384), torch.arange(640, 768)])
        device_indices = token_mapping(device_pages, 256)
        for host, device in ((main, pool.main_pool), (index, pool.index_k_pool)):
            host.backup_from_device_all_layer(
                device, host_indices, device_indices, "kernel"
            )

        # MHA backends store the anchor page as a K and a V object; together
        # they cover the page's layer rows in order.
        ptrs, sizes = main.get_page_buffer_meta(host_indices)
        assert len(ptrs) == len(sizes) == 2 * len(device_pages)
        for i, page in enumerate(device_pages):
            stored = b"".join(
                ctypes.string_at(ptr, size)
                for ptr, size in zip(ptrs[2 * i : 2 * i + 2], sizes[2 * i : 2 * i + 2])
            )
            assert stored == pool.main_pool.slab[:, page].numpy().tobytes()

        # The index page is one contiguous object.
        ptrs, sizes = index.get_page_buffer_meta(host_indices)
        assert sizes == [index.page_bytes] * len(device_pages)
        for ptr, size, page in zip(ptrs, sizes, device_pages):
            assert (
                ctypes.string_at(ptr, size)
                == pool.index_k_pool.slab[:, page].numpy().tobytes()
            )
        with pytest.raises(ValueError, match="full pages"):
            main.get_page_buffer_meta(host_indices[:100])
    finally:
        index.destroy()
        main.destroy()


@pytest.mark.parametrize("case", ["pp", "layout", "budget"])
def test_unsupported_stack_fails_before_allocating_host_memory(case):
    pool = make_pool()
    params = CacheInitParams(False, None, None, 128, pp_size=2 if case == "pp" else 1)
    memory = SimpleNamespace(
        hicache_size=0,
        hicache_ratio=2,
        hicache_io_backend="kernel",
        hicache_mem_layout="page_head" if case == "layout" else "layer_first",
    )
    with (
        patch.object(assembler, "get_memory", return_value=memory),
        patch(_MAIN_HOST) as allocate,
        patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=1 if case == "budget" else 10**9,
        ),
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            build_stack(params, pool)
        allocate.assert_not_called()


def dspark_params(target, draft):
    return CacheInitParams(
        False,
        # A request table without recurrent-state slots.
        SimpleNamespace(holds_state_slots=False),
        # A static allocator: no unified KV storage.
        SimpleNamespace(get_kvcache=lambda: target, storage=None),
        128,
        sidecar_draft_device_pools=(draft,),
    )


def build_stack(params, target, storage_backend=None):
    return assembler.build_minimax_sparse_hicache_stack(
        params=params,
        sparse_pool=target,
        load_cache_event=None,
        storage_backend=storage_backend,
    )


def make_draft_pool(size=768, page_size=128, layer_num=5):
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )


@pytest.fixture
def cpu_host_allocation(monkeypatch):
    # Keep CPU tests independent of CUDA host registration and DMA streams.
    monkeypatch.setattr(
        "sglang.srt.mem_cache.pool_host.base._cuda_host_unregister", lambda buffer: None
    )
    monkeypatch.setitem(
        ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, dtype, device, **kwargs: torch.empty(
            dims, dtype=dtype, device=device
        ),
    )
    monkeypatch.setattr(
        assembler,
        "HybridCacheController",
        lambda allocator, group, page_size, tp_group, **kwargs: SimpleNamespace(
            mem_pool_host=group,
            page_size=page_size,
            transfer_layer_id_max=kwargs["transfer_layer_id_max"],
        ),
    )


@pytest.mark.parametrize(
    "layout,mode",
    [
        ("layer_first", "fixed"),
        ("layer_first", "ratio"),
        ("layer_first", "small"),
        ("page_first_direct", "ratio"),
    ],
)
def test_dspark_hicache_shares_capacity_and_total_budget(
    cpu_host_allocation, layout, mode
):
    target, draft = make_pool(layer_num=6), make_draft_pool()
    params = dspark_params(target, draft)
    # Six NVFP4 layers: main K/V (two heads), K-only index (one head),
    # each with one byte of scale for every eight payload bytes. Five BF16 draft layers.
    bytes_per_token = 6 * (2 * 2 + 1) * (64 + 8) + 5 * 2 * (128 + 128) * 2
    expected_slots = 128 if mode == "small" else 512
    budget = expected_slots * bytes_per_token
    with (
        get_context().override_server_args(
            hicache_size=500 * bytes_per_token / 1e9 if mode == "fixed" else 0,
            hicache_ratio=0.1 if mode == "small" else 0.5,
            hicache_io_backend="direct" if layout == "page_first_direct" else "kernel",
            hicache_mem_layout=layout,
        ),
        host_memory_budget_scope(budget),
    ):
        group, controller = build_stack(params, target)
        specs, entries = assembler.build_full_draft_pools(
            draft_kv_pool=draft, tree_cache=SimpleNamespace(cache_controller=controller)
        )
        assert specs[0].indices_from_pool == PoolName.KV
        for entry in entries:
            group.add_entry(entry)
        try:
            assert {entry.host_pool.size for entry in group.entries} == {expected_slots}
            assert (
                sum(entry.host_pool.kv_buffer.nbytes for entry in group.entries)
                == budget
            )
            assert host_memory_budget_bytes() == 0
            draft_entry = group.get_entry(PoolName.DRAFT)
            layers = [draft_entry.layer_mapper(i) for i in range(6)]
            assert layers == [0, 1, 2, 3, 4, None]
            slots = group.alloc(expected_slots)
            assert slots is not None and len(slots) == expected_slots
            assert group.alloc(128) is None
            group.free(slots)
            assert group.available_size() == expected_slots
        finally:
            for entry in group.entries:
                entry.host_pool.destroy()


def test_dspark_hicache_budget_rejected_before_any_host_allocation():
    target, draft = make_pool(layer_num=6), make_draft_pool()
    params = dspark_params(target, draft)
    bytes_per_token = 6 * 5 * 72 + 5 * 2 * 256 * 2
    with (
        get_context().override_server_args(hicache_ratio=0.5, hicache_size=0),
        host_memory_budget_scope(512 * bytes_per_token - 1),
        patch(_MAIN_HOST) as allocate,
        pytest.raises(ValueError, match="Not enough host memory"),
    ):
        build_stack(params, target)
    allocate.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"size": 640}, {"page_size": 64}, {"layer_num": 7}])
def test_dspark_hicache_rejects_incompatible_draft_before_allocation(kwargs):
    target, draft = make_pool(layer_num=6), make_draft_pool(**kwargs)
    params = dspark_params(target, draft)
    with (
        get_context().override_server_args(hicache_size=0, hicache_ratio=0.5),
        patch(_MAIN_HOST) as allocate,
        pytest.raises(ValueError, match="dense MHA draft"),
    ):
        build_stack(params, target)
    allocate.assert_not_called()


def test_draft_host_capacity_uses_dcp_logical_slots(cpu_host_allocation):
    draft = make_draft_pool()
    controller = SimpleNamespace(
        mem_pool_host=SimpleNamespace(size=256, logical_size=512),
        page_size=256,
        transfer_layer_id_max=6,
    )
    with get_context().override_server_args(hicache_mem_layout="layer_first"):
        _, entries = assembler.build_full_draft_pools(
            draft_kv_pool=draft, tree_cache=SimpleNamespace(cache_controller=controller)
        )
    try:
        assert entries[0].host_pool.size == 512
        assert entries[0].host_pool.page_num == 2
    finally:
        entries[0].host_pool.destroy()


def test_draft_layers_must_fit_target_transfer_stages(cpu_host_allocation):
    controller = SimpleNamespace(
        mem_pool_host=SimpleNamespace(logical_size=512),
        page_size=128,
        transfer_layer_id_max=4,
    )
    with (
        patch.object(assembler, "_build_mha_mla_host_pool") as allocate,
        pytest.raises(ValueError, match="exceed the target transfer stages"),
    ):
        assembler.build_full_draft_pools(
            draft_kv_pool=make_draft_pool(),
            tree_cache=SimpleNamespace(cache_controller=controller),
        )
    allocate.assert_not_called()


def test_dspark_default_hicache_auto_size_fits_all_pools(cpu_host_allocation):
    target, draft = make_pool(layer_num=6), make_draft_pool()
    params = dspark_params(target, draft)
    plan = SimpleNamespace(mode="sidecar", device_pools=(draft,))
    rank_budget = 4_000_000
    with (
        get_context().override_server_args(enable_hierarchical_cache=True),
        patch.object(base, "ranks_per_host", return_value=1),
        patch.object(
            base,
            "available_host_memory_bytes",
            return_value=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + rank_budget,
        ),
        auto_size_hicache(params, plan, enabled=True),
    ):
        group, controller = build_stack(params, target)
        _, entries = assembler.build_full_draft_pools(
            draft_kv_pool=draft, tree_cache=SimpleNamespace(cache_controller=controller)
        )
        for entry in entries:
            group.add_entry(entry)
        try:
            actual_bytes = sum(
                entry.host_pool.kv_buffer.nbytes for entry in group.entries
            )
            assert {entry.host_pool.size for entry in group.entries} == {384}
            assert actual_bytes <= 0.8 * rank_budget
            assert host_memory_budget_bytes() == int(0.8 * rank_budget) - actual_bytes
        finally:
            for entry in group.entries:
                entry.host_pool.destroy()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
