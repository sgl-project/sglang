import json
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_deepseek_v4_device_pool_group,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("split_indexer", [False, True])
def test_unified_deepseek_v4_direct_group_uses_only_compressed_pools(split_indexer):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

    page_size = 2
    c4_buffers = [
        torch.zeros((4, 5), dtype=torch.uint8),
        torch.zeros((4, 7), dtype=torch.uint8),
    ]
    c128_buffers = [torch.zeros((4, 11), dtype=torch.uint8)]
    indexer_buffers = [
        torch.zeros((4, 13), dtype=torch.uint8),
        torch.zeros((4, 17), dtype=torch.uint8),
    ]
    indexer_scale_buffers = None
    if split_indexer:
        payload = [torch.zeros((4, 1, 4, 1, 16), dtype=torch.uint8) for _ in range(2)]
        scale = [torch.zeros((4, 1, 4, 1), dtype=torch.uint8) for _ in range(2)]
        indexer = SimpleNamespace(
            index_k_with_scale_buffer=None,
            index_k_payload_buffer=payload,
            index_k_scale_buffer=scale,
        )
        indexer_buffers = [buffer.flatten(1) for buffer in payload]
        indexer_scale_buffers = [buffer.flatten(1) for buffer in scale]
    else:
        indexer = SimpleNamespace(index_k_with_scale_buffer=indexer_buffers)
    unified_pool = object()

    def unified_region_buffers(ratio):
        return (c4_buffers, 7) if ratio == 4 else (c128_buffers, 11)

    kvcache = SimpleNamespace(
        _unified_kv=True,
        swa_is_index_addressed=False,
        start_layer=0,
        end_layer=3,
        layer_mapping=[
            DeepSeekV4LayerItem(4, 1),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 0),
        ],
        # These pools intentionally do not exist under unified-KV.
        swa_kv_pool=None,
        c4_kv_pool=None,
        c128_kv_pool=None,
        swa_page_size=page_size + 1,
        unified_kv_pool=unified_pool,
        c4_indexer_kv_pool=indexer,
        unified_region_buffers=MagicMock(side_effect=unified_region_buffers),
    )
    group = _build_deepseek_v4_device_pool_group(kvcache, page_size)

    expected_names = {
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
    }
    if split_indexer:
        expected_names.add(PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE)
    assert set(group.entry_map) == expected_names
    assert set(group.sources.values()) == {PoolName.KV}
    assert group.rank_replicated
    assert group.entry_map[PoolName.DEEPSEEK_V4_C4].device_pool is unified_pool
    assert group.entry_map[PoolName.DEEPSEEK_V4_C128].device_pool is unified_pool
    assert all(
        actual is expected
        for actual, expected in zip(
            group.entry_map[PoolName.DEEPSEEK_V4_C4].components[0], c4_buffers
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            group.entry_map[PoolName.DEEPSEEK_V4_C128].components[0], c128_buffers
        )
    )
    assert all(
        actual.data_ptr() == expected.data_ptr() and actual.shape == expected.shape
        for actual, expected in zip(
            group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER].components[0],
            indexer_buffers,
        )
    )
    if split_indexer:
        scale_entry = group.entry_map[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE]
        assert all(
            actual.data_ptr() == expected.data_ptr() and actual.shape == expected.shape
            for actual, expected in zip(
                scale_entry.components[0], indexer_scale_buffers
            )
        )
    assert [item.args for item in kvcache.unified_region_buffers.call_args_list] == [
        (4,),
        (128,),
    ]

    resolved = group.resolve_transfers(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page-0"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert {transfer.name for transfer in resolved} == set(group.entry_map)


def test_deepseek_v4_direct_group_still_rejects_hisparse():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

    kvcache = SimpleNamespace(
        c4_kv_pool=HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
    )

    with pytest.raises(ValueError, match="does not support HiSparse"):
        _build_deepseek_v4_device_pool_group(kvcache, 2)


def test_layout_tag_survives_a_model_whose_ratios_outnumber_its_kv_layers():
    """62 ratios against a 61-layer stage: a real launch failed on a helper that
    derived one from the other. The tag must not, and must match across ranks."""
    import os
    from unittest.mock import patch

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        with_direct_linker_cache_layout_tag,
    )

    # A real instance with __init__ skipped: the helper gates on isinstance().
    kvcache = object.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = True
    kvcache.compression_ratios = [4] * 62
    kvcache._stage_start, kvcache._stage_end = 0, 61
    kvcache.c4_indexer_kv_pool = SimpleNamespace(use_fp4_indexer=False)
    group = SimpleNamespace(entry_map={PoolName.DEEPSEEK_V4_C4: object()})

    for env, pp_size in (("61", 1), ("30,31", 2)):
        with patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": env}):
            tags = {
                with_direct_linker_cache_layout_tag(
                    {},
                    kvcache=kvcache,
                    pool_group=group,
                    pp_rank=rank,
                    pp_size=pp_size,
                )["extra_backend_tag"]
                for rank in range(pp_size)
            }
        assert len(tags) == 1, f"tag differs across PP ranks: {tags}"
        assert env.replace(",", ".") in tags.pop()


@pytest.fixture
def dsv4_layout_pool():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    kvcache = object.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = True
    kvcache.c4_indexer_kv_pool = SimpleNamespace(use_fp4_indexer=False)
    entry = DevicePoolEntry(
        name=PoolName.DEEPSEEK_V4_C4,
        indices_from_pool=PoolName.KV,
        device_pool=kvcache,
        components=[[torch.zeros((32, 1, 4), dtype=torch.uint8) for _ in range(3)]],
        layer_mapping={layer: layer for layer in range(3)},
        page_size=2,
        rows_are_pages=False,
    )
    return kvcache, DevicePoolGroup([entry], 3, 2, rank_replicated=True)


def test_dsv4_layout_tag_is_appended_to_umbp_storage_config(
    monkeypatch, dsv4_layout_pool
):
    from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    kvcache, group = dsv4_layout_pool
    extra_config = {"extra_backend_tag": "tenant-a", "custom_option": "kept"}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("SGLANG_PP_LAYER_PARTITION", raising=False)
    monkeypatch.setattr(
        umbp_direct_linker, "resolve_hybrid_device_pool_group", lambda **_: group
    )
    monkeypatch.setattr(
        umbp_direct_linker,
        "get_memory",
        lambda: SimpleNamespace(
            hicache_storage_backend_extra_config=json.dumps(extra_config)
        ),
    )
    monkeypatch.setattr(
        umbp_direct_linker,
        "get_model",
        lambda: SimpleNamespace(model_path="test-model"),
    )
    monkeypatch.setattr(umbp_direct_linker.device_module, "Event", MagicMock())
    captured = {}

    def make_storage_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(umbp_direct_linker, "HiCacheStorageConfig", make_storage_config)
    mode = Enum("DeploymentMode", ["StandaloneProcess", "Local"])
    storage = MagicMock()
    storage._disable_zero_copy_register = False
    storage.client.get_deployment_mode.return_value = mode.StandaloneProcess
    storage.client.get_backend_mode.return_value = mode.Local
    params = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kvcache),
        tp_cache_group=None,
        pp_rank=1,
        pp_size=4,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    linker = umbp_direct_linker.UMBPDirectLinker(
        SimpleNamespace(tp_size=1),
        params,
        components={ComponentType.FULL},
        _storage=storage,
    )
    try:
        assert captured["extra_config"]["custom_option"] == "kept"
        assert captured["extra_config"]["extra_backend_tag"] == (
            "tenant-a__ucdl-dsv4-v1-layout-unified-bf16-indexer-int8-pp4-layers-auto"
        )
    finally:
        linker.close()


def test_dsv4_layout_tag_is_identical_across_pp_ranks_for_extkv(
    monkeypatch, dsv4_layout_pool
):
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        with_direct_linker_cache_layout_tag,
    )

    kvcache, group = dsv4_layout_pool
    monkeypatch.setenv("SGLANG_PP_LAYER_PARTITION", "30,31")
    tags = [
        with_direct_linker_cache_layout_tag(
            {"extra_backend_tag": "tenant-a"},
            kvcache=kvcache,
            pool_group=group,
            pp_rank=rank,
            pp_size=2,
        )["extra_backend_tag"]
        for rank in range(2)
    ]
    assert tags[0] == tags[1]
    assert tags[0].endswith("-pp2-layers-30.31")
