import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dsv4_layout_tag_is_appended_to_mooncake_storage_config(monkeypatch):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
    )

    class PlatformDeepSeekV4TokenToKVPool(DeepSeekV4TokenToKVPool):
        pass

    kvcache = PlatformDeepSeekV4TokenToKVPool.__new__(PlatformDeepSeekV4TokenToKVPool)
    kvcache._unified_kv = False
    kvcache.c4_indexer_kv_pool = SimpleNamespace(use_fp4_indexer=True)

    entry = SimpleNamespace(
        name=PoolName.DEEPSEEK_V4_C4,
        indices_from_pool=PoolName.KV,
        get_hybrid_pool_buffer=lambda: [],
    )
    pool_group = DevicePoolGroup(
        [entry], num_layers=30, page_size=2, rank_replicated=True
    )
    monkeypatch.setattr(
        mooncake_direct_linker,
        "resolve_hybrid_device_pool_group",
        lambda **_: pool_group,
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    captured = {}

    def make_storage_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        mooncake_direct_linker,
        "HiCacheStorageConfig",
        make_storage_config,
    )
    monkeypatch.setattr(
        mooncake_direct_linker,
        "get_memory",
        lambda: SimpleNamespace(
            hicache_storage_backend_extra_config=json.dumps(
                {"extra_backend_tag": "tenant-b", "custom_option": "kept"}
            )
        ),
    )
    monkeypatch.setattr(
        mooncake_direct_linker,
        "get_model",
        lambda: SimpleNamespace(model_path="test-model"),
    )
    monkeypatch.delenv("SGLANG_PP_LAYER_PARTITION", raising=False)

    params = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kvcache),
        tp_cache_group=None,
        attn_tp_cache_group=None,
        pp_rank=1,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        req_to_token_pool=MagicMock(),
    )
    server_args = SimpleNamespace(tp_size=1)
    storage = MagicMock()
    linker = MooncakeDirectLinker(
        server_args,
        params,
        components={ComponentType.FULL},
        storage=storage,
    )
    try:
        assert captured["extra_config"]["custom_option"] == "kept"
        assert captured["extra_config"]["extra_backend_tag"] == (
            "tenant-b__ucdl-dsv4-v1-layout-paged-indexer-fp4-pp2-layers-auto"
        )
    finally:
        linker.close()
