import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.pool_host.host_pool_decl import make_kv_pool_decl
from sglang.srt.mem_cache.pool_host.qsa import qsa_indexer_bytes_per_token_per_layer
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _qsa_pool_stub(*, full_layers, ratio=4, start_layer=0):
    """QSATokenToKVPool shape without CUDA: a full-KV sub-pool plus compressed
    key buffers on the hybrid pool itself."""
    pool = object.__new__(QSATokenToKVPool)
    full = SimpleNamespace(layer_num=len(full_layers), size=256)
    full.host_pool_decls = lambda: (make_kv_pool_decl(full),)
    pool.full_kv_pool = full
    pool.start_layer = start_layer
    pool.full_attention_layer_id_mapping = {
        layer: i for i, layer in enumerate(full_layers)
    }
    pool.qsa_index_kv_heads = 1
    pool.qsa_index_head_dim = 128
    pool.qsa_compress_ratio = ratio
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(
            full.size // ratio, 1, 128, dtype=QSATokenToKVPool.index_state_dtype
        )
        for _ in full_layers
    ]
    pool.layer_transfer_counter = None
    return pool


def test_compressed_key_getter_waits_for_the_layer_transfer():
    """The indexer reads compressed keys before attention reads full KV; a
    host restore still in flight must be fenced at this getter too (#39830)."""
    pool = _qsa_pool_stub(full_layers=[7, 11], start_layer=4)
    pool.layer_transfer_counter = Mock()

    buffer = pool.get_qsa_compressed_k_buffer(11)

    pool.layer_transfer_counter.wait_until.assert_called_once_with(11 - 4)
    assert buffer is pool.qsa_compressed_k_buffer_pool[1]


def test_host_pool_decls_put_kv_on_the_sub_pool_and_compressed_keys_on_the_hybrid():
    pool = _qsa_pool_stub(full_layers=[3, 7])

    kv, indexer = pool.host_pool_decls()

    assert (kv.pool_name, kv.device_pool) == (PoolName.KV, pool.full_kv_pool)
    assert (indexer.pool_name, indexer.device_pool) == (PoolName.INDEXER, pool)
    assert (indexer.indices_from_pool, indexer.layout_source) == (
        PoolName.KV,
        PoolName.KV,
    )
    # kv_heads 1 x head_dim 128 x bf16 = 256 B per group of 4 tokens; one
    # 64-token page is 16 groups = 4096 B, the byte row the mirror moves.
    assert indexer.storage_info.bytes_per_token_per_layer == 64
    assert indexer.storage_info.page_bytes(64) == 4096


def test_compressed_group_must_split_into_whole_bytes_per_token():
    with pytest.raises(ValueError, match="compress ratio 3"):
        qsa_indexer_bytes_per_token_per_layer(
            kv_heads=1, head_dim=128, compress_ratio=3, dtype=torch.bfloat16
        )


def test_qsa_allocations_follow_parent_mooncake_scope(monkeypatch):
    active_scopes = set()
    allocations = 0
    original_zeros = torch.zeros

    @contextmanager
    def scope(name):
        active_scopes.add(name)
        try:
            yield
        finally:
            active_scopes.remove(name)

    def init_parent(pool, **_):
        pool.full_kv_pool = SimpleNamespace(
            memory_saver_adapter=SimpleNamespace(
                region=lambda _: scope("memory_saver")
            ),
            enable_custom_mem_pool=True,
            custom_mem_pool=object(),
        )

    def allocate(*args, **kwargs):
        nonlocal allocations
        assert active_scopes == {"memory_saver", "custom_pool"}
        allocations += 1
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(HybridLinearKVPool, "__init__", init_parent)
    monkeypatch.setattr(QSATokenToKVPool, "get_kv_size_bytes", lambda _: (0, 0))
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _: scope("custom_pool"))
    monkeypatch.setattr("sglang.srt.mem_cache.qsa_kv_pool.torch.zeros", allocate)

    QSATokenToKVPool(
        size=8,
        dtype=torch.bfloat16,
        page_size=4,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=[1, 3],
        device="cpu",
        mamba_pool=object(),
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
        qsa_compress_ratio=2,
        qsa_token_topk=4,
        num_request_slots=3,
    )

    assert allocations == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
