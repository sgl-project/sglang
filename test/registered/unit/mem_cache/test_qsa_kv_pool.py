import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.kv_cache_configurator import (
    _qsa_cache_sharding_graph_reservation_gb,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_qsa_cache_sharding_budget_covers_graph_and_ple_prefill_output():
    reserve_gb = _qsa_cache_sharding_graph_reservation_gb(
        sharding_size=4,
        is_qsa=True,
        cuda_graph_enabled=True,
        prefill_tokens=8192,
        hidden_size=2560,
        hyper_connection_count=4,
        activation_element_size=2,
    )

    graph_private_gb = 4 * 0.5
    ple_tensor_gb = 8192 * 2560 * 4 * 2 / (1 << 30)
    assert reserve_gb == pytest.approx(graph_private_gb + 3 * ple_tensor_gb)


@pytest.mark.parametrize(
    ("sharding_size", "is_qsa", "cuda_graph_enabled"),
    [(1, True, True), (4, False, True), (4, True, False)],
)
def test_qsa_cache_sharding_budget_preserves_unsharded_or_eager_behavior(
    sharding_size, is_qsa, cuda_graph_enabled
):
    assert (
        _qsa_cache_sharding_graph_reservation_gb(
            sharding_size=sharding_size,
            is_qsa=is_qsa,
            cuda_graph_enabled=cuda_graph_enabled,
            prefill_tokens=8192,
            hidden_size=2560,
            hyper_connection_count=4,
            activation_element_size=2,
        )
        == 0.0
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
