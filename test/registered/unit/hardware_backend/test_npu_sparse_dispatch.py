"""CPU contracts for selecting the NPU sparse-KV implementations."""

import sys
from types import ModuleType

import pytest

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import config, dispatch


@pytest.mark.parametrize("fused", [False, True])
def test_manager_and_attention_follow_one_switch(monkeypatch, fused):
    prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
    calls = []
    expected = {}
    for mode, manager_name, attention_name in (
        (False, "manager", "attention"),
        (True, "fused_manager", "fused_attention"),
    ):
        manager_module = ModuleType(prefix + manager_name)
        attention_module = ModuleType(prefix + attention_name)

        def manager(*args, _mode=mode, **kwargs):
            calls.append((_mode, args, kwargs))
            return _mode

        def attention(*args, _mode=mode):
            return _mode

        manager_module.SparseKVCacheManager = manager
        attention_module.forward_sparsity_driven_kv_offload = attention
        monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
        monkeypatch.setitem(sys.modules, attention_module.__name__, attention_module)
        expected[mode] = attention

    result = dispatch.create_sparse_kv_manager(
        "pool", "allocator", 2048, fused=fused, options={"graph_batch_capacity": 4}
    )
    assert result is fused
    assert calls == [
        (
            fused,
            ("pool", "allocator", 2048),
            {"graph_batch_capacity": 4},
        )
    ]
    assert dispatch.get_sparse_kv_attention(fused=fused) is expected[fused]


def test_legacy_budget_preserves_topk_capacity(monkeypatch):
    monkeypatch.setattr(config, "get_eager_max_batch_size", lambda _: 4)
    with envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(False):
        budget = config.get_sparse_kv_budget(
            request_rows=3,
            context_len=1024,
            num_layers=2,
            top_k=512,
            kv_bytes_per_token=1152,
            graph_batch_capacity=4,
        )
    assert budget.cache_tokens == 512
    assert budget.device_kv_bytes == 3 * 2 * 512 * 1152
    assert budget.slot_map_bytes == 2 * 4 * ((1024 // 8 + 1) * 8) * 4
    assert budget.managed_hbm_bytes >= budget.device_kv_bytes + budget.slot_map_bytes


def test_fused_budget_keeps_its_constraints(monkeypatch):
    monkeypatch.setattr(config, "get_eager_max_batch_size", lambda _: 4)
    monkeypatch.setattr(config, "get_sparse_kv_max_query_width", lambda: 4)
    with envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(True):
        with pytest.raises(ValueError, match="K=2048"):
            config.get_sparse_kv_budget(
                request_rows=3,
                context_len=1024,
                num_layers=2,
                top_k=512,
                kv_bytes_per_token=1152,
                graph_batch_capacity=4,
            )
